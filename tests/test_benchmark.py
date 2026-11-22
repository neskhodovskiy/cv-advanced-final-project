import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from benchmark import average_precision, score_pair
from experiment import FIELDS, load_rows, read_image, prepare_clips


class BenchmarkTests(unittest.TestCase):
    def test_ap_with_ties(self):
        self.assertAlmostEqual(average_precision([1, 0, 1], [3, 2, 1]), 5/6)
        self.assertAlmostEqual(average_precision([0, 1], [0, 0]), 0.5)
        self.assertAlmostEqual(average_precision([1, 0], [0, 0]), 0.5)

    def test_missing_flat_and_tiny_frames(self):
        for method in ("ORB", "AKAZE"):
            for image in (None, np.empty((0, 0), np.uint8), np.ones((1, 5), np.uint8),
                          np.zeros((95, 160), np.uint8)):
                self.assertEqual(score_pair(image, image, method, 2)["score"], 0)

    def test_reproducible_identity(self):
        image = np.random.default_rng(42).integers(0, 255, (190, 250), dtype=np.uint8)
        for method in ("ORB", "AKAZE"):
            first = score_pair(image, image, method)
            self.assertGreater(first["score"], 4)
            self.assertEqual(first["points"], score_pair(image, image, method)["points"])

    def test_manifest_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            def write(rows):
                with manifest.open("w", newline="") as f:
                    csv.writer(f).writerows([FIELDS, *rows])
            a = ["sample", "dev", "seq", 0, "q", "p", "n"]
            with patch("experiment.Path.is_file", return_value=True):
                write([a, [*a[:1], "test", *a[2:]]])
                with self.assertRaisesRegex(ValueError, "overlapping"):
                    load_rows(manifest)
                write([[*a[:-1], ""]])
                with self.assertRaisesRegex(ValueError, "Missing annotation"):
                    load_rows(manifest)
            write([a])
            with self.assertRaises(FileNotFoundError):
                load_rows(manifest)

    def test_aligned_clean_and_encoded_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            image = np.random.default_rng(5).integers(0, 255, (64, 120), dtype=np.uint8)
            path = str(folder / "frame.png")
            cv2.imwrite(path, image)
            row = dict(query=path, dataset="fixture", sequence="a")
            [(rows, clips)] = prepare_clips([row, row], folder)
            for name, clip in clips.items():
                cap = cv2.VideoCapture(str(clip))
                try:
                    for _ in rows:
                        ok, frame = cap.read()
                        self.assertTrue(ok)
                        if name == "clean":
                            self.assertTrue(np.array_equal(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), read_image(path)))
                    self.assertFalse(cap.read()[0])
                finally:
                    cap.release()
