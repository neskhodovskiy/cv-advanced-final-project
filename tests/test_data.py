import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import BadZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_data
import prepare_data


class DataPreparationTests(unittest.TestCase):
    def test_archive_rejects_invalid_container(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(prepare_data, "DATA", Path(tmp)):
            target = Path(tmp) / "archives/vpair_sample.zip"
            target.parent.mkdir()
            target.write_bytes(b"corrupt zip response")
            with self.assertRaises(BadZipFile):
                prepare_data.vpair_pairs()

    def test_missing_annotation_does_not_shift_selected_window(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(fetch_data, "RAW", Path(tmp)):
            target = Path(tmp) / "05/05.csv"
            target.parent.mkdir()
            target.write_text("num\n" + "\n".join(str(i) for i in range(220, 254) if i != 233))
            with self.assertRaisesRegex(ValueError, "Missing or duplicate annotation"):
                fetch_data.selected_rows("05")

    def test_cached_file_with_wrong_digest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "image.png"
            target.write_bytes(b"stale")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                fetch_data.fetch("unused", target, expected_size=5, expected_sha="0" * 64)

    def test_fixed_pairs_and_entire_sequence_split(self):
        path = Path(__file__).resolve().parents[1] / "datasets/pairs.csv"
        with path.open() as data:
            rows = list(csv.DictReader(data))
        self.assertEqual(len(rows), 96)
        keys = {(r["dataset"], r["sequence"], int(r["frame"])) for r in rows}
        self.assertEqual(len(keys), len(rows))
        splits = {(r["dataset"], r["sequence"], r["split"]) for r in rows}
        self.assertEqual(splits, {("UAV-VisLoc", "05", "dev"), ("UAV-VisLoc", "06", "dev"),
                                  ("UAV-VisLoc", "10", "test"), ("VPAIR", "sample", "test")})


if __name__ == "__main__":
    unittest.main()
