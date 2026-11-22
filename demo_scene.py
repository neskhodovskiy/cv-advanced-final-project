"""Generate the original, fictional recording used by the offline viewer."""
import json
import math
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


SIZE = 768
WIDTH, HEIGHT = 320, 192
COUNT, FPS = 72, 6
VARIANTS = ("clean", "h264_28", "h264_43", "blur_1", "blur_3")
ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "data" / "demo"


def terrain():
    rng = np.random.default_rng(742)
    img = np.empty((SIZE, SIZE, 3), np.uint8)
    palette = [(78, 126, 134), (80, 137, 116), (84, 148, 146), (62, 101, 98),
               (64, 130, 141), (91, 135, 119), (83, 123, 157)]
    y = 0
    while y < SIZE:
        height = int(rng.integers(40, 95))
        x = 0
        while x < SIZE:
            width = int(rng.integers(45, 126))
            color = palette[rng.integers(len(palette))]
            cv2.rectangle(img, (x, y), (x + width, y + height), color, -1)
            for xx in range(x + 4, min(SIZE, x + width), 5):
                cv2.line(img, (xx, y), (xx, y + height - 2),
                         tuple(max(0, c - 5) for c in color), 1)
            x += width
        y += height
    for _ in range(70):
        x, y = rng.integers(0, SIZE, 2)
        for _ in range(30):
            cx, cy = np.round(rng.normal([x, y], 10)).astype(int)
            r = int(rng.integers(2, 8))
            shade = int(rng.integers(0, 25))
            cv2.circle(img, (cx + 1, cy + 1), r + 1, (47, 73, 62), -1)
            cv2.circle(img, (cx, cy), r, (42 + shade, 78 + shade, 51 + shade), -1)
    river = [(int(410 + 95 * math.sin(y / 102.0)), y) for y in range(-5, SIZE + 8, 5)]
    cv2.polylines(img, [np.int32(river)], False, (80, 111, 93), 24, cv2.LINE_AA)
    cv2.polylines(img, [np.int32(river)], False, (122, 123, 80), 18, cv2.LINE_AA)
    road = [(x, int(0.48 * x + 125 + 32 * math.sin(x / 58))) for x in range(0, SIZE + 4, 4)]
    cv2.polylines(img, [np.int32(road)], False, (97, 107, 105), 8, cv2.LINE_AA)
    cv2.polylines(img, [np.int32(road)], False, (159, 165, 160), 4, cv2.LINE_AA)
    for x, y in road[4::4]:
        if abs(x - 410 - 95 * math.sin(y / 102)) < 40:
            continue
        y += int(rng.choice([-1, 1]) * rng.integers(10, 24))
        c = int(rng.integers(110, 177))
        cv2.rectangle(img, (x, y), (x + 8, y + 7), (75, 78, 71), -1)
        cv2.rectangle(img, (x, y), (x + 7, y + 5), (c - 20, c - 20, c), -1)
    noise = rng.normal(0, 3.8, (SIZE, SIZE, 1))
    return np.uint8(np.clip(img + noise, 0, 255))


def make_assets():
    if (ASSETS / "scene.json").is_file():
        return
    ASSETS.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ASSETS.parent) as tmp:
        folder = Path(tmp)
        for name in (*VARIANTS, "ref"):
            (folder / name).mkdir()
        image = terrain()
        cv2.imwrite(str(folder / "terrain.jpg"), image)
        route = []
        rng = np.random.default_rng(31)
        for i in range(COUNT):
            t = i / (COUNT - 1)
            x = int(178 + 398 * t)
            y = int(271 + 162 * t + 72 * math.sin(2 * math.pi * t))
            left, top = x - WIDTH // 2, y - HEIGHT // 2
            ref = image[top:top + HEIGHT, left:left + WIDTH].copy()
            matrix = cv2.getRotationMatrix2D((WIDTH / 2, HEIGHT / 2), 5 * math.sin(3*t), 1.08)
            query = cv2.warpAffine(ref, matrix, (WIDTH, HEIGHT), borderMode=cv2.BORDER_REFLECT)
            query = np.uint8(np.clip(query * .79 + 19 + rng.normal(0, 2, query.shape), 0, 255))
            if 48 <= i < 53:  # Intentional loss of texture for a visible failure example.
                query[:] = (80, 86, 79)
            for name, frame in [("clean", query), ("ref", ref),
                                ("blur_1", cv2.GaussianBlur(query, (0, 0), 1)),
                                ("blur_3", cv2.GaussianBlur(query, (0, 0), 3))]:
                cv2.imwrite(str(folder / name / f"{i:03d}.png"), frame)
            route.append([x, y])
        for crf in (28, 43):
            name = f"h264_{crf}"
            movie = folder / f"{name}.mp4"
            run_ffmpeg(["-framerate", str(FPS), "-i", str(folder / "clean" / "%03d.png"),
                        "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", str(movie)])
            run_ffmpeg(["-i", str(movie), "-start_number", "0", str(folder / name / "%03d.png")])
        (folder / "scene.json").write_text(json.dumps({
            "route": route, "width": WIDTH, "height": HEIGHT, "size": SIZE, "fps": FPS,
            "source": "Синтетична тестова сцена",
        }))
        # Rename only a complete generation. An interrupted setup can be safely retried.
        if ASSETS.exists():
            raise RuntimeError(f"Неповний каталог демо: {ASSETS}")
        folder.rename(ASSETS)


def run_ffmpeg(args):
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *args], check=True)


if __name__ == "__main__":
    make_assets()
