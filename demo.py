"""Serve a fixed set of annotated recordings and their pairwise image scores."""
import base64
import json
import os
import tempfile
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from urllib.parse import parse_qs, urlsplit

import cv2

from benchmark import score_pair
from demo_scene import ASSETS, ROOT, VARIANTS, make_assets
from experiment import SIZE, prepare_clips, read_image


def prepared_path(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to((ROOT / "data").resolve()) or not path.is_file():
        raise ValueError(f"Недоступний підготовлений файл {name}")
    return path


def configure(temp):
    make_assets()
    info = json.loads((ASSETS / "scene.json").read_text())
    info["ids"] = list(range(len(info["route"])))
    info["mapWidth"] = info["mapHeight"] = info["size"]
    info["mapUrl"] = "/terrain.jpg"
    info["synthetic"] = True
    recordings = {"synthetic": (info, None, None)}
    path = ROOT / "data" / "replay.json"
    if path.is_file():
        meta = json.loads(path.read_text())
        rows = meta["frames"]
        for row in rows:
            row["query"] = str(prepared_path(row["query"]))
            row["positive"] = str(prepared_path(row["positive"]))
            row["dataset"] = meta["dataset"]
            row["sequence"] = meta["sequence"]
        (_, clips), = prepare_clips(rows, temp)
        recordings["uav"] = (dict(
            route=[[f["x"], f["y"]] for f in rows], rects=[f["rect"] for f in rows],
            width=SIZE[0], height=SIZE[1], mapWidth=meta["width"], mapHeight=meta["height"],
            fps=10, ids=[f["frame"] for f in rows], synthetic=False,
            mapUrl="/public-terrain.jpg", source=f'{meta["dataset"]} · {meta["sequence"]}'),
            rows, clips)
    return recordings


def data_url(img):
    success, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise ValueError("Помилка кодування кадру")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def read_pair(recording, index, variant):
    _, rows, clips = RECORDINGS[recording]
    if rows is None:
        return tuple(cv2.imread(str(ASSETS / name / f"{index:03d}.png")) for name in (variant,"ref"))
    video = cv2.VideoCapture(str(clips[variant]))
    try:
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, img = video.read()
        if not ok:
            raise ValueError("Помилка декодування кадру")
    finally:
        video.release()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), read_image(rows[index]["positive"])


@lru_cache(maxsize=128)
def snapshot(recording, index, variant, method, clahe):
    start = perf_counter()
    query, ref = read_pair(recording, index, variant)
    read_ms = (perf_counter() - start) * 1000
    scored = score_pair(query, ref, method, CLAHE_CLIP if clahe else 0)
    return json.dumps({"frame": index, "variant": variant, "method": method,
                       "clahe": clahe, "decode_ms": read_ms, "query": data_url(query),
                       "reference": data_url(ref), **scored}).encode()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlsplit(self.path)
        p = parse_qs(url.query)
        recording = p.get("recording", ["synthetic"])[0]
        if recording not in RECORDINGS:
            self.send_error(404)
            return
        info = RECORDINGS[recording][0]
        if url.path == "/api/scene":
            self.serve(json.dumps({**info, "recordings": {k:v[0]["source"] for k,v in RECORDINGS.items()},
                                   "clip": CLAHE_CLIP}).encode(), "application/json")
            return
        if url.path == "/api/frame":
            try:
                i = int(p.get("i", ["0"])[0])
                variant = p.get("variant", ["clean"])[0]
                method = p.get("method", ["ORB"])[0]
                clahe = p.get("clahe", ["0"])[0]
                if not (0 <= i < len(info["route"]) and variant in VARIANTS and method in ("ORB", "AKAZE") and clahe in ("0", "1")):
                    raise ValueError
            except ValueError:
                self.send_error(400, "Invalid recording settings")
                return
            self.serve(snapshot(recording, i, variant, method, clahe == "1"), "application/json")
            return
        files = {"/": (ROOT / "web/index.html", "text/html; charset=utf-8"),
                 "/app.js": (ROOT / "web/app.js", "text/javascript; charset=utf-8"),
                 "/style.css": (ROOT / "web/style.css", "text/css; charset=utf-8"),
                 "/terrain.jpg": (ASSETS / "terrain.jpg", "image/jpeg")}
        if PUBLIC_MAP:
            files["/public-terrain.jpg"] = (PUBLIC_MAP, "image/jpeg")
        if url.path not in files:
            self.send_error(404)
            return
        path, mime = files[url.path]
        self.serve(path.read_bytes(), mime)

    def serve(self, body, mime):
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    cv2.setNumThreads(1)
    os.chdir(ROOT)
    with tempfile.TemporaryDirectory() as tmp:
        RECORDINGS = configure(Path(tmp))
        meta = ROOT / "data" / "replay.json"
        PUBLIC_MAP = prepared_path(json.loads(meta.read_text())["map"]) if meta.exists() else None
        frozen = ROOT / "results" / "frozen.json"
        CLAHE_CLIP = json.loads(frozen.read_text())["clip_limit"] if frozen.exists() else 2.0
        port = int(os.getenv("PORT", "8080"))
        server = ThreadingHTTPServer((os.getenv("DEMO_HOST", "0.0.0.0"), port), Handler)
        print(f"Демо на http://localhost:{port}", flush=True)
        server.serve_forever()
