"""Fetch the exact public benchmark subset. Run from the repository root."""
import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

DATA = Path("data")
RAW = DATA / "raw" / "UAV-VisLoc"
MAP_CSV = "satellite_ coordinates_range.csv"
# This selection precedes scoring. Ten is held out as a complete flight.
FLIGHTS = {"05": "dev", "06": "dev", "10": "test"}
FRAME_COUNT = 24


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path, expected_size, expected_sha):
    if path.stat().st_size != expected_size:
        raise ValueError(f"Size mismatch: {path}")
    if digest(path) != expected_sha:
        raise ValueError(f"Checksum mismatch: {path}")


def fetch(url, target, expected_size, expected_sha):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        verify(target, expected_size, expected_sha)
    else:
        for attempt in range(3):
            # Keep the published cache intact on short reads or failed checks.
            with tempfile.TemporaryDirectory(dir=target.parent) as staging:
                partial = Path(staging) / target.name
                try:
                    with urllib.request.urlopen(url, timeout=120) as source, partial.open("wb") as sink:
                        shutil.copyfileobj(source, sink, length=2**20)
                    verify(partial, expected_size, expected_sha)
                    partial.replace(target)
                    break
                except (OSError, ValueError):
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)
    return dict(path=target.as_posix(), size=expected_size, sha256=expected_sha, source=url,
                received_utc=datetime.fromtimestamp(target.stat().st_mtime, timezone.utc).isoformat())


def read_table(path):
    for encoding in ("utf-8-sig", "gb18030", "cp1252"):
        try:
            with path.open(encoding=encoding, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unknown CSV encoding: {path}")


def selected_rows(sequence):
    rows = sorted(read_table(RAW / sequence / f"{sequence}.csv"), key=lambda r: int(r["num"]))
    # Freeze the initial midpoint selection. A missing annotation must never
    # cause later examples to silently shift into the evaluation manifest.
    first = {"05": 225, "06": 161, "10": 61}[sequence]
    selected = [r for r in rows if first <= int(r["num"]) < first + FRAME_COUNT]
    if [int(r["num"]) for r in selected] != list(range(first, first + FRAME_COUNT)):
        raise ValueError(f"Missing or duplicate annotation: {sequence}")
    return selected


def fetch_record(item):
    return fetch(item["source"], Path(item["path"]), item["size"], item["sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["uav", "vpair", "all"], default="all", nargs="?")
    args = parser.parse_args()
    for name in ("uav", "vpair"):
        if args.dataset not in ("all", name):
            continue
        manifest = Path("datasets") / f"source-{name}.json"
        sources = json.loads(manifest.read_text())
        with ThreadPoolExecutor(max_workers=4) as pool:
            records = list(pool.map(fetch_record, sources))
        (DATA / f"source-{name}.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"Verified {len(records)} {name} source files", flush=True)


if __name__ == "__main__":
    main()
