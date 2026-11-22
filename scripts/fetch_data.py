"""Fetch the exact public benchmark subset. Run from the repository root."""
import csv
import hashlib
import json
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

DATA = Path("data")
RAW = DATA / "raw" / "UAV-VisLoc"
REV = "64001825da73fa63e966054f9ee6f7269b9b92cf"
HUB = "https://huggingface.co"
DS = "mvidem/UAV-VisLoc"
VPAIR = "https://zenodo.org/api/records/6473989/files/vpair_sample.zip/content"
MAP_CSV = "satellite_ coordinates_range.csv"
# This selection precedes scoring. Ten is held out as a complete flight.
FLIGHTS = {"05": "dev", "06": "dev", "10": "test"}
FRAME_COUNT = 24


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, target, expected_size, expected_sha=None, algorithm="sha256"):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        part = target.with_suffix(target.suffix + ".partial")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=120) as source, part.open("wb") as sink:
                    shutil.copyfileobj(source, sink, length=2**20)
                if part.stat().st_size != expected_size:
                    raise OSError(f"Incomplete download: {target}")
                part.replace(target)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
    if target.stat().st_size != expected_size:
        raise ValueError(f"Size mismatch: {target}")
    checksum = digest(target)
    if expected_sha and (checksum if algorithm == "sha256" else digest(target, algorithm)) != expected_sha:
        raise ValueError(f"Checksum mismatch: {target}")
    return dict(path=target.as_posix(), size=expected_size, sha256=checksum,
                source=url, received_utc=datetime.fromtimestamp(target.stat().st_mtime, timezone.utc).isoformat())


def list_hub(folder=""):
    url = f"{HUB}/api/datasets/{DS}/tree/{REV}/{folder}?limit=1000"
    with urllib.request.urlopen(url, timeout=120) as f:
        return {item["path"]: item for item in json.load(f) if item["type"] == "file"}


def fetch_entry(entry):
    key = entry["path"]
    return fetch(f"{HUB}/datasets/{DS}/resolve/{REV}/{quote(key)}", RAW / key,
                 entry["size"], entry.get("lfs", {}).get("oid"))


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


def fetch_uav():
    entries = list_hub()
    records = [fetch_entry(entries[MAP_CSV])]
    for seq in FLIGHTS:
        entries = list_hub(seq)
        records.append(fetch_entry(entries[f"{seq}/{seq}.csv"]))
        images = list_hub(f"{seq}/drone")
        selected = [images[f"{seq}/drone/{r['filename']}"] for r in selected_rows(seq)]
        selected.append(entries[f"{seq}/satellite{seq}.tif"])
        with ThreadPoolExecutor(max_workers=4) as pool:
            records.extend(pool.map(fetch_entry, selected))
        print(f"Verified UAV-VisLoc {seq}", flush=True)
    return records


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=["uav", "vpair", "all"], default="all", nargs="?")
    args = p.parse_args()
    for name in ("uav", "vpair"):
        if args.dataset not in ("all", name):
            continue
        if name == "uav":
            records = fetch_uav()
        else:
            records = [fetch(VPAIR, DATA / "archives/vpair_sample.zip", 861249439,
                             "96342d9acb9b16fccad19408a8b29665", "md5")]
        lock = Path("datasets") / f"source-{name}.json"
        if lock.exists():
            def index(items):
                return {item["path"]: item["sha256"] for item in items}
            if index(records) != index(json.loads(lock.read_text())):
                raise ValueError(f"Source files differ from {lock}")
        out = DATA / f"source-{name}.json"
        out.write_text(json.dumps(records, indent=2) + "\n")

if __name__ == "__main__":
    main()
