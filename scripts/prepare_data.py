"""Build fixed evaluation pairs from published annotations and VPAIR pair IDs."""
import csv
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from fetch_data import DATA, FLIGHTS, FRAME_COUNT, MAP_CSV, RAW, read_table, selected_rows

PREP = DATA / "prepared"
COLUMNS = ["dataset", "split", "sequence", "frame", "query", "positive", "negative"]


def resized(im):
    frame = im.convert("RGB")
    frame.thumbnail((512, 512), Image.Resampling.LANCZOS)
    return frame


def save(path, im):
    image = resized(im)
    image.save(path, quality=94, subsampling=0)
    return path.as_posix()


def path_for(dataset, sequence, frame, role):
    folder = PREP / dataset / sequence
    folder.mkdir(exist_ok=True, parents=True)
    return folder / f"{frame:05d}-{role}.jpg"


def box(cx, cy, side, width, height):
    if side > min(width, height):
        raise ValueError("Reference is smaller than the test tile")
    left = max(0, min(round(cx - side / 2), width - side))
    top = max(0, min(round(cy - side / 2), height - side))
    return left, top, left + side, top + side


def gap_to_box(point, bounds):
    x, y = point
    x0, y0, x1, y1 = bounds
    return math.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1))


def uav_pairs():
    dataset = "UAV-VisLoc"
    maps = {r["mapname"]: r for r in read_table(RAW / MAP_CSV)}
    rows = []
    for seq, split in FLIGHTS.items():
        name = f"satellite{seq}.tif"
        extent = maps[name]
        west, east = float(extent["LT_lon_map"]), float(extent["RB_lon_map"])
        north, south = float(extent["LT_lat_map"]), float(extent["RB_lat_map"])
        annotations = selected_rows(seq)
        ids = [int(r["num"]) for r in annotations]
        if ids != list(range(ids[0], ids[0] + FRAME_COUNT)):
            raise ValueError(f"Nonconsecutive annotations: {seq}")
        if len({r["filename"] for r in annotations}) != len(annotations):
            raise ValueError(f"Duplicate annotated image: {seq}")
        with Image.open(RAW / seq / name) as atlas:
            width, height = atlas.size
            replay_frames = []
            if split == "test":
                preview = atlas.copy()
                preview.thumbnail((1440, 1440), Image.Resampling.LANCZOS)
                map_path = PREP / "replay-map.jpg"
                preview.convert("RGB").save(map_path, quality=88)
                sx, sy = preview.width / width, preview.height / height
            for record in annotations:
                frame = int(record["num"])
                x = width * (float(record["lon"]) - west) / (east - west)
                y = height * (north - float(record["lat"])) / (north - south)
                if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
                    raise ValueError(f"Published centre outside map for {seq}/{frame}")
                with Image.open(RAW / seq / "drone" / record["filename"]) as query:
                    # Fixed nominal GSD ratio (0.15 m / 0.3 m) from dataset documentation.
                    # This is a centre-overlap label, not a calibrated camera footprint.
                    side = round(max(query.size) / 2)
                    pos = box(x, y, side, width, height)
                    neg = box((x + width / 2) % width, (y + height / 2) % height,
                              side, width, height)
                    # Even the image's half-diagonal at the stated upper GSD
                    # (0.2 m / 0.3 m) must stop before the nearest negative pixel.
                    max_radius = math.hypot(*query.size) / 3
                    if gap_to_box((x, y), neg) <= max_radius:
                        raise ValueError(f"Ambiguous negative for {seq}/{frame}")
                    paths = dict(query=save(path_for(dataset, seq, frame, "q"), query),
                                 positive=save(path_for(dataset, seq, frame, "p"), atlas.crop(pos)),
                                 negative=save(path_for(dataset, seq, frame, "n"), atlas.crop(neg)))
                rows.append(dict(dataset=dataset, split=split, sequence=seq, frame=frame, **paths))
                if split == "test":
                    rect = [round(v * scale, 3) for v, scale in zip(pos, (sx, sy, sx, sy))]
                    replay_frames.append(dict(frame=frame, x=round(x*sx, 3), y=round(y*sy, 3),
                                              rect=rect, **paths))
            if split == "test":
                replay = dict(dataset=dataset, sequence=seq, map=map_path.as_posix(),
                              width=preview.width, height=preview.height, frames=replay_frames)
                (DATA / "replay.json").write_text(json.dumps(replay, indent=2) + "\n")
    return rows


def vpair_pairs():
    rows = []
    dataset, sequence = "VPAIR", "sample"
    with ZipFile(DATA / "archives/vpair_sample.zip") as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Corrupt VPAIR archive member: {bad}")
        names = set(archive.namelist())
        queries = sorted(f for f in names if f.startswith("vpair_sample/queries/") and f.endswith(".png"))
        distractors = sorted(f for f in names if f.startswith("vpair_sample/distractors/") and f.endswith(".png"))
        if min(len(queries), len(distractors)) < FRAME_COUNT:
            raise ValueError("Incomplete VPAIR sample")
        start = (len(queries) - FRAME_COUNT) // 2
        for i, q in enumerate(queries[start:start + FRAME_COUNT]):
            frame = int(Path(q).stem)
            p = f"vpair_sample/reference_views/{Path(q).name}"
            n = distractors[i * len(distractors) // FRAME_COUNT]
            paths = {}
            for role, src in (("query", q), ("positive", p), ("negative", n)):
                if src not in names:
                    raise ValueError(f"Missing VPAIR member {src}")
                with Image.open(io.BytesIO(archive.read(src))) as im:
                    paths[role] = save(path_for(dataset, sequence, frame, role[0]), im)
            rows.append(dict(dataset=dataset, split="test", sequence=sequence, frame=frame, **paths))
    return rows


def main():
    rows = uav_pairs() + vpair_pairs()
    target = DATA / "manifest.csv"
    temp = target.with_suffix(".partial")
    with temp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    frozen = Path("datasets/pairs.csv")
    if frozen.exists() and temp.read_bytes() != frozen.read_bytes():
        raise ValueError(f"Prepared membership differs from {frozen}")
    temp.replace(target)
    print(f"Prepared {len(rows)} queries, {len(rows)*2} labeled pairs at {target}")


if __name__ == "__main__":
    main()
