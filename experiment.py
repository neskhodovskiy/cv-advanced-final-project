"""Run a paired, sequence-held-out image verification experiment (CPU only)."""
import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from benchmark import average_precision, score_pair

CONDITIONS = ("clean", "h264_28", "h264_43", "blur_1", "blur_3")
METHODS = ("ORB", "AKAZE")
FIELDS = ("dataset", "split", "sequence", "frame", "query", "positive", "negative")
SIZE = (384, 288)
MIN_MATCHES = 4


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path):
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Manifest is missing required columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty manifest")
    seen = set()
    splits = {}
    for r in rows:
        if any(not r.get(k) for k in FIELDS):
            raise ValueError("Missing annotation or image path")
        key = (r["dataset"], r["sequence"])
        if r["split"] not in ("dev", "test") or splits.setdefault(key, r["split"]) != r["split"]:
            raise ValueError(f"Invalid or overlapping split: {key}")
        r["frame"] = int(r["frame"])
        frame = (*key, r["frame"])
        if frame in seen or r["positive"] == r["negative"]:
            raise ValueError(f"Duplicate frame or reference: {frame}")
        seen.add(frame)
        for k in ("query", "positive", "negative"):
            if not Path(r[k]).is_file():
                raise FileNotFoundError(r[k])
    return sorted(rows, key=lambda r: (r["dataset"], r["sequence"], r["frame"]))


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot decode {path}")
    # Letterbox identically for both sides, preserving feature geometry.
    h, w = img.shape
    scale = min(SIZE[0] / w, SIZE[1] / h)
    w, h = max(1, round(w * scale)), max(1, round(h * scale))
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    x, y = (SIZE[0] - w) // 2, (SIZE[1] - h) // 2
    return cv2.copyMakeBorder(img, y, SIZE[1]-h-y, x, SIZE[0]-w-x, cv2.BORDER_REPLICATE)


def prepare_clips(rows, directory):
    groups = []
    for i, (_, group) in enumerate(itertools.groupby(rows, lambda r: (r["dataset"], r["sequence"]))):
        frames = list(group)
        original = [read_image(r["query"]) for r in frames]
        clips = {}
        for condition in CONDITIONS:
            path = directory / f"{i}-{condition}.mkv"
            is_h264 = condition.startswith("h264")
            codec = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", condition.split("_")[1],
                     "-preset", "medium"] if is_h264 else ["-c:v", "ffv1"]
            cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
                   "-s", f"{SIZE[0]}x{SIZE[1]}", "-r", "10", "-i", "pipe:0", "-threads", "1", *codec, str(path)]
            sigma = int(condition[-1]) if condition.startswith("blur") else 0
            images = (cv2.GaussianBlur(a, (0, 0), sigma) if sigma else a for a in original)
            subprocess.run(cmd, input=b"".join(im.tobytes() for im in images), check=True)
            clips[condition] = path
        groups.append((frames, clips))
    return groups


def run_condition(frames, clip, condition, method, strength):
    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        raise ValueError(f"Could not open {clip}")
    values = []
    start = perf_counter()
    try:
        for row in frames:
            begin = perf_counter()
            ok, img = capture.read()
            if not ok:
                raise ValueError(f"Frame missing: {row['frame']} in {clip}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            query_ms = 1000 * (perf_counter() - begin)
            for label, key in ((1, "positive"), (0, "negative")):
                begin = perf_counter()
                reference = read_image(row[key])
                ref_ms = 1000 * (perf_counter() - begin)
                result = score_pair(img, reference, method, strength)
                values.append(dict(dataset=row["dataset"], split=row["split"], sequence=row["sequence"],
                                   frame=row["frame"], label=label, condition=condition, method=method,
                                   clip_limit=strength, score=result["score"], matches=result["inliers"],
                                   insufficient=int(result["inliers"] < MIN_MATCHES),
                                   query_decode_ms=query_ms, reference_load_ms=ref_ms,
                                   algorithm_ms=result["algorithm_ms"]))
        if capture.read()[0]:
            raise ValueError(f"Unexpected extra frames in {clip}")
    finally:
        capture.release()
    return values, perf_counter() - start


def summarize(values, seconds):
    times = [r["algorithm_ms"] for r in values]
    return dict(ap=average_precision([r["label"] for r in values], [r["score"] for r in values]),
                pairs=len(values), insufficient_rate=np.mean([r["insufficient"] for r in values]),
                algorithm_p50_ms=np.median(times), algorithm_p95_ms=np.percentile(times, 95),
                query_decode_p50_ms=np.median([r["query_decode_ms"] for r in values]),
                reference_load_p50_ms=np.median([r["reference_load_ms"] for r in values]),
                pairs_per_s=len(values) / seconds, frames_per_s=len(values) / (2 * seconds))


def evaluate(groups, split, strengths):
    samples, summary = [], []
    for dataset in sorted({frames[0]["dataset"] for frames, _ in groups if frames[0]["split"] == split}):
        for method, strength, condition in itertools.product(METHODS, strengths, CONDITIONS):
            values, seconds = [], 0
            for frames, clips in groups:
                if (frames[0]["dataset"], frames[0]["split"]) == (dataset, split):
                    batch, elapsed = run_condition(frames, clips[condition], condition, method, strength)
                    values.extend(batch)
                    seconds += elapsed
            summary.append(dict(dataset=dataset, split=split, method=method, clip_limit=strength,
                                condition=condition, **summarize(values, seconds)))
            samples.extend(values)
    return samples, summary


def plot_results(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    datasets = sorted({r["dataset"] for r in rows})
    fig, axes = plt.subplots(len(datasets), 1, figsize=(8, 3.7 * len(datasets)), squeeze=False)
    for ax, dataset in zip(axes.flat, datasets):
        subset = [r for r in rows if r["dataset"] == dataset]
        for method, color in zip(METHODS, ("#00756b", "#b55217")):
            for strength in sorted({r["clip_limit"] for r in subset}):
                points = {r["condition"]: r["ap"] for r in subset if (r["method"], r["clip_limit"]) == (method,strength)}
                ax.plot(CONDITIONS, [points[c] for c in CONDITIONS], marker="o", color=color,
                        linestyle="--" if strength else "-", label=method + (" + CLAHE" if strength else ""))
        ax.set(title=dataset, ylim=(0, 1.04), ylabel="Average precision")
        ax.grid(axis="y", alpha=.22)
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run(manifest, output):
    cv2.setNumThreads(1)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(manifest)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if {r["split"] for r in rows} != {"dev", "test"}:
        raise ValueError("Need independent dev and test sequences")
    with tempfile.TemporaryDirectory() as temp:
        groups = prepare_clips(rows, Path(temp))
        _, dev_summary = evaluate(groups, "dev", (1.0, 2.0, 4.0))
        # Maximize macro-AP across methods and conditions, tie → smaller clip limit.
        selected = max((1.0, 2.0, 4.0), key=lambda c: np.mean([r["ap"] for r in dev_summary if r["clip_limit"] == c]))
        save_csv(output / "development.csv", dev_summary)
        config = dict(clip_limit=selected, grid=[1, 2, 4], manifest_sha256=digest, image_size=SIZE,
                      minimum_matches=MIN_MATCHES, methods=METHODS, conditions=CONDITIONS)
        (output / "frozen.json").write_text(json.dumps(config, indent=2) + "\n")
        samples, summary = evaluate(groups, "test", (0, selected))
    lookup = {(r["dataset"],r["method"],r["condition"],r["clip_limit"]): r["ap"] for r in summary}
    for r in summary:
        ds, m, c, strength = r["dataset"], r["method"], r["condition"], r["clip_limit"]
        r["delta_vs_clean"] = r["ap"] - lookup[ds, m, "clean", strength]
        r["delta_vs_no_clahe"] = r["ap"] - lookup[ds, m, c, 0]
    save_csv(output / "samples.csv", samples)
    save_csv(output / "summary.csv", summary)
    save_csv(output / "pairs.csv", rows)
    plot_results(summary, output / "quality.png")
    # Best/worst ranking examples with identifiers for offline inspection.
    failures = [r for r in samples if (r["label"] == 1 and r["insufficient"]) or
                (r["label"] == 0 and not r["insufficient"])]
    if failures:
        save_csv(output / "errors.csv", failures)
    else:
        (output / "errors.csv").write_text(",".join(samples[0]) + "\n")
    cpu = os.environ.get("BENCHMARK_CPU") or platform.processor() or platform.machine()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[-1].strip()
                break
    env = dict(cpu=cpu, platform=platform.platform(), python=platform.python_version(),
               opencv=cv2.__version__, numpy=np.__version__, threads=cv2.getNumThreads(),
               ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0])
    (output / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    print(f"Evaluated {len(samples)} labeled observations; results in {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.manifest, args.output)
