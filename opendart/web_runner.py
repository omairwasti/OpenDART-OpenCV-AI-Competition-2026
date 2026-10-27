"""Bounded deployment adapter; the scientific simulation remains unchanged."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

PRESETS = ("NOMINAL", "BLACKOUT", "NO_MEASUREMENT", "BALLISTIC")
ROOT = Path(__file__).resolve().parents[1]


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def source_digest():
    files = sorted((ROOT / "opendart").glob("*.py"))
    h = hashlib.sha256()
    for path in files:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def preset_config(preset):
    from .planar import PlanarConfig
    if preset not in PRESETS:
        raise ValueError("Unknown preset")
    cfg = PlanarConfig(**json.loads((ROOT / "configs/visual_2d.json").read_text()))
    cfg.seed = 7
    if preset == "BLACKOUT":
        cfg.blackout_s = [[200., 240.]]
    elif preset == "NO_MEASUREMENT":
        cfg.measurements = False
    elif preset == "BALLISTIC":
        cfg.control = False
    cfg.validate()
    return cfg


def run(preset, output):
    if os.name == "posix":
        import signal
        signal.alarm(180)  # Also bound an orphan if its web supervisor exits.
    # Set these before NumPy/SciPy are imported, also for direct CLI execution.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    import cv2
    from .planar import simulate_planar
    from .vision import annotate_candidates
    cv2.setNumThreads(1)
    distribution = importlib.metadata.version("opencv-python")
    if distribution != "5.0.0.93" or cv2.__version__ != "5.0.0":
        raise RuntimeError("The pinned OpenCV 5 distribution/runtime is required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError("Refusing to overwrite completed evidence")
    started = time.perf_counter()
    frames = []
    snapshots = {0, 100, 199, 200, 210, 220, 239, 240, 241, 400, 550}
    latest = None

    def capture(cfg, binary, row, image, candidates, selected, rows):
        nonlocal latest
        elapsed = int(row["elapsed_s"])
        latest = (row.copy(), image.copy(), candidates, selected)
        if elapsed in snapshots:
            filename = f"camera_{elapsed:03d}.png"
            if not cv2.imwrite(str(output / filename), annotate_candidates(image, candidates, selected)):
                raise RuntimeError("Camera export failed")
            frames.append(dict(file=filename, elapsed_s=row["elapsed_s"], mode=row["mode"],
                               tracker=row["tracker"], accepted=row["accepted"]))
            write_json(output / "progress.json", dict(elapsed_s=row["elapsed_s"], mode=row["mode"]))
        return None

    cfg = preset_config(preset)
    result = simulate_planar(cfg, frame_callback=capture)
    row, image, candidates, selected = latest
    cv2.imwrite(str(output / "camera_final.png"), annotate_candidates(image, candidates, selected))
    frames.append(dict(file="camera_final.png", elapsed_s=row["elapsed_s"], mode=row["mode"],
                       tracker=row["tracker"], accepted=row["accepted"]))
    rows = result["telemetry"]
    with (output / "telemetry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    trajectory = [dict(t=r["elapsed_s"], truth=[r["relative_x_m"], r["relative_y_m"]],
                       estimate=[r["estimate_x_m"], r["estimate_y_m"]]) for r in rows]
    peak_rss = None
    if os.name == "posix":
        import resource
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    payload = dict(preset=preset, generated_utc=datetime.now(timezone.utc).isoformat(),
                   config=asdict(cfg), summary=result["summary"], frames=frames, trajectory=trajectory,
                   runtime=dict(python=platform.python_version(), platform=platform.system(),
                                architecture=platform.machine(), opencv=cv2.__version__,
                                opencv_distribution=distribution, opencv_threads=cv2.getNumThreads(),
                                wall_s=time.perf_counter()-started, peak_rss_mib=peak_rss,
                                source_sha256=source_digest(), release_sha256=os.getenv("OPENDART_RELEASE_SHA256", "local")),
                   recovery=dict(accepted_after_blackout=sum(r["accepted"] for r in rows if r["elapsed_s"] >= cfg.blackout_s[0][1]),
                                 reacquired_samples=sum(r["tracker"] == "REACQUIRED" for r in rows)),
                   max_force_n=max(r["force_n"] for r in rows))
    write_json(output / "result.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.preset, args.out)
    print(json.dumps(dict(preset=args.preset, outcome=result["summary"]["status"], runtime=result["runtime"])))


if __name__ == "__main__":
    main()
