"""Offline renderer/detector diagnostics; truth is used only to score pixels.

Run ``python -m opendart.calibrate_size --out results/new_calibration``.
No plant, estimator or guidance is run or tuned. The fixed pixel-radius bias
is fitted on training subpixel offsets and scored on held-out offsets/seeds.
This deliberately small experiment is not a visual Monte Carlo campaign.
"""
import argparse
import csv
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

from .model import Binary
from .planar import CameraRenderer, PlanarConfig
from .vision import AsteroidDetector, require_opencv


@dataclass
class CalibrationConfig:
    seed: int = 741
    radii_px: list = field(default_factory=lambda: [3., 5., 8., 12., 20., 40., 80., 160.])
    orbit_fractions: list = field(default_factory=lambda: [0., .125, .5, .625])
    noise_dn: list = field(default_factory=lambda: [0., 1., 2.])
    train_offsets_px: list = field(default_factory=lambda: [0., .5])
    validation_offsets_px: list = field(default_factory=lambda: [.25, .75])
    # Optional local appearance only; no archive download/parsing.
    reference_image: str | None = None

    def validate(self):
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Calibration seed must be a nonnegative integer")
        for name in ("radii_px", "orbit_fractions", "noise_dn",
                     "train_offsets_px", "validation_offsets_px"):
            values = np.asarray(getattr(self, name), float)
            if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} requires a nonempty finite vector")
        if min(self.radii_px) < 2 or max(self.radii_px) > 200 or min(self.noise_dn) < 0:
            raise ValueError("Require 2 <= radius <= 200 pixels and nonnegative noise")
        if set(self.train_offsets_px) & set(self.validation_offsets_px):
            raise ValueError("Training and validation pixel offsets must be disjoint")


def residual_statistics(values):
    values = np.asarray(values, float)
    if not values.size:
        return None
    return dict(n=int(values.size), mean=float(np.mean(values)),
                median=float(np.median(values)), std=float(np.std(values)),
                rmse=float(np.sqrt(np.mean(values**2))),
                p05=float(np.percentile(values, 5)), p95=float(np.percentile(values, 95)),
                absolute_p95=float(np.percentile(np.abs(values), 95)))


def summarize_calibration(rows):
    """Fit on training data only; never choose a coefficient by impact outcome."""
    summaries = {}
    for appearance in sorted({row["appearance"] for row in rows}):
        matching = [r for r in rows if r["appearance"] == appearance]
        usable = [r for r in matching if r["status"] == "MEASURED"]
        training = [r for r in usable if r["split"] == "train"]
        if not training:
            raise ValueError(f"No usable training samples for {appearance}")
        bias = float(np.median([r["radius_residual_px"] for r in training]))
        splits = {}
        for split in ("train", "validation"):
            samples = [r for r in usable if r["split"] == split]
            corrected = [np.log((r["measured_radius_px"]-bias)/r["truth_radius_px"])
                         for r in samples if r["measured_radius_px"] > bias]
            coverage = []
            for r in samples:
                radius = r["measured_radius_px"]-bias
                if radius <= 0:
                    continue
                sigma = max(.5/radius, .10)  # existing declared likelihood, not fitted
                coverage.append(abs(np.log(radius/r["truth_radius_px"])) <= 1.96*sigma)
            splits[split] = dict(
                centroid_u_residual_px=residual_statistics([r["centroid_u_residual_px"] for r in samples]),
                centroid_v_residual_px=residual_statistics([r["centroid_v_residual_px"] for r in samples]),
                radius_residual_px=residual_statistics([r["radius_residual_px"] for r in samples]),
                raw_log_radius_residual=residual_statistics([r["log_radius_residual"] for r in samples]),
                corrected_log_radius_residual=residual_statistics(corrected),
                corrected_log_residual_within_declared_95_fraction=float(np.mean(coverage)) if coverage else None,
                invalid_corrected_radius_count=len(samples)-len(corrected))
        by_size = {}
        for size in sorted({r["truth_radius_px"] for r in usable}):
            samples = [r for r in usable if r["truth_radius_px"] == size and r["split"] == "validation"]
            by_size[str(size)] = dict(
                raw=residual_statistics([r["log_radius_residual"] for r in samples]),
                corrected=residual_statistics([np.log((r["measured_radius_px"]-bias)/size)
                                               for r in samples if r["measured_radius_px"] > bias]))
        summaries[appearance] = dict(
            attempted=len(matching), measured=len(usable),
            status_counts={s:sum(r["status"] == s for r in matching) for s in sorted({r["status"] for r in matching})},
            fitted_radius_bias_px=bias, splits=splits, validation_by_ideal_size_px=by_size)
    training = [r for r in rows if r["status"] == "MEASURED" and r["split"] == "train"]
    pooled_bias = float(np.median([r["radius_residual_px"] for r in training]))
    rounded_bias = round(pooled_bias, 2)
    validations = {}
    for appearance in sorted(summaries):
        samples = [r for r in rows if r["appearance"] == appearance
                   and r["status"] == "MEASURED" and r["split"] == "validation"]
        errors = [np.log(r["measured_radius_px"]/(r["truth_radius_px"]+rounded_bias)) for r in samples]
        validations[appearance] = dict(
            log_likelihood_residual=residual_statistics(errors),
            within_declared_95_fraction=float(np.mean([
                abs(e) <= 1.96*max(.5/r["measured_radius_px"], .10)
                for r, e in zip(samples, errors)])) if samples else None)
    return dict(appearance_results=summaries, pooled_model=dict(
        training_samples=len(training), fitted_radius_bias_px=pooled_bias,
        rounded_radius_bias_px=rounded_bias,
        model="log(r_detector) = log(f*R/Z + bias_px)",
        uncertainty="Retain max(0.5/r_detector, 0.10) log-size sigma; no persistent bias state",
        validation=validations))


def calibrate(config=None):
    config = config or CalibrationConfig()
    config.validate()
    cv = require_opencv()
    if int(cv.__version__.split(".")[0]) != 5:
        raise RuntimeError("Offline calibration requires OpenCV 5")
    base = PlanarConfig()
    binary = Binary(period_s=base.period_s, primary_spin_period_s=base.spin_period_s)
    rows = []
    appearances = [("analytic", None)]
    if config.reference_image:
        appearances.append(("reference", config.reference_image))
    case = 0
    for appearance, reference in appearances:
        # Reuse expensive reference extraction, but reset image noise per case.
        renderer = CameraRenderer(replace(base, reference_image=reference))
        for split, offsets in [("train", config.train_offsets_px),
                               ("validation", config.validation_offsets_px)]:
            for radius in config.radii_px:
                for fraction in config.orbit_fractions:
                    for noise in config.noise_dn:
                        for offset in offsets:
                            seed = config.seed+case
                            cfg = replace(base, seed=seed, noise_dn=noise)
                            renderer.cfg = cfg
                            renderer.rng = np.random.default_rng(seed)
                            time = base.epoch_s+fraction*base.period_s
                            depth = base.camera_f_px*88./radius
                            truth_uv = np.array([(base.camera_width-1)/2+offset,
                                                 (base.camera_height-1)/2])
                            relative = np.array([offset*depth/base.camera_f_px, depth])
                            observer = binary.states(time)[0][1, :2]-relative
                            image = renderer.render(time, observer, np.pi/2, binary)
                            candidates = AsteroidDetector().detect(image)
                            # Truth association is explicitly OFFLINE scoring, not flight acquisition.
                            chosen = min(candidates, key=lambda c:np.linalg.norm(c.centroid_px-truth_uv)) if candidates else None
                            status = "NO_MATCH"
                            if chosen is not None and np.linalg.norm(chosen.centroid_px-truth_uv) <= 3.:
                                status = "CLIPPED" if chosen.touches_border else "MEASURED"
                            row = dict(case=case, split=split, appearance=appearance, seed=seed,
                                       noise_dn=noise, orbit_fraction=fraction, offset_px=offset,
                                       truth_depth_m=depth, truth_radius_px=radius,
                                       truth_u_px=float(truth_uv[0]), truth_v_px=float(truth_uv[1]),
                                       status=status, candidate_count=len(candidates))
                            for key in ("measured_radius_px", "radius_residual_px", "log_radius_residual",
                                        "centroid_u_residual_px", "centroid_v_residual_px"):
                                row[key] = None
                            if status == "MEASURED":
                                row.update(measured_radius_px=float(chosen.radius_px),
                                           radius_residual_px=float(chosen.radius_px-radius),
                                           log_radius_residual=float(np.log(chosen.radius_px/radius)),
                                           centroid_u_residual_px=float(chosen.centroid_px[0]-truth_uv[0]),
                                           centroid_v_residual_px=float(chosen.centroid_px[1]-truth_uv[1]))
                            rows.append(row)
                            case += 1
    return rows, summarize_calibration(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="Must be a new directory")
    parser.add_argument("--reference-image", type=Path)
    args = parser.parse_args()
    data = json.loads(args.config.read_text()) if args.config else {}
    if args.reference_image:
        data["reference_image"] = str(args.reference_image.resolve())
    config = CalibrationConfig(**data)
    if args.out.exists():
        parser.error("Output directory already exists; use a new path")
    rows, summary = calibrate(config)
    args.out.mkdir(parents=True)
    (args.out/"config.json").write_text(json.dumps(asdict(config), indent=2)+"\n")
    with (args.out/"samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    source = {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in ("calibrate_size.py", "planar.py", "vision.py", "model.py")}
    report = dict(scope="Offline synthetic pixel residuals; no trajectory/controller tuning or consistency claim",
                  opencv_version=require_opencv().__version__, numpy_version=np.__version__,
                  python_version=platform.python_version(), python_executable=sys.executable,
                  reference_image_sha256=hashlib.sha256(Path(config.reference_image).read_bytes()).hexdigest()
                                         if config.reference_image else None,
                  source_sha256=source, conditions=asdict(config), **summary)
    (args.out/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
