"""Frozen local visual campaign: initialize, run isolated cases, then aggregate.

See docs/VISUAL_EVALUATION.md. Evaluation truth never enters the estimator.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata as metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
from scipy.stats import chi2

from .planar import PlanarConfig, simulate_planar
from .vision import require_opencv

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("NOMINAL", "IMAGE_DEGRADED", "EXTENDED_BLACKOUT", "RANGE_PRIOR_ERROR")
EPOCHS = (0., 30., 100., 199., 220., 240., 300., 500., 580.)
OUTCOMES = ("TARGET_IMPACT", "MISS", "WRONG_BODY_IMPACT", "TIMEOUT")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_info():
    import scipy
    cv = require_opencv()
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                logical_cpus=os.cpu_count(), numpy=np.__version__, scipy=scipy.__version__,
                opencv=cv.__version__, opencv_threads=cv.getNumThreads(),
                opencv_distributions={d.metadata["Name"]:d.version for d in metadata.distributions()
                                      if d.metadata["Name"].lower().startswith("opencv-")},
                package_version=metadata.version("opendart-dynamics"), pid=os.getpid(),
                thread_environment={key:os.environ.get(key) for key in
                                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    group: str
    appearance: str
    condition: str
    seed: int
    radius_factor: float
    config: dict

    def relative_path(self):
        parent = "radius_sensitivity" if self.group == "radius_sensitivity" else "runs"
        return Path(parent)/self.run_id


def campaign_plan(base, reference_image):
    """Only the documented perturbations differ; range errors are prior-only."""
    base.validate()
    if (base.size_radius_bias_px != 1.07 or not base.require_opencv5
            or base.navigation_radius_m != 88. or not base.control
            or not base.measurements or not base.use_apparent_size):
        raise ValueError("Campaign requires the accepted Task 1 bias and OpenCV 5")
    reference = str(Path(reference_image).resolve())
    specs = []
    for appearance in ("ANALYTIC", "REFERENCE"):
        initial = replace(base, reference_image=reference if appearance == "REFERENCE" else None)
        for condition in CONDITIONS:
            for seed in range(12):
                cfg = replace(initial, seed=seed)
                if condition == "IMAGE_DEGRADED":
                    cfg = replace(cfg, noise_dn=3., dropout_probability=.10)
                elif condition == "EXTENDED_BLACKOUT":
                    cfg = replace(cfg, blackout_s=[[200., 240.]])
                elif condition == "RANGE_PRIOR_ERROR":
                    error = list(cfg.estimate_error)
                    error[1] = (.1 if seed % 2 == 0 else -.1)*cfg.initial_range_m
                    cfg = replace(cfg, estimate_error=error)
                name = f"{appearance.lower()}_{condition.lower()}_s{seed:02d}"
                specs.append(RunSpec(name, "guided", appearance, condition, seed, 1., asdict(cfg)))
        for condition in ("BALLISTIC", "NO_MEASUREMENT"):
            cfg = replace(initial, seed=7, control=condition != "BALLISTIC",
                          measurements=condition != "NO_MEASUREMENT")
            name = f"{appearance.lower()}_{condition.lower()}_s07"
            specs.append(RunSpec(name, "control", appearance, condition, 7, 1., asdict(cfg)))
        for factor in (.90, .95, 1., 1.05, 1.10):
            cfg = replace(initial, seed=7, navigation_radius_m=base.navigation_radius_m*factor)
            name = f"{appearance.lower()}_radius_{round(100*factor):03d}_s07"
            specs.append(RunSpec(name, "radius_sensitivity", appearance, "RADIUS_SENSITIVITY", 7,
                                 factor, asdict(cfg)))
    assert len({spec.run_id for spec in specs}) == len(specs) == 110
    for spec in specs:
        PlanarConfig(**spec.config).validate()
    return specs


def wilson(k, n, z=1.959963984540054):
    if not 0 <= k <= n or n < 1:
        raise ValueError("Wilson requires 0 <= k <= n and n > 0")
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0., float(center-half)), min(1., float(center+half))]


def distribution(values):
    a = np.array([float(x) for x in values if x is not None and np.isfinite(float(x))])
    if not a.size:
        return dict(n=0, mean=None, median=None, q1=None, q3=None, p05=None, p95=None, min=None, max=None)
    return dict(n=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                q1=float(np.percentile(a,25)), q3=float(np.percentile(a,75)),
                p05=float(np.percentile(a,5)), p95=float(np.percentile(a,95)),
                min=float(a.min()), max=float(a.max()))


def episodes(rows, predicate, end_time):
    found, start = [], None
    for row in rows:
        active = predicate(row)
        t = float(row["elapsed_s"])
        if active and start is None:
            start = t
        elif not active and start is not None:
            found.append(dict(start_s=start, end_s=t, duration_s=t-start, recovered=True))
            start = None
    if start is not None:
        found.append(dict(start_s=start, end_s=end_time, duration_s=end_time-start, recovered=False))
    return found


def analyze_run(result, spec, wall_s):
    """Offline scoring of posterior state/P at matching truth epochs."""
    rows, summary = result["telemetry"], result["summary"]
    cfg = spec.config
    diagnostics = []
    for row in rows:
        error = np.array([row["estimate_x_m"]-row["eval_relative_x_m"],
                          row["estimate_y_m"]-row["eval_relative_y_m"],
                          row["estimate_vx_mps"]-row["eval_relative_vx_mps"],
                          row["estimate_vy_mps"]-row["eval_relative_vy_mps"]])
        covariance = np.zeros((4,4))
        for i in range(4):
            for j in range(i,4):
                covariance[i,j] = covariance[j,i] = row[f"estimate_cov_{i}{j}"]
        valid = np.all(np.isfinite(covariance)) and np.linalg.eigvalsh(covariance).min() > 0
        nees = float(error@np.linalg.solve(covariance,error)) if valid else None
        pos_nees = float(error[:2]@np.linalg.solve(covariance[:2,:2],error[:2])) if valid else None
        z = float(row["eval_range_error_sigma"])
        selected_error = closer_primary = None
        if np.isfinite(row["measured_u_px"]):
            theta = row["theta_rad"]
            right, forward = np.array([np.sin(theta),-np.cos(theta)]), np.array([np.cos(theta),np.sin(theta)])
            pixels = []
            for body in ("primary", "target"):
                delta = np.array([row[f"{body}_x_m"]-row["spacecraft_x_m"],
                                  row[f"{body}_y_m"]-row["spacecraft_y_m"]])
                depth = forward@delta
                pixels.append(cfg["camera_f_px"]*(right@delta)/depth+(cfg["camera_width"]-1)/2 if depth > 0 else None)
            if pixels[1] is not None:
                selected_error = float(abs(row["measured_u_px"]-pixels[1]))
                if pixels[0] is not None:
                    closer_primary = bool(abs(row["measured_u_px"]-pixels[0]) < selected_error)
        diagnostics.append(dict(elapsed_s=float(row["elapsed_s"]),
            nees4=nees, position_nees2=pos_nees, covariance_positive_definite=bool(valid),
            range_95_covered=bool(abs(z)<=1.959963984540054) if np.isfinite(z) else None,
            state_95_covered=bool(nees<=chi2.ppf(.95,4)) if nees is not None else None,
            position_95_covered=bool(pos_nees<=chi2.ppf(.95,2)) if pos_nees is not None else None,
            eval_selected_target_u_error_px=selected_error,
            eval_selected_closer_primary=closer_primary))
    outage = episodes(rows, lambda r:not r["accepted"], summary["elapsed_s"])
    tracker_loss = episodes(rows, lambda r:r["tracker"]=="LOST", summary["elapsed_s"])
    blackout_recovery = []
    for start,end in cfg["blackout_s"]:
        accepted = next((r for r in rows if r["elapsed_s"]>=end and r["accepted"]), None)
        blackout_recovery.append(dict(start_s=start,end_s=end,
            first_accepted_after_s=accepted["elapsed_s"] if accepted else None,
            recovery_delay_s=accepted["elapsed_s"]-end if accepted else None))
    weights = np.diff([r["elapsed_s"] for r in rows]+[summary["elapsed_s"]])
    coast = float(sum(dt for dt,r in zip(weights,rows) if r["mode"]=="TERMINAL_COAST"))
    metrics = dict(run_id=spec.run_id, group=spec.group, appearance=spec.appearance,
        condition=spec.condition, seed=spec.seed, radius_factor=spec.radius_factor,
        status=summary["status"], success=summary["success"], elapsed_s=summary["elapsed_s"],
        wall_s=wall_s, accepted_updates=summary["accepted_updates"], missing_updates=summary["missing_updates"],
        rejected_updates=summary["rejected_updates"], update_reason_counts=summary["update_reason_counts"],
        nav_hold_s=summary["nav_hold_s"], terminal_coast_s=coast, saturation_s=summary["saturation_s"],
        delta_v_mps=summary["delta_v_mps"], propellant_used_kg=summary["propellant_used_kg"],
        final_position_error_m=summary["final_estimation_error_m"],
        peak_position_error_m=summary["eval_peak_position_error_m"],
        final_range_error_m=summary["estimate_final_range_m"]-summary["eval_final_true_range_m"],
        abs_final_range_error_m=abs(summary["estimate_final_range_m"]-summary["eval_final_true_range_m"]),
        peak_abs_range_error_m=max(abs(r["eval_range_error_m"]) for r in rows),
        final_true_range_m=summary["eval_final_true_range_m"], final_estimated_range_m=summary["estimate_final_range_m"],
        final_radial_sigma_m=summary["estimate_final_range_sigma_m"],
        final_standardized_range_error=summary["eval_final_range_error_sigma"],
        max_abs_standardized_range_error=summary["eval_max_abs_sampled_range_error_sigma"],
        final_sample_position_sigma_m=rows[-1]["estimate_position_sigma_m"],
        nis=distribution([r["nis"] for r in rows]),
        accepted_nis=distribution([r["nis"] for r in rows if r["accepted"]]),
        standardized_range_error=distribution([r["eval_range_error_sigma"] for r in rows]),
        nees4=distribution([d["nees4"] for d in diagnostics]),
        range_95_frame_fraction=float(np.mean([d["range_95_covered"] for d in diagnostics if d["range_95_covered"] is not None])),
        position_95_frame_fraction=float(np.mean([d["position_95_covered"] for d in diagnostics if d["position_95_covered"] is not None])),
        state_95_frame_fraction=float(np.mean([d["state_95_covered"] for d in diagnostics if d["state_95_covered"] is not None])),
        invalid_covariance_frames=sum(not d["covariance_positive_definite"] for d in diagnostics),
        candidate_count=distribution([r["candidates"] for r in rows]),
        tracker_status_counts={s:sum(r["tracker"]==s for r in rows) for s in sorted({r["tracker"] for r in rows})},
        reacquired_samples=sum(r["tracker"]=="REACQUIRED" for r in rows),
        no_update_episodes=outage, tracker_lost_episodes=tracker_loss,
        longest_no_update_s=max((e["duration_s"] for e in outage),default=0.),
        blackout_recovery=blackout_recovery,
        selected_target_u_error_px=distribution([d["eval_selected_target_u_error_px"] for d in diagnostics]),
        selected_closer_primary_samples=sum(d["eval_selected_closer_primary"] is True for d in diagnostics),
        matched_epochs=[d for d in diagnostics if d["elapsed_s"] in EPOCHS],
        max_force_n=max(r["force_n"] for r in rows),
        mode_counts={s:sum(r["mode"]==s for r in rows) for s in sorted({r["mode"] for r in rows})})
    if spec.group == "control":
        assert summary["delta_v_mps"] == 0 and metrics["max_force_n"] == 0
        if spec.condition == "NO_MEASUREMENT":
            assert summary["accepted_updates"] == 0
    return metrics, diagnostics


def frozen_paths(base_path, reference):
    return sorted(set(list((ROOT/"opendart").glob("*.py"))+
                      [Path(base_path).resolve(),Path(reference).resolve(),ROOT/"pyproject.toml"]))


def initialize(out, base_path, reference):
    out = Path(out).resolve()
    base = PlanarConfig(**json.loads(Path(base_path).read_text()))
    specs = campaign_plan(base, reference)
    if not Path(reference).is_file():
        raise ValueError("Bundled reference image not found")
    info = runtime_info()
    if int(info["opencv"].split(".")[0]) != 5:
        raise RuntimeError("Campaign requires actual OpenCV 5")
    out.mkdir(parents=True, exist_ok=False)
    (out/"runs").mkdir(); (out/"radius_sensitivity").mkdir()
    paths = frozen_paths(base_path, reference)
    hashes = {str(p):digest(p) for p in paths}
    for p in paths:
        relative = p.relative_to(ROOT) if p.is_relative_to(ROOT) else Path("external_inputs")/p.name
        target = out/"frozen_source"/relative
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(p.read_bytes())
    write_json(out/"machine.json",info)
    write_json(out/"manifest.json",dict(schema_version=1,
        design="96 guided + 4 negative controls; 10 separate navigation-radius sensitivity cases",
        base_config=asdict(base), baseline_range_component_m=base.initial_range_m,
        baseline_geometric_range_m=float(np.hypot(base.initial_cross_m,base.initial_range_m)),
        paired_seed_note="Seeds are independent within each appearance/condition; conditions and appearances use matched seeds. Do not pool them as independent trials.",
        frozen_sha256=hashes, runs=[asdict(s) for s in specs],
        execution=dict(default_workers=2,opencv_threads_per_worker=1,blas_threads_per_worker=1,
                       fidelity="Unchanged 512px camera, dt/integrator/EKF/controller; no dashboard/video overhead")))
    return specs


def verify_frozen(manifest):
    changed = [p for p,h in manifest["frozen_sha256"].items() if not Path(p).is_file() or digest(p)!=h]
    if changed:
        raise RuntimeError(f"Frozen source/input changed; stop and use a new campaign version: {changed}")


def worker_init():
    require_opencv().setNumThreads(1)


def run_one(out, spec_data):
    spec = RunSpec(**spec_data)
    destination = Path(out)/spec.relative_path()
    destination.mkdir(exist_ok=False)
    write_json(destination/"config.json",spec.config)
    write_json(destination/"case.json",asdict(spec))
    start = time.monotonic()
    info = runtime_info()
    write_json(destination/"runtime.json",info)
    try:
        with (destination/"telemetry.csv").open("x", newline="") as stream:
            csv_writer = None
            def callback(cfg,binary,row,image,candidates,selected,rows):
                nonlocal csv_writer
                if csv_writer is None:
                    csv_writer = csv.DictWriter(stream,fieldnames=list(row)); csv_writer.writeheader()
                csv_writer.writerow(row)
                if len(rows)%60 == 0:
                    stream.flush()
                return None
            result = simulate_planar(PlanarConfig(**spec.config),callback)
        elapsed = time.monotonic()-start
        write_json(destination/"summary.json",result["summary"])
        metrics, diagnostics = analyze_run(result,spec,elapsed)
        write_json(destination/"metrics.json",metrics)
        with (destination/"diagnostics.csv").open("x",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(diagnostics[0])); writer.writeheader();writer.writerows(diagnostics)
        if not require_opencv().imwrite(str(destination/"last_camera.png"),result["final_camera"]):
            raise OSError("Failed to save camera output")
        files = ("config.json","case.json","runtime.json","summary.json","metrics.json","telemetry.csv","diagnostics.csv","last_camera.png")
        write_json(destination/"complete.json",dict(wall_s=time.monotonic()-start,sha256={f:digest(destination/f) for f in files}))
        return dict(run_id=spec.run_id,status=metrics["status"],wall_s=elapsed,path=str(destination))
    except Exception:
        write_json(destination/"execution_error.json",dict(traceback=traceback.format_exc(),wall_s=time.monotonic()-start))
        raise


def execute(out, workers=2, ids=None):
    out = Path(out).resolve()
    manifest = json.loads((out/"manifest.json").read_text())
    verify_frozen(manifest)
    specs = [RunSpec(**s) for s in manifest["runs"]]
    if ids and not set(ids).issubset({s.run_id for s in specs}):
        raise ValueError("Unknown run id")
    pending = []
    for spec in specs:
        if ids and spec.run_id not in ids:
            continue
        path = out/spec.relative_path()
        if path.exists():
            if not (path/"complete.json").is_file():
                raise RuntimeError(f"Incomplete run preserved; no overwrite/retry in place: {path}")
            if json.loads((path/"config.json").read_text()) != spec.config:
                raise RuntimeError("Completed config differs from manifest")
            complete = json.loads((path/"complete.json").read_text())
            if any(digest(path/f)!=h for f,h in complete["sha256"].items()):
                raise RuntimeError("Completed run artifact changed")
        else:
            pending.append(spec)
    if not 1 <= workers <= 4:
        raise ValueError("Use one to four local workers; default two")
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    batch = len(list(out.glob("batch_*.json")))+1
    start = time.monotonic(); completions=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),initializer=worker_init) as pool:
        futures = [pool.submit(run_one,str(out),asdict(s)) for s in pending]
        try:
            for future in as_completed(futures):
                record = future.result();completions.append(record)
                with (out/"progress.jsonl").open("a") as stream:
                    stream.write(json.dumps(record)+"\n")
                print(f"{len(completions)}/{len(pending)} {record['run_id']}: {record['status']} ({record['wall_s']:.2f}s)",flush=True)
        except Exception:
            for future in futures:
                future.cancel()
            raise
    verify_frozen(manifest)
    write_json(out/f"batch_{batch:03d}.json",dict(workers=workers,wall_s=time.monotonic()-start,completed=completions))


def aggregate_groups(metrics):
    if any(m["status"] not in OUTCOMES for m in metrics):
        raise ValueError("Unknown/exception outcome cannot be folded into campaign statistics")
    groups=[]
    guided=[m for m in metrics if m["group"]=="guided"]
    fields=("delta_v_mps","final_position_error_m","peak_position_error_m","abs_final_range_error_m",
            "peak_abs_range_error_m","nav_hold_s","terminal_coast_s","accepted_updates","missing_updates",
            "rejected_updates","reacquired_samples","longest_no_update_s","final_radial_sigma_m",
            "max_abs_standardized_range_error","range_95_frame_fraction","position_95_frame_fraction",
            "state_95_frame_fraction","wall_s")
    for appearance,condition in sorted({(m["appearance"],m["condition"]) for m in guided}):
        ms=[m for m in guided if m["appearance"]==appearance and m["condition"]==condition]
        successes=sum(m["status"]=="TARGET_IMPACT" for m in ms)
        groups.append(dict(appearance=appearance,condition=condition,n=len(ms),
            outcome_counts={s:sum(m["status"]==s for m in ms) for s in OUTCOMES},success_fraction=successes/len(ms),
            wilson_95_interval=wilson(successes,len(ms)),
            distributions={key:distribution([m[key] for m in ms]) for key in fields},
            update_reason_counts={r:sum(m["update_reason_counts"].get(r,0) for m in ms) for r in sorted(ms[0]["update_reason_counts"])},
            per_run_nis_median=distribution([m["nis"]["median"] for m in ms]),
            per_run_nees4_median=distribution([m["nees4"]["median"] for m in ms])))
    return groups


def aggregate(out):
    out=Path(out).resolve();manifest=json.loads((out/"manifest.json").read_text());verify_frozen(manifest)
    metrics=[]
    for data in manifest["runs"]:
        spec=RunSpec(**data);path=out/spec.relative_path()
        if not (path/"complete.json").is_file():
            raise RuntimeError(f"Aggregate only after every run is complete: {spec.run_id}")
        complete=json.loads((path/"complete.json").read_text())
        if any(digest(path/f)!=h for f,h in complete["sha256"].items()):
            raise RuntimeError(f"Run artifact changed: {spec.run_id}")
        metrics.append(json.loads((path/"metrics.json").read_text()))
    dest=out/"aggregate";dest.mkdir(exist_ok=False)
    groups=aggregate_groups(metrics)
    write_json(dest/"groups.json",groups)
    write_json(dest/"run_metrics.json",metrics)
    scalar_keys=[k for k,v in metrics[0].items() if not isinstance(v,(dict,list))]
    with (dest/"runs.csv").open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=scalar_keys);writer.writeheader()
        writer.writerows({k:m[k] for k in scalar_keys} for m in metrics)
    matched=[]
    for group in groups:
        ms=[m for m in metrics if m["group"]=="guided" and m["appearance"]==group["appearance"] and m["condition"]==group["condition"]]
        for epoch in EPOCHS:
            ds=[d for m in ms for d in m["matched_epochs"] if d["elapsed_s"]==epoch]
            matched.append(dict(appearance=group["appearance"],condition=group["condition"],elapsed_s=epoch,n=len(ds),
                range_95_fraction=float(np.mean([d["range_95_covered"] for d in ds])) if ds else None,
                position_95_fraction=float(np.mean([d["position_95_covered"] for d in ds])) if ds else None,
                state_95_fraction=float(np.mean([d["state_95_covered"] for d in ds])) if ds else None,
                nees4=distribution([d["nees4"] for d in ds])))
    write_json(dest/"matched_epoch_coverage.json",matched)
    from .visual_campaign_analysis import create_plots_and_report
    create_plots_and_report(out,metrics,groups,matched)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest="command",required=True)
    init=subs.add_parser("init");init.add_argument("--out",type=Path,required=True)
    init.add_argument("--base-config",type=Path,default=ROOT/"configs/visual_2d.json")
    init.add_argument("--reference-image",type=Path,default=ROOT/"assets/dart_0401929788_46871_01_iof.png")
    run=subs.add_parser("run");run.add_argument("--out",type=Path,required=True)
    run.add_argument("--workers",type=int,default=2);run.add_argument("--ids",nargs="+")
    agg=subs.add_parser("aggregate");agg.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="init":initialize(args.out,args.base_config,args.reference_image)
    elif args.command=="run":execute(args.out,args.workers,args.ids)
    else:aggregate(args.out)


if __name__=="__main__":
    main()
