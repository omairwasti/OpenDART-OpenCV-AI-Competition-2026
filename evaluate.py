"""Reproducible truth-state Monte Carlo and targeted ablations; no cloud required."""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from opendart.simulation import Scenario, simulate


def wilson(k, n, z=1.96):
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-half, center+half]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--out", type=Path, default=Path("results/evaluation"))
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows, configs = [], []
    for i in range(args.trials):
        cfg = Scenario(name=f"mc_{i:03d}", seed=args.seed+i,
                       transverse_position_m=rng.uniform(-1500, 1500, 2).tolist(),
                       transverse_velocity_mps=rng.uniform(-1, 1, 2).tolist(),
                       phase_rad=float(rng.uniform(0, 2*np.pi)),
                       mass_kg=float(rng.uniform(490, 510)))
        result = simulate(cfg)
        row = {k: v for k, v in result["summary"].items() if not isinstance(v, list)}
        rows.append(row)
        configs.append(result["config"])
        print(f'{i+1}/{args.trials}: {row["status"]}', flush=True)
    with (args.out/"trials.csv").open("w", newline="") as f:
        keys = sorted(set().union(*(r.keys() for r in rows)))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (args.out/"trial_configs.json").write_text(json.dumps(configs, indent=2)+"\n")
    successes = sum(r["success"] for r in rows)
    report = dict(
        description="Truth-state Monte Carlo; not perception or flight performance",
        seed=args.seed, n=args.trials, successes=successes,
        success_fraction=successes/args.trials, wilson_95_interval=wilson(successes, args.trials),
        sampled_parameters=dict(cross_position_m=[-1500, 1500],
                                cross_velocity_mps=[-1, 1], phase_rad=[0, float(2*np.pi)],
                                spacecraft_mass_kg=[490, 510]),
        delta_v_mps_median=float(np.median([r["delta_v_mps"] for r in rows])),
        delta_v_mps_max=float(max(r["delta_v_mps"] for r in rows)))
    (args.out/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
