import argparse
import csv
import json
import platform
from pathlib import Path
import numpy as np
import scipy
from .simulation import Scenario, simulate


def save(result, out, plots=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out/"config.json").write_text(json.dumps(result["config"], indent=2)+"\n")
    (out/"summary.json").write_text(json.dumps(result["summary"], indent=2, allow_nan=False)+"\n")
    (out/"runtime.json").write_text(json.dumps(dict(python=platform.python_version(),
         numpy=np.__version__, scipy=scipy.__version__), indent=2)+"\n")
    rows = result["telemetry"]
    with (out/"telemetry.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        t = np.array([r["t_s"] for r in rows])
        for axis in ("x", "y"):
            axs[0, 0].plot(t, [r[f"r_{axis}"] for r in rows], label=f"True {axis}")
            if result["config"]["navigation"] == "bearing":
                axs[0, 0].plot(t, [r[f"rhat_{axis}"] for r in rows], "--", label=f"Estimate {axis}")
        axs[0, 0].set(ylabel="Transverse relative position (m)")
        axs[0, 0].legend()
        axs[0, 1].semilogy(t, [r["range_m"] for r in rows], color="#205493")
        axs[0, 1].set(ylabel="Target center range (m)")
        for axis in ("x", "y", "z"):
            axs[1, 0].plot(t, [r[f"force_{axis}"] for r in rows], label=axis)
        axs[1, 0].set(ylabel="Applied force (N)")
        axs[1, 0].legend()
        if result["config"]["navigation"] == "bearing":
            axs[1, 1].semilogy(t, [max(r["cross_sigma_m"], 1e-6) for r in rows],
                              label="Predicted cross-track sigma")
            axs[1, 1].semilogy(t, [max(r["range_sigma_m"], 1e-6) for r in rows],
                              label="Range sigma")
            axs[1, 1].set(ylabel="Navigation uncertainty (m)")
            axs[1, 1].legend()
        else:
            axs[1, 1].plot(t, [result["config"]["mass_kg"]-r["mass_kg"] for r in rows])
            axs[1, 1].set(ylabel="Propellant consumed (kg)")
        for ax in axs.flat:
            ax.set_xlabel("Time (s)")
            ax.grid(alpha=.25)
        fig.suptitle(f'{result["config"]["name"]} | {result["summary"]["status"]}', fontsize=15)
        fig.savefig(out/"diagnostics.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="OpenDART standalone terminal-interception simulator")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/run"))
    parser.add_argument("--guidance", choices=["zem", "pd", "none"])
    parser.add_argument("--navigation", choices=["truth", "bearing"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.config.read_text()) if args.config else {}
    for key in ("guidance", "navigation", "seed"):
        if getattr(args, key) is not None:
            data[key] = getattr(args, key)
    result = simulate(Scenario(**data))
    save(result, args.out, not args.no_plots)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
