"""Offline campaign figures and numeric report; no simulation feedback or tuning."""
import csv
import json
import os
from pathlib import Path

import numpy as np

CONDITIONS=("NOMINAL","IMAGE_DEGRADED","EXTENDED_BLACKOUT","RANGE_PRIOR_ERROR")
LABELS=("Nominal","Image\ndegraded","Extended\nblackout","Range\nprior error")
COLORS={"ANALYTIC":"#2867a1","REFERENCE":"#ce7b1d"}


def create_plots_and_report(out,metrics,groups,matched):
    out=Path(out);dest=out/"aggregate"
    os.environ["MPLCONFIGDIR"]=str(dest/"matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":.22})
    def save(fig,name):
        fig.savefig(dest/name,dpi=160);plt.close(fig)
    def box(axis,key,title,scale=None):
        for i,condition in enumerate(CONDITIONS):
            for appearance,shift in [("ANALYTIC",-.18),("REFERENCE",.18)]:
                values=[m[key] for m in metrics if m["group"]=="guided" and m["condition"]==condition
                        and m["appearance"]==appearance and m[key] is not None]
                axis.boxplot([values],positions=[i+shift],widths=.28,patch_artist=True,
                             boxprops=dict(facecolor=COLORS[appearance],alpha=.55),
                             medianprops=dict(color="black"),manage_ticks=False)
        axis.set_xticks(range(4),LABELS);axis.set_title(title)
        if scale:axis.set_yscale(scale)
    fig,axis=plt.subplots(figsize=(10,4),constrained_layout=True)
    for appearance,shift in [("ANALYTIC",-.18),("REFERENCE",.18)]:
        gs=[next(g for g in groups if g["appearance"]==appearance and g["condition"]==c) for c in CONDITIONS]
        x=np.arange(4)+shift;y=np.array([g["success_fraction"] for g in gs])
        low=np.array([g["wilson_95_interval"][0] for g in gs]);high=np.array([g["wilson_95_interval"][1] for g in gs])
        axis.bar(x,y,width=.3,color=COLORS[appearance],label=appearance)
        axis.errorbar(x,y,yerr=np.vstack([y-low,high-y]),fmt="none",color="black",capsize=4)
        for xi,g in zip(x,gs):axis.text(xi,1.04,f"{g['outcome_counts']['TARGET_IMPACT']}/{g['n']}",ha="center",fontsize=9)
    axis.set_xticks(range(4),LABELS);axis.set(ylim=(0,1.15),ylabel="Target-impact fraction",
        title="Guided cases only: 12 seeds per appearance/condition; Wilson 95% intervals")
    axis.legend(loc="lower right");save(fig,"success_rates.png")
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for axis,key,title,scale in zip(axes,["final_position_error_m","peak_position_error_m","delta_v_mps"],
                                   ["Final position error (m)","Peak position error (m)","Correction delta-v (m/s)"],["log","log",None]):
        box(axis,key,title,scale)
    fig.suptitle("Guided-run distributions: blue analytic; orange reference; boxes = IQR")
    save(fig,"accuracy_and_correction.png")
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for axis,key,title,scale in zip(axes,["max_abs_standardized_range_error","range_95_frame_fraction","state_95_frame_fraction"],
             ["Max |range error / sigma|","Range ±1.96 sigma frame fraction","4-state 95% ellipsoid frame fraction"],["log",None,None]):
        box(axis,key,title,scale)
    axes[0].axhline(1.96,color="gray",ls="--",lw=.8)
    for axis in axes[1:]:axis.set_ylim(-.03,1.03)
    fig.suptitle("Per-run diagnostic distributions; frames are correlated, not independent trials")
    save(fig,"uncertainty_diagnostics.png")
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    box(axes[0],"nav_hold_s","NAV_HOLD duration (s)")
    box(axes[1],"longest_no_update_s","Longest sampled update outage (s)")
    for appearance,shift in [("ANALYTIC",-.18),("REFERENCE",.18)]:
        for i,condition in enumerate(CONDITIONS):
            vals=[e["recovery_delay_s"] for m in metrics if m["group"]=="guided" and m["appearance"]==appearance
                  and m["condition"]==condition for e in m["blackout_recovery"] if e["recovery_delay_s"] is not None]
            axes[2].boxplot([vals],positions=[i+shift],widths=.28,patch_artist=True,
                boxprops=dict(facecolor=COLORS[appearance],alpha=.55),medianprops=dict(color="black"),manage_ticks=False)
    axes[2].set_xticks(range(4),LABELS);axes[2].set_title("Accepted-update delay after blackout (s)")
    fig.suptitle("Loss/recovery evidence: blue analytic; orange reference")
    save(fig,"loss_recovery.png")
    failures=[m for m in metrics if m["group"]=="guided" and not m["success"]]
    chosen=[next(m for m in metrics if m["run_id"]=="analytic_nominal_s07"),
            next(m for m in metrics if m["run_id"]=="reference_image_degraded_s07")]
    chosen.append(max(failures,key=lambda m:m["final_position_error_m"]) if failures else
                  next(m for m in metrics if m["run_id"]=="analytic_no_measurement_s07"))
    fig,axes=plt.subplots(len(chosen),3,figsize=(13,8),constrained_layout=True)
    for axes_row,m in zip(axes,chosen):
        folder=out/"runs"/m["run_id"]
        rows=list(csv.DictReader((folder/"telemetry.csv").open()))
        ts=[float(r["elapsed_s"]) for r in rows]
        axes_row[0].plot(ts,[float(r["eval_position_error_m"]) for r in rows]);axes_row[0].set_yscale("log")
        axes_row[1].plot(ts,[float(r["eval_range_error_sigma"]) for r in rows]);axes_row[1].axhline(0,color="gray",lw=.6)
        axes_row[2].plot(ts,[float(r["force_n"]) for r in rows],label="force N")
        axes_row[2].fill_between(ts,0,10,where=[r["mode"]=="NAV_HOLD" for r in rows],color="orange",alpha=.2,label="NAV_HOLD")
        axes_row[0].set_ylabel(m["run_id"].replace("_"," ")+"\n"+m["status"],fontsize=8)
        for axis in axes_row:axis.set_xlabel("Elapsed time (s)")
    for axis,title in zip(axes[0],["Position error (m)","Signed range error / sigma","Applied force (N)"]):axis.set_title(title)
    axes[0,2].legend(fontsize=8)
    fig.suptitle("Predetermined nominal/degraded seed 7; third row = worst-error guided failure or no-image control")
    save(fig,"representative_runs.png")
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for appearance in COLORS:
        ms=sorted([m for m in metrics if m["group"]=="radius_sensitivity" and m["appearance"]==appearance],key=lambda m:m["radius_factor"])
        x=[100*(m["radius_factor"]-1) for m in ms]
        for axis,key in zip(axes,["final_position_error_m","final_range_error_m","final_standardized_range_error"]):
            axis.plot(x,[m[key] for m in ms],"o-",label=appearance,color=COLORS[appearance])
        for xi,m in zip(x,ms):axes[0].annotate(m["status"].replace("TARGET_",""),(xi,m["final_position_error_m"]),fontsize=7,xytext=(0,5),textcoords="offset points",ha="center")
    for axis,title in zip(axes,["Final position error (m)","Signed final range error (m)","Signed final range error / sigma"]):
        axis.set(title=title,xlabel="Navigation radius assumption error (%)");axis.axhline(0,color="gray",lw=.6)
    axes[0].legend();fig.suptitle("Separate sensitivity diagnostic: seed 7; rendered radius always 88 m")
    save(fig,"radius_sensitivity.png")
    lines=["# Frozen synthetic visual campaign: numeric results", "",
           "96 guided cases; 4 negative controls; 10 separate radius-assumption cases.",
           "Primary Wilson intervals use independent seeds within each appearance/condition. Matched seeds across appearances/conditions are paired; do not pool all cases as independent trials.","",
           "| Appearance | Condition | Impact / n | MISS | Wrong body | Timeout | Wilson 95% |",
           "|---|---|---:|---:|---:|---:|---|"]
    for g in groups:
        c=g["outcome_counts"];lo,hi=g["wilson_95_interval"]
        lines.append(f"| {g['appearance']} | {g['condition']} | {c['TARGET_IMPACT']}/{g['n']} | {c['MISS']} | {c['WRONG_BODY_IMPACT']} | {c['TIMEOUT']} | [{lo:.4f}, {hi:.4f}] |")
    for group in ["control","radius_sensitivity"]:
        lines += ["",f"## {group}","","| Run | Outcome | Final position error m | Delta-v m/s | Max standardized range error |","|---|---|---:|---:|---:|"]
        for m in metrics:
            if m["group"]==group:
                lines.append(f"| {m['run_id']} | {m['status']} | {m['final_position_error_m']:.3f} | {m['delta_v_mps']:.3f} | {m['max_abs_standardized_range_error']:.3f} |")
    lines += ["","Run metrics, robust spreads, NIS/NEES and coverage: `run_metrics.json`, `groups.json`, `matched_epoch_coverage.json`, `runs.csv`.",
              "Frame-level diagnostics are descriptive; full covariance/measurement assumptions are not validated by a nominal confidence threshold.",
              "Physical-radius diagnostic and controls are excluded from guided success statistics."]
    (dest/"NUMERIC_RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
