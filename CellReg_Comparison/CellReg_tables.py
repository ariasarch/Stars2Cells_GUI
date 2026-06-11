#!/usr/bin/env python3
"""
cellreg_plots.py

Generates a publication-quality dumbbell chart comparing CellReg vs
Stars2Cells F1 scores across all benchmark conditions and neuron counts.

Usage:
    python cellreg_plots.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
BASE_DIR = Path(r"C:\Users\ariAccount\Desktop\Stars2CellsPaper\CellRegComparison")
NEURON_COUNTS = [100, 250, 500, 1000]
JSON_PATHS = {
    nc: BASE_DIR / f"CellReg_{nc}_batch_perturbed_moderate" / f"all_results_{nc}n_perturbed_moderate.json"
    for nc in NEURON_COUNTS
}
OUT_DIR = BASE_DIR / "benchmark_plots"

COND_LABELS = {
    "A_rot":      "A: Rotation",
    "A_trans":    "A: Translation",
    "A_combined": "A: Combined",
    "B_dropout":  "B: Dropout",
    "B_drift":    "B: Drift",
    "B_rot":      "B: Rotation",
    "B_combined": "B: Combined",
    "C_walk":     "C: Random Walk",
}

S2C_COLOR = "#2BA4A0"
CR_COLOR  = "#E8734A"
GAP_COLOR = "#D0D0D0"


def load_all():
    data = {}
    for nc in NEURON_COUNTS:
        p = JSON_PATHS[nc]
        if p.exists():
            with open(p) as f:
                data[nc] = json.load(f)
    return data


def get_f1(data, nc, pipeline, condition):
    try:
        v = data[nc]["_summary"][pipeline][condition]["f1"]
        return v if v is not None else np.nan
    except (KeyError, TypeError):
        return np.nan


def get_overall_f1(data, nc, pipeline):
    try:
        v = data[nc]["_summary"][pipeline]["_overall"]["f1"]
        return v if v is not None else np.nan
    except (KeyError, TypeError):
        return np.nan


def plot_dumbbell(data):
    ncs = [nc for nc in NEURON_COUNTS if nc in data]
    n_panels = len(ncs)

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]

    for ax, nc in zip(axes, ncs):
        conditions = data[nc]["_meta"]["conditions"]
        # Reverse so first condition is at top
        conditions = conditions[::-1]
        n = len(conditions)
        y_pos = np.arange(n)

        cr_vals  = [get_f1(data, nc, "cellreg", c) * 100 for c in conditions]
        s2c_vals = [get_f1(data, nc, "stars2cells", c) * 100 for c in conditions]

        # Draw connecting lines (the "dumbbells")
        for i in range(n):
            if not (np.isnan(cr_vals[i]) or np.isnan(s2c_vals[i])):
                ax.plot([cr_vals[i], s2c_vals[i]], [i, i],
                        color=GAP_COLOR, linewidth=2.5, zorder=1)

        # Dots
        ax.scatter(cr_vals, y_pos, color=CR_COLOR, s=70, zorder=2,
                   label="ROI Matching", edgecolors="white", linewidths=0.5)
        ax.scatter(s2c_vals, y_pos, color=S2C_COLOR, s=70, zorder=2,
                   label="Stars2Cells", edgecolors="white", linewidths=0.5)

        # Overall row
        cr_ov  = get_overall_f1(data, nc, "cellreg") * 100
        s2c_ov = get_overall_f1(data, nc, "stars2cells") * 100
        ov_y = -1.2
        if not (np.isnan(cr_ov) or np.isnan(s2c_ov)):
            ax.plot([cr_ov, s2c_ov], [ov_y, ov_y],
                    color=GAP_COLOR, linewidth=2.5, zorder=1)
        ax.scatter([cr_ov], [ov_y], color=CR_COLOR, s=90, zorder=2,
                   marker="D", edgecolors="white", linewidths=0.5)
        ax.scatter([s2c_ov], [ov_y], color=S2C_COLOR, s=90, zorder=2,
                   marker="D", edgecolors="white", linewidths=0.5)

        # Separator line
        ax.axhline(y=-0.6, color="#cccccc", linewidth=0.5, linestyle="--")

        # Labels — each panel gets its own independent y-axis
        labels = [COND_LABELS.get(c, c) for c in conditions] + ["OVERALL"]
        all_y = list(y_pos) + [ov_y]
        ax.set_yticks(all_y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_ylim(ov_y - 0.5, n - 0.5)
        ax.set_xlim(-2, 105)
        ax.set_xlabel("F1 Score (%)", fontsize=10)
        ax.set_title(f"{nc} Neurons", fontsize=12, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
        ax.set_axisbelow(True)

    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    data = load_all()
    if not data:
        print("No result JSONs found.")
        exit(1)

    OUT_DIR.mkdir(exist_ok=True, parents=True)

    fig = plot_dumbbell(data)
    fig.savefig(OUT_DIR / "dumbbell_f1_moderate.png", dpi=300, bbox_inches="tight")
    print(f"  Saved → {OUT_DIR / 'dumbbell_f1_moderate.png'}")

#!/usr/bin/env python3
# """Compare CellReg F1: original footprints vs perturbed (moderate)."""

# import json
# import numpy as np
# from pathlib import Path

# BASE_DIR = Path(r"C:\Users\ariAccount\Desktop")
# NCS = [100, 250, 500, 1000]

# ORIG = {nc: BASE_DIR / f"CellReg_{nc}_batch" / f"all_results_{nc}n.json" for nc in NCS}
# PERT = {nc: BASE_DIR / f"CellReg_{nc}_batch_perturbed_moderate" / f"all_results_{nc}n_perturbed_moderate.json" for nc in NCS}

# ALL_CONDS = ["A_rot","A_trans","A_combined","B_dropout","B_drift","B_rot","B_combined","C_walk"]

# def load(paths):
#     d = {}
#     for nc, p in paths.items():
#         if p.exists():
#             with open(p) as f: d[nc] = json.load(f)
#     return d

# def f1(data, nc, cond):
#     try:
#         v = data[nc]["_summary"]["cellreg"][cond]["f1"]
#         return v if v is not None else np.nan
#     except: return np.nan

# def ov(data, nc):
#     try:
#         v = data[nc]["_summary"]["cellreg"]["_overall"]["f1"]
#         return v if v is not None else np.nan
#     except: return np.nan

# def pct(v):
#     return f"{v*100:5.1f}%" if not np.isnan(v) else "    —"

# orig, pert = load(ORIG), load(PERT)
# ncs = sorted(set(orig) & set(pert))

# for nc in ncs:
#     conds = orig[nc]["_meta"]["conditions"]
#     print(f"\n{'═'*52}")
#     print(f"  {nc}n — CellReg F1: Original vs Perturbed (moderate)")
#     print(f"{'═'*52}")
#     print(f"  {'Condition':<15} {'Orig':>8} {'Pert':>8} {'Δ':>8}")
#     print(f"  {'─'*42}")
#     for c in conds:
#         o, p = f1(orig, nc, c), f1(pert, nc, c)
#         d = f"{(p-o)*100:+5.1f}pp" if not (np.isnan(o) or np.isnan(p)) else "    —"
#         print(f"  {c:<15} {pct(o):>8} {pct(p):>8} {d:>8}")
#     print(f"  {'─'*42}")
#     o, p = ov(orig, nc), ov(pert, nc)
#     d = f"{(p-o)*100:+5.1f}pp" if not (np.isnan(o) or np.isnan(p)) else "    —"
#     print(f"  {'OVERALL':<15} {pct(o):>8} {pct(p):>8} {d:>8}")

# print()