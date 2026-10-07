"""Journal-style figures for PAPER_v4.3 (2026-10-04). No embedded headline or subtitle: the
caption carries them. Wording follows the manuscript ("baseline", not "comparator"; the logistic
regression's tuning is described in Methods, not in the legend).

    python3 make_figures_v43.py        # -> ../figs/figure2_saturation.{png,pdf,svg}
                                       #    ../figs/figureS3_weissman_list.{png,pdf,svg}

Same data and computation as make_figure1_combined.py (Figure 2) and make_figure1.py (Figure S3):
means over ten random outcome-stratified 75/25 splits, bars t(0.975, 9)*sd/sqrt(10). The existing
figure1_* files are not touched.
"""
import os
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(HERE, "out")
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "..", "figs")); os.makedirs(OUT, exist_ok=True)
TIERS = ["T1", "T2", "T3", "T4", "T5"]
INK_2, INK_3, SURFACE = "#52514e", "#8a8983", "#ffffff"
OBS = f"{U}/observable_v2/W1_repeated_reselect.csv"
WL  = f"{U}/w1_repeated_v2/W1_repeated_reselect.csv"
BLUE, ORANGE = "#2a78d6", "#eb6834"
XLAB = "Number of structured predictors in the baseline"
YLAB = "Gain in AUROC from adding notes"

def curve(path, mk):
    r = pd.read_csv(path)
    if "arm" in r.columns: r = r[r.arm == "rechosen"]
    r = r.drop_duplicates(["draw", "tier", "model"], keep="last"); r = r[r.model == mk]
    p = r.pivot(index="draw", columns="tier", values="increment").dropna(subset=["T1", "T5"])
    nv = r.groupby("tier").n_vars.first()
    m = np.array([p[t].mean() for t in TIERS]); tq = stats.t.ppf(0.975, len(p) - 1)
    ci = np.array([tq * p[t].std(ddof=1) / np.sqrt(len(p)) for t in TIERS])
    return np.array([nv[t] for t in TIERS], float), m, ci, 100 * ((p.T1 - p.T5) / p.T1).mean()

def style(ax):
    ax.set_facecolor(SURFACE); ax.axhline(0, color=INK_3, lw=0.8, zorder=1)
    ax.set_xscale("log"); ax.set_xticks([20, 50, 100, 200])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.minorticks_off()
    ax.grid(axis="y", color="#e8e7e2", lw=0.8, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(INK_3); ax.spines[sp].set_lw(0.8)
    ax.tick_params(colors=INK_2, labelsize=9.5, length=3)
    ax.set_xlabel(XLAB, fontsize=10, color=INK_2, labelpad=7)
    ax.set_ylim(-0.010, 0.080); ax.set_yticks(np.arange(0.00, 0.081, 0.02))

def label(ax, x, m, pct, color, dy):
    ax.annotate(f"−{pct:.0f}%", xy=(x, m), xytext=(14, dy), textcoords="offset points", fontsize=10,
                color=color, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.9, shrinkA=1, shrinkB=3))

# ---- Figure 2: observable specification, both learners on one axis
fig, ax = plt.subplots(figsize=(7.0, 4.6)); fig.patch.set_facecolor(SURFACE); style(ax)
for mk, lab, col, mkr, ls, dodge, dy in [("gbm", "Gradient boosting", BLUE, "o", "-", 0.965, 13),
                                          ("lrcv", "Logistic regression", ORANGE, "s", (0, (4.5, 2.2)), 1.035, -15)]:
    x, m, ci, pct = curve(OBS, mk)
    ax.errorbar(x * dodge, m, yerr=ci, color=col, lw=2.0, linestyle=ls, marker=mkr, markersize=7,
                capsize=3, capthick=1.2, elinewidth=1.2, zorder=3, label=lab)
    label(ax, x[-1] * dodge, m[-1], pct, col, dy)
ax.set_xlim(17, 400); ax.set_ylabel(YLAB, fontsize=10, color=INK_2, labelpad=8)
ax.legend(frameon=False, fontsize=9.5, loc="upper right", labelcolor=INK_2, handlelength=2.8)
fig.tight_layout()
for ext in ("png", "pdf", "svg"): fig.savefig(f"{OUT}/figure2_saturation.{ext}", dpi=300, facecolor=SURFACE)

# ---- Figure S3: observable specification vs Weissman's predictor list, one panel per learner
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True); fig.patch.set_facecolor(SURFACE)
for ax, mk, title in zip(axes, ["gbm", "lrcv"], ["Gradient boosting", "Logistic regression"]):
    style(ax); ax.set_xlim(17, 460); ax.set_title(title, fontsize=11, loc="left", pad=8)
    for path, lab, col, primary in [(OBS, "Observable specification", BLUE, True),
                                    (WL, "Weissman's predictor list", ORANGE, False)]:
        x, m, ci, pct = curve(path, mk)
        ax.errorbar(x, m, yerr=ci, color=col, lw=2.0, marker="o", markersize=7 if primary else 6,
                    markerfacecolor=col if primary else SURFACE, markeredgecolor=col, markeredgewidth=1.8,
                    capsize=3, capthick=1.2, elinewidth=1.2, zorder=3 if primary else 2, label=lab,
                    linestyle="-" if primary else (0, (4.5, 2.2)))
        label(ax, x[-1], m[-1], pct, col, 13 if primary else -17)
axes[0].set_ylabel(YLAB, fontsize=10, color=INK_2, labelpad=8)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=9.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0), labelcolor=INK_2)
fig.tight_layout(rect=(0, 0, 1, 0.92))
for ext in ("png", "pdf", "svg"): fig.savefig(f"{OUT}/figureS3_weissman_list.{ext}", dpi=300, facecolor=SURFACE)
print("wrote", OUT)
