"""Supplementary Figures S1 (coverage test) and S2 (per-partition curves).
Reads the per-draw result CSVs only; no patient-level data."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
U = os.environ.get("W1_OUT", os.path.join(HERE, "out"))
FIG = os.environ.get("FIG_OUT", os.path.join(HERE, "..", "figs"))
os.makedirs(FIG, exist_ok=True)
INK_1, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
BLUE, ORANGE, GRID = "#2a78d6", "#eb6834", "#e8e7e2"
TIERS = ["T1", "T2", "T3", "T4", "T5"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})

def style(ax):
    ax.axhline(0, color=INK_3, lw=0.8, zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(INK_3); ax.spines[sp].set_lw(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=3)

def tci(x):
    x = np.asarray(x, float); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), h

# ---------------- Figure S2: every partition's curve
SPECS = [("Observable specification", f"{U}/observable_v2/W1_repeated_reselect.csv", BLUE),
         ("Original predictor list", f"{U}/w1_repeated_v2/W1_repeated_reselect.csv", ORANGE)]  # Weissman list: notes-no-discharge (2026-10-03)
LEARN = [("gbm", "Gradient boosting"), ("lrcv", "Logistic regression")]
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharex=False, sharey=True)
for i, (sname, path, col) in enumerate(SPECS):
    r = pd.read_csv(path)
    r = r[r.arm == "rechosen"].drop_duplicates(["draw", "tier", "model"], keep="last")
    for j, (mk, lname) in enumerate(LEARN):
        ax = axes[i, j]; style(ax)
        q = r[r.model == mk]
        p = q.pivot(index="draw", columns="tier", values="increment")[TIERS]
        nv = q.groupby("tier").n_vars.first()[TIERS].to_numpy(float)
        for _, row in p.iterrows():
            ax.plot(nv, row.to_numpy(), color=col, alpha=0.28, lw=1.0, zorder=2)
        ax.plot(nv, p.mean().to_numpy(), color=col, lw=2.2, marker="o", ms=6.5, zorder=3,
                markeredgecolor="white", markeredgewidth=1.2)
        coll = (p.T1 - p.T5) / p.T1 * 100
        ax.set_xscale("log"); ax.set_xticks(nv); labs = ax.set_xticklabels([f"{int(v)}" for v in nv])
        labs[3].set_horizontalalignment("right"); labs[4].set_horizontalalignment("left")
        ax.minorticks_off()
        ax.set_title(f"{sname} — {lname}", fontsize=10, color=INK_1, loc="left", pad=6)
        ax.text(0.98, 0.95, f"decline {coll.mean():.1f}%\nrange {coll.min():.1f}–{coll.max():.1f}%\nfalls in {(p.T5 < p.T1).sum()}/10",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK_2, linespacing=1.4)
        if j == 0: ax.set_ylabel("Gain in AUROC from adding notes", color=INK_2)
        if i == 1: ax.set_xlabel("Structured predictors in the baseline (log scale)", color=INK_2)
from matplotlib.lines import Line2D
h = [Line2D([], [], color=INK_3, alpha=0.5, lw=1.0), Line2D([], [], color=INK_3, lw=2.2, marker="o", ms=6)]
fig.legend(h, ["one repetition", "mean of ten repetitions"], frameon=False, ncol=2, loc="lower center",
           bbox_to_anchor=(0.5, -0.005), labelcolor=INK_2, fontsize=9)
fig.tight_layout(rect=(0, 0.035, 1, 1))
for ext in ("png", "pdf"): fig.savefig(f"{FIG}/figureS2_partitions.{ext}", dpi=220, facecolor="white")
plt.close(fig)

# ---------------- Figure S1: coverage test, both fine-tuning runs (2026-10-03)
RUNS = [("Earlier version: fine-tuned on \u2264 2 windows, read out on \u2264 4",
         f"{U}/w1_ftsplit_v2_observable/W1_coverage_split.csv"),
        ("Full-window version: every window, fine-tuning and readout",
         f"{U}/w1_ftfull_v2_observable/W1_ftfull_coverage.csv")]
SET = [("text", "Text only"), ("T1", "Added to T1\n(21 predictors)"), ("T5", "Added to T5\n(223 predictors)")]
GRP = [("full", "Admissions with \u2264 4 windows", BLUE, "o", -0.12),
       ("partial", "Admissions with > 4 windows", ORANGE, "s", 0.12)]
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharey=True)
for i, (rlab, path) in enumerate(RUNS):
    cv = pd.read_csv(path)
    cv = cv[cv.arm == "ft_cls"].drop_duplicates(["draw", "setting", "model", "arm", "group"], keep="last")
    for j, (mk, lname) in enumerate(LEARN):
        ax = axes[i, j]; style(ax)
        for gk, glab, col, mkr, off in GRP:
            for k, (sk, _) in enumerate(SET):
                x = cv[(cv.model == mk) & (cv.setting == sk) & (cv.group == gk)].sort_values("draw")["diff"]
                m, hw = tci(x)
                ax.errorbar(k + off, m, yerr=hw, color=col, marker=mkr, ms=7.5, lw=1.6, capsize=3,
                            markerfacecolor=col if gk == "full" else "white", markeredgewidth=1.6, zorder=3,
                            label=glab if k == 0 else None)
        ax.set_xticks(range(3)); ax.set_xticklabels([s_[1] for s_ in SET], color=INK_2)
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(f"{lname}\n{rlab}", fontsize=9.5, color=INK_1, loc="left", pad=6, linespacing=1.4)
        if j == 0: ax.set_ylabel("Fine-tuned encoder minus bag-of-words\n(\u0394AUROC, same test admissions)", color=INK_2)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.005), labelcolor=INK_2, fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 1))
for ext in ("png", "pdf"): fig.savefig(f"{FIG}/figureS1_coverage.{ext}", dpi=220, facecolor="white")
print("written", os.listdir(FIG))
