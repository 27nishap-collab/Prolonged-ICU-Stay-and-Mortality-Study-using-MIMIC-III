"""Cohort flow diagram (CONSORT-style) for the paper.

    python3 make_cohort_figure.py        # writes ../figs/figure_cohort.{png,pdf,svg}

Every count is read from the files build_cohort_W1.py writes -- nothing is typed in:
  out/W1_funnel_clinical.csv   steps 1-4 (clinical eligibility)
  out/cohort_W1.csv            the final cohort and its outcomes
  out/text_meta_W1_v2.parquet  notes left once discharge summaries are removed
The 75/25 split sizes follow draw_split() in run_repeated_reselect_W1.py (25% of each
outcome class, rounded).
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(HERE, "out")
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "..", "figs")); os.makedirs(OUT, exist_ok=True)
INK_1, INK_2, INK_3, LINE = "#0b0b0b", "#52514e", "#8a8983", "#8a8983"
FILL_MAIN, FILL_EXCL, FILL_KEY = "#ffffff", "#f4f3ef", "#eaf2fc"
plt.rcParams.update({"font.family": "DejaVu Sans"})

f = pd.read_csv(f"{U}/W1_funnel_clinical.csv").n.tolist()          # 4 steps
d = pd.read_csv(f"{U}/cohort_W1.csv")
m = pd.read_parquet(f"{U}/text_meta_W1_v2.parquet")
N = len(d); P = d.subject_id.nunique()
death, los7, comp = int(d.outcome_death.sum()), int(d.outcome_icu_los_7d.sum()), int(d.labels.sum())
both = int(((d.outcome_death == 1) & (d.outcome_icu_los_7d == 1)).sum())
no_icu = int(d.icu_los_sum.isna().sum())
no_text = int((m.n_notes.fillna(0) == 0).sum())
n1 = comp; n0 = N - comp
test = int(round(0.25 * n0)) + int(round(0.25 * n1)); train = N - test
pc = lambda k: f"{100 * k / N:.1f}%"
fmt = lambda k: f"{k:,}"

main = [(f"MIMIC-III v1.4 hospital admissions, 2001–2012", f[0]),
        ("Adult admissions", f[1]),
        ("Hospital stay of at least 48 hours", f[2]),
        ("No limitation of life-sustaining therapy", f[3]),
        ("Study cohort", N)]
excl = [("Age under 18 years", f[0] - f[1]),
        ("Discharged or died within 48 hours\nof hospital admission", f[1] - f[2]),
        ("Code status other than full code\nrecorded in the first 48 hours", f[2] - f[3]),
        ("No clinical note in the first 48 hours\n(radiology, ECG and echo reports\ndo not count)", f[3] - N)]

fig, ax = plt.subplots(figsize=(8.6, 10.2)); ax.set_xlim(0, 100); ax.set_ylim(-2, 120); ax.axis("off")
fig.patch.set_facecolor("white")

def box(x, y, w, h, title, sub=None, fill=FILL_MAIN, bold=True, fs=10.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fill, ec=LINE, lw=1.0))
    if sub is None:
        ax.text(x, y, title, ha="center", va="center", fontsize=fs, color=INK_1,
                fontweight="bold" if bold else "normal", linespacing=1.35)
    else:
        ax.text(x, y + 1.6, title, ha="center", va="center", fontsize=fs - 0.5, color=INK_2, linespacing=1.3)
        ax.text(x, y - 2.0, sub, ha="center", va="center", fontsize=fs + 1, color=INK_1, fontweight="bold")

def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.1, shrinkA=0, shrinkB=0, mutation_scale=11))

XM, W, H = 30, 42, 8.2
ys = [112, 92, 72, 52, 30]
for i, (t, n) in enumerate(main):
    sub = f"{fmt(n)} admissions" if i < 4 else f"{fmt(N)} admissions · {fmt(P)} patients"
    box(XM, ys[i], W, H if i < 4 else 9.6, t, sub, fill=FILL_KEY if i == 4 else FILL_MAIN)
for i in range(4):
    arrow(XM, ys[i] - H / 2 - 0.6, XM, ys[i + 1] + (H if i < 3 else 9.6) / 2 + 0.6)
    ym = (ys[i] + ys[i + 1]) / 2
    ax.plot([XM, 57], [ym, ym], color=LINE, lw=1.0)
    t, n = excl[i]
    hh = 6.4 + 2.4 * t.count("\n")
    box(77, ym, 37, hh, f"Excluded: {fmt(n)}\n{t}", fill=FILL_EXCL, bold=False, fs=9.2)
    arrow(57, ym, 77 - 37 / 2 - 0.6, ym)

# outcomes and partitions: both hang off the study cohort
yb = 6.5; hb = 13.0
ax.plot([XM, XM], [ys[4] - 9.6 / 2 - 0.6, 19.5], color=LINE, lw=1.0)
ax.plot([XM, 77], [19.5, 19.5], color=LINE, lw=1.0)
arrow(XM, 19.5, XM, yb + hb / 2 + 0.6)
arrow(77, 19.5, 77, yb + hb / 2 + 0.6)
box(XM, yb, 42, hb,
    f"Composite outcome: {fmt(comp)} ({pc(comp)})\n"
    f"In-hospital death: {fmt(death)} ({pc(death)})\n"
    f"ICU stay ≥ 7 days: {fmt(los7)} ({pc(los7)})\n"
    f"({fmt(both)} had both)", bold=False, fs=9.4)
box(77, yb, 37, hb,
    f"Ten repetitions, each a random\n75/25 division stratified on the outcome:\n{fmt(train)} training\n{fmt(test)} test admissions",
    bold=False, fs=9.2, fill=FILL_EXCL)

ax.text(0, -3.5,
        f"Notes: {fmt(no_text)} admissions have no note left once discharge summaries are removed from the text (their only in-window "
        f"note\nwas a discharge summary); they stay in the cohort with empty text features. {fmt(no_icu)} admissions have no "
        f"ICU stay record.\nAll windows run from hospital admission. ICU stay is the sum of the admission's ICU stays.",
        fontsize=8.0, color=INK_2, ha="left", va="top", linespacing=1.45)
fig.tight_layout()
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{OUT}/figure_cohort.{ext}", dpi=220, facecolor="white", bbox_inches="tight")
print(dict(funnel=f, N=N, patients=P, comp=comp, death=death, los7=los7, both=both,
           no_text=no_text, no_icu=no_icu, train=train, test=test))
