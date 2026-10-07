"""Design 3 -- split-sample fine-tuning. The fine-tuned arm WITHOUT contamination.

RUN ON THE MAC, native Python (python3.11, MPS). Three modes:

    cd /Users/Shared/MIMIC_git/Analysis/replicate_weissman

    python3.11 finetune_split_W1.py check
        # no GPU, ~1-2 min: loads everything, checks the saved halves, times one
        # downstream cell. Also runs in the Cowork VM.

    caffeinate -i env HF_HUB_OFFLINE=1 W1_SMOKE=1 python3.11 finetune_split_W1.py run
        # ~10 min timing test on draw 0: 300 training steps + 400 admissions
        # encoded, then prints the projected hours per draw. Writes NO results.

    caffeinate -i env HF_HUB_OFFLINE=1 python3.11 finetune_split_W1.py run 2>&1 | tee ft_split.log
        # the real run, ~2.6 h per draw, ~13 h for five. Resumable per draw:
        # Ctrl-C and restart is safe; W1_DRAWS=0,1,2 one night and 3,4 the next.

    python3.11 finetune_split_W1.py summary

--------------------------------------------------------------------------
WHY. finetune_bert_W1.py (Design 1, 2026-09-23) fine-tuned Bio_ClinicalBERT on
each draw's training patients and then fitted the downstream GBM / LR on the SAME
patients. diag_contamination_W1.py measured what that does: the downstream model
sees text features that were trained to predict its own training labels, which
inflates the fine-tuned arm by +0.047 AUROC (95% CI +0.040 to +0.054) -- larger
than the effect in question. So F17 reports that arm in direction only.

Here each draw's training patients are split in two, once, stratified on the
outcome (saved by check_halfsize_W1.py as halves_draw{r}.npz, seed SEED+5000+r):

    half A  -> fine-tunes the encoder (with a 5% validation slice carved from A)
    half B  -> fits every downstream model, and every train-only preprocessing
               step (rare-level collapsing, imputation, scaling, C selection)
    test    -> the same 25% as every other draw-r result; touched by nothing

The encoder never sees a half-B or test label, so the downstream model's text
features carry no memory of its own training labels. Bakumenko et al. 2025
(arXiv 2511.15847) fit their stacker on held-out patients for the same reason.

check_halfsize_W1.py showed half-size downstream fitting does not distort the
collapse for GBM (5 draws) or tuned LR (1 draw); default LR failed it and is not
used here.

--------------------------------------------------------------------------
FINE-TUNING -- identical to Design 1 except for the rows it sees.
  Bio_ClinicalBERT (the checkpoint Deznabi 2021 and Lyu 2022 fine-tune), all
  layers trained, AdamW lr 2e-5, 10% linear warm-up then linear decay, 2 epochs,
  batch 8 (see the calibration note in finetune_bert_W1.py; do not raise it).
  Up to 2 windows of 510 wordpieces per admission, each inheriting the admission's
  label (as Deznabi's notes and Huang's chunks do), no class weighting.

READOUTS -- three, from ONE forward pass over up to 4 windows per admission.
The literature check (2026-09-27) found Design 1 read out something the
fine-tuning did not directly optimise: the classification head is trained on the
[CLS] token, but Design 1 averaged every token's final state. So:

  ft_cls   mean over windows of the final-layer [CLS] vector (768). This is
           Deznabi's readout ("average embedding of all the clinical notes", each
           note's CLS). PRIMARY, fixed before any result.
  ft_tok   mean over all content tokens (768) -- Design 1's readout, for continuity.
  ft_prob  mean over windows of the fine-tuned head's P(outcome) (1 column) --
           the model's own prediction, close to Huang et al.'s aggregation.

Frozen comparators on the same half B: frozen_tok (the frozen arm used throughout
the project) and frozen_cls (same shards, s_cls / n_win). Bag-of-words (bow) as
the anchor. Each fine-tuned readout is compared with the frozen arm of the same
readout; ft_prob, which has no frozen twin, with frozen_cls.

WHAT IS REPORTED, fixed before any result:
  * per arm and learner (GBM, tuned LR): T1 and T5 increments over structured-only,
    collapse (T1 - T5) / T1, 95% CI over 5 draws (t, 4 df), draws in which it falls;
  * fine-tuned minus frozen at T1 and at T5, paired within draw;
  * text-only held-out AUROC of every arm (fitted on B), plus ft_prob scored
    directly -- the clean version of Design 1's +0.021 fine-tuned-over-frozen gain.
  Design 1's numbers are NOT differenced against these: different downstream
  sample size (half vs full). Compare within this script only.

--------------------------------------------------------------------------
TIME. Design 1's clean draws took 3.6 h to fine-tune on ~35,400 windows and
0.8 h to encode. Half A has ~17,700 windows, so ~1.8 h + 0.8 h = ~2.6 h a draw.
Two of Design 1's draws stalled for 11 and 35 extra hours -- the Mac slept or was
busy. Keep it plugged in, use caffeinate, and run nothing else heavy.

Outputs in out/w1_ftsplit_v2_observable/ (gitignored):
  emb_draw{r}.npz          the three fine-tuned readouts for all admissions
  W1_ftsplit.csv           increments, one row per draw x tier x learner x arm
  W1_ftsplit_textonly.csv  text-only held-out AUROCs
  W1_ftsplit_val.csv       window-level validation AUROC per epoch (monitoring only)
"""
import os, sys, json, glob, time, warnings
import numpy as np, pandas as pd
from scipy import sparse, stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
D  = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(D, "out")
sys.path.insert(0, D)
from lr_tuned import lr_tuned_scores

MODE = sys.argv[1] if len(sys.argv) > 1 else "run"
assert MODE in ("check", "run", "summary"), "mode: check | run | summary"

SEED = 20180701
GBM  = dict(max_depth=10, learning_rate=0.05, max_iter=500, min_samples_leaf=15)
TAG, HF = "bio_clinicalbert", "emilyalsentzer/Bio_ClinicalBERT"
DRAWS  = [int(x) for x in os.environ.get("W1_DRAWS", "0,1,2,3,4").split(",")]
EPOCHS, BATCH, TUNE_BAND, ENC_BAND, LR, VAL_FRAC = 2, 8, 2, 4, 2e-5, 0.05
TIERS, LEARNERS = ["T1", "T5"], ["gbm", "lrcv"]
ARM_NAMES = ["bow", "frozen_tok", "frozen_cls", "ft_cls", "ft_tok", "ft_prob"]
TWIN = {"ft_cls": "frozen_cls", "ft_tok": "frozen_tok", "ft_prob": "frozen_cls"}
SMOKE = os.environ.get("W1_SMOKE") == "1"
SMOKE_STEPS, SMOKE_ENC = 300, 400

SFX, SUF = "_observable", "_v2"            # observable spec, notes-no-discharge
HALVES = f"{U}/w1_splitsample{SUF}{SFX}"
OUT = f"{U}/w1_ftsplit{SUF}{SFX}"; os.makedirs(OUT, exist_ok=True)
CSV, TXT, VAL = (f"{OUT}/W1_ftsplit.csv", f"{OUT}/W1_ftsplit_textonly.csv",
                 f"{OUT}/W1_ftsplit_val.csv")
t0 = time.time()
def log(m): print(f"[{(time.time()-t0)/3600:5.2f}h] {m}", flush=True)
def append(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)

# ---------------------------------------------------------------- DeLong
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T_ = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T_[i:j] = 0.5*(i+j-1)+1; i = j
    T2 = np.empty(N); T2[J] = T_; return T2

def delong(y, scores):
    y = np.asarray(y); s = np.atleast_2d(np.asarray(scores, float))
    o = np.argsort(-y, kind="mergesort"); s = s[:, o]
    m = int(y.sum()); n = len(y) - m
    tx = np.vstack([_midrank(s[r, :m]) for r in range(s.shape[0])])
    ty = np.vstack([_midrank(s[r, m:]) for r in range(s.shape[0])])
    tz = np.vstack([_midrank(s[r])     for r in range(s.shape[0])])
    aucs = tz[:, :m].sum(1)/m/n - (m+1.0)/2.0/n
    return aucs, (np.atleast_2d(np.cov((tz[:, :m]-tx)/n))/m
                  + np.atleast_2d(np.cov(1.0-(tz[:, m:]-ty)/m))/n)

def paired(y, a, b):
    au, c = delong(y, [a, b]); v = c[0, 0]+c[1, 1]-2*c[0, 1]
    return float(au[0]), float(au[1]), float(au[1]-au[0]), float(np.sqrt(max(v, 0)))

# ---------------------------------------------------------------- data
if MODE != "summary":
    tiers = json.load(open(f"{U}/W1_tier_columns{SFX}.json"))
    T   = pd.read_parquet(f"{U}/tiers_W1{SFX}.parquet")
    tab = pd.read_parquet(f"{U}/model_tabular_W1{SUF}.parquet")
    Xo  = sparse.load_npz(f"{U}/text_open_W1{SUF}_aligned.npz").toarray().astype(np.float32)
    tab = tab.set_index("hadm_id"); order = tab.index.to_numpy()
    assert (np.sort(order) == order).all()
    T = T.set_index("hadm_id").reindex(order)
    cvc = [c for c in tab.columns if c.startswith("cv_")]
    FIX = np.hstack([tab[cvc].to_numpy(np.float32), tab[["n_words_raw"]].to_numpy(np.float32)])
    y = tab.labels.to_numpy().astype(int); N = len(y)

    z = np.load(f"{U}/tok16_{TAG}_W1.npz")
    ids, wlen, nwin, hadm = z["ids"], z["wlen"], z["nwin"], z["hadm"]
    woff = np.concatenate([[0], np.cumsum(wlen.astype(np.int64))])
    aoff = np.concatenate([[0], np.cumsum(nwin.astype(np.int64))])
    tok_pos = pd.Index(hadm).get_indexer(order)
    assert not (tok_pos < 0).any(), "tok16 does not cover every admission"
    vocab = open(os.path.join(D, "hf", TAG, "vocab.txt")).read().split("\n")
    CLS, SEP, PAD = vocab.index("[CLS]"), vocab.index("[SEP]"), vocab.index("[PAD]")

    fs = sorted(glob.glob(f"{U}/bert_{TAG}_w0-{ENC_BAND}/b_*.npz")); assert fs
    g = {k: np.concatenate([np.load(f)[k] for f in fs])
         for k in ("hadm", "s_tok", "n_tok", "s_cls", "n_win")}
    gp = pd.Index(g["hadm"]).get_indexer(order); assert not (gp < 0).any()
    BASE = {
        "bow":        np.hstack([FIX, Xo]),
        "frozen_tok": np.hstack([FIX, (g["s_tok"][gp] / np.maximum(g["n_tok"][gp], 1)[:, None]
                                       ).astype(np.float32)]),
        "frozen_cls": np.hstack([FIX, (g["s_cls"][gp] / np.maximum(g["n_win"][gp], 1)[:, None]
                                       ).astype(np.float32)]),
    }
    log(f"{N:,} admissions | outcome {y.mean():.4f} | draws {DRAWS} | mode {MODE}"
        + (" | SMOKE" if SMOKE else ""))

# ------------------------------------- partitions: verbatim from run_repeated_splits_W1.py
def draw_split(r):
    rng = np.random.RandomState(SEED + 1000 * (r + 1))
    tr = np.ones(len(y), bool)
    for lab in (0, 1):
        idx = np.flatnonzero(y == lab); rng.shuffle(idx)
        tr[idx[: int(round(0.25 * len(idx)))]] = False
    return tr

def halves(r, tr):
    """The halves check_halfsize_W1.py saved. Never regenerated here."""
    f = f"{HALVES}/halves_draw{r}.npz"
    assert os.path.exists(f), f"{f} missing -- it must come from check_halfsize_W1.py"
    z = np.load(f); A, B = z["A"], z["B"]
    assert len(A) == N and not (A & B).any() and ((A | B) == tr).all()
    assert not ((A | B) & ~tr).any(), "a half touches the test set"
    return A, B

def encode_tier(cols, fit):
    out = []
    for c in cols:
        s = T[c]
        if not pd.api.types.is_numeric_dtype(s):
            v = s.astype(object).where(s.notna(), "__na__")
            k = v[fit].value_counts(); lv = set(k[k >= 0.01*fit.sum()].index)
            v = v.where(v.isin(lv), "__other__")
            for L in sorted(lv | {"__other__"}): out.append((v == L).to_numpy(np.float32))
        else: out.append(s.to_numpy(np.float32))
    return np.column_stack(out)

def score(X, fit, mk):
    """Scores for every row; fitted on `fit` rows only (half B here)."""
    if mk == "gbm":
        return HistGradientBoostingClassifier(early_stopping=False, random_state=SEED,
                                              **GBM).fit(X[fit], y[fit]).decision_function(X)
    assert mk == "lrcv", mk
    return lr_tuned_scores(X, fit, y)[0]

def windows_of(rows, band):
    out = []
    for i in np.nonzero(rows)[0]:
        t = tok_pos[i]
        for j in range(aoff[t], aoff[t] + min(int(nwin[t]), band)):
            out.append((j, i))
    return np.array(out, np.int64).reshape(-1, 2)

# ---------------------------------------------------------------- the encoder
def finetune_and_encode(r, A):
    """Fine-tune on half A only; encode every admission; return the three readouts."""
    cache = f"{OUT}/emb_draw{r}.npz"
    if os.path.exists(cache) and not SMOKE:
        log(f"draw {r}: fine-tuned readouts cached, skipping"); return dict(np.load(cache))

    import torch
    from transformers import AutoModelForSequenceClassification
    if not torch.backends.mps.is_available():
        sys.exit("MPS unavailable. Run with native python3.11 on the Mac, not in the VM.")
    DEV = torch.device("mps")

    def make_batch(js):
        ws = [ids[woff[j]:woff[j+1]].astype(np.int64) for j in js]
        L = max(len(w) for w in ws) + 2
        X = np.full((len(ws), L), PAD, np.int64); M = np.zeros((len(ws), L), np.int64)
        for k, w in enumerate(ws):
            s = [CLS] + list(w) + [SEP]; X[k, :len(s)] = s; M[k, :len(s)] = 1
        return torch.from_numpy(X).to(DEV), torch.from_numpy(M).to(DEV)

    torch.manual_seed(SEED + r); rng = np.random.RandomState(SEED + r)
    a_idx = np.nonzero(A)[0]; rng.shuffle(a_idx)
    nval = int(round(VAL_FRAC * len(a_idx)))
    val_rows = np.zeros(N, bool); val_rows[a_idx[:nval]] = True
    fit_rows = A & ~val_rows
    W, WV = windows_of(fit_rows, TUNE_BAND), windows_of(val_rows, TUNE_BAND)
    assert A[W[:, 1]].all() and A[WV[:, 1]].all(), "fine-tuning saw a row outside half A"

    model = AutoModelForSequenceClassification.from_pretrained(HF, num_labels=2).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    nsteps = EPOCHS * (len(W) // BATCH); warm = max(1, nsteps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        s / warm if s < warm else max(0.0, (nsteps - s) / max(1, nsteps - warm))))
    log(f"draw {r}: fine-tune on {len(W):,} windows ({fit_rows.sum():,} half-A admissions), "
        f"val {len(WV):,} windows, {nsteps:,} steps")

    step, ts = 0, time.time()
    for ep in range(EPOCHS):
        model.train(); perm = rng.permutation(len(W))
        for b in range(len(W) // BATCH):
            k = perm[b*BATCH:(b+1)*BATCH]
            X, M = make_batch(W[k, 0])
            yb = torch.from_numpy(y[W[k, 1]]).to(DEV)
            model(input_ids=X, attention_mask=M, labels=yb).loss.backward()
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 500 == 0:
                el = time.time() - ts
                log(f"  draw {r} ep{ep} step {step:,}/{nsteps:,}  {el/60:.0f} min, "
                    f"{(nsteps-step)*el/step/60:.0f} min left")
            if SMOKE and step >= SMOKE_STEPS: break
        if SMOKE:
            per_step = (time.time() - ts) / step
            log(f"SMOKE: {per_step*1000:.0f} ms/step -> fine-tuning ~{per_step*nsteps/3600:.2f} h")
            break
        model.eval(); sv = []
        with torch.no_grad():
            for b in range(0, len(WV), BATCH):
                X, M = make_batch(WV[b:b+BATCH, 0])
                sv.append(torch.softmax(model(input_ids=X, attention_mask=M).logits.float(), -1
                                        )[:, 1].cpu().numpy())
        v = roc_auc_score(y[WV[:, 1]], np.concatenate(sv))
        append(VAL, dict(draw=r, epoch=ep, val_auroc=round(v, 4)))
        log(f"  draw {r} epoch {ep}: window-level val AUROC {v:.4f}  (half-A val slice)")

    # ---- encode: one pass, three readouts
    rows = np.ones(N, bool)
    if SMOKE:
        rows = np.zeros(N, bool); rows[:SMOKE_ENC] = True
    W_all = windows_of(rows, ENC_BAND)
    H = model.config.hidden_size
    s_tok, s_cls = np.zeros((N, H)), np.zeros((N, H))
    s_prob, n_tok, n_w = np.zeros(N), np.zeros(N, np.int64), np.zeros(N, np.int64)
    o = np.argsort(wlen[W_all[:, 0]]); te = time.time()
    model.eval()
    with torch.no_grad():
        for b in range(0, len(o), BATCH):
            k = W_all[o[b:b+BATCH]]
            X, M = make_batch(k[:, 0])
            out = model(input_ids=X, attention_mask=M, output_hidden_states=True)
            h = out.hidden_states[-1].float().cpu().numpy()      # = base last_hidden_state
            p = torch.softmax(out.logits.float(), -1)[:, 1].cpu().numpy()
            for q, (j, i) in enumerate(k):
                m = int(wlen[j])
                s_tok[i] += h[q, 1:1+m].sum(0); n_tok[i] += m
                s_cls[i] += h[q, 0]; s_prob[i] += p[q]; n_w[i] += 1
    if SMOKE:
        per_adm = (time.time() - te) / SMOKE_ENC
        log(f"SMOKE: encoding ~{per_adm*N/3600:.2f} h for all {N:,} admissions")
        log("SMOKE: nothing written. If the projection is ~2.6 h a draw, start the real run.")
        sys.exit(0)
    wn = np.maximum(n_w, 1)
    E = dict(ft_tok=(s_tok / np.maximum(n_tok, 1)[:, None]).astype(np.float32),
             ft_cls=(s_cls / wn[:, None]).astype(np.float32),
             ft_prob=np.where(n_w > 0, s_prob / wn, np.nan).astype(np.float32)[:, None])
    np.savez(cache, **E)
    del model, opt
    try: torch.mps.empty_cache()
    except Exception: pass
    log(f"draw {r}: {(time.time()-ts)/3600:.2f} h -> {cache}")
    return E

# ---------------------------------------------------------------- modes
def check():
    for r in range(5):
        tr = draw_split(r); A, B = halves(r, tr)
        log(f"draw {r}: train {tr.sum():,} = A {A.sum():,} + B {B.sum():,} "
            f"| events A {y[A].mean():.4f} B {y[B].mean():.4f} test {y[~tr].mean():.4f} "
            f"| fine-tune windows ~{len(windows_of(A, TUNE_BAND)):,}")
    tr = draw_split(0); A, B = halves(0, tr)
    S = encode_tier(tiers["T1"], B); ts = time.time()
    a, b, d, se = paired(y[~tr], score(S, B, "gbm")[~tr],
                         score(np.hstack([S, BASE["frozen_cls"]]), B, "gbm")[~tr])
    log(f"draw 0/T1/gbm/frozen_cls on half B: struct {a:.4f} -> {b:.4f}, inc {d:+.4f} "
        f"({time.time()-ts:.0f}s)")
    S = encode_tier(tiers["T5"], B); ts = time.time()
    score(np.hstack([S, BASE["frozen_cls"]]), B, "lrcv")
    log(f"one tuned-LR fit at T5 + 768 dims on half B: {time.time()-ts:.0f}s "
        f"(the slowest downstream cell; a draw has ~40 downstream fits)")
    log("check OK")

def run():
    done = set()
    if os.path.exists(CSV):
        o = pd.read_csv(CSV); done = set(zip(o.draw, o.tier, o.model, o.arm))
        log(f"resuming: {len(done)} cells on disk")
    for r in DRAWS:
        need = [(t, m, a) for t in TIERS for m in LEARNERS for a in ARM_NAMES
                if (r, t, m, a) not in done]
        if not need and not SMOKE:
            log(f"draw {r}: complete, skipping"); continue
        tr = draw_split(r); A, B = halves(r, tr); test = ~tr
        log(f"draw {r}: A {A.sum():,} (fine-tune) | B {B.sum():,} (downstream) "
            f"| test {test.sum():,}")
        E = finetune_and_encode(r, A)
        ARMS = dict(BASE, **{k: np.hstack([FIX, E[k]]) for k in ("ft_cls", "ft_tok", "ft_prob")})

        # text only, fitted on B, scored on test -- once per draw
        if not os.path.exists(TXT) or r not in set(pd.read_csv(TXT).draw):
            append(TXT, dict(draw=r, arm="ft_prob_direct", model="none",
                             auroc=round(roc_auc_score(y[test], np.nan_to_num(E["ft_prob"][test, 0])), 4)))
            for arm in ARM_NAMES:
                for mk in LEARNERS:
                    au = roc_auc_score(y[test], score(ARMS[arm], B, mk)[test])
                    append(TXT, dict(draw=r, arm=arm, model=mk, auroc=round(au, 4)))
                    log(f"  text-only {arm:10s}/{mk}: {au:.4f}")

        for tier in TIERS:
            S = encode_tier(tiers[tier], B)
            for mk in LEARNERS:
                sb = score(S, B, mk)
                for arm in ARM_NAMES:
                    if (r, tier, mk, arm) in done: continue
                    a, b, d, se = paired(y[test], sb[test],
                                         score(np.hstack([S, ARMS[arm]]), B, mk)[test])
                    append(CSV, dict(draw=r, tier=tier, n_vars=len(tiers[tier]), model=mk,
                                     arm=arm, n_fit=int(B.sum()), ev_fit=int(y[B].sum()),
                                     struct=round(a, 4), with_text=round(b, 4),
                                     increment=round(d, 4), se=round(se, 4)))
                    log(f"  {tier}/{mk}/{arm:10s}: struct {a:.4f} -> {b:.4f}  inc {d:+.4f}")
    summary()

def ci(v):
    v = np.asarray(v, float); n = len(v)
    h = stats.t.ppf(0.975, n-1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return v.mean(), v.mean() - h, v.mean() + h

def summary():
    if not os.path.exists(CSV): sys.exit("no results yet")
    res = pd.read_csv(CSV).drop_duplicates(["draw", "tier", "model", "arm"])
    print("\n=== collapse by arm (half-B downstream, 95% CI t over draws) ===")
    for mk in LEARNERS:
        for arm in ARM_NAMES:
            q = res[(res.model == mk) & (res.arm == arm)].pivot(
                index="draw", columns="tier", values="increment").dropna()
            if q.empty or not {"T1", "T5"} <= set(q.columns): continue
            m, lo, hi = ci(100 * (q.T1 - q.T5) / q.T1)
            print(f"  {mk:4s}/{arm:10s} n={len(q)}  T1 {q.T1.mean():+.4f}  T5 {q.T5.mean():+.4f}"
                  f"  collapse {m:5.1f}% [{lo:5.1f}, {hi:5.1f}]  falls {(q.T1 > q.T5).sum()}/{len(q)}")
    print("\n=== fine-tuned minus frozen, paired within draw ===")
    for mk in LEARNERS:
        for ft, fz in TWIN.items():
            for t in TIERS:
                a = res[(res.model == mk) & (res.tier == t)].pivot(
                    index="draw", columns="arm", values="increment")
                if not {ft, fz} <= set(a.columns): continue
                dd = (a[ft] - a[fz]).dropna()
                if dd.empty: continue
                m, lo, hi = ci(dd)
                print(f"  {mk:4s} {t} {ft:8s} - {fz:10s}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                      f"  ft ahead in {(dd > 0).sum()}/{len(dd)}")
    if os.path.exists(TXT):
        print("\n=== text-only held-out AUROC (mean over draws) ===")
        print(pd.read_csv(TXT).drop_duplicates(["draw", "arm", "model"], keep="last").pivot_table(index="arm", columns="model", values="auroc"
                                           ).round(4).to_string())
    log("done -- tell Claude and it will read the CSVs back")

{"check": check, "run": run, "summary": summary}[MODE]()
