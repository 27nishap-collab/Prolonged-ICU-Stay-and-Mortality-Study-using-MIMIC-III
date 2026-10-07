"""Design 3 again, with the encoder reading ALL of each admission's note text.

RUN ON THE MAC, native Python (python3.11, MPS). Prerequisite, once (Cowork VM or Mac):
    W1_MAXWIN=all W1_TOKMODELS=bio_clinicalbert python3 tokenize_W1.py         # repeat until ALL PARTS DONE
    W1_MAXWIN=all W1_TOKMODELS=bio_clinicalbert python3 tokenize_W1.py merge   # -> out/tokall_bio_clinicalbert_W1.npz
Then:
    python3.11 finetune_full_W1.py check                                       # no GPU, ~2 min
    caffeinate -dis env HF_HUB_OFFLINE=1 W1_SMOKE=1 python3.11 finetune_full_W1.py run   # ~15 min timing test
    caffeinate -dis env HF_HUB_OFFLINE=1 python3.11 finetune_full_W1.py run 2>&1 | tee ft_full.log
    python3.11 finetune_full_W1.py summary

WHY. finetune_split_W1.py (2026-09-28) fine-tuned on at most 2 and read out at most
4 windows of 510 wordpieces per admission: 15% and 27% of the note text; 45% of
admissions were read in full. coverage_split_W1.py then showed the fine-tuned
encoder trails bag-of-words by more on admissions it only partly read -- text only
and at T1 -- but not at T5. This run removes the truncation: every window of every
admission, in fine-tuning and in the readout (328,408 windows; 121,107 in a half A).

WHAT CHANGES FROM finetune_split_W1.py, and nothing else:
  * windows: all of them (W1_TUNE_BAND / W1_ENC_BAND default "all"), from
    tokall_bio_clinicalbert_W1.npz -- same tokenizer, same 510-wordpiece windows,
    same nested even-spread order;
  * frozen comparators re-encoded on all windows too (frozen_tok, frozen_cls), once,
    so fine-tuned vs frozen stays like for like;
  * bag-of-words comparators are the clean ones: bow_rs (500 phrases chosen on half A,
    the same patients the encoder learns from) and bow_rt (chosen on the partition's
    whole training set); reselect_split_W1.py made both. The original single-choice
    bow is not used;
  * the fine-tuned weights are SAVED (out/w1_ftfull_v2_observable/model_draw{r}/), and
    fine-tuning CHECKPOINTS every 1,000 steps, so a stall, a restart or a reboot costs
    at most ~3 minutes of training, not a partition;
  * the coverage split is recomputed on the same groups as coverage_split_W1.py
    (admissions of <= 4 windows vs more): with nothing unread, the interaction that
    test found should disappear if truncation caused it.
Everything else is identical: the five Design 3 partitions and their saved halves;
fine-tuning on half A only (Bio_ClinicalBERT, all layers, AdamW 2e-5, 10% warm-up then
linear decay, 2 epochs, batch 8, windows inherit the admission label, no weighting);
downstream GBM (fixed) and tuned LR fitted on half B, scored on the untouched test set;
[CLS] readout primary, token mean and predicted probability as sensitivity.

READ-OFF, fixed before any result:
  * collapse of ft_cls (and the other readouts) under both learners, 95% t CI over 5;
  * ft_cls minus bow_rs at T1 and T5, paired within partition -- the ranking question;
  * the coverage interaction for ft_cls - bow_rs (text only, T1): near zero now?
  * fine-tuned minus frozen, same readout, both reading all windows.
Compare with finetune_split_W1.py's numbers only as "4-window vs all-window" on the same
partitions, halves and machine; never across machines.

TIME (to be replaced by the smoke test): fine-tuning ~121,000 half-A windows x 2 epochs
at ~0.17 s per window ~ 11.5 h per partition; encoding all 328,408 windows ~ 3 h per
partition, plus ~3 h once for the frozen encoder. About 75 h for five partitions --
run them over several nights with W1_DRAWS=0 ... 4.

Outputs in out/w1_ftfull_v2_observable/ (gitignored):
  frozen_all.npz, emb_draw{r}.npz, model_draw{r}/, ckpt_draw{r}.pt (deleted when done)
  W1_ftfull.csv, W1_ftfull_textonly.csv, W1_ftfull_val.csv, W1_ftfull_coverage.csv
"""
import os, sys, json, glob, time, warnings
import numpy as np, pandas as pd
from scipy import sparse, stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
D  = os.path.dirname(os.path.abspath(__file__)); U = os.environ.get("W1_DATA", os.path.join(D, "out"))
sys.path.insert(0, D)
from lr_tuned import lr_tuned_scores

MODE = sys.argv[1] if len(sys.argv) > 1 else "run"
assert MODE in ("check", "run", "summary"), "mode: check | run | summary"

SEED = 20180701
GBM  = dict(max_depth=10, learning_rate=0.05, max_iter=500, min_samples_leaf=15)
TAG, HF = "bio_clinicalbert", "emilyalsentzer/Bio_ClinicalBERT"
DRAWS  = [int(x) for x in os.environ.get("W1_DRAWS", "0,1,2,3,4").split(",")]
def _band(v): return 10**9 if v == "all" else int(v)
TUNE_BAND = _band(os.environ.get("W1_TUNE_BAND", "all"))
ENC_BAND  = _band(os.environ.get("W1_ENC_BAND", "all"))
EPOCHS, BATCH, LR, VAL_FRAC = 2, 8, 2e-5, 0.05
CKPT_EVERY = 1000
TIERS, LEARNERS = ["T1", "T5"], ["gbm", "lrcv"]
FT = ["ft_cls", "ft_tok", "ft_prob"]
ARM_NAMES = ["bow_rs", "bow_rt", "frozen_tok", "frozen_cls"] + FT
TWIN = {"ft_cls": "frozen_cls", "ft_tok": "frozen_tok", "ft_prob": "frozen_cls"}
SMOKE = os.environ.get("W1_SMOKE") == "1"
SMOKE_STEPS, SMOKE_ENC = 300, 400
COVER_K = 4                                   # coverage groups, as in coverage_split_W1.py

SFX, SUF = "_observable", "_v2"
HALVES = f"{U}/w1_splitsample{SUF}{SFX}"
BOWD   = f"{U}/w1_ftsplit{SUF}{SFX}"          # bow_rs / bow_rt from reselect_split_W1.py
OUT = f"{U}/w1_ftfull{SUF}{SFX}"; os.makedirs(OUT, exist_ok=True)
CSV, TXT, VAL, COV = (f"{OUT}/W1_ftfull.csv", f"{OUT}/W1_ftfull_textonly.csv",
                      f"{OUT}/W1_ftfull_val.csv", f"{OUT}/W1_ftfull_coverage.csv")
t0 = time.time()
def log(m): print(f"[{(time.time()-t0)/3600:5.2f}h] {m}", flush=True)
def append(path, row, cols=None):
    df = pd.DataFrame([row]); df = df.reindex(columns=cols) if cols else df
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

# ---------------------------------------------------------------- DeLong
def _midrank(x):
    J = np.argsort(x); Z = x[J]; n = len(x); T_ = np.zeros(n); i = 0
    while i < n:
        j = i
        while j < n and Z[j] == Z[i]: j += 1
        T_[i:j] = 0.5*(i+j-1)+1; i = j
    T2 = np.empty(n); T2[J] = T_; return T2

def paired(yt, a, b):
    """AUROC of a, of b, b - a, and the DeLong se of the difference."""
    s = np.vstack([a, b]).astype(float); o = np.argsort(-yt, kind="mergesort"); s = s[:, o]
    m = int(yt.sum()); n = len(yt) - m
    tx = np.vstack([_midrank(s[k, :m]) for k in range(2)])
    ty = np.vstack([_midrank(s[k, m:]) for k in range(2)])
    tz = np.vstack([_midrank(s[k]) for k in range(2)])
    au = tz[:, :m].sum(1)/m/n - (m+1.0)/2.0/n
    c = np.cov((tz[:, :m]-tx)/n)/m + np.cov(1.0-(tz[:, m:]-ty)/m)/n
    return float(au[0]), float(au[1]), float(au[1]-au[0]), float(np.sqrt(max(c[0,0]+c[1,1]-2*c[0,1], 0)))

# ---------------------------------------------------------------- data
if MODE != "summary":
    tiers = json.load(open(f"{U}/W1_tier_columns{SFX}.json"))
    T   = pd.read_parquet(f"{U}/tiers_W1{SFX}.parquet")
    tab = pd.read_parquet(f"{U}/model_tabular_W1{SUF}.parquet").set_index("hadm_id")
    order = tab.index.to_numpy(); assert (np.sort(order) == order).all()
    T = T.set_index("hadm_id").reindex(order)
    cvc = [c for c in tab.columns if c.startswith("cv_")]
    FIX = np.hstack([tab[cvc].to_numpy(np.float32), tab[["n_words_raw"]].to_numpy(np.float32)])
    y = tab.labels.to_numpy().astype(int); N = len(y)

    TOKTAG = os.environ.get("W1_TOK", "all")          # "16" only for testing the script
    TOKF = f"{U}/tok{TOKTAG}_{TAG}_W1.npz"
    assert os.path.exists(TOKF), f"{TOKF} missing -- run tokenize_W1.py with W1_MAXWIN=all, then merge"
    z = np.load(TOKF)
    ids, wlen, nwin, nfull, hadm = z["ids"], z["wlen"], z["nwin"], z["nfull"], z["hadm"]
    woff = np.concatenate([[0], np.cumsum(wlen.astype(np.int64))])
    aoff = np.concatenate([[0], np.cumsum(nwin.astype(np.int64))])
    tok_pos = pd.Index(hadm).get_indexer(order); assert not (tok_pos < 0).any()
    NFULL = nfull[tok_pos].astype(int)
    if TOKTAG == "all":
        assert (nwin == nfull).all(), "tokall file does not hold every window"
    GROUP = np.where(NFULL == 0, "none", np.where(NFULL <= COVER_K, "full", "partial"))
    vocab = open(os.path.join(D, "hf", TAG, "vocab.txt")).read().split("\n")
    CLS, SEP, PAD = vocab.index("[CLS]"), vocab.index("[SEP]"), vocab.index("[PAD]")
    log(f"{N:,} admissions | {int(np.minimum(nwin, ENC_BAND).sum()):,} windows to encode | "
        f"tune band {'all' if TUNE_BAND >= 10**9 else TUNE_BAND} | draws {DRAWS} | mode {MODE}"
        + (" | SMOKE" if SMOKE else ""))

def draw_split(r):
    rng = np.random.RandomState(SEED + 1000 * (r + 1))
    tr = np.ones(len(y), bool)
    for lab in (0, 1):
        idx = np.flatnonzero(y == lab); rng.shuffle(idx)
        tr[idx[: int(round(0.25 * len(idx)))]] = False
    return tr

def halves(r, tr):
    f = f"{HALVES}/halves_draw{r}.npz"; assert os.path.exists(f), f
    z_ = np.load(f); A, B = z_["A"], z_["B"]
    assert len(A) == N and not (A & B).any() and ((A | B) == tr).all()
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
    if mk == "gbm":
        return HistGradientBoostingClassifier(early_stopping=False, random_state=SEED,
                                              **GBM).fit(X[fit], y[fit]).decision_function(X)
    return lr_tuned_scores(X, fit, y)[0]

def windows_of(rows, band):
    out = []
    for i in np.nonzero(rows)[0]:
        t = tok_pos[i]
        for j in range(aoff[t], aoff[t] + min(int(nwin[t]), band)):
            out.append((j, i))
    return np.array(out, np.int64).reshape(-1, 2)

def bow(tag, r):
    f = f"{BOWD}/{tag}_draw{r}.npz"
    assert os.path.exists(f), f"{f} missing -- run reselect_split_W1.py select (and W1_SELECT_ON=train)"
    return np.hstack([FIX, sparse.load_npz(f).toarray().astype(np.float32)])

# ---------------------------------------------------------------- the encoder
def _torch():
    import torch
    if not torch.backends.mps.is_available():
        sys.exit("MPS unavailable. Run with native python3.11 on the Mac, not in the VM.")
    return torch, torch.device("mps")

def _batch(torch, DEV, js):
    ws = [ids[woff[j]:woff[j+1]].astype(np.int64) for j in js]
    L = max(len(w) for w in ws) + 2
    X = np.full((len(ws), L), PAD, np.int64); M = np.zeros((len(ws), L), np.int64)
    for k, w in enumerate(ws):
        s = [CLS] + list(w) + [SEP]; X[k, :len(s)] = s; M[k, :len(s)] = 1
    return torch.from_numpy(X).to(DEV), torch.from_numpy(M).to(DEV)

def encode_all(model, torch, DEV, with_head, rows=None):
    """One pass over every window of every admission (or of `rows`, for the smoke test).
    Returns token-mean, [CLS]-mean and (if with_head) mean P(outcome) per admission."""
    rows = np.ones(N, bool) if rows is None else rows
    W_all = windows_of(rows, ENC_BAND)
    H = model.config.hidden_size
    s_tok, s_cls = np.zeros((N, H)), np.zeros((N, H))
    s_prob, n_tok, n_w = np.zeros(N), np.zeros(N, np.int64), np.zeros(N, np.int64)
    o = np.argsort(wlen[W_all[:, 0]]); ts = time.time()
    model.eval()
    with torch.no_grad():
        for b in range(0, len(o), BATCH):
            k = W_all[o[b:b+BATCH]]
            X, M = _batch(torch, DEV, k[:, 0])
            out = model(input_ids=X, attention_mask=M, output_hidden_states=with_head)
            h = (out.hidden_states[-1] if with_head else out.last_hidden_state).float().cpu().numpy()
            p = torch.softmax(out.logits.float(), -1)[:, 1].cpu().numpy() if with_head else None
            for q, (j, i) in enumerate(k):
                m = int(wlen[j])
                s_tok[i] += h[q, 1:1+m].sum(0); n_tok[i] += m
                s_cls[i] += h[q, 0]; n_w[i] += 1
                if with_head: s_prob[i] += p[q]
            if b and b % (BATCH * 5000) == 0:
                el = time.time() - ts
                log(f"    encoded {b:,}/{len(o):,} windows, {(len(o)-b)*el/b/60:.0f} min left")
    wn = np.maximum(n_w, 1)
    E = dict(tok=(s_tok / np.maximum(n_tok, 1)[:, None]).astype(np.float32),
             cls=(s_cls / wn[:, None]).astype(np.float32))
    if with_head: E["prob"] = np.where(n_w > 0, s_prob / wn, np.nan).astype(np.float32)[:, None]
    return E, (time.time() - ts) / max(len(W_all), 1)

def frozen_all():
    f = f"{OUT}/frozen_all.npz"
    if os.path.exists(f) and not SMOKE: return dict(np.load(f))
    torch, DEV = _torch()
    from transformers import AutoModel
    model = AutoModel.from_pretrained(HF).to(DEV)
    log("frozen encoder: encoding every window once")
    if SMOKE:
        rows = np.zeros(N, bool); rows[:SMOKE_ENC] = True
        _, per_w = encode_all(model, torch, DEV, False, rows)
        log(f"SMOKE: frozen encode ~{per_w * int(np.minimum(nwin, ENC_BAND).sum()) / 3600:.2f} h (once)")
        return None
    E, _ = encode_all(model, torch, DEV, False)
    np.savez(f, **E); log(f"frozen readouts -> {f}")
    del model
    try: torch.mps.empty_cache()
    except Exception: pass
    return E

def finetune_and_encode(r, A):
    cache = f"{OUT}/emb_draw{r}.npz"
    if os.path.exists(cache) and not SMOKE:
        log(f"draw {r}: fine-tuned readouts cached, skipping"); return dict(np.load(cache))
    torch, DEV = _torch()
    from transformers import AutoModelForSequenceClassification

    rng = np.random.RandomState(SEED + r)                     # the same val slice rule as Design 3
    a_idx = np.nonzero(A)[0]; rng.shuffle(a_idx)
    nval = int(round(VAL_FRAC * len(a_idx)))
    val_rows = np.zeros(N, bool); val_rows[a_idx[:nval]] = True
    fit_rows = A & ~val_rows
    W, WV = windows_of(fit_rows, TUNE_BAND), windows_of(val_rows, TUNE_BAND)
    assert A[W[:, 1]].all() and A[WV[:, 1]].all(), "fine-tuning saw a row outside half A"

    model = AutoModelForSequenceClassification.from_pretrained(HF, num_labels=2).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    spe = len(W) // BATCH; nsteps = EPOCHS * spe; warm = max(1, nsteps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        s / warm if s < warm else max(0.0, (nsteps - s) / max(1, nsteps - warm))))
    ck = f"{OUT}/ckpt_draw{r}.pt"; step0 = 0
    if os.path.exists(ck) and not SMOKE:
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step0 = st["step"]; log(f"draw {r}: resuming fine-tuning at step {step0:,}/{nsteps:,}")
    log(f"draw {r}: fine-tune on {len(W):,} windows ({fit_rows.sum():,} half-A admissions), "
        f"val {len(WV):,} windows, {nsteps:,} steps")

    def save_ckpt(step):
        tmp = ck + ".tmp"
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step), tmp)
        os.replace(tmp, ck)

    ts = time.time(); done_here = 0
    for ep in range(EPOCHS):
        # a fixed permutation per epoch, so a resumed run continues the same order
        perm = np.random.RandomState(SEED + 100 * r + ep).permutation(len(W))
        model.train()
        for b in range(spe):
            step = ep * spe + b
            if step < step0: continue
            k = perm[b*BATCH:(b+1)*BATCH]
            X, M = _batch(torch, DEV, W[k, 0])
            yb = torch.from_numpy(y[W[k, 1]]).to(DEV)
            model(input_ids=X, attention_mask=M, labels=yb).loss.backward()
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); done_here += 1
            if (step + 1) % 500 == 0:
                el = time.time() - ts
                log(f"  draw {r} ep{ep} step {step+1:,}/{nsteps:,}  "
                    f"{el/done_here*1000:.0f} ms/step, {(nsteps-step-1)*el/done_here/60:.0f} min left")
            if SMOKE and done_here >= SMOKE_STEPS:
                per_step = (time.time() - ts) / done_here
                log(f"SMOKE: {per_step*1000:.0f} ms/step -> fine-tuning ~{per_step*nsteps/3600:.2f} h per partition")
                rows = np.zeros(N, bool); rows[:SMOKE_ENC] = True
                _, per_w = encode_all(model, torch, DEV, True, rows)
                log(f"SMOKE: encoding ~{per_w * int(np.minimum(nwin, ENC_BAND).sum()) / 3600:.2f} h per partition")
                log("SMOKE: nothing written."); sys.exit(0)
            if not SMOKE and (step + 1) % CKPT_EVERY == 0: save_ckpt(step + 1)
        if (ep + 1) * spe <= step0: continue          # this epoch finished before the restart
        model.eval(); sv = []
        with torch.no_grad():
            for b in range(0, len(WV), BATCH):
                X, M = _batch(torch, DEV, WV[b:b+BATCH, 0])
                sv.append(torch.softmax(model(input_ids=X, attention_mask=M).logits.float(), -1)[:, 1].cpu().numpy())
        v = roc_auc_score(y[WV[:, 1]], np.concatenate(sv))
        append(VAL, dict(draw=r, epoch=ep, val_auroc=round(v, 4)))
        log(f"  draw {r} epoch {ep}: window-level val AUROC {v:.4f}")
        save_ckpt((ep + 1) * spe)

    md = f"{OUT}/model_draw{r}"
    model.save_pretrained(md); log(f"draw {r}: fine-tuned weights -> {md}")
    log(f"draw {r}: encoding every window")
    E, _ = encode_all(model, torch, DEV, True)
    E = {"ft_tok": E["tok"], "ft_cls": E["cls"], "ft_prob": E["prob"]}
    np.savez(cache, **E)
    try: os.remove(ck)
    except OSError: pass
    del model, opt
    try: torch.mps.empty_cache()
    except Exception: pass
    log(f"draw {r}: {(time.time()-ts)/3600:.2f} h -> {cache}")
    return E

# ---------------------------------------------------------------- modes
COLS = ["draw", "tier", "n_vars", "model", "arm", "n_fit", "ev_fit", "struct", "with_text", "increment", "se"]
CCOLS = ["draw", "setting", "model", "arm", "group", "n", "events", "auc_arm", "auc_ref", "diff", "se"]

def check():
    import collections
    tot = 0
    for r in range(5):
        tr = draw_split(r); A, B = halves(r, tr)
        nA = int(np.minimum(nwin[tok_pos[A]], TUNE_BAND).sum()); tot += nA
        log(f"draw {r}: A {A.sum():,} / B {B.sum():,} | fine-tune windows {nA:,} "
            f"-> {EPOCHS * nA // BATCH:,} steps | bow_rs {os.path.exists(f'{BOWD}/bow_rs_draw{r}.npz')} "
            f"bow_rt {os.path.exists(f'{BOWD}/bow_rt_draw{r}.npz')}")
    g = collections.Counter(GROUP); log(f"coverage groups: {dict(g)}")
    log(f"projected at 0.171 s per window-step: fine-tuning ~{tot/5*EPOCHS*0.171/3600:.1f} h per partition")
    log("check OK")

def run():
    done = set()
    if os.path.exists(CSV):
        o = pd.read_csv(CSV); done = set(zip(o.draw, o.tier, o.model, o.arm))
    FR = frozen_all()
    for r in DRAWS:
        tr = draw_split(r); A, B = halves(r, tr); te = ~tr
        log(f"draw {r}: A {A.sum():,} (fine-tune) | B {B.sum():,} (downstream) | test {te.sum():,}")
        E = finetune_and_encode(r, A)
        ARMS = {"bow_rs": bow("bow_rs", r), "bow_rt": bow("bow_rt", r),
                "frozen_tok": np.hstack([FIX, FR["tok"]]), "frozen_cls": np.hstack([FIX, FR["cls"]]),
                **{k: np.hstack([FIX, E[k]]) for k in FT}}
        have_txt = os.path.exists(TXT) and r in set(pd.read_csv(TXT).draw)
        for setting in ("text", "T1", "T5"):
            S = None if setting == "text" else encode_tier(tiers[setting], B)
            for mk in LEARNERS:
                todo = [a for a in ARM_NAMES if (r, setting, mk, a) not in done]
                if setting == "text" and have_txt: continue
                if setting != "text" and not todo: continue
                sb = None if S is None else score(S, B, mk)
                sc = {}
                for arm in ARM_NAMES:
                    sc[arm] = score(ARMS[arm] if S is None else np.hstack([S, ARMS[arm]]), B, mk)
                    if S is None:
                        append(TXT, dict(draw=r, arm=arm, model=mk,
                                         auroc=round(roc_auc_score(y[te], sc[arm][te]), 4)))
                    elif arm in todo:
                        a, b, d, se = paired(y[te], sb[te], sc[arm][te])
                        append(CSV, dict(draw=r, tier=setting, n_vars=len(tiers[setting]), model=mk, arm=arm,
                                         n_fit=int(B.sum()), ev_fit=int(y[B].sum()), struct=round(a, 4),
                                         with_text=round(b, 4), increment=round(d, 4), se=round(se, 4)), COLS)
                        log(f"  {setting}/{mk}/{arm:10s}: struct {a:.4f} -> {b:.4f}  inc {d:+.4f}")
                # coverage split, as coverage_split_W1.py: arm minus bow_rs within each group
                for arm in ("ft_cls", "frozen_cls"):
                    for grp in ("full", "partial", "all"):
                        m = te & (GROUP != "none") if grp == "all" else te & (GROUP == grp)
                        a1, a2, d, se = paired(y[m], sc["bow_rs"][m], sc[arm][m])
                        append(COV, dict(draw=r, setting=setting, model=mk, arm=arm, group=grp, n=int(m.sum()),
                                         events=int(y[m].sum()), auc_arm=round(a2, 4), auc_ref=round(a1, 4),
                                         diff=round(d, 4), se=round(se, 4)), CCOLS)
        if not have_txt:
            append(TXT, dict(draw=r, arm="ft_prob_direct", model="none",
                             auroc=round(roc_auc_score(y[te], np.nan_to_num(E["ft_prob"][te, 0])), 4)))
    summary()

def ci(v):
    v = np.asarray(v, float); n = len(v)
    h = stats.t.ppf(0.975, n-1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return v.mean(), v.mean() - h, v.mean() + h

def summary():
    if not os.path.exists(CSV): sys.exit("no results yet")
    res = pd.read_csv(CSV).drop_duplicates(["draw", "tier", "model", "arm"], keep="last")
    print("\n=== collapse by arm (all windows; half-B downstream; 95% CI t over draws) ===")
    for mk in LEARNERS:
        for arm in ARM_NAMES:
            q = res[(res.model == mk) & (res.arm == arm)].pivot(index="draw", columns="tier", values="increment").dropna()
            if q.empty or not {"T1", "T5"} <= set(q.columns): continue
            m, lo, hi = ci(100 * (q.T1 - q.T5) / q.T1)
            print(f"  {mk:4s}/{arm:10s} n={len(q)}  T1 {q.T1.mean():+.4f}  T5 {q.T5.mean():+.4f}"
                  f"  collapse {m:5.1f}% [{lo:5.1f}, {hi:5.1f}]  falls {(q.T1 > q.T5).sum()}/{len(q)}")
    print("\n=== paired within draw ===")
    for mk in LEARNERS:
        for a_, b_ in list(TWIN.items()) + [("ft_cls", "bow_rs"), ("ft_cls", "bow_rt")]:
            for t in TIERS:
                p = res[(res.model == mk) & (res.tier == t)].pivot(index="draw", columns="arm", values="increment")
                if not {a_, b_} <= set(p.columns): continue
                d = (p[a_] - p[b_]).dropna()
                if len(d) < 2: continue
                m, lo, hi = ci(d)
                print(f"  {mk:4s} {t} {a_:8s} - {b_:10s}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  first ahead in {(d > 0).sum()}/{len(d)}")
    if os.path.exists(COV):
        c = pd.read_csv(COV).drop_duplicates(["draw", "setting", "model", "arm", "group"], keep="last")
        print("\n=== coverage interaction: (arm - bow_rs) partly read minus fully read ===")
        for arm in ("ft_cls", "frozen_cls"):
            for st in ("text", "T1", "T5"):
                for mk in LEARNERS:
                    p = c[(c.arm == arm) & (c.setting == st) & (c.model == mk)].pivot(index="draw", columns="group", values="diff")
                    if not {"full", "partial"} <= set(p.columns) or len(p) < 2: continue
                    m, lo, hi = ci(p.partial - p.full)
                    print(f"  {arm:10s} {st:4s} {mk:4s} full {p.full.mean():+.4f} partial {p.partial.mean():+.4f}"
                          f"  interaction {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    if os.path.exists(TXT):
        print("\n=== text-only held-out AUROC (mean over draws) ===")
        print(pd.read_csv(TXT).drop_duplicates(["draw", "arm", "model"], keep="last")
              .pivot_table(index="arm", columns="model", values="auroc").round(4).to_string())
    log("done -- tell Claude and it will read the CSVs back")

{"check": check, "run": run, "summary": summary}[MODE]()
