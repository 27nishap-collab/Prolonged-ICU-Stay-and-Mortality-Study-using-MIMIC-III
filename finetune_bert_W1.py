"""Does the collapse survive a FINE-TUNED encoder? -- RUN THIS ON THE MAC.

    cd /Users/Shared/MIMIC_git/Analysis/replicate_weissman
    caffeinate -i env HF_HUB_OFFLINE=1 python3 finetune_bert_W1.py 2>&1 | tee ft.log

~21.5 h for five draws. Resumable at draw granularity -- Ctrl-C and restart is
safe, and W1_DRAWS=0,1,2 one night and 3,4 the next is the intended way to run
it. `caffeinate -i` keeps the Mac awake; without it the run stops at sleep.

--------------------------------------------------------------------------
WHY. Every text arm in this project is count-based or FROZEN: bow, LSA, and two
frozen encoders under two poolings. The two published studies with the largest
text gains -- Deznabi 2021 (+0.064) and Lyu 2022 (+0.050) -- both fine-tune a
clinical BERT on the outcome, and the four studies that do not (Jin, Khadanga,
J. Zhang, Husmann -- the last verified frozen: "we did not train end-to-end due
to resource constraints") land between +0.016 and +0.021. Lyu's own ablation
puts frozen ClinicalBERT at AUCPR 0.265 and fine-tuned at 0.482 on identical
data.

So the obvious reading of our near-null at T5 is "your text representation was
weak", not "the baseline saturated". Thirteen representations collapsing 84-132%
does not answer it, because all thirteen sit in the bottom row of that table.
This fills the one empty cell.

Bio_ClinicalBERT, because it is the encoder Deznabi and Lyu both fine-tuned.

--------------------------------------------------------------------------
THE MEASUREMENT IT SUPPORTS, and the one it does NOT.

Reported: for each draw, the text increment at T1 and at T5, for the fine-tuned
arm alongside the frozen arm and bow, all on the SAME partition. The comparison
is within-draw and paired, so partition variance -- the thing F3 showed is worth
6 points -- largely cancels.

Five draws, not ten. The collapse is a within-draw difference (T1 minus T5), the
frozen arm fell in 10 of 10, and 5 of 5 is a sign test at p=0.03. That is enough
for a directional claim, which is what this experiment is for.

NOT reported as a headline CI. In the frozen design the representation is
identical across draws and only the classifier is refit, so the spread is pure
partition variance. Here each draw trains its OWN encoder, so the spread also
contains head initialisation, batch order and MPS nondeterminism. That is the
more honest pipeline variance, but it is not the same quantity, and the two
intervals must not be printed side by side in Figure 1 as if they were.

--------------------------------------------------------------------------
CALIBRATED, not assumed (out/calibrate_finetune_bio_clinicalbert.json):

  batch 8      171.2 ms/window   <- the only usable training configuration
  batch 16     359.0 ms/window   memory-pressure artifact, NOT faster
  batch 24     OOM at 19.93 GiB against a 20.13 GiB ceiling

Per-window cost DOUBLING from batch 8 to 16 is the allocator thrashing below the
limit. Do not "optimise" BATCH upward. Forward is linear in padded length at
0.066 ms/token, which reproduces the 31 ms/window encode baseline exactly, so
the backward factor is 5.5x, not the 3x that was assumed before measuring.

Nothing else heavy should run on the Mac during this.

--------------------------------------------------------------------------
DESIGN NOTES worth knowing before reading a result off this.

  * Fine-tuning reads the OUTCOME. It therefore happens strictly inside each
    draw's own training admissions, and the encoder is retrained per draw. The
    val slice for epoch monitoring is carved out of TRAIN, never test.

  * Band w0-2 for tuning, w0-4 for the final encode. The frozen arm we compare
    against is w0-4 mean-pooled, so the fine-tuned arm must be too. Tuning on
    the first two windows keeps the cost survivable.

  * Window-level labels inherit the admission's label, so an admission with two
    windows contributes twice. This is what Deznabi does (a softmax head per
    note) and the distortion is bounded at 2x by the band. Not corrected, on
    purpose -- correcting it would make us less comparable to the precedent.

  * No class reweighting, for the same reason. Event rate is 21%.

  * Cached artifact is the pooled embedding, not the weights: 79 MB per draw
    against 440 MB, and it is what the fit stage actually consumes. A crash in
    fitting therefore never costs a re-tune. out/ is gitignored.

Nothing leaves the machine: local tok16 integer ids and the existing
~/.cache/huggingface weights, exactly as encode_bert_local.py.
"""
import os, sys, json, time, glob, warnings
import numpy as np, pandas as pd, torch
from scipy import sparse, stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification

warnings.filterwarnings("ignore")
D  = os.path.dirname(os.path.abspath(__file__))
U  = os.environ.get("W1_DATA", os.path.join(D, "out"))
SEED = 20180701
GBM  = dict(max_depth=10, learning_rate=0.05, max_iter=500, min_samples_leaf=15)

TAG    = os.environ.get("W1_MODEL", "bio_clinicalbert")
HF     = {"bio_clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
          "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"}
DRAWS  = [int(x) for x in os.environ.get("W1_DRAWS", "0,1,2,3,4").split(",")]
EPOCHS = int(os.environ.get("W1_EPOCHS", 2))
BATCH  = int(os.environ.get("W1_BATCH", 8))        # see the calibration note
TUNE_BAND = int(os.environ.get("W1_TUNE_BAND", 2))
ENC_BAND  = int(os.environ.get("W1_ENC_BAND", 4))
LR     = float(os.environ.get("W1_LR", 2e-5))
VAL_FRAC = 0.05
TIERS  = ["T1", "T5"]

SPEC = os.environ.get("W1_SPEC", "observable")
SPEC = {"leakfree": "observable", "published": "replication"}.get(SPEC, SPEC)
assert SPEC in ("observable", "replication"), SPEC
SFX  = "" if SPEC == "replication" else "_observable"
CORPUS = os.environ.get("W1_CORPUS", "v2")
assert CORPUS in ("v1", "v2"), CORPUS
SUF = "" if CORPUS == "v1" else "_v2"

OUT = os.path.join(U, f"w1_finetune{SUF}{SFX}"); os.makedirs(OUT, exist_ok=True)
CSV = f"{OUT}/W1_finetune.csv"

if not torch.backends.mps.is_available():
    sys.exit("MPS unavailable. This is a 21-hour job; it is not worth starting on CPU.")
DEV = torch.device("mps")
t0 = time.time()
def log(m): print(f"[{(time.time()-t0)/3600:5.2f}h] {m}", flush=True)

# ---------------------------------------------------------------- DeLong
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5*(i+j-1)+1; i = j
    T2 = np.empty(N); T2[J] = T; return T2

def delong(y, scores):
    y = np.asarray(y); s = np.atleast_2d(np.asarray(scores, float))
    o = np.argsort(-y, kind="mergesort"); s = s[:, o]
    m = int(y.sum()); n = len(y) - m; k = s.shape[0]
    tx = np.vstack([_midrank(s[r, :m]) for r in range(k)])
    ty = np.vstack([_midrank(s[r, m:]) for r in range(k)])
    tz = np.vstack([_midrank(s[r])     for r in range(k)])
    aucs = tz[:, :m].sum(1)/m/n - (m+1.0)/2.0/n
    return aucs, (np.atleast_2d(np.cov((tz[:, :m]-tx)/n))/m
                  + np.atleast_2d(np.cov(1.0-(tz[:, m:]-ty)/m))/n)

def paired(y, a, b):
    au, c = delong(y, [a, b]); v = c[0,0]+c[1,1]-2*c[0,1]
    d = au[1]-au[0]; se = float(np.sqrt(max(v, 0)))
    return float(au[0]), float(au[1]), float(d), float(se)

# ---------------------------------------------------------------- data
tiers = json.load(open(f"{U}/W1_tier_columns{SFX}.json"))
T   = pd.read_parquet(f"{U}/tiers_W1{SFX}.parquet")
tab = pd.read_parquet(f"{U}/model_tabular_W1{SUF}.parquet")
Xo  = sparse.load_npz(f"{U}/text_open_W1{SUF}{'_aligned' if SUF else ''}.npz"
                      ).toarray().astype(np.float32)
tab = tab.set_index("hadm_id"); order = tab.index.to_numpy()
assert (np.sort(order) == order).all()
T = T.set_index("hadm_id").reindex(order)
cvc  = [c for c in tab.columns if c.startswith("cv_")]
FIX  = np.hstack([tab[cvc].to_numpy(np.float32),
                  tab[["n_words_raw"]].to_numpy(np.float32)])
BOW  = np.hstack([FIX, Xo])
y    = tab.labels.to_numpy().astype(int)
N    = len(y)

z = np.load(f"{U}/tok16_{TAG}_W1.npz")
ids, wlen, nwin, hadm = z["ids"], z["wlen"], z["nwin"], z["hadm"]
woff = np.concatenate([[0], np.cumsum(wlen.astype(np.int64))])
aoff = np.concatenate([[0], np.cumsum(nwin.astype(np.int64))])
tok_pos = pd.Index(hadm).get_indexer(order)          # tok16 row for each tabular row
assert not (tok_pos < 0).any(), "tok16 does not cover every admission"

vocab = open(os.path.join(D, "hf", TAG, "vocab.txt")).read().split("\n")
CLS, SEP, PAD = vocab.index("[CLS]"), vocab.index("[SEP]"), vocab.index("[PAD]")

# frozen mean-pooled embedding, from the shards already on disk -- the arm the
# fine-tuned one has to beat
fs = sorted(glob.glob(f"{U}/bert_{TAG}_w0-{ENC_BAND}/b_*.npz"))
assert fs, f"no frozen shards in out/bert_{TAG}_w0-{ENC_BAND}/ -- run encode_bert_local.py first"
g = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("hadm", "s_tok", "n_tok")}
gp = pd.Index(g["hadm"]).get_indexer(order); assert not (gp < 0).any()
FROZEN = np.hstack([FIX, (g["s_tok"][gp] /
                    np.maximum(g["n_tok"][gp], 1)[:, None]).astype(np.float32)])
log(f"{N:,} admissions | outcome {y.mean():.4f} | bow {BOW.shape[1]} cols "
    f"| frozen {FROZEN.shape[1]} cols | draws {DRAWS} | {EPOCHS} epochs, batch {BATCH}")

# ------------------------------------------- partitions, verbatim from
# run_repeated_splits_W1.py so these ARE draws 0-4 of the frozen arm
def draw_split(r):
    rng = np.random.RandomState(SEED + 1000 * (r + 1))
    tr = np.ones(len(y), bool)
    for lab in (0, 1):
        idx = np.flatnonzero(y == lab); rng.shuffle(idx)
        tr[idx[: int(round(0.25 * len(idx)))]] = False
    return tr

def encode_tier(cols, tr):
    out = []
    for c in cols:
        s = T[c]
        if not pd.api.types.is_numeric_dtype(s):
            v = s.astype(object).where(s.notna(), "__na__")
            k = v[tr].value_counts(); lv = set(k[k >= 0.01*tr.sum()].index)
            v = v.where(v.isin(lv), "__other__")
            for L in sorted(lv | {"__other__"}): out.append((v == L).to_numpy(np.float32))
        else: out.append(s.to_numpy(np.float32))
    return np.column_stack(out)

def score(X, tr, mk):
    Xtr, ytr = X[tr], y[tr]
    if mk == "gbm":
        return HistGradientBoostingClassifier(early_stopping=False, random_state=SEED,
                                              **GBM).fit(Xtr, ytr).decision_function(X)
    mu = np.nanmean(Xtr, 0); mu = np.where(np.isnan(mu), 0.0, mu)
    A = np.where(np.isnan(Xtr), mu, Xtr); B = np.where(np.isnan(X), mu, X)
    sc = StandardScaler().fit(A)
    return LogisticRegression(max_iter=5000).fit(sc.transform(A), ytr
           ).decision_function(sc.transform(B))

# ---------------------------------------------------------------- the encoder
def make_batch(js):
    ws = [ids[woff[j]:woff[j+1]].astype(np.int64) for j in js]
    L = max(len(w) for w in ws) + 2
    X = np.full((len(ws), L), PAD, np.int64); A = np.zeros((len(ws), L), np.int64)
    for r_, w in enumerate(ws):
        s = [CLS] + list(w) + [SEP]
        X[r_, :len(s)] = s; A[r_, :len(s)] = 1
    return torch.from_numpy(X).to(DEV), torch.from_numpy(A).to(DEV)

def windows_of(rows, band):
    """(window index, tabular row) for each window of each admission in `rows`."""
    out = []
    for i in np.nonzero(rows)[0]:
        t = tok_pos[i]
        for j in range(aoff[t], aoff[t] + min(int(nwin[t]), band)):
            out.append((j, i))
    return np.array(out, np.int64)

def finetune_and_encode(r, tr):
    """Fine-tune on draw r's training admissions, then mean-pool-encode everyone."""
    cache = f"{OUT}/emb_ft_{TAG}_draw{r}.npy"
    if os.path.exists(cache):
        log(f"draw {r}: fine-tuned embedding cached, skipping {EPOCHS} epochs")
        return np.load(cache)

    torch.manual_seed(SEED + r); rng = np.random.RandomState(SEED + r)
    # val slice carved out of TRAIN admissions only
    tr_idx = np.nonzero(tr)[0]; rng.shuffle(tr_idx)
    nval = int(round(VAL_FRAC * len(tr_idx)))
    val_rows = np.zeros(N, bool); val_rows[tr_idx[:nval]] = True
    fit_rows = tr.copy(); fit_rows[val_rows] = False

    W  = windows_of(fit_rows, TUNE_BAND)
    WV = windows_of(val_rows,  TUNE_BAND)
    model = AutoModelForSequenceClassification.from_pretrained(HF[TAG], num_labels=2).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    nsteps = EPOCHS * (len(W) // BATCH); warm = max(1, nsteps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        s / warm if s < warm else max(0.0, (nsteps - s) / max(1, nsteps - warm))))
    log(f"draw {r}: fine-tune on {len(W):,} windows ({fit_rows.sum():,} admissions), "
        f"val {len(WV):,} windows, {nsteps:,} steps")

    step, tstart = 0, time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(len(W))
        for b in range(len(W) // BATCH):
            k = perm[b*BATCH:(b+1)*BATCH]
            X, A = make_batch(W[k, 0])
            yb = torch.from_numpy(y[W[k, 1]]).to(DEV)
            model(input_ids=X, attention_mask=A, labels=yb).loss.backward()
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 500 == 0:
                el = time.time() - tstart
                log(f"  draw {r} ep{ep} step {step:,}/{nsteps:,}  "
                    f"{el/60:.0f} min, {(nsteps-step)*el/step/60:.0f} min left")
        # window-level val AUROC: is the second epoch buying anything?
        model.eval(); sv = []
        with torch.no_grad():
            for b in range(0, len(WV), BATCH):
                X, A = make_batch(WV[b:b+BATCH, 0])
                # rank by P(outcome), which depends only on the GAP between the two
                # logits. Ranking by logits[:, 1] alone (the original line, used in the
                # 2026-09-23 run) ignores the "no outcome" logit and can mis-order
                # windows; it affected only this monitoring number, never the embeddings.
                lg = model(input_ids=X, attention_mask=A).logits.float()
                sv.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
        sv = np.concatenate(sv)
        log(f"  draw {r} epoch {ep}: window-level val AUROC "
            f"{roc_auc_score(y[WV[:, 1]], sv):.4f}")

    # encode EVERY admission at the frozen arm's band, mean over content tokens
    log(f"draw {r}: encoding all {N:,} admissions at w0-{ENC_BAND}")
    base = model.base_model
    W_all = windows_of(np.ones(N, bool), ENC_BAND)
    s_tok = np.zeros((N, base.config.hidden_size), np.float64)
    n_tok = np.zeros(N, np.int64)
    o = np.argsort(wlen[W_all[:, 0]])              # length-sorted: less padding
    model.eval()
    with torch.no_grad():
        for b in range(0, len(o), BATCH):
            k = W_all[o[b:b+BATCH]]
            X, A = make_batch(k[:, 0])
            h = base(input_ids=X, attention_mask=A).last_hidden_state.float().cpu().numpy()
            for r_, (j, i) in enumerate(k):
                m = int(wlen[j])
                s_tok[i] += h[r_, 1:1+m].sum(0); n_tok[i] += m
    emb = (s_tok / np.maximum(n_tok, 1)[:, None]).astype(np.float32)
    np.save(cache, emb)
    del model, opt
    try: torch.mps.empty_cache()
    except Exception: pass
    log(f"draw {r}: {(time.time()-tstart)/3600:.2f} h -> {cache}")
    return emb

# ---------------------------------------------------------------- run
done = set()
if os.path.exists(CSV):
    old = pd.read_csv(CSV)
    done = set(zip(old.draw, old.tier, old.model, old.arm))
    log(f"resuming: {len(done)} cells on disk")

for r in DRAWS:
    tr = draw_split(r)
    need = [(t, m, a) for t in TIERS for m in ("gbm", "lr")
            for a in ("bow", "frozen", "finetuned") if (r, t, m, a) not in done]
    if not need:
        log(f"draw {r}: complete, skipping"); continue
    log(f"draw {r}: train {tr.sum():,} / test {(~tr).sum():,} "
        f"| test outcome {y[~tr].mean():.4f}")
    ARMS = {"bow": BOW, "frozen": FROZEN}
    if any(a == "finetuned" for _, _, a in need):
        ARMS["finetuned"] = np.hstack([FIX, finetune_and_encode(r, tr)])

    for tier in TIERS:
        S = encode_tier(tiers[tier], tr)
        for mk in ("gbm", "lr"):
            sb = score(S, tr, mk)
            for arm in ("bow", "frozen", "finetuned"):
                if (r, tier, mk, arm) in done: continue
                a, b, d, se = paired(y[~tr], sb[~tr],
                                     score(np.hstack([S, ARMS[arm]]), tr, mk)[~tr])
                pd.DataFrame([dict(draw=r, tier=tier, n_vars=len(tiers[tier]), model=mk,
                                   arm=arm, struct=round(a, 4), with_text=round(b, 4),
                                   increment=round(d, 4), se=round(se, 4))]
                             ).to_csv(CSV, mode="a", header=not os.path.exists(CSV),
                                      index=False)
                log(f"  {tier}/{mk}/{arm:9s}: struct {a:.4f} -> +text {b:.4f}  inc {d:+.4f}")

# ---------------------------------------------------------------- summary
res = pd.read_csv(CSV).drop_duplicates(["draw", "tier", "model", "arm"])
pd.set_option("display.width", 200)
print("\n=== increment by tier and arm (mean over draws) ===")
print(res.pivot_table(index=["model", "arm"], columns="tier",
                      values="increment").round(4).to_string())
print("\n=== THE TEST: does the collapse survive fine-tuning? ===")
for mk in ("gbm", "lr"):
    for arm in ("bow", "frozen", "finetuned"):
        q = res[(res.model == mk) & (res.arm == arm)].pivot(
            index="draw", columns="tier", values="increment")
        if not {"T1", "T5"} <= set(q.columns): continue
        q = q.dropna(subset=["T1", "T5"])
        coll = (q.T1 - q.T5) / q.T1
        print(f"  {mk:3s}/{arm:9s}  T1 {q.T1.mean():+.4f}  T5 {q.T5.mean():+.4f}  "
              f"collapse {100*coll.mean():5.1f}%  "
              f"falls in {(q.T1 > q.T5).sum()}/{len(q)} draws")
log("done -- tell Claude and it will read the CSV back")
