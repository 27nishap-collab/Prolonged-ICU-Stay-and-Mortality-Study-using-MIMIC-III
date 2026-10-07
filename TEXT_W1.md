# W1 text pipeline — build, validation, deviations

Built by `build_text_W1.py` (`build` → `vocab` → `open`). Outputs under `out/`:
`text_parts/` (26 parquet parts, cleaned documents + closed-vocabulary counts),
`text_meta_W1.parquet` (labels and splits), `W1_openvocab_full.csv`,
`text_open_W1.npz` + `text_open_vocab_W1.csv`, `W1_openvocab_l1_path.csv`.

**No deduplication.** Weissman does not deduplicate, and neither do Huang,
Husmann or Hsu. Rules A and B stay in `Analysis/notes_extract` and are applied
only as a sensitivity test on our own cohorts.


> **LEAK FOUND AND FIXED (2026-09-17).** Discharge summaries carry a NULL `charttime`, so the `COALESCE(charttime, chartdate)` convention above stamps them at midnight of the discharge DAY and pulls 1,454 of them (1,398 admissions, 5.4%) inside the 48h window while the patient is still in hospital. A `--v2` corpus drops the Discharge summary category; it changes the measured text increment by about 2%. See RESULTS_ablation.md section 1. Every v2 artifact carries a `_v2` suffix.

---

## 1. Corpus

| | |
|---|---|
| admissions | 25,692 |
| notes | 235,101 |
| raw text | 548 MB |
| category filter | everything except `Radiology` |
| timestamp | `COALESCE(charttime, chartdate)` in `[admittime, admittime+48h)` |
| error filter | `iserror IS NULL` |

Against Table 1:

| | ours | published |
|---|---|---|
| notes per admission | **6 (4–11)** | 6 (4–11) |
| notes per admission, outcome-yes | **8 (6–12)** | 8 (6–12) |
| total raw word count | 1,061 (623–2,699) | 1,023 (562–2,266) |
| raw word count, outcome-yes | 1,340 (900–2,152) | 1,410 (956–2,292) |

Note counts match exactly. Word counts are within 4% on the median; the wider
IQR is expected, since the published word count is computed on whatever
tokenisation their R pipeline used and ours is whitespace tokens.

The corpus definition here (exclude Radiology only) is deliberately **different
from the eligibility filter** used to build the cohort (exclude Radiology, ECG
and Echo). The paper does not distinguish the two, but only this pairing
reproduces both the cohort size and the notes-per-admission rows. See
`METHODS_weissman.md` gap #7.

---

## 2. `preprocess()`

Exactly as in the authors' released code: lowercase → delete
`` !"#$%&()*+,:;<=>?@[\]^_`{|}~ `` and all digits → replace `- . / '` with a
space → drop NLTK English stopwords → whitespace split, no stemming.

Median word count 1,061 raw → **768 cleaned** (72% of tokens retained).

Example of the output:

> chief complaint gib hotn saw examined patient physically present icu resident
> key portions services provided agree note including assessment plan hpi yo man
> hep c cirrhosis presents one episode hematemesis ed vs stable initial hct prior
> one month ago transferred micu became hypotensive started dopamine repeat hct
> came back unable get ngt central line placed note u ed showed ascites allergies
> known

The MIMIC de-identification scaffolding `[** … **]` is destroyed by this regex
(brackets, asterisks and digits are all deleted), but residue survives —
`hospital`, `name`, `md` — and those tokens are frequent enough to enter the
open vocabulary. That is a property of their pipeline, not a bug in ours.

### Stopword-list version — a declared deviation

The list is vendored as `nltk_stopwords_english.txt` (NLTK 3.10.3, **198
words**) because the shell this runs in cannot reach the NLTK data mirror, and
pinning it makes the run reproducible. Weissman ran Python 3.5.1 in 2016–17,
when NLTK's English list had **153 words** and did not contain the contraction
fragments (`wouldn`, `doesn`, `t`, `s`, `d`, `ll`, …).

This has a visible consequence for the closed vocabulary, below, and is the most
likely explanation for our open vocabulary being smaller than theirs: a longer
stopword list strips more tokens, so fewer phrases clear the 5% threshold.

---

## 3. Closed vocabulary — 21 a-priori terms

Prevalence in the corpus, matched whole-word on the **raw** text (primary):

| term | % of admissions | | term | % |
|---|---|---|---|---|
| family | 67.8 | | palliative | 0.8 |
| code status | 27.8 | | palliative care | 0.5 |
| comfortable | 22.8 | | hospice | 0.4 |
| family meeting | 5.5 | | mortality | 0.3 |
| high risk | 2.7 | | doesn't want | 0.2 |
| death | 2.2 | | comfort care | 0.1 |
| wishes | 2.2 | | survival | 0.1 |
| goals of care | 1.1 | | palliation | 0.0 |
| die | 1.0 | | pall care | 0.0 |
| poor prognosis | 0.8 | | wouldn't want | 0.0 |
| | | | mortal | 0.0 |

`mortal` is 0.0% because whole-word matching does not count `mortality`; both
terms are on the list separately, so this is correct, not a miss.

### Why the raw text is matched, not the preprocessed text

Running the terms through `preprocess()` — which is what their pipeline would
actually have seen — mangles several of them:

| term | after preprocess() | |
|---|---|---|
| goals of care | `goals care` | "of" is a stopword |
| doesn't want | `want` | "doesn" and "t" are stopwords |
| wouldn't want | `want` | **collides with the above** |

The collision is not cosmetic. `doesn't want` and `wouldn't want` are 0.2% and
0.0% prevalent as written; the merged `want` is **9.1%** — a term that no longer
means anything close to what was specified a priori. Both variants are computed
and stored (`cv_*` raw, `cvp_*` post-preprocess) so the choice is auditable, and
the raw match is primary because it preserves the terms as the authors defined
them.

Under the 153-word stopword list of their era, `wouldn` and `t` are not
stopwords, so `wouldn't want` would have become `wouldn t want` and not
collided. Their pipeline was probably not affected the way ours would be —
which is precisely why the raw match is the safer primary.

---

## 4. Open vocabulary

### 4.1 The vocabulary

1-, 2- and 3-grams with document frequency ≥ 5% **in the training set only**:

| | ours | published |
|---|---|---|
| unigrams | 2,111 (of 138,002 distinct) | — |
| bigrams | 1,641 | — |
| trigrams | 1,073 | — |
| **total** | **4,825** | **5,790** |

Within 17% of their count, and smaller in the direction the stopword-list
difference predicts.

`CountVectorizer(ngram_range=(1,3), min_df=0.05)` cannot be used here: it builds
the full n-gram vocabulary before applying `min_df`, and 1-3 grams over ~90M
tokens is hundreds of millions of distinct trigrams in a 3.9 GB shell. The
vocabulary is instead built level by level using downward closure — a bigram
with df ≥ t requires both of its unigrams to have df ≥ t, and likewise trigrams
from bigrams. This is **exact**, not an approximation: it yields the same
vocabulary `CountVectorizer` would, in bounded memory. The candidate counts show
how much work that saves: 975,206 bigram candidates instead of every bigram in
the corpus, and 4,618 trigram candidates instead of every trigram.

### 4.2 Selecting 500

L1 logistic regression, 10-fold CV on the training set, scored by AUROC.
`glmnet`'s lambda path is walked as an sklearn `C` grid:

| C | CV AUROC | non-zero |
|---|---|---|
| 0.0005 | 0.501 | 2 |
| 0.001 | 0.709 | 5 |
| 0.005 | 0.847 | 45 |
| 0.01 | 0.865 | 102 |
| **0.05** | **0.881** | **610** |
| 0.1 | 0.875 | 1,194 |
| 0.5 | 0.842 | 2,740 |

The path peaks at C = 0.05 with 610 non-zero coefficients, so the 500 selected
are the largest |coef| from the CV-optimal fit rather than from a truncated
grid. Selected mix: 417 unigrams, 62 bigrams, 21 trigrams; training document
frequency 5.0%–90.7%.

`log1p` is applied to the counts **for the selector only**. The features shipped
downstream are raw integer counts, as the paper specifies.

### 4.3 The selected block is mostly unigram, though the spec is trigram

| | candidate pool | selected 500 | share of \|coef\| |
|---|---|---|---|
| 1-gram | 2,034 (43%) | **419 (84%)** | **85%** |
| 2-gram | 1,593 (34%) | 60 (12%) | 12% |
| 3-gram | 1,060 (23%) | 21 (4%) | 3% |

2,653 multi-word phrases were available and 81 were taken. At `min_df >= 5%` a
qualifying n-gram is nearly always collinear with its parts, and L1 keeps one of
a correlated set; the survivors are the phrases whose meaning is not their
parts' — `transfer floor` (-0.287), `cvp mmhg` (-0.190) (a central line is in),
`intubated sedated` (-0.151) (one state, not two), `vap comments` (-0.139) and
`per protocol` (-0.138) (documentation artifacts rather than clinical content),
`insulin gtt` (-0.098).

Worth carrying into any claim about "the text representation": it is effectively
bag-of-unigrams plus ~80 compositional phrases. The same holds in v2 and in the
disposition-free v3 selection (85% / 12% / 3%).

### 4.4 What the selected phrases are

Most positive: *angio, intubation, tachycardia, intubated, open, amber, drain,
vent, rising, ffp, abg, specifics, proxy, movement, mixed*

Most negative: *oob, without, access, cooperative, clear, cvp mmhg, transfer,
chair, ew, diet, floor, transfer floor, extubate, extubation, extubated*

The negative end is almost entirely **disposition and de-escalation language** —
extubated, transfer floor, chair, diet, out of bed. The model is not reading
severity so much as reading whether the team has started writing about sending
the patient home. Worth holding onto when the models are compared: it bears
directly on our own question about what the text advantage is made of.

---

## 5. The split — and the leak in it

Created here, once, so vocabulary selection and modelling cannot disagree.

| | train | test | outcome rate |
|---|---|---|---|
| `split` (paper's protocol) | 19,269 | 6,423 | 0.2096 / 0.2096 |
| `split_grouped` (deviation) | 19,316 | 6,376 | 0.2107 / 0.2061 |

`caret::createDataPartition(df$Y, p=0.75)` is stratified on the outcome and
**not grouped by patient**, and the paper never mentions repeat admissions. The
consequence is measurable: **23.1% of test admissions belong to a patient who
also appears in training.** Both splits are stored; the paper's is primary and
the grouped one is reported as a deviation.

---

## 6. Feature blocks now available

| block | shape | file |
|---|---|---|
| structured (18 predictors) | 25,692 × 18 | `design_W1.parquet` |
| closed vocabulary | 25,692 × 21 counts | `text_parts/*.parquet`, `cv_*` |
| closed vocabulary, post-preprocess | 25,692 × 20 counts | `text_parts/*.parquet`, `cvp_*` |
| open vocabulary | 25,692 × 500 counts | `text_open_W1.npz` |
| word count covariate | 25,692 × 1 | `text_meta_W1.parquet`, `n_words_raw` |

Text-inclusive models take structured + 21 + 500 + word count = **540
predictors**, matching the paper's description.
