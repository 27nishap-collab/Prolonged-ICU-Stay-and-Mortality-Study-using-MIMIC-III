# Specification decision — observable is primary

**Decided 2026-09-20, before any observable numbers were computed.**

That date matters. The audit (`AUDIT_48h.md`) found four predictors not
observable at hour 48. The choice of which specification carries the primary
analysis was made on principle, with only the *exposure* of each leak known
(100% / 6.3% / 0.2% / 0.05%) and none of the resulting AUROCs. Running both and
then choosing would be specification-shopping on the primary outcome — the exact
methodological failure this project criticises in others.

## The rule

> A predictor enters the analysis only if its value is knowable at hour 48 of
> the admission being predicted.

Prior admissions' discharge codes satisfy this: they were finalised before this
admission began. The current admission's discharge codes do not.

## What changes

| | replication specification | **observable specification (primary)** |
|---|---|---|
| `elixhauser` | ICD-9 from *this* admission's discharge coding | **removed** |
| — replaced by | — | `has_prior_admission` (binary) + `prior_elixhauser` (score from the patient's earlier admissions, NaN for the 78.2% with none) |
| `n_icu_stays` | count over the whole admission | **stays beginning before hour 48** |
| `first_careunit`, `pre_icu_hours` | earliest stay, any time | **earliest stay beginning before hour 48** |
| `first_service`, `surgical_service` | earliest transfer, any time | **earliest transfer before hour 48** |
| everything else (199 measurement variables + 19 T1 predictors) | unchanged | unchanged |

## Where each specification is used

- **F1 / Table 2 replication** — replication specification, discharge-coded Elixhauser
  included, explicitly labelled as reproducing Weissman. A replication that
  silently changes the feature set is not a replication.
- **F3 onward — the curve, ablation, outcome split, LSA, encoders, repeated
  splits** — observable specification, primary.
- **One sensitivity row** reporting the curve on the replication specification alongside, and a
  faint companion trace in the main figure. The near-identity of the two curves
  is evidence that the collapse is not a leak artifact; that belongs in the
  figure, not the supplement.

## Why

1. **The leak favours our own conclusion.** An unobservable predictor makes the
   structured baseline artificially strong, which shrinks the text increment,
   which inflates the collapse. Bias in the direction of one's own hypothesis is
   the kind to remove rather than quarantine.
2. **The downsides are asymmetric.** "Your primary baseline contains a
   100%-exposure unobservable feature" is not answerable in one sentence. "T1 is
   Weissman's 18 minus the one not knowable at the prediction time; the published
   version is in the supplement" is.
3. **The audit is on-thesis.** The paper argues this literature builds structured
   comparators carelessly. It should visibly model building one carefully. Under
   this decision the audit is a Methods strength; under the alternative it is a
   limitations paragraph.
4. **The substance is not expected to move.** `RESULTS_saturation.md` §7 already
   ran the curve without Elixhauser: 91% → 89%. The three context leaks affect
   0.05–6.3% of admissions and `n_icu_stays` is near-constant (mean 1.07 vs
   1.00). This is a change of specification, not of finding — which is precisely
   why it is cheap to make it the honest one.

## Stated in advance, so it can be checked afterwards

- The collapse is expected to stay in the 85–91% band. If it moves outside that,
  something is wrong with the rebuild and not with the earlier result.
- Removing Elixhauser **raises** the T1 text increment more than the T5 one
  (Elixhauser is worth +0.0154 at 19 structured variables and +0.0025 at 221), so
  the collapse should fall slightly rather than rise.
- The three context fixes should move T5 by ≲0.002 and T1–T4 not at all, since
  all four leaks live in T5.
- The encoder and LSA conclusions should be unchanged in direction. If the
  encoders move from "slightly negative at T5" to "slightly positive," that is a
  rewording of `RESULTS_bert.md` §6, not a reversal.

## Declared limitation

MIMIC-III observes only BIDMC 2001–2012, so 21.8% is a floor on true prior
history and `prior_elixhauser` does not restore the comorbidity construct — for
most of the cohort it is absent, and `has_prior_admission` largely encodes "has
been to this hospital before." **The observable baseline is therefore genuinely
thinner on comorbidity than a deployed model would be.** That biases against
this project's conclusion — thinner baseline, larger text increment, smaller
collapse — which is the safe direction, and is declared here rather than left
implicit.

Separately: `insurance`, `marital_status`, `language` and `religion` are treated
as known at admission. In some hospital systems these are updated during the
stay. Not checkable from the MIMIC-III schema; flagged, not resolved.

---

# Addendum, 2026-09-21 — checking the predictions

*Appended after the observable numbers existed. Nothing above this line has been
edited; a pre-registration that gets revised is not one.*

## The four stated expectations

| stated 2026-09-20 | outcome |
|---|---|
| removing Elixhauser raises the T1 increment more than the T5 one | **met** — GBM composite T1 +0.0105, T5 +0.0012 |
| the three context fixes move T5 by ≲0.002 and T1–T4 not at all | **met** — all four leaks are in T5; T5 moved +0.0012 |
| encoder and LSA conclusions unchanged in direction | **met**, with one sub-claim withdrawn (`RESULTS_observable_refit.md` §4a) |
| **the collapse stays in the 85–91% band** | **see below — the band held on its own estimator** |

## The band

The fourth line said:

> The collapse is expected to stay in the 85–91% band. If it moves outside that,
> something is wrong with the rebuild and not with the earlier result.

The headline observable collapse is **83.6%**, which reads as 1.4 points below
the floor. It is not, because the two figures are different estimators.

The band's inputs are named in §4 above — 91% from the main curve and 89% from
the no-Elixhauser sensitivity. Both are **single-split with GBM re-tuned per
tier** (the `run_saturation_W1.py` procedure). The 83.7% is a **ten-draw mean
with GBM held fixed** (the `run_repeated_splits_W1.py` procedure). Running the
observable curve through the first procedure closes the comparison:

| GBM collapse | replication | observable | spec effect |
|---|---|---|---|
| single split, re-tuned — *the band's own estimator* | 90.9% | **91.4%** | +0.5 pts |
| ten draws, fixed config — *the headline* | 85.4% | 83.7% | −1.7 pts |
| *estimator effect* | *−5.5 pts* | *−7.7 pts* | |

Both columns are the `notes-all` corpus, matching the band's own inputs — like for like. (On
the clean `notes-no-discharge` corpus the two observable cells are 90.1% and 83.6%; see F16 in
`FINDINGS.md`. The corpus moves this figure by about a point and changes nothing
here.)

The specification is worth ±0.5 to −1.7 points. The estimator is worth −5.5 to
−7.7. On the estimator this band was set with, the observable specification gives
**91.4%** — 0.4 points above the ceiling, which for a single split with no
interval is nothing.

**So the tripwire did not fire.** What it was for — "something is wrong with the
rebuild" — is not what happened, and the rebuild is sound.

## What this does put on the record

The prediction was **under-specified**: it named a range without naming the
procedure, written at a point in the project when both procedures were in use and
differ by nearly 8 points. That is a defect in the pre-registration, and the
lesson for the next one is that a predicted quantity has to name its estimator.

Two further items, kept because they are less flattering:

1. The reading taken when the number first came in — "the band was too tight" —
   was formed *after* seeing it, and is **withdrawn rather than vindicated.** It
   pointed the right way for the wrong reason. The resolution above is arithmetic
   over inputs named in this document before the fact; that one was a judgement
   call made afterwards, and the difference is the whole point.
2. The directional prediction — "the collapse should fall slightly rather than
   rise" — is only **weakly** met. It falls on ten draws (85.4% → 83.7%) and on
   LR at a single split (124% → 113%), but GBM at a single split ticks *up* 0.5
   points. All three are inside noise. The direction is not resolved by this
   data, in either direction.

## Consequence for how the collapse is quoted

Every quoted collapse figure must name its procedure. **The headline is a single
number, 83.6%** — ten draws, GBM fixed, observable specification, the `notes-no-discharge` corpus —
with the re-tuned single-split figure (90.1%) in a footnote. No range: the two
procedures differ by an analyst choice, not by sampling, and 83.6% is the
smallest of the four, so reporting it is the conservative call.

*Correction, same day.* A first version of this section attributed the gap
between the two procedures to the tuning choice, and quoted a 84–91% range. Both
were wrong. The comparison set *re-tuned + single split* against *fixed + ten
draws* — two changes labelled as one. Isolating them (observable spec, `notes-no-discharge`):

| held fixed | comparison | effect |
|---|---|---|
| same split, same corpus | re-tuned → fixed config | **−0.4 pts** (90.1% → 89.7%) |
| same config, same corpus | original split → mean of 10 draws | **−6.1 pts** (89.7% → 83.6%) |

Tuning is worth well under half a point; the gap is **partition**. Which
sharpens rather than weakens the reconciliation above — the band's inputs were
single-split figures, and the single split is exactly what turns out to carry
the variation.

Two further items land on the record from this:

1. The original 75/25 split gives an 89.7% collapse against a ten-draw range of
   **80.8–88.5%**. It is not a typical partition, and every single-split result
   in the project rests on it. Relative comparisons within it stand; collapse
   percentages quoted off it do not. See F3 in `FINDINGS.md`.
2. A separate corpus audit (F16) found that the two scripts producing the curve
   were still reading the `notes-all` corpus, in which 1,454 discharge summaries fall
   inside the 48h window. Both were re-run on `notes-no-discharge`. That was worth 0.1 points on
   the headline — but it means this document's rule ("knowable at hour 48")
   finally applies to the notes as well as the predictors, which it did not when
   the rule was written.
