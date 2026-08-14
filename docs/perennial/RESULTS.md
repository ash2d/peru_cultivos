# Perennial vs annual vs pasture/fallow — results

> Companion to [`plan.md`](plan.md) (the blueprint) and [`../PIPELINE.md`](../PIPELINE.md)
> (the 12-class pipeline this reuses). **Status: Phases 1–6 complete. Phase 5 (the panel) is
> extracted, assembled and inferred — 215,320 parcel-years. Phase 7 ran its validation gate
> and THE GATE FAILED (S4 and S5), so no trajectories, transitions or area estimates
> exist and none should be produced.** See §7. Everything below is measured, not projected.
>
> **For the narrative version — both strands, all the gates, with figures — read
> [`../SUMMARY_FULL.md`](../SUMMARY_FULL.md).** This document is the numbers-of-record and
> the audit trail; that one explains them.
>
> Last updated: 2026-08-07 (§8.3 added — the ≥1999 no-El-Niño retrain: S4 passes, S5 still
> fails; §4.1 rules control re-analysed + the `BSI_max` variant run and persisted; §8.2 added
> — the El Niño *mechanism* and the BSI feature-availability audit).
> Earlier: 2026-08-06 (§4.6 ablation + re-selection; panel assembled and inferred; §7
> Phase-7 gate FAILED; §8.1 El Niño confound test failed).

## 0. Headline

**The classifier works.** Locked-test **macro-F1 0.681** (accuracy 0.748) on 9,671 held-out
parcels — against a majority baseline of 0.257 macro-F1, a transparent phenology rule at
0.395, and MapBiomas Peru at 0.385. All four success criteria that can be evaluated without
the panel have passed:

| # | criterion | result |
|---|---|---|
| S1 | beat majority baseline by ≥ 0.15 macro-F1 | **PASS** (+0.393 on CV) |
| S2 | best ML beats the rule-based control | **PASS** (+0.257) |
| S3 | beat MapBiomas at predicting PETT labels | **PASS** (+0.296) — but see §5 for why |
| S4 | temporal transfer degrades gracefully | **FAIL** — 0.146 drop at k = −3; LTAE 0.116 (§7.0). **Passes (0.063) on a ≥1999 panel — §8.3** |
| S5 | trajectories are not noise (flicker < 15 %) | **FAIL** — 0.428 PERENNIAL; **LTAE 0.780** (§7.0). Still fails at 0.356 on ≥1999 — §8.3 |
| S6 | trend direction agrees with MapBiomas | **not evaluable as written** — see §5/§7.6 |

> ## ⛔ THE HEADLINE HAS CHANGED (2026-08-06)
>
> **The classifier works. The panel does not.** S1–S3 still pass and the locked-test 0.681
> still stands as a *single-year* result. But the panel validation gate — run for the first
> time on 2026-08-06 against all 28 assembled years — **fails S4 and S5**, and the §8.1 El
> Niño confound test fails as well. **No trend has been produced and none should be.**
>
> **The panel was re-inferred with a second architecture (LTAE, no statics at all) and the
> gate fails harder — flicker 0.780 vs 0.428 (§7.0.3). This is a data limitation, not a model
> choice.** "Try the other model" was the cheapest hypothesis and it is now closed.
>
> Three findings, one story:
> 1. **`frac_l7` (§4.6)** would have manufactured the trend from satellite availability
>    alone. **Fixed** — ablated, re-selected on `lightgbm_nometa`, exclusion enforced at
>    inference. Removing it cost 0.0013 macro-F1, i.e. nothing.
> 2. **The 1998 baseline is broken (§8.1, §7.0.3).** Trained on 1999+2000 and tested on 1998
>    with a proper control arm, **`PERENNIAL` recall falls to 0.016 (LightGBM) and 0.033
>    (LTAE)** against controls of 0.54 and 0.66 — effectively zero under *both* architectures.
>    1996–98 are the panel's baseline years, so the perennial baseline is near-zero for
>    artefactual reasons and any rise from it is uninterpretable.
> 3. **Per-parcel series flicker (§7.0).** 43 % of `PERENNIAL`-labelled parcels change class
>    more than once per five observed years against a 15 % criterion — 78 % for LTAE — and it
>    is not a coverage artefact.
>
> What survives: mission-boundary radiometry is clean (0 of 30 boundary steps exceed 2× the
> within-era year-to-year movement), forward transfer at k = +1…+3 is within 0.010, and
> `PERENNIAL` recall holds at 0.65–0.75 across the k range. The perennial *signal* is sound;
> reading annual *change* in it per parcel is what fails. §7.0.2 lists the options.
>
> **§7.0.2's option 2 has now been tested (§8.3, 2026-08-07) and it is not sufficient.**
> Retraining on ≥1999 only (`--train-years`, El Niño cohort excluded) and re-inferring a
> 1999–2023 panel **turns S4 into a PASS (0.063)** at *zero* CV cost on the years it still
> covers (pooled CV on ≥1999 validation parcels: 0.6412 vs 0.6415) — but **S5 still fails at
> 0.356**, and a control gate on the *baseline* model truncated to the same years shows most
> of the S4 gain is the shorter panel, not the retrain. **The gate still fails; no trend was
> produced.**

---

## 1. What was built

A **3-class land-state classifier** — `PERENNIAL` / `ANNUAL` / `PASTURE_FALLOW` — as a
proxy for the research question *"has land in these Piura parcels shifted from non-export
to export crops?"* (perennial ≈ export: mango, lime, coffee, banana, cacao, avocado;
annual ≈ domestic: rice, maize, cotton, beans, wheat).

Four models on identical spatially-blocked folds:

1. **`rules`** — an explicit depth-2 phenology rule on (NDVI level, NDVI amplitude), the
   *scientific control*. Registered as a normal model, so it goes through the identical
   CV protocol and is directly comparable. A second, better-featured variant
   (`rules_3c_bsi`) exists but is **not** the registered control — **§4.1**.
2. **LightGBM**, **LTAE**, **PSE-LTAE** — the three existing architectures, unchanged,
   retrained on the 3-class label space.

New code lives in `src/crop_classifier/perennial/`; everything else is reused through a
`CC_PROC` workspace switch (`paths.py`) so the completed 12-class work — whose locked test
set is still unspent — is untouched.

## 2. Workspace layout

| workspace | `CC_PROC` | contents |
|---|---|---|
| 12-class (existing, untouched) | *unset* | `data/processed/` |
| 3-class (primary) | `data/processed/perennial` | 56,419 parcels, 3 classes |
| 4-class diagnostic (D2) | `data/processed/perennial4` | same parcels, PASTURE/FALLOW split |

The **pixel feature store is shared** (`data/processed/features/`) — same parcels, same
years, same pixels; only the label column differs.

**The refactor was verified before anything else ran:** rebuilding the 12-class label table
with the new path resolution reproduces the existing table exactly — 49,648 parcels,
identical labels, geometry, `label_map.json` and exclusion counts.

## 3. Label build (Phase 1)

**56,419 parcels, 3 classes** (vs 49,648 at 12 classes — the rare-class and unmerged-
intercrop drops largely disappear at group level):

| class | parcels | share |
|---|---|---|
| `ANNUAL` | 36,102 | 64.0 % |
| `PASTURE_FALLOW` | 12,218 | 21.7 % |
| `PERENNIAL` | 8,099 | 14.4 % |

3,027 multi-crop parcels were resolved by group priority (`PERENNIAL` > `ANNUAL` >
`PASTURE_FALLOW`) rather than dropped. Exclusions: 9,390 on the area gate, 503 woody
non-crop (`ALGARROBO`, `GUAYAQUIL`, `COBERTURA ARBOREA`, …), 40 unmappable.

**Lexicon audit.** The top 88 tokens (≥15 records, 79,164 of 80,618 records) are assigned
by hand; the tail resolves through `category_default` or, for `crop`-category tokens, a
documented `crop_fallback: ANNUAL` guess. Tokens resting on that guess are **0.76 % of
records** — well inside the 2 % budget the build asserts. Full trail in
`class_lexicon_resolved.csv` and `unassigned_tokens.csv`.

*Deviation from the plan:* `PERENNIAL` came in at 14.4 %, below the plan's anticipated
20–25 % but well above its 10 % "the lexicon has a bug" threshold. It is consistent with
the 12-class table (CAFE + MANGO_LIMON + PLATANO = 11.4 % of parcels), so the plan's
expectation was simply optimistic, not the build wrong.

### 3.1 Gap-fill extraction

The 3-class map admits parcels the 12-class policy had dropped, and **6,113 of them had
never been extracted** — including **2,339 `PERENNIAL`**, our scarcest class. Without
filling that gap the model would have trained on a subset and the "more parcels" advantage
would have evaporated silently. `perennial/gapfill.py` extracts them into the shared store
(with statics taken from the *union* of both workspaces' parcel tables, so the 12-class
runs' feature rows are not blanked).

### 3.2 Spatial splits — leakage is worse here, not better

As the plan anticipated, three broad classes are *more* spatially autocorrelated than
twelve crops:

| distance | 3-class agreement | (12-class) |
|---|---|---|
| < 100 m | **0.845** | 0.75 |
| 4–5 km | **0.610** | 0.37 |
| random baseline | 0.477 | 0.24 |

Agreement never falls to baseline within the 5 km audit range, so `buffer_m = 1500` controls
only part of the leakage. Locked test = 10,161 parcels in 29 contiguous 5 km regions
(18.0 %). A `buffer_m = 3000` variant config (`config/split_b3000.yaml`) exists to bound
how optimistic CV is.

## 4. Model comparison (Phases 2–4)

Pooled spatial CV over 44,022 validation parcels (majority baseline macro-F1 **0.257**,
accuracy 0.627). Figures: `runs/perennial/comparison/*.png`; table:
`runs/perennial/comparison/model_comparison.csv`.

| model | pooled macro-F1 | bal. acc | acc | κ | CV macro-F1 |
|---|---|---|---|---|---|
| LTAE | **0.652** | 0.697 | 0.691 | 0.472 | 0.652 ± 0.038 |
| PSE-LTAE | 0.650 | 0.695 | 0.681 | 0.460 | 0.649 ± 0.028 |
| LightGBM | 0.650 | 0.666 | **0.701** | 0.455 | 0.647 ± 0.035 |
| LightGBM (swept) | 0.648 | 0.664 | 0.698 | 0.451 | 0.644 ± 0.034 |
| **rules** (control) | 0.395 | 0.467 | 0.367 | 0.106 | 0.388 ± 0.055 |

Per-class F1 (pooled CV):

| class | LightGBM | LTAE | PSE-LTAE | rules |
|---|---|---|---|---|
| `ANNUAL` | **0.784** | 0.769 | 0.756 | 0.327 |
| `PERENNIAL` | **0.683** | 0.656 | 0.673 | 0.493 |
| `PASTURE_FALLOW` | 0.482 | **0.532** | 0.521 | 0.367 |

* **S1 (beat the majority baseline by ≥ 0.15): PASSED** — +0.393.
* **S2 (best ML beats the rule): PASSED** — +0.257.
* The plan's "if 3-class macro-F1 is not comfortably above 0.65, something is wrong" bar is
  met, and it is a large jump over the 12-class state of the art (0.427).
* **The three ML models are within 0.003 of each other.** The attention advantage seen at
  12 classes came from rescuing rare classes; with three balanced-ish classes there is
  nothing to rescue — exactly as the plan predicted.
* **Hyper-parameter tuning did not help.** A 30-trial Optuna sweep landed at CV 0.644,
  *below* the 0.647 default, so the defaults were kept. LTAE was subsequently swept too
  (2026-08-05) — same conclusion, and the tuned model was then run against the locked test
  as an explicitly exploratory second use. See **§4.5**.
* **`PASTURE_FALLOW` is the weak class everywhere** (0.48–0.53) — the D2 concern that
  merging two spectrally *opposite* states creates a heterogeneous class.
* **The `rules` row understates a fair control.** Its accuracy (0.367) is *below* the 0.627
  majority baseline because `PASTURE_FALLOW` is its fall-through branch, and a one-feature
  swap lifts it to pooled 0.451 / accuracy 0.519. That variant is **not** adopted — see
  **§4.1**, which also explains why S2's "+0.257" is better read as ~+0.20.

### 4.1 The rule-based control is genuinely worse — and that is informative

*(re-analysed 2026-08-07; the `BSI_max` variant is now a persisted run,
`runs/perennial/rules_3c_bsi/`)*

**What it is.** An explicit depth-2 rule (`perennial/rules.py`), registered as a normal
model so it goes through the identical spatial-CV protocol:

```
if   NDVI_p25 >= t_hi  and NDVI_amp <= t_amp:   PERENNIAL
elif NDVI_amp > t_amp  or  NDVI_max >= t_peak:  ANNUAL
else:                                            PASTURE_FALLOW
```

`p25` not `min`, because the minimum is one cloud-edge pixel from garbage. **Structure and
features were fixed a priori from agronomy and never searched** — only the three thresholds
are fitted, by exhaustive grid over 25 quantiles of each feature's *training-fold*
distribution (~15.6k combinations), maximising macro-F1. `val_ds` is accepted for interface
compatibility and deliberately never read.

**Result: pooled macro-F1 0.395** (CV 0.388 ± 0.055) — clearly above the 0.257 majority
baseline, 0.26 below the ML models. Its thresholds are **stable across all five spatial
folds** (level 0.548–0.568, amplitude 0.631–0.663, peak 0.499–0.529), which is itself a
result: the rule generalises spatially, it is simply not expressive enough.

**But its failure mode is structural, and the headline number hides it.** Accuracy is
**0.367 — far *below* the 0.627 majority baseline** — because it predicts `PASTURE_FALLOW`
for **69.3 % of parcels against a true share of 22.0 %**, dropping `ANNUAL` F1 to 0.327.
`PASTURE_FALLOW` is the rule's *fall-through* branch, so every parcel failing both explicit
tests lands there, and a macro-F1 objective is content to trade away the majority class to
lift a small one. Read the 0.395 as "a rule with a broken third branch", not "a phenology
rule's ceiling".

#### The `BSI_max` swap — the control was under-specified

`rules.py`'s own docstring names "low BSI maximum" as part of the perennial physics, but
the registered rule reads **only its three configured columns** — so `BSI_max` sits unused
in the same flat store the rule is already reading. That is a configuration choice, not a
data limitation. Swapping the peak slot `NDVI_max` → `BSI_max` (same directional semantics:
higher = more bare soil = annual), everything else identical:

| variant | CV macro-F1 | pooled | acc | **bal. acc** | κ |
|---|---|---|---|---|---|
| `NDVI_p25, NDVI_amp, NDVI_max` (registered) | 0.3885 ± 0.0551 | 0.3955 | 0.3669 | **0.4675** | 0.106 |
| `NDVI_p25, NDVI_amp, BSI_max` | **0.4366 ± 0.0655** | **0.4507** | **0.5188** | 0.4550 | **0.149** |

Per-fold Δ: **+0.108, +0.001, +0.065, +0.048, +0.019** — 5/5 folds, mean **+0.048** (though
fold 1 is a tie at +0.0008). The pathological class mix is also repaired: predicted shares
go 0.196 / 0.693 / 0.111 → **0.549 / 0.315 / 0.136** against true 0.627 / 0.220 / 0.153.

**Three reasons it has NOT been adopted as the registered control:**

1. **It costs the class the research question depends on.** The gain is entirely in `ANNUAL`
   (F1 0.327 → 0.635); **`PERENNIAL` F1 *falls* 0.493 → 0.379** and `PASTURE_FALLOW`
   0.367 → 0.338. On the headline metric it is better; on perennial detection it is worse.
2. **Balanced accuracy falls** (0.4675 → 0.4550) even as raw accuracy rises 0.367 → 0.519 —
   it is trading rare-class recall for majority-class correctness, which is exactly what
   macro-F1 was chosen to prevent.
3. **It gives up the control's one clean win — threshold stability.** The level threshold
   spreads **0.426–0.590** across folds, a **0.164-wide** band against the registered rule's
   **0.020** (0.548–0.568); the amplitude band is 0.059 wide against 0.032. A control whose
   thresholds wander is worth less as a transparent, defensible rule for years with no
   ground truth, which is the whole reason it exists.
   *(The amplitude threshold also relocates entirely — 0.631–0.663 → 0.267–0.326 — because
   once the peak branch actually discriminates, amplitude no longer has to carry the whole
   `ANNUAL` test on its own. That part is a re-balancing, not instability.)*

And it is **post-hoc tuning of a control** — the ML models each got a 30-trial sweep, the
control got hand-picked features, so the comparison was already unfair in the models'
favour; fixing that asymmetry has to be declared, not slipped in. Both runs are kept
(`rules_3c`, `rules_3c_bsi`) so either can be reported.

The larger available improvement is not a feature swap at all: the per-class seasonal
profiles (`docs/figures/profiles_3class.png`) show the separation lives in the **Aug–Dec dry
season**, and a whole-year `NDVI_p25` averages that signal away. A dry-season-window feature
would likely beat all of the above — but it is not in the feature store, so it needs an
assembly change.

**The conclusion is unchanged either way: a transparent phenology rule does not get you most
of the way here** — but the honest gap to the ML models is ~0.20, not the 0.26 the headline
table implies.

### 4.2 The D2 diagnostic: merging PASTURE and FALLOW was right

The 4-class variant (`PERENNIAL` / `ANNUAL` / `PASTURE` / `FALLOW`) reached CV macro-F1
0.581. Collapsing its probabilities post-hoc to three classes gives macro-F1 **0.643** vs
**0.650** for the directly-trained 3-class model (accuracy 0.691 vs 0.701). Splitting the
class does not help: separately, `PASTURE` only reaches F1 0.404 and `FALLOW` 0.484, both
below the merged class. **Keep the merge.**

### 4.3 Binary variant: folding PASTURE_FALLOW into ANNUAL does **not** help

`PASTURE_FALLOW` is the weakest class (test F1 0.44) and 21 % of parcels, so the obvious
question is whether it earns its keep — the research question is really *perennial vs not*.
A 2-class workspace (`data/processed/perennial_bin`, config `perennial_binary.yaml`) was
built and LightGBM retrained on it.

Its headline CV macro-F1 is **0.812 ± 0.019** vs 0.647 for 3 classes — but that comparison
is meaningless, because macro-F1 over two easy classes is not macro-F1 over three. The
correct test is the same one used for the 4-class variant: **collapse the 3-class model's
predictions post-hoc and score both on identical parcels** (44,022; PERENNIAL prevalence
15.3 %):

| setup | PERENNIAL precision | recall | **PERENNIAL F1** | macro-F1 | accuracy | **ROC AUC** |
|---|---|---|---|---|---|---|
| binary (directly trained) | 0.652 | 0.718 | **0.6835** | 0.8115 | 0.8983 | **0.9368** |
| 3-class, collapsed post-hoc | 0.624 | 0.755 | **0.6830** | 0.8093 | 0.8929 | **0.9368** |

**Identical ROC AUC to four decimal places, and a PERENNIAL F1 difference of 0.0005.** The
3-class model already contains all the perennial-detection skill the binary model has; the
only difference is a marginal precision/recall trade (binary is slightly more precise, the
3-class slightly more sensitive). Training binary buys nothing and throws away the
pasture/fallow decomposition.

**Decision: keep the 3-class model.** Report the binary collapse as a *derived view* of it
when the analysis wants perennial-vs-rest.

The genuinely useful number here is **ROC AUC 0.937**: the model separates perennial from
non-perennial very well, and the F1 of 0.68 reflects the 15.3 % prevalence and a fixed
argmax threshold, not poor discrimination. That is an argument for leaning on the
probability-weighted and bias-corrected area estimators rather than raw argmax counts in
the trend — and for tuning the operating threshold if perennial detection is the goal.

### 4.4 Model selection and the locked test set

Selected: **LightGBM (default parameters)** — recorded in
`runs/perennial/selected_model.json`. LTAE leads by 0.0027, far inside the plan's ~0.02
tie-break threshold, so the documented tie-break toward the cheaper model applies: LightGBM
is ~100× cheaper for the 220k-parcel-year panel inference, and it also has the best accuracy
and the best `PERENNIAL` F1. LTAE is better only on `PASTURE_FALLOW`.

**Temperature scaling** (D8) on held-out fold-0 validation probabilities: **T = 1.440**,
NLL 0.6757 → 0.6455, **ECE 0.0934 → 0.0372**. Argmax is preserved (asserted), so accuracy
and macro-F1 are untouched — only the confidences move. The model was indeed overconfident,
as the 12-class work suspected.

**The locked test set was then spent exactly once** (`runs/perennial/lightgbm_3c_final`):

| metric | locked test (n = 9,671) |
|---|---|
| **macro-F1** | **0.681** |
| weighted F1 | 0.744 |
| balanced accuracy | 0.681 |
| accuracy | 0.748 (majority baseline 0.671) |
| Cohen's κ | 0.478 |
| Brier (top class) | 0.178 |

Per class: `ANNUAL` F1 0.833 (n=6,494), `PERENNIAL` 0.769 (n=1,157), `PASTURE_FALLOW`
0.440 (n=2,020).

**The test score (0.681) is *higher* than CV (0.647).** That is the reassuring direction:
the accepted residual spatial leakage did not inflate the CV estimate, so the plan's
`buffer_m = 3000` sensitivity run is no longer needed to defend the headline number (the
config remains available for anyone who wants it).

The test confusion matrix is saved as
`runs/perennial/lightgbm_3c_final/test_confusion.csv` and is the error matrix for the
Olofsson area correction (D8):

| | ref ANNUAL | ref PASTURE_FALLOW | ref PERENNIAL |
|---|---|---|---|
| **map ANNUAL** | 5,486 | 1,011 | 174 |
| **map PASTURE_FALLOW** | 886 | 842 | 80 |
| **map PERENNIAL** | 122 | 167 | 903 |

### 4.5 LTAE sweep, and a **second, exploratory** use of the locked test (2026-08-05)

The plan called for sweeping all three ML models; only LightGBM had been swept, so LTAE was
swept to close that gap (`runs/perennial/sweep_ltae_20260805_141641`, 30 trials, 3 folds).

**The sweep does not overturn the LightGBM selection.** Trial distribution:

| | value |
|---|---|
| mean / median | 0.6393 / 0.6403 |
| std | 0.0057 |
| min / max | 0.6279 / **0.6501** |
| LightGBM on the *same 3 folds* | 0.6422 |
| LTAE trials beating it | **9 of 30** |

LightGBM sits near the 70th percentile of the LTAE distribution. The best-of-30 (0.6501)
is what selection noise alone predicts — for 30 draws at σ = 0.0057 the expected maximum is
≈ mean + 2σ ≈ 0.650. The two top trials (3 and 29) differ by 0.0004 at different dropouts,
which says the same thing.

**Then, at the user's explicit request and purely for interest, the tuned LTAE was refit on
all 5 folds and run against the locked test** (`runs/perennial/ltae_3c_tuned_test`,
`d_model=256, dropout=0.315, lr=2.36e-4`). ⚠️ **This is the SECOND use of the locked test.**
It was not used to select anything — LightGBM remains the selected model on the CV evidence
above — but the test set is no longer a clean held-out estimate for *any* future selection.
Recorded here so the provenance is not lost.

| | LightGBM (default) | LTAE (tuned) |
|---|---|---|
| per-fold CV | 0.654 / 0.676 / 0.597 / 0.681 / 0.625 | 0.656 / 0.683 / 0.612 / 0.690 / 0.648 |
| **CV mean** | 0.6466 ± 0.0353 | **0.6578 ± 0.0310** |
| **locked test** | **0.6807** | 0.6610 |
| test − CV | **+0.0341** | +0.0032 |
| accuracy / κ | 0.748 / 0.478 | 0.720 / 0.442 |

**The ranking reverses between CV and test.** Tuned LTAE beats LightGBM on *all five* folds
— including folds 3 and 4, which the 3-fold sweep never saw, so the CV advantage is real and
not a selection artefact. Yet it loses the locked test by 0.020, uniformly across all three
classes (`ANNUAL` 0.813 vs 0.833, `PASTURE_FALLOW` 0.420 vs 0.440, `PERENNIAL` 0.751 vs
0.769) — not a class-specific failure.

The informative quantity is **test − CV**: +0.034 for LightGBM against +0.003 for LTAE. The
higher-capacity attention model fits the trainval regions better and transfers to unseen
regions worse. Spatial CV did **not** catch this; only the held-out regions did. That is a
direct argument for keeping a locked test — and a caution that CV rank alone would have
picked the model that generalises less well.

**Net effect: none on the pipeline.** LightGBM stays selected, on stronger evidence than the
original cheap-model tie-break. No temperature was fitted for the LTAE run (argmax-preserving,
so it cannot change any figure above).

### 4.6 ⚠️ Feature audit — `frac_l7` manufactures the headline trend (2026-08-06)

**This must be resolved before any panel inference.** Found while writing the Phase-7 handoff.

`data.py:80-81` drops only `COD_PREDIO` and `label_id`, so **every other column in
`features_lightgbm.parquet` is a model input** — including acquisition metadata that
describes *which satellite was overhead*, not the land.

`frac_l7` (fraction of a parcel-year's observations from L7 rather than L5) is confounded
with the label in training, because mission availability and titling year move together:

| label year | frac_l7 | perennial rate |
|---|---|---|
| 1998 | 0.000 | **0.4 %** |
| 1999 | 0.397 | 13.9 % |
| 2000 | 0.469 | 33.9 % |
| 2002+ | 1.000 | 3–62 % |

corr(`frac_l7`, is_perennial) = **0.33**, and the model took it: **`frac_l7` is the 2nd most
important feature by gain (6.56 % of 140)**.

**The panel trajectory is a monotonic ramp:** 1996–98 = **0.000**, 1999 = 0.428,
2000 = 0.511, 2001 = 0.800, **2002–2023 = 1.000**.

So at inference the model sees `frac_l7 = 0` for every parcel across the entire 1996–98
**baseline** and `frac_l7 = 1` from 2002 on — suppressing perennial early and inflating it
later, **producing a step change caused by satellite availability alone**. It manufactures
the annual→perennial shift the research is looking for, in the right direction, at roughly
the right time. A trend from this model is uninterpretable.

Four more acquisition features share the character (vary over the panel for non-land-use
reasons): `n_valid_obs` 2.03 %, `n_dates` 0.89 %, `max_gap` 0.85 %, `n_valid_pixels` 0.56 %
— **10.89 % of gain in total**.

Separately, **`centroid_lat` is the single largest feature at 20.12 %**. It is
*time-invariant per parcel*, so it cannot manufacture a trend, but it is heavy spatial
memorisation and a candidate for the same ablation. Genuine spectral features hold ~69 %.

#### The attention models are structurally immune

`build_tensors` feeds LTAE/PSE-LTAE **only** `x = [N, T, 11]` (6 raw bands + NDVI/EVI/NDWI/
NDMI/BSI), `doy`, and `mask`. **No statics are passed at all** — no `frac_l7`, no
`centroid_lat`, no `n_valid_obs`. LTAE cannot use the confound because it never sees it.

This reframes the §4.5 selection. LightGBM won the locked test by 0.020, but its failure
mode (**manufacturing the deliverable**) is fatal, while LTAE's (~2 points of macro-F1) is
benign. **Accuracy is the wrong sole criterion for a model whose output is a time trend.**

#### Residual sensor exposure that ablation does *not* remove (both models)

* **Raw bands are more sensor-sensitive than indices.** L5 TM and L7 ETM+ have different
  spectral response functions. Ratio indices (NDVI etc.) partially cancel calibration
  differences; raw B/G/R/NIR/SWIR1/SWIR2 summaries do not — and both models consume them.
* **Acquisition density is implicit in the mask.** Effective sequence length varies with how
  many clear dates a year had, which varies systematically across the panel.
* **`doy` distribution shifts with mission.** L5 and L7 fly the same 16-day cycle 8 days
  apart, so overpass timing — and hence the sampled `doy` pattern — differs by era.

No *label* leakage was found: `label_id` is dropped and no label-derived column reaches
either model. The issue is confounding with time, not with the target.

#### The ablation and re-selection — **DONE (2026-08-06)**

Gain ≠ contribution, and there was precedent: the 2026-08-03 12-class mission-mixing audit
found metadata at **29.8 % of gain worth only 0.007 macro-F1** when ablated. That is exactly
what happened again.

`data.py` now takes a `drop_features` spec (column names, or the aliases `meta` =
the five acquisition columns and `location` = `centroid_lat`), exposed as
`train --drop-features`. Two variants were trained on the identical 5 spatial folds. **The
locked test was not touched** — it has been spent twice already, so it is not a valid basis
for this or any future selection.

| model | CV macro-F1 | pooled CV | acc | `PERENNIAL` F1 | trend-safe |
|---|---|---|---|---|---|
| `ltae_3c_tuned_test` | **0.6578 ± 0.031** | 0.6577 | 0.693 | 0.678 | ✔ (no statics at all) |
| **`lightgbm_nometa`** ⭐ | 0.6479 ± 0.036 | 0.6503 | 0.699 | **0.685** | ✔ |
| `lightgbm_3c_final` | 0.6466 ± 0.035 | 0.6496 | 0.701 | 0.683 | ✘ **disqualified** |
| `lightgbm_nometa_nolat` | 0.6418 ± 0.039 | 0.6429 | 0.698 | 0.663 | ✔ |

**Dropping the acquisition metadata is free.** `lightgbm_nometa` − `lightgbm_3c_final` =
**+0.0013** mean macro-F1 across the five paired folds (3/5 folds won, paired *t* p = 0.85).
The 10.89 % of gain those five columns carried was not contribution. Note what this does
*not* say: it is not evidence the model was ignoring them. It used them heavily; they simply
duplicated information the spectral features already carried, **and at inference they would
still have driven the trend**. Removal is mandatory on the trend argument alone.

Dropping `centroid_lat` as well costs 0.0061 CV and 0.021 `PERENNIAL` F1 and buys nothing a
trend needs — it is time-invariant per parcel, so it contributes a constant offset and
cannot manufacture a slope. Not selected.

> **⚠️ 2026-08-08 — the national data reverses this decision, and shows why Piura could not
> have made it.** With 14 departments, leave-one-department-out finds `centroid_lat` worth
> **+0.047 macro-F1 on spatial CV and −0.060 on unseen departments**; **12 of 14 departments
> improve without it**, Piura most of all (+0.178). Spatial CV holds out 5 km cells *inside
> departments the model has already seen*, so it cannot separate spatial memorisation from
> signal — the experiment that settles it needs more than one department. The decision above
> was not wrong on the evidence available here; it was **unfalsifiable** here. See
> [`../all_peru/RESULTS.md`](../all_peru/RESULTS.md) §6.2. (Its share of gain rose 20.1 % → 22.9 % in
`lightgbm_nometa`, absorbing the metadata's old role.)

**Selected for the panel: `lightgbm_nometa`** (`runs/perennial/selected_model_panel.json`,
which supersedes `selected_model.json` *for panel inference only*). LTAE leads by 0.0099 —
inside the plan §6.3 tie-break threshold of ~0.02, and it now wins only **3 of 5 folds**
against 5 of 5 versus the un-ablated LightGBM, so de-metadata-ing LightGBM removed most of
LTAE's paired advantage. No pair among the three admissible models is statistically
distinguishable on five folds (all p > 0.3). The cheap-model tie-break therefore applies
again: LightGBM is ~100× cheaper over 215k parcel-years, and §4.5 measured `test − CV` at
+0.034 for LightGBM against +0.003 for LTAE — it transferred to unseen regions *better*
than CV implied, not worse. Temperature refitted: **T = 1.409**, ECE 0.0897 → 0.0308.

**The exclusion is enforced at inference, not just at training.** The panel feature store
still contains all 141 columns (`frac_l7` is 0.0 in 1996, exactly as §4.6 describes), so a
model that merely *was* trained without them would silently get them back. `infer()` now
pins the column list to the model's own saved `feature_names`, and raises if the store
cannot supply them. Verified against the assembled 1996 panel bundle: 135 model features,
`frac_l7` present in the store and absent from the model. Regression:
`tests/test_pipeline.py::TestFeatureExclusion`.

**Recommended sensitivity if the Phase-7 gate passes:** re-run the headline trend under
`ltae_3c_tuned_test` too. The two models share no statics and differ in architecture, so a
trend that holds under both is materially stronger than one that does not.

## 5. MapBiomas Peru benchmark (Phase 6) — the decisive negative finding

**MapBiomas Peru Collection 3 cannot separate perennial from annual cropland in Piura.**

The legend was verified two independent ways rather than assumed:

1. the published MapBiomas Collection-3 legend-code table (shared RAISG/Andean scheme), and
2. **empirically**, by cross-tabulating codes inside parcels whose PETT crop we already
   know — code 40 covers 54 % of `ARROZ` parcels' pixels but ~1 % of orchards', confirming
   40 = Rice.

Codes actually present over the Piura parcel bbox, 1990–2024:

```
3, 4, 5, 9, 11, 12, 13, 21, 23, 24, 25, 27, 29, 30, 32, 33, 40, 66, 68, 72
```

The only agricultural codes are **21 (mosaic of uses), 40 (rice), 72 (other crops)**.
**Codes 36 / 46 / 47 / 48 (perennial crop, coffee, citrus, other perennial) and 15
(pasture) never appear anywhere in Piura.** Inside our parcels MapBiomas is ~63 % class 21,
with mango/lime orchards (84 %), coffee (45 %) and fallow (58 %) all landing in that same
class; shade-coffee reads as forest (27 % class 3) and the dry-forest class 4 covers half of
our pasture parcels.

### 5.1 Criterion S3 — the head-to-head

Both scored against PETT labels on the **same locked-test parcels**, at the label year:

| predictor | macro-F1 | accuracy | n | coverage |
|---|---|---|---|---|
| **our model** | **0.681** | 0.748 | 9,671 | 100 % |
| MapBiomas, mosaic → `PASTURE_FALLOW` | 0.385 | 0.580 | 9,175 | 90 % |
| MapBiomas, mosaic → `PERENNIAL` | 0.325 | 0.495 | 9,175 | 90 % |
| MapBiomas, mosaic excluded | 0.315 | 0.897 | 4,221 | 42 % |
| MapBiomas, mosaic → `ANNUAL` | 0.275 | 0.703 | 9,175 | 90 % |

**S3: PASSED by +0.296** over the best MapBiomas configuration. (The "mosaic excluded"
row's high *accuracy* with low macro-F1 is the giveaway: excluding class 21 leaves mostly
rice parcels, so it predicts `ANNUAL` well and nothing else, on 42 % of parcels.)

Stratified by parcel area — and this **contradicts the plan's expectation** that MapBiomas
would look much better on parcels ≥ 3 ha:

| area (ha) | n | ours | MapBiomas |
|---|---|---|---|
| 0–0.5 | 5,768 | 0.687 | 0.387 |
| 0.5–1 | 2,219 | 0.695 | 0.394 |
| 1–3 | 1,438 | 0.661 | 0.369 |
| 3–50 | 246 | 0.583 | **0.289** |

MapBiomas gets *worse* on large parcels, not better. Mixed pixels were never its binding
constraint here — the missing perennial class is, and parcel size cannot fix that. (Our own
model also dips on the 246 largest parcels, which are few and more internally
heterogeneous.)

**Consequence for criterion S3.** The result is settled in the custom model's favour for a
structural reason, not because we won a fair contest: the benchmark has no perennial class
to compare against. That is the honest framing and it should not be presented as a
performance victory. The useful conclusion is the positive one — **a parcel-level model is
necessary here precisely because the off-the-shelf product does not resolve the distinction
the research question depends on.**

**Independence caveat (restate wherever these numbers appear):** MapBiomas Peru is itself
Landsat-derived at 30 m, so it shares sensors, cloud regimes and mixed-pixel problems with
our model. It is a benchmark, not ground truth, and agreement is not evidence of
correctness.

## 6. Multi-year panel (Phase 5) — *in progress*

The panel machinery is built and unit-tested, and the three landmines the plan identified
in the existing extraction code are fixed:

1. **`assemble` merged years silently.** `per_date_medians` groups on `(COD_PREDIO, doy)`,
   which is only sound while a parcel has one year of pixels. `assemble()` now takes a
   `years` filter and a per-year `out_dir`, and `assert_one_year_per_parcel` raises rather
   than producing plausible-looking, wrong features. Regression test:
   `tests/test_assemble_years.py` (a synthetic two-year parcel must yield year-isolated
   tensors identical to the single-year build).
2. **`run_coverage` deduped on `COD_PREDIO`** — now called once per year with an explicit
   `out=`.
3. **`run_pixels` hard-coded the shared feature dir** — now takes `feat_dir`; the panel
   writes to `features/panel/`, keeping the audited training store immutable.

Also applied: **L9 added** to `missions_for_year` (2021+, without it 2022–24 would run on
L8 + SLC-off L7 alone) and **OLI→ETM+ harmonisation** (Roy et al. 2016 coefficients) at
assembly time, so the raw store stays raw and the correction stays revisable.

### 6.1 Two further bugs the timing probe exposed — both fixed and regression-tested

The probe was worth running for the bugs alone, never mind the budget:

4. **Cross-year coverage contamination.** `run_coverage` globbed *every* `cov_*.parquet` in
   its chunk directory and deduped on `COD_PREDIO` alone, so two years whose `out=` files
   share a parent directory returned the **first** year's numbers for every parcel. The
   plan's fix ("call it once per year with an explicit `out=`") is **not sufficient**,
   because the chunk directory is derived from `out.parent` — and `extract_panel` put every
   year in `FEAT/panel/`, so the real 27-hour run would have produced 29 identical years.
   It surfaced as 1995, 1996 and 2005 all reporting an identical 4.7 % gate pass.
   Now the combine filters by requested year and dedupes on `(COD_PREDIO, year)`.
   Regression: `tests/test_coverage_years.py`.
5. **Empty-year crash.** A year with no acquisitions produced a band-less count image and
   `unmask` raised *"If one image has no bands…"* — real for early-1990s Piura. Fixed by
   merging the zero base image first, so an empty year counts 0 instead of crashing.

### 6.2 Mission policy: TM/ETM+ only, no OLI (user decision, 2026-08-04)

The training data is 52.8 % L5 (TM) + 47.2 % L7 (ETM+) and **530 OLI observations out of
5.65 M** — the classifier has effectively never seen L8/L9. Under the original 1996–2024
plan, everything from 2013 on would have been inferred on unseen radiometry, with the sensor
step at 2013 sitting exactly where an export-crop expansion would appear, resting on an
unvalidated correction. **The panel is now restricted to `{L5, L7}`**, so every year is
inferred on radiometry the model was trained on and the Roy et al. harmonisation is switched
off (still implemented, re-enable by clearing `panel.PANEL_MISSIONS`).

Measured L7 coverage (300 parcels, gate = ≥4 clear acquisitions) — full table in
[`l7_coverage.md`](l7_coverage.md):

* **usable 2000–2023**, typically 0.86–1.00 gate pass;
* **2024 and 2025 are 0.000** (mean 0.07 and 0.00 clear obs/parcel) — L7 acquisitions stop,
  so **without OLI there is no 2024+**;
* adding L5 back **does not rescue** the weak years: 2009 (0.470) and 2011 (0.373) are
  identical with or without it — Landsat 5 contributed essentially nothing over Piura in its
  degraded final years before the November 2011 decommissioning;
* **SLC-off costs less than feared**: median pixels per parcel-date 6.0 (2001, SLC-on) →
  5.0 (2015, SLC-off), single-pixel parcel-dates 7.9 % → 10.9 %, and *no* parcel is
  systematically lost. Post-2003 per-date medians are noisier, not missing.

Resulting composition: **1996–1998** L5 only · **1999–2011** L5 + L7 · **2012–2023** L7
only. Panel range **1996–2023 (28 years)**.

**Years to flag in every figure rather than interpolate over:** 1997 (48.3 %, El Niño),
2009 (47.0 %), 2011 (37.3 %), 2012 (80.3 %, SLC-off L7 alone).

The trade is explicit: one year of range and a slightly thinner post-2003 pixel count, in
exchange for eliminating the single largest threat to the headline trend.

### 6.3 The panel starts in 1996, not 1990

Coverage probe on one identical 300-parcel sample (`docs/perennial/panel_budget.md`):

| year | gate pass | median clear obs |
|---|---|---|
| 1990 | 3.7 % | 2 |
| **1992** | **0.0 %** | **0** |
| 1995 | 4.7 % | 2 |
| **1996** | **100 %** | **8** |
| 1998 | 95.0 % | 6 |
| 2015 | 100 % | 13 |

There is a **hard Landsat-archive boundary at 1996** over Piura (the well-known pre-1999
international-ground-station gap in the L5 archive). Extracting 1990–95 would buy nothing
and would plant a spurious rise in *every* class at the head of the headline figure.
`DEFAULT_YEARS` is now **1996–2024** and the pre-1996 gap is stated, not silently cropped.

Cost also rises 11× across the record (0.069 s/parcel-year in 1995 → 0.774 in 2023), so the
1998-derived 0.19 s/parcel-year rate must not be extrapolated — the plan was right to insist
on measuring. Panel sized from the measured rate: **7,690 parcels × 29 years ≈ 27 h**
(2,999 forced locked-test parcels for §7.3 + 4,691 stratified across all 256 regions).

### 6.3 Status — **EXTRACTION COMPLETE** (2026-08-05 19:37)

**All 28 years extracted: 16,259,158 pixel-obs, 707 MB, 5,431 chunks, 0 corrupt.** Wall
clock ~8.7 h against a projected 27 h. The panel is ~2.9× the entire 5.65 M single-year
training store.

Parcels reaching the pixel stage per year track the gate exactly — the three thin years are
**1997 (3,686), 2009 (3,758), 2011 (3,294)**; healthy years run 6,400–7,575 of 7,690. 2023
yields the fewest pixel-obs of any healthy year (356,872) because L7 acquisitions thin out
at the end of the record — consistent with 2024 being empty and bounding the panel.

**Measured gate pass, against the L7-only predictions in
[`l7_coverage.md`](l7_coverage.md):**

| year | predicted | **measured** | | year | predicted | **measured** |
|---|---|---|---|---|---|---|
| 1996 | — | 99.9 % | | 2009 | 47.0 % | **49.2 %** |
| 1997 | — | **48.3 %** | | 2010 | 97.0 % | 96.4 % |
| 1998 | — | 92.4 % | | 2011 | 37.3 % | **43.3 %** |
| 1999 | 50.0 % | 99.7 % | | 2012 | 80.3 % | **83.6 %** |
| 2000 | 97.0 % | 100.0 % | | 2013 | 97.3 % | 98.7 % |
| 2006 | 100.0 % | 99.9 % | | 2014 | 99.0 % | 98.8 % |
| 2008 | 87.0 % | 84.8 % | | 2015 | 88.3 % | 87.0 % |

Every prediction held. All **four flagged years are now measured** — 1997 (48.3 %, El Niño),
2009 (49.2 %), 2011 (43.3 %), 2012 (83.6 %) — and all must be flagged in figures, never
interpolated over. 1999/2000 come in far above the L7-only prediction because L5 is admitted
in the panel and contributes heavily in those years.

Note 2009 and 2011 sit either side of 2010, so a trajectory through that stretch rests on a
single observed year — a place where the ≥3-consecutive-year rule will legitimately refuse
to call a transition.

**⚠️ Operational note — silent GEE hangs are RECURRING, not a one-off.** Three in one run:

| # | when | stalled at | cost |
|---|---|---|---|
| 1 | 2026-08-04 | mid-2006 pixels | **13.4 h** (unsupervised) |
| 2 | 2026-08-05 18:08 | mid-2022 pixels | ~31 min |
| 3 | 2026-08-05 18:46 | entering 2023 | ~32 min |

Every one: process alive, zero output, no exception, no retry. `landsat_gee.py` sets
`socket.setdefaulttimeout(120)` and has a 5-try backoff; **neither ever fired**, because the
GEE client blocks somewhere the socket timeout does not reach. **`_retry` only handles errors
that *raise* — a hang is not an error.** They hit at unrelated points in unrelated years, so
this is not a bad chunk or a bad year; budget roughly one hang per 30–40 min of sustained
GEE work.

**The durable fix — a wall-clock deadline per chunk inside `_retry` — is still not
implemented.** On this evidence it should be: any long extraction in this repo has the same
exposure, including the 12-class pipeline.

Interim mitigation was a stall-watchdog (kill + restart on log silence, parquet integrity
sweep between attempts). Two lessons if it is reused:

* **Restart with only the outstanding years.** Relaunching the full `--years 1996-2023`
  replayed ~27 cached years (~7 min) on every recovery; with a 30-min stall threshold a cycle
  could spend most of its time on overhead. Narrowing to `2023-2023` and dropping the
  threshold to 10 min cut recovery cost from ~40 min to ~10.
* **The CLI parses `--years` as `lo-hi`**, so a single year must be written `2023-2023`; a
  bare `2023` raises `ValueError: not enough values to unpack`.

Resumption is safe: both stages skip existing chunk files and the combine step globs
per-year, so no cross-year mixing. Chunk integrity is verified between restarts because a
`kill -9` mid-write would leave a truncated file that resume would **silently skip** rather
than error. Verified clean at every restart (final sweep: 5,431 chunks, 0 corrupt).

### 6.4 Assembly and inference — **DONE (2026-08-06)**

All 28 years assembled (`panel assemble`, ~2 h, zero errors; each year a self-contained
bundle of `features_lightgbm.parquet` + per-date and pixel-set tensors). Panel inference with
`lightgbm_nometa` produced **`panel_predictions.parquet`, 215,320 parcel-years** over 7,690
parcels × 28 years, temperature-calibrated at T = 1.409. Abstention tracks the coverage gate
exactly: ~115–150 parcels in healthy years (mostly `no_features`), rising to 850–960 in
1997/2009/2011.

## 7. ⛔ The Phase-7 gate — **RUN, AND IT FAILS** (2026-08-06)

**`perennial diagnostics` → GATE: FAIL (S4 FAIL, S5 FAIL).** Recorded in
`data/processed/perennial/phase7_gate.json`; figures in `runs/perennial/diagnostics/`.

**No trajectories, transitions or area estimates have been produced, and none should be.**
That code is built and unit-tested (`tests/test_trajectories.py`,
`tests/test_area_estimation.py`) and it stays unrun: a trend from a panel that failed
validation is not a weaker finding, it is a wrong one. The per-year predicted class share —
the headline deliverable — was deliberately **not** computed, because it is exactly the
quantity these diagnostics say is untrustworthy.

### 7.0 What the four diagnostics found

| # | diagnostic | LightGBM `nometa` | LTAE tuned |
|---|---|---|---|
| S4 | abs(k) ≤ 3 within 0.10 of k=0 | **FAIL** — 0.146 at k = −3 | **FAIL** — 0.116 at k = −3 |
| S5 | flicker < 15 % for PERENNIAL | **FAIL** — 0.428 | **FAIL** — **0.780** |
| — | drift at mission boundaries | **PASS on levels** (shared: model-independent) | — |
| — | 1999+2000 → 1998 El Niño (§8.1) | **FAIL** — PERENNIAL → 0.016 | **FAIL** — PERENNIAL → 0.033 |

Run for both architectures — see **§7.0.3**, which is the reason to believe the failure is in
the data rather than in a model choice. The detail below is LightGBM's unless stated.

#### S4 — temporal transfer (n = 3,553 locked-test parcel-years at k = 0)

| k | −5 | −4 | **−3** | −2 | −1 | **0** | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.695 | 0.661 | **0.600** | 0.774 | 0.728 | **0.746** | 0.736 | 0.738 | 0.740 | 0.715 | 0.739 |
| macro-F1 | 0.551 | 0.602 | 0.507 | 0.594 | 0.613 | 0.674 | 0.582 | 0.578 | 0.596 | 0.578 | 0.617 |

Forward transfer is excellent — k = +1…+3 sit within 0.010 of k = 0. **The failure is
entirely at k = −3**, and decomposing it says why: that bin is **66 % panel-year 1996**
(1,064 of 1,605 obs, accuracy 0.632) and contains **panel-year 1997 at accuracy 0.326** —
the El Niño year. The S4 failure and the §8.1 confound are the same phenomenon seen twice.

Two composition controls were added, because k bins are *not* the same parcel mix (which
parcels have a panel year at distance k depends on their label year, and label year is
confounded with label — §8). Neither changes the verdict; both are diagnosis:

* **prior-standardised** (per-class recalls re-weighted to the k = 0 class prior): worst
  deviation **0.063 → would pass**. So a good part of the raw failure is that the k = −3 bin
  is 29.4 % `PASTURE_FALLOW` against 20.8 % at k = 0, i.e. a harder mix, not purely worse
  transfer.
* **balanced accuracy** (all classes equal): worst deviation **0.156 → fails harder**, and
  *every* k ≠ 0 sits 0.066–0.156 below k = 0.

Per-class recall makes the mechanism plain:

| k | −3 | −1 | **0** | +1 | +3 |
|---|---|---|---|---|---|
| `ANNUAL` | 0.882 | 0.928 | **0.849** | 0.929 | 0.931 |
| `PASTURE_FALLOW` | 0.125 | 0.215 | **0.412** | 0.110 | 0.137 |
| `PERENNIAL` | 0.535 | 0.652 | **0.749** | 0.750 | 0.724 |

`PASTURE_FALLOW` recall is only ever respectable *at the label year* (0.41) and collapses to
0.11–0.22 everywhere else, while `ANNUAL` recall goes **up** away from k = 0. The model
drifts toward calling everything `ANNUAL` the further it gets from the label year — the same
failure the El Niño test found, generalised.

**An important honest caveat cuts the other way:** fallow is a genuinely *transient* state. A
parcel fallow in 1998 is often not fallow in 2003, so part of that collapse is **real land-use
change**, which is exactly why plan §7.3 calls S4 a *lower bound* on model stability. S4
cannot separate the two. What it can say is that the criterion as written is not met.

**The one encouraging number:** `PERENNIAL` recall holds at 0.65–0.75 across the whole
k range — the class the research question actually depends on is the most temporally stable.

#### S5 — flicker (the harder failure)

| PETT label | n | raw | smoothed | flagged years dropped |
|---|---|---|---|---|
| `ANNUAL` | 4,938 | 0.210 | 0.078 | 0.203 |
| `PASTURE_FALLOW` | 1,666 | **0.583** | 0.265 | 0.584 |
| **`PERENNIAL`** | 1,086 | **0.428** | 0.221 | 0.421 |
| ALL | 7,690 | 0.322 | 0.139 | 0.317 |

**0.428 against a 0.15 threshold — nearly 3× over.** Two supplementary views were computed
to diagnose it, neither of which rescues it:

* **It is not a coverage artefact.** Dropping the four flagged years (1997, 2009, 2011, 2012)
  moves PERENNIAL flicker from 0.428 to 0.421. The instability is spread across the record,
  not concentrated in the thin years.
* **Smoothing halves it but does not fix it** (0.428 → 0.221, still above 0.15). This is the
  case plan §9.2 explicitly warned about: if raw flicker is high, the minimum-duration rule
  **hides** the problem rather than solving it. A ≥3-consecutive-year transition rule applied
  to a series that changes class this often would mostly be filtering its own noise, and
  whatever survived would be indistinguishable from the residue of that filtering.

An orchard does not appear and vanish; a `PERENNIAL`-labelled parcel whose predicted class
changes more than once every five observed years is being misread in some of those years.
At 0.43, roughly two in five perennial parcels are.

### 7.0.1 Sensor drift — the one diagnostic that passes

Measured on a **balanced panel of 3,434 parcels present in every non-flagged year**, so a
step cannot be a composition artefact (the coverage gate drops ~half the panel in 1997, 2009
and 2011, and a naive per-year median would mix drift with a changing parcel set).

**0 of 30 mission-boundary steps exceed 2× the within-era year-to-year movement**, for raw
bands and indices alike. The largest is `NIR_amp` at the 2011/2012 boundary at 1.40×; every
index feature is under 0.60×. Raw B/G/R/NIR/SWIR summaries — the exposure the §4.6 ablation
explicitly does *not* remove, since ratio indices partly cancel calibration differences and
both models consume raw bands — show **no detectable TM→ETM+ level step** at this resolution.
Figures: `runs/perennial/diagnostics/drift_indices.png`, `drift_raw_bands.png`.

**But levels are not the whole story.** Cross-parcel spread (p90 − p10) grows across the
record even on the balanced panel:

| feature kind | L5 only | L5 + L7 | L7 only | ratio |
|---|---|---|---|---|
| raw band `_amp` | 0.106 | 0.111 | **0.145** | **1.37×** |
| index (`NDVI_amp`, …) | 0.273 | 0.324 | 0.354 | 1.30× |
| raw band `_median` | 0.079 | 0.088 | 0.086 | 1.10× |

and it accelerates sharply at the end: mean raw-`_amp` spread runs ~0.15 through 2020, then
**0.196 (2021), 0.269 (2022), 0.294 (2023)** — the thinning L7 record, where each parcel's
seasonal amplitude is estimated from fewer clear dates. The *centre* holds; the *noise* does
not. That extra per-parcel noise is a plausible direct contributor to the S5 flicker, and it
means the last three panel years are the least trustworthy, not merely the sparsest.

### 7.0.3 The LTAE cross-check — the panel fails for *data* reasons, not model reasons

The obvious objection to §7.0 is that it indicts one model. So the whole panel was re-inferred
with **`ltae_3c_tuned_test`** — a different architecture, trained on per-date sequences, which
receives **no statics whatsoever** (`build_tensors` passes only `[N,T,11]` spectral + `doy` +
`mask`). Same 28 years, same parcels, 215,320 parcel-years, temperature refitted (T = 1.329,
ECE 0.072 → 0.020). Artifacts carry the `_ltae` suffix; figures in
`runs/perennial/diagnostics_ltae/`.

**`perennial diagnostics --preds panel_predictions_ltae.parquet --tag ltae` → GATE: FAIL,
on both criteria, worse than LightGBM.**

| | LightGBM `nometa` | **LTAE tuned** |
|---|---|---|
| S4 accuracy at k = 0 | 0.746 | 0.712 |
| S4 worst deviation, abs(k) ≤ 3 | 0.146 (FAIL) | 0.116 (FAIL) |
| **S5 flicker, `PERENNIAL`** | **0.428** | **0.780** |
| S5 flicker, all parcels | 0.322 | 0.842 |
| S5 flicker, smoothed | 0.221 | 0.440 |

So the answer to "is the flicker a LightGBM artefact?" is **no — it is worse without
LightGBM**, which turns out to be the informative result.

> **✅ 2026-08-09 — INDEPENDENTLY REPLICATED ON ALL OF PERU, and it was pre-registered.**
> The ordering below was written down as a prediction *before* the national panel was gated
> (`../all_peru/plan.md` §7b), and it came out exactly right on 14 departments, 25 years and
> 114,125 parcel-years per arm:
>
> | arm | statics | Piura flicker | **all-Peru flicker** |
> |---|---|---|---|
> | `lightgbm_nometa` | 3 | 0.428 | **0.517** |
> | `lightgbm_nometa_nolat` | 2 | 0.547 | **0.744** |
> | `ltae` (no statics) | 0 | 0.780 | **0.980** |
>
> **Every rung is worse nationally**, and k = 0 accuracy again barely moves across the three
> (0.586 / 0.544 / 0.581). The national replication is the stronger evidence, because that
> panel has **no El Niño baseline** (it starts in 1999), **no thin years** (minimum coverage
> gate pass 93.4 % vs Piura's 43.3 %) and **S4 actually passes** on both LightGBM arms — so
> the three excuses available for the Piura result are all removed and the flicker remains.
> See `../all_peru/RESULTS.md` §7.4–§7.5.

#### Why: time-invariant features manufacture *stability*

A third panel inference, with `lightgbm_nometa_nolat` (no `centroid_lat`), completes a clean
monotone ordering in how much time-invariant information the model holds:

| model | time-invariant inputs | flicker `PERENNIAL` | flicker all | S4 acc at k=0 |
|---|---|---|---|---|
| `lightgbm_nometa` | `centroid_lat`, `area_ha`, `n_pixels_est` | **0.428** | 0.322 | 0.746 |
| `lightgbm_nometa_nolat` | `area_ha`, `n_pixels_est` | 0.547 | 0.490 | 0.752 |
| `ltae_3c_tuned_test` | **none** | **0.780** | 0.842 | 0.712 |

**The more time-invariant information a model has, the flatter its trajectory — while its
accuracy barely moves** (0.746 / 0.752 / 0.712).

*(2026-08-08: the all-Peru strand found the same feature doing the same kind of damage on a
third axis — manufacturing **accuracy that does not leave the training departments**. See
[`../all_peru/RESULTS.md`](../all_peru/RESULTS.md) §6.2.)* A feature that cannot change between years
cannot contribute a class change, so it damps the series mechanically. This is the exact
mirror image of the `frac_l7` problem in §4.6: **`frac_l7` manufactured change; `centroid_lat`
manufactures stability.** Both are the model answering from something other than this year's
land.

Two consequences worth stating plainly:

1. **S5 is gameable.** A model could be made to "pass" by leaning harder on time-invariant
   features — predicting the same class every year, which is maximally stable and carries no
   temporal information at all. `lightgbm_nometa`'s 0.428 is therefore an *optimistic* figure,
   not a neutral one, and the honest estimate of what per-year spectral data supports is
   closer to LTAE's 0.780.
2. **The failure is in the data, not the model.** Two architectures with no shared inputs,
   no shared feature representation and no shared statics both fail, and the one with the
   cleanest inputs fails hardest. Per-year 30 m spectral summaries over ~0.5 ha parcels do not
   determine the 3-class state reliably enough to support parcel-level annual trajectories.

#### The El Niño arm under both architectures

`elnino_confound_test` was refactored to go through the normal model registry, so it now asks
the question of the actual architecture instead of always fitting its own LightGBM. (The
numbers therefore differ slightly from the first §8.1 run, which used a fixed
`LGBMClassifier(n_estimators=400)` and no validation split; the effect is unchanged.)

| class | LGBM control | LGBM 1998 | **Δ** | LTAE control | LTAE 1998 | **Δ** |
|---|---|---|---|---|---|---|
| `ANNUAL` | 0.833 | 0.908 | **+0.076** | 0.774 | 0.342 | **−0.432** |
| `PASTURE_FALLOW` | 0.626 | 0.142 | **−0.484** | 0.632 | 0.501 | −0.130 |
| **`PERENNIAL`** | 0.544 | **0.016** | **−0.527** | 0.660 | **0.033** | **−0.627** |
| accuracy | 0.760 | 0.804 | +0.043 | 0.728 | 0.362 | −0.367 |

The two architectures fail **differently but both catastrophically**. LightGBM funnels its
1998 errors into `ANNUAL` (accuracy even *rises*, because 1998 is 86 % `ANNUAL` by label);
LTAE instead collapses broadly, calling many true-`ANNUAL` parcels `PASTURE_FALLOW`.

**What they agree on is the finding that matters: `PERENNIAL` recall on 1998 is 0.016 and
0.033 — effectively zero under both models, against controls of 0.54 and 0.66.** That is
architecture-independent, so it is a property of 1997–98 El Niño imagery, not of a classifier.
Since 1996–98 are the panel's baseline years, **the perennial baseline is near-zero for
artefactual reasons under any model**, and any measured rise from it is uninterpretable.

### 7.0.2 What this means, and what would have to change

The three failures are **not independent** — they are one story told three ways: a model
supervised by a single ~1998 titling snapshot, in which registration year is confounded with
label and 1997–98 is a catastrophic El Niño, does not hold a stable class for a parcel across
28 years, and is least reliable precisely in the baseline years against which any trend would
be measured. `frac_l7` (§4.6) would have added a fourth, independent artefact; removing it was
necessary but nowhere near sufficient.

**§7.0.3 establishes that this is a property of the data, not of one model.** Two
architectures sharing no inputs, no feature representation and no statics both fail both
criteria, and the one with the cleanest inputs (LTAE, no statics at all) fails hardest —
flicker 0.780 against LightGBM's 0.428. That gap is time-invariant features damping the
series, not better temporal discrimination, so **0.428 is the optimistic end of the range and
0.780 is the honest one.** Both put `PERENNIAL` recall on 1998 at ~0.02–0.03.

What is *not* broken: mission-boundary radiometry (7.0.1), forward temporal transfer at
k = +1…+3 (within 0.010), and `PERENNIAL` recall across k (0.65–0.75 for LightGBM, 0.62–0.82
for LTAE). The perennial signal itself is the most robust thing in this analysis. What fails
is the ability to read a *change* in it, year by year, at parcel level.

**One route is now closed.** "Try the other architecture" was the cheapest hypothesis and it
has been tested and rejected; no model choice recovers this panel. What remains has to change
the data or the estimand, not the classifier.

Options, roughly in order of expected value:

1. **Abandon per-parcel annual trajectories; report an aggregate share with uncertainty.**
   Flicker is a per-parcel property; a population share can be stable while individual series
   are not. This changes the deliverable from "which parcels converted and when" to "how much
   of the area was perennial each year, ± CI" — which is closer to the actual research
   question and far more defensible. It still requires fixing the baseline problem (§8.1).
2. **Restrict the baseline.** Start the panel at 1999 rather than 1996 and state that no
   pre-El-Niño baseline is recoverable. This removes the worst years but also removes the
   comparison point the research wants.
3. **Get supervision at a second time point.** The 2012 CENAGRO census is reachable through
   the name link (CLAUDE.md Chain B, weak on exact parcel). Even noisy 2012 labels would
   convert this from single-snapshot extrapolation into something testable.
4. **Threshold on confidence and abstain aggressively.** The model is now calibrated
   (T = 1.409, ECE 0.031), so a probability floor is meaningful. Untested whether it cuts
   flicker enough to matter.

Re-running the gate is one command (`perennial diagnostics`) and is the acceptance test for
any of these.

### 7.1 The ≥3-consecutive-observed-year transition rule *(built, unrun — blocked by the gate)*

### 7.1 The ≥3-consecutive-observed-year transition rule

The panel gives one predicted class per parcel per year. Taking every year-to-year change
at face value would count classifier noise as land-use change — and with three classes and
per-year accuracy around 0.70, noise would dominate. The rule that defines a *real* change:

> A transition `A → B` at year *t* is recorded only if class `A` held for **≥ 3 consecutive
> observed years** immediately before *t*, **and** class `B` holds for **≥ 3 consecutive
> observed years** from *t* onward.

Three properties matter:

1. **"Observed", not calendar.** Years where the parcel abstained (failed the coverage gate,
   no features, below the confidence threshold) are carried as explicit gaps and **skipped**
   in the count, never interpolated. So `ANNUAL, gap, ANNUAL, ANNUAL → PERENNIAL, PERENNIAL,
   gap, PERENNIAL` counts as 3 before and 3 after and *does* fire. This matters because the
   thin years (1997, 2009, 2011) would otherwise silently break every run that spans them.
2. **It is a physical claim, not a tuned parameter.** A mango or lime orchard takes 2–3
   years to close canopy and does not revert next season; clearance is equally durable. So a
   one- or two-year excursion is noise *by construction* — not because 3 scored best on
   some metric. `min_duration` is configurable, but the default encodes the agronomy.
3. **It is deliberately conservative.** It cannot detect a change in the last two observed
   years of the panel, and it discards genuine short-lived changes. That is the right trade
   for a trend claim: the cost is missed detections, not fabricated ones.

Applied to the *smoothed* series (a centred, gap-aware 3-year mode filter that keeps the
parcel's own class on a tie, so a smoother can never invent a change). Both raw and smoothed
transition counts are reported.

The companion diagnostic is the **flicker rate** — the fraction of parcels whose *raw*
series changes class more than `n_observed / 5` times. Criterion S5 requires it under 15 %
for parcels whose PETT label is PERENNIAL. If flicker is high, the per-year features are too
weak and the trend must not be run on raw predictions at all; the minimum-duration rule
would then be hiding the problem rather than solving it.

Realism checks that catch a broken model faster than any metric: `ANNUAL → PERENNIAL` should
substantially exceed `PERENNIAL → ANNUAL` (a symmetric matrix means noise); establishment
should look like a **ramp**, not a step, in mean `P(PERENNIAL)` around detected transitions
(canopy closure takes years); and `PERENNIAL → ANNUAL` should be rare and spatially
clustered.

### 7.2 Area estimation *(built, unrun — blocked by the gate)*

⚠️ **An unresolved problem even if the gate is later passed:** the Olofsson correction needs
an error matrix from held-out data, and the one on disk
(`lightgbm_3c_final/test_confusion.csv`) belongs to the **disqualified** model. The locked
test cannot be re-run for `lightgbm_nometa` — it is spent twice. The honest substitute is that
model's pooled spatial-CV confusion matrix (`preds_cv.parquet`, 44,022 parcels), which is
held-out per fold but carries the residual spatial leakage §3.2 documents, so the CIs would be
mildly optimistic and must be labelled as CV-derived rather than test-derived.

Three estimators per year per class: naive argmax, probability-weighted, and **Olofsson et
al. (2014) bias-corrected with 95 % CIs** from the locked-test error matrix (D8), all
expanded to the population by the panel sampling weights. Given ROC AUC 0.937 but argmax F1
0.68 (§4.3), the probability-weighted and bias-corrected estimators are expected to be
materially more trustworthy than the naive count.

### 7.5 Is the panel adequate for the tenure → transition analysis? (2026-08-05)

The downstream question is whether land tenure at registration predicts subsequent
`ANNUAL → PERENNIAL` conversion. **Verdict: adequate for a weighted descriptive association;
not adequate for a causal claim. The binding constraint is confounding, not sample size.**

Tenure is `ESTADO en RRPP` from the raw SSET xlsx (`COND_JUR` is degenerate — 188,131 of
190,098 share one value). ⚠️ **It is not carried into any processed table** — `load_sset()`
hard-codes a `usecols` list that omits it, so it must be read from the raw sheet. Panel
coverage is **100 %** (1,783 INSCRITO / 5,907 NO INSCRITO).

The at-risk pool for the transition analysis is the ANNUAL parcels:

| tenure | raw n | Kish n_eff | deff |
|---|---|---|---|
| INSCRITO | 949 | 721 | 1.32 |
| NO INSCRITO | 3,989 | 2,843 | 1.40 |

Kish effective n matters because weights run 1–17 (forced test parcels at 1.0, sampled up to
17). Design effects of 1.32–1.40 are mild; this powers detection of a **~4 pp** tenure
difference at 80 %, against a cross-sectional gap of 24.9 % vs 10.9 % perennial. **Power is
not the problem.** Four other things are:

1. **PERENNIAL base rate is confounded with registration year** — 1998 is 0.4 % perennial,
   1999 13.7 %, 2000 33.8 % — *and* tenure tracks year (17.2 % INSCRITO in 1998, 80.9 % in
   2005). Tenure ↔ year ↔ baseline perennial propensity are mutually entangled. See §8.
2. **Tenure is spatially clustered and so is crop suitability.** Titling swept region by
   region; mango and grape need particular soils and irrigation.
3. **Tenure is time-invariant here** — observed once at registration. Parcels may have
   formalised later, which attenuates any estimate toward zero.
4. **Reverse causality is live.** A farmer intending a 20-year orchard has strong reason to
   seek formal title *first* (collateral, security). Cross-sectional data cannot separate
   this from tenure→perennial.

**The main fix needs no re-sampling.** Tenure varies *within* region:

```
regions with ANNUAL parcels : 230
  tenure-MIXED regions      :  77
  ANNUAL parcels in mixed   : 4,579 of 4,938  (92.7%)
```

So a **region fixed-effects / within-region matched contrast** is available today on 93 % of
the at-risk pool, which differences out confound (2) — the largest. Adding registration-year
fixed effects handles (1).

If the panel *were* re-drawn: stratify on **tenure × label** (current balance — 19.2 %
INSCRITO among panel ANNUAL vs 21.1 % weighted-population — came out fine by luck, not
design); **oversample ANNUAL** (only 4,938 of 7,690 parcels are in the at-risk denominator);
and match on region × area × baseline crop. None of this fixes (4), which needs time-varying
tenure (a later titling wave → difference-in-differences) or an instrument.

### 7.6 S6 is not evaluable as written — proposed replacement

S6 asks whether our perennial-share trend agrees in direction with MapBiomas. **It cannot be
run**: §5 established that MapBiomas Peru never assigns a perennial-crop code (36/46/47/48)
or pasture (15) anywhere in Piura, across 1,974,665 parcel-years. There is no perennial share
to compare against. This was not known when the plan was written.

Two substitutes, both testable with data already in hand:

1. **Compare the annual side.** Our `ANNUAL` share trend vs MapBiomas's temporary-crop share
   (code 40 rice + 72 other crops). Those codes are well populated, giving a genuine external
   directional check on half the picture.
2. **Test class migration** (the stronger test). If a parcel converts to orchard, MapBiomas
   should stop calling it rice — drifting from 40/72 toward 21 (mosaic). So: do parcels we
   detect as `ANNUAL → PERENNIAL` show a decline in MapBiomas rice/other-crops fraction
   around the transition year, relative to parcels we say did not change? This needs no
   perennial class in the benchmark at all.

Recommendation: replace S6 with (2) and report (1) alongside.

## 8. Limitations

* **Labels are titling dates, not verified field observations** (`CLAUDE.md` §6). The whole
  supervision signal is one ~1998–99 snapshot per parcel.
* **Residual spatial leakage.** Class agreement stays above baseline past 5 km while the
  buffer is 1.5 km, so CV is optimistic; contiguous 5 km test regions confine the leak to
  region borders.
* **~0.5 ha parcels are ~5 Landsat pixels.** Every metric should be read stratified by
  area; the ≥ 3 ha subset is the honest upper bound on achievable accuracy.
* **`PASTURE_FALLOW` is a deliberately heterogeneous class** and is the weakest of the
  three; see the D2 diagnostic.
* **Woody non-crops are excluded from training but not from the world.** At inference,
  algarrobo/bamboo/plantation parcels will be predicted `PERENNIAL` and are a known
  false-positive source for the export-crop reading.
* **2012 is L7-SLC-off only** — flagged in every figure, never interpolated over.
* **Sensor shift is the main threat to the headline trend**, which is why D7 harmonisation
  and the §7.4 drift diagnostics are mandatory before any trend is believed.
* **⚠️ Registration year is severely confounded with the label — and with the split.**
  Discovered 2026-08-05, not previously recorded. The label mix differs wildly by cohort:

  | year | ANNUAL | PASTURE_FALLOW | **PERENNIAL** |
  |---|---|---|---|
  | 1998 | 83.2 % | 16.4 % | **0.4 %** |
  | 1999 | 54.2 % | 32.1 % | **13.7 %** |
  | 2000 | 51.9 % | 14.4 % | **33.8 %** |

  Because titling campaigns swept region by region, year and region are confounded — so the
  *purely spatial* split (`splits.py` never reads `year`) produced an **accidental temporal
  split**: the locked test is **48.3 % 1998** against trainval's 32.9 % (and 8.4 % vs 2.8 %
  for 2005). Nobody designed this and nothing controls it.

  The compounding risk: **1997–98 was the catastrophic Piura El Niño**, so 1998
  simultaneously has anomalous imagery (1997 gate passes only 48.3 %) *and* almost no
  perennial labels. The model may have learned "El Niño-looking radiometry → ANNUAL", a rule
  that transfers to no other year — and 1996–1998 are the panel's **baseline** years against
  which all trend is measured.

  **⛔ THE DIAGNOSTIC WAS RUN (2026-08-06) AND IT FAILED — see §8.1 below. The 1998
  baseline confound is real, large and robust.** The spec that follows is what was
  implemented (`perennial/diagnostics.py::elnino_confound_test`); the result is §8.1.

  **The "1999+2000 → 1998" test — full spec:**

  * **Restrict to the 18 regions containing both 1998 and 1999 parcels** (17,668 parcels).
    Year and region are confounded, so this holds region fixed while year varies.
  * **Test arm:** train LightGBM on the 1999+2000 parcels in those regions, test on the
    1998 parcels in those regions.
  * **Control arm (required):** same training set, but test on *held-out 1999+2000 parcels
    in the same regions*. Without it you cannot tell "1998 is a different year" from "1998
    parcels are simply harder". **The quantity of interest is test-arm − control-arm**, per
    class, not the raw 1998 score.
  * **Direction matters:** do *not* train on 1998 — it has only 51 perennial parcels in the
    shared regions, so a failure to transfer would be unsurprising and uninformative.
  * **Report per-class recall**, not macro-F1: cohort priors differ enormously by
    construction, so an aggregate mostly measures prior shift.

  | outcome | reading |
  |---|---|
  | ANNUAL recall holds, **PASTURE recall collapses** | the smoking gun — 1998 flood/bare radiometry is being read as the ANNUAL signature |
  | all classes degrade ≈ equally vs control | generic temporal-transfer decay; consistent with S4, far less alarming |
  | little degradation vs control | confound is not operating through radiometry; the panel baseline is safer than feared |

  **Limit:** one year cannot separate "1998 is a *different* year" from "1998 is an *El
  Niño* year" — they are collinear here (1996, the natural non-El-Niño L5-only comparator,
  has only 4 labelled parcels). A positive result condemns the baseline without saying
  exactly why, which is still decisive for whether to trust the trend.

  **Complementary to S4, not redundant:** S4 measures decay with distance from the label
  year but cannot separate model drift from the El Niño confound; this can. Run both.
* **`PERENNIAL` F1 is 0.769 — better than the 0.681 macro-F1 suggests.** Macro-F1 is dragged
  down by `PASTURE_FALLOW` (0.440). For a perennial-vs-rest trend, the relevant number is
  the perennial one, with balanced precision (0.758) and recall (0.780).

### 8.1 ⛔ The 1998 confound test — **RUN, and it is the smoking gun** (2026-08-06)

Implemented exactly to the §8 spec in `perennial/diagnostics.py::elnino_confound_test`,
including the required control arm. Trained on the 1999+2000 cohort within regions that also
contain 1998 parcels (region held fixed, year varied), with the model's own ablated feature
set (`drop_features="meta"`), `class_weight="balanced"`.

Primary run — 24 shared regions, 6,262 train / 2,088 control / 15,408 test-1998 parcels:

| class | control arm (held-out 1999+2000) | test arm (1998) | **Δ = test − control** |
|---|---|---|---|
| `ANNUAL` | 0.880 | 0.930 | **+0.049** |
| `PASTURE_FALLOW` | 0.589 | 0.100 | **−0.490** |
| `PERENNIAL` | 0.361 | **0.000** | **−0.361** |

**This is the §8 outcome table's first row — the smoking gun.** `ANNUAL` recall *rises*
while `PASTURE_FALLOW` collapses by 49 points and `PERENNIAL` goes to **exactly zero on 61
parcels**. The model applied to 1998 imagery calls almost everything `ANNUAL`. That is the
predicted signature of 1998 flood/bare El Niño radiometry being read as the annual-crop
signature.

Recall is computed *within* each true class, so the fact that 1998 is 86 % `ANNUAL` by label
cannot produce this — that is precisely why §8 specified recall over macro-F1.

Robustness — the finding does not move:

| variant | Δ`ANNUAL` | Δ`PASTURE_FALLOW` | Δ`PERENNIAL` |
|---|---|---|---|
| primary (24 regions, ablated features) | +0.049 | −0.490 | −0.361 |
| strict §8 wording (18 regions, 1998 ∩ 1999) | +0.026 | −0.477 | −0.345 |
| un-ablated features (metadata present) | +0.052 | −0.518 | −0.370 |
| seeds 1 / 2 / 3 | +0.012 / +0.034 / +0.066 | −0.545 / −0.493 / −0.538 | −0.315 / −0.368 / −0.359 |

**⚠️ Superseded numbers.** The table above is the first run, which used a fixed
`LGBMClassifier(n_estimators=400)` with no validation split. `elnino_confound_test` now goes
through the model registry so it can test any architecture; **the current numbers, for both
LightGBM and LTAE, are in §7.0.3.** They are materially the same, except that `PERENNIAL`
recall on 1998 lands at 0.016 rather than 0.000 and the control is higher. The conclusion is
unchanged and now architecture-independent.

Two things follow. First, **the §4.6 metadata ablation neither causes nor fixes this** — the
numbers are the same with and without the metadata, so it is a genuinely separate failure
mode from `frac_l7`. Second, **1996–1998 are the panel's baseline years.** If panel-era
radiometry in those years resembles the 1998 training cohort's, the baseline will be
spuriously `ANNUAL`-heavy and near-zero `PERENNIAL`, and *every* subsequent year will look
like a shift toward perennial — the exact artefact this project must not produce, arriving
by a second, independent route.

The §8 limit stands and is now load-bearing: one year cannot separate "1998 is a *different*
year" from "1998 is an *El Niño* year", because 1996 — the natural non-El-Niño L5-only
comparator — has only 4 labelled parcels. The result condemns the baseline without saying
exactly why, which is still decisive for whether to trust the trend.

### 8.2 *Why* 1998 breaks it — the signature is a contrast, and the flood erased it (2026-08-07)

§8.1 establishes *that* the model fails on 1998. This is the mechanism. Numbers from
`perennial/report_figures.py::elnino_signature_collapse`, persisted to
`docs/figures/elnino_signature_collapse.csv`; figure `docs/figures/elnino_mechanism.png`.

**A perennial parcel is not identified by an absolute value — it is identified by a contrast
with its annual neighbours.** Measuring that contrast *within* each cohort, on the same
regions (class-median gap in within-cohort SD units):

| discriminating feature | 1999+2000 (normal) | 1998 (El Niño) |
|---|---|---|
| `NDVI_p25` — "does it stay green?" | **+1.468 SD** | **+0.457 SD** |
| `NDVI_amp` — "does it senesce?" | −0.602 SD | **−0.105 SD** |
| `BSI_max` — "is soil ever exposed?" | −0.757 SD | **−0.904 SD** |

**The two classes move toward each other from both directions:**

* **Perennials became less perennial-looking** — NDVI floor 0.518 → 0.448, amplitude
  0.261 → 0.298. Consistent with flood damage and defoliation.
* **Annuals became more perennial-looking** — NDVI floor 0.363 → 0.404, amplitude
  0.344 → 0.311. The flooded ground simply stayed green year-round.

So the strongest discriminator drops from a 1.47 SD separation to 0.46 SD and the
seasonal-amplitude signal is annihilated (0.60 → 0.10 SD). **The model is not broken in
1998 — the information it depends on is not in the imagery.**

`BSI_max` is the one feature that **holds up, and even sharpens**. That is not an artefact
of the within-cohort normalisation: the *raw* class gap also widens, −0.049 → −0.072. §4.1's
rule-control experiment independently found the same feature to be the more robust
discriminator. Both point at **bare-soil exposure being sturdier than greenness here**.

*Caveat:* only **61** labelled `PERENNIAL` parcels fall in the 1998 shared-region cohort (vs
420 in 1999+2000), so the 1998 column is a small sample. The direction is corroborated by
§8.1's recall collapse under both architectures.

#### ⚠️ This is not a missing-feature problem — a BSI availability audit

The natural misreading is "the models were missing the robust feature". They were not. What
each model actually receives:

| model | raw per-date `BSI` | whole-year `BSI_max` | uses it? |
|---|---|---|---|
| LightGBM (`lightgbm_nometa`) | via summaries | **yes** — in `feature_names` | **yes** — rank **9 of 135**, 1.46 % of gain |
| `rules` | n/a | present in the store, **never read** | no — reads only 3 configured columns (§4.1) |
| LTAE / PSE-LTAE | **yes**, channel 11 of 11 | **no** precomputed summary | must derive an equivalent through attention |

All 12 `BSI_*` columns are in the flat store, and **the `meta`/`location` ablations do not
touch any of them** (`resolve_drop_features("meta,location")` returns only the five
acquisition columns plus `centroid_lat`).

So **LightGBM had `BSI_max`, ranked it 9th, and still lost `PERENNIAL` almost entirely on
1998.** What the models lack is not the feature but the **weighting**: all `BSI_*` features
together are **6.51 % of LightGBM's gain**, while the NDVI/NDWI/NDMI greenness family
dominates — and greenness is exactly what the flood destroyed.

**Whether deliberately reweighting toward bare-soil features would buy El Niño robustness is
a testable hypothesis, not a demonstrated fix.** §4.1's result only shows it helps a
3-feature rule *where BSI was genuinely absent*, and even there it cost `PERENNIAL` F1 — the
class this would need to rescue. Do not cite the two results as mutually confirming.

## 8.3 The ≥1999 (no-El-Niño) retrain — **S4 now passes, S5 still fails** (2026-08-07)

§8.1/§8.2 say the model cannot read 1997–98 El Niño imagery, and 1996–98 are the panel's
baseline. This tests the obvious response: **drop the El Niño cohort from supervision**.
New flag `train --train-years` (`data.parse_year_spec` / `restrict_years`) restricts the
**training** indices only — in both the CV folds and the final refit — leaving validation
and locked-test membership untouched so CV stays comparable. It is recorded in
`cv_metrics.json`. Regression: `tests/test_pipeline.py::TestTrainYears`. **The locked test
was not touched** (spent twice already, §4.4/§4.5).

Run: `runs/perennial/lightgbm_nometa_from1999` (`--drop-features meta --train-years
1999-2023`), panel `panel_predictions_from1999.parquet` (192,250 parcel-years, 1999–2023),
gate artifacts tagged `_from1999`.

**What the restriction costs.** Final-refit training cohort 33,334 → **24,096** parcels,
221 → 218 regions; class mix **60.9/22.1/16.9 → 52.7/24.0/23.3** (ANNUAL/PASTURE/PERENNIAL).
`PERENNIAL` loses only **40 of 5,647** train parcels (0.7 %) — 1998 supplies the ANNUAL bulk,
almost none of the scarce class.

### CV — free on the years it still trains for

| run | CV mean ± std | pooled CV, **all** val | pooled CV, val `year ≥ 1999` | acc (all) | `PERENNIAL` F1 (all) |
|---|---|---|---|---|---|
| `lightgbm_nometa` | 0.6479 ± 0.036 | **0.6503** | **0.6415** | 0.699 | 0.685 |
| `lightgbm_nometa_from1999` | 0.6240 ± 0.035 | 0.6266 | **0.6412** | 0.696 | 0.652 |

Read the two pooled columns together: on the **≥1999 validation parcels the two models are
indistinguishable (0.6415 vs 0.6412, −0.0003)**. The entire −0.024 headline drop is the
model losing accuracy on the 14,302 **1998** validation parcels it is no longer trained for
— which is the intended trade, not a regression. Per-fold: 0.645/0.652/0.578/0.651/0.594.
Temperature refitted **T = 1.258** (ECE 0.0714 → 0.0312).

### The gate — and the control that says how much of it is the retrain

A second gate run was added that the task did not ask for but that the result requires:
the **baseline model's own predictions truncated to ≥1999** (`--tag trunc1999`). Without it
"S4 now passes" cannot be attributed, because shortening the panel *removes the years that
were failing*.

| | baseline `nometa`, 1996–2023 | **control**: same model, 1999–2023 | **`from1999`**, 1999–2023 |
|---|---|---|---|
| S4 worst dev, abs(k) ≤ 3 | **0.1456 FAIL** | 0.0764 **PASS** | **0.0627 PASS** |
| S4 accuracy at k = 0 | 0.7456 | 0.6905 | 0.6877 |
| S4 prior-standardised / bal-acc | 0.063 / 0.156 | 0.065 / 0.087 | 0.055 / 0.070 |
| **S5 flicker `PERENNIAL`** | **0.428 FAIL** | 0.397 FAIL | **0.356 FAIL** |
| S5 flicker `PASTURE_FALLOW` | 0.583 | 0.547 | 0.514 |
| S5 flicker ALL | 0.322 | 0.302 | 0.285 |
| S5 smoothed `PERENNIAL` | 0.221 | 0.204 | 0.173 |
| sensor drift | PASS | PASS | PASS (model-independent) |
| El Niño arm, `PERENNIAL` recall 1998 | 0.016 (control 0.544) | — | **0.016 (control 0.544)** |
| **verdict** | FAIL | FAIL | **FAIL (S4 pass, S5 FAIL)** |

**S4's pass is mostly mechanical.** The control passes too, at 0.0764. The failing k = −3
bin shrank from 1,605 to 368 observations and its composition changed from **66 % panel-year
1996 + 9 % 1997** to **48 % 2002** — the years that were failing are simply no longer in the
panel. The retrain adds a further 0.0764 → 0.0627. Note k = 0 accuracy *falls* 0.746 → 0.688:
1998-labelled parcels no longer have a k = 0 panel year, so the easy 1998-in-1998
observations are gone from the numerator as well.

**S5 fails, as expected, and the retrain owns about half the improvement it did get.**
0.428 → 0.397 from truncation, 0.397 → **0.356** from the retrain. Against a 0.15 criterion
this is still 2.4× over, and smoothing (0.173) still does not clear it. Per §7.0.3, 0.356 is
also the *optimistic* end — `lightgbm_nometa` retains `centroid_lat`, and time-invariant
features damp the series mechanically.

**The El Niño arm is bit-identical to §7.0.3's LightGBM column** (control 0.833/0.626/0.544,
1998 0.908/0.142/0.016). That is correct and worth stating: `elnino_confound_test` trains its
*own* model on the 1999+2000 cohort, so `--train-years` cannot reach it. It re-confirms that
1998 unreadability is a property of the imagery, not of the supervision cohort.

**Verdict: the gate still fails, so no trajectories, transitions or area estimates were
produced.** This was the expected outcome. What it buys is a cleaner diagnosis: the temporal
-transfer failure was *localised to 1996–98* and is removable, while the flicker failure is
**not** — it is spread across the whole 1999–2023 record and survives both the retrain and
the truncation.

### The two cheap checks

**Does it change 2017 and 2023?** Yes, in the predicted direction, and they are the years it
changes most — but the effect is small. `PERENNIAL` share vs the baseline model, all 25
years: **2023 +0.018** and **2017 +0.016** are 2 of the 3 largest positive deltas (2008 and
2011 also +0.017); the median year moves +0.004. In 2023 `ANNUAL` falls 0.838 → 0.816 and
`PERENNIAL` rises 0.126 → 0.144; 2023's baseline `PERENNIAL` share was the lowest of the
2015+ era. Per-year prediction agreement between the two models is **lowest in 2017 (0.900)**
against a non-flood mean of 0.921. So a model no longer taught "flooded ⇒ ANNUAL" does behave
differently on flood years — consistent with §8.2's `BSI_max` mechanism — but by ~1.5
percentage points of share, not enough to matter to a trend.

**Does dropping 1998 drop a place?** Essentially no. 13 regions are >90 % 1998 (9,267
parcels, 17.0 % of the labelled set) but only **4 are 100 % 1998**, so the final-refit
training set loses just **3 of 221 regions**. The training bounding box is unchanged
(−81.072…−79.355, −5.707…−4.572). Distance from each panel parcel to its nearest retained
training parcel rises only **median 1.21 → 1.52 km, p90 2.99 → 3.43 km**, and the >10 km tail
is unchanged at 0.77 %. **No spatial coverage gap** — the 1998-heavy regions are interleaved
with ≥1999 ones, not a separate part of Piura.

### Where this leaves §7.0.2's option list

Option 2 ("restrict the baseline") is now **tested, and it is not sufficient on its own**. It
does fix what it was aimed at — S4, and the baseline-year contamination — at a CV cost of
zero on the years it still covers. It does not touch S5, which is the harder failure. The
remaining live options are unchanged: **option 1** (drop per-parcel annual trajectories,
report an aggregate share with CIs) and **option 3** (2012 CENAGRO supervision at a second
time point). A ≥1999 panel is the right substrate for option 1 when it is attempted.

## 9. Reproducing

```bash
export CC_PROC=data/processed/perennial CC_RUNS=runs/perennial

uv run python -m crop_classifier.cli perennial labels
uv run python -m crop_classifier.cli splits assign
uv run python -m crop_classifier.perennial.gapfill        # newly-admitted parcels
uv run python -m crop_classifier.cli features assemble

for m in rules lightgbm ltae psetae; do
  uv run python -m crop_classifier.cli train --model $m --run-name ${m}_3c
  uv run python -m crop_classifier.cli perennial pool-cv runs/perennial/${m}_3c
done
uv run python -m crop_classifier.cli perennial compare runs/perennial/*_3c \
    --out runs/perennial/comparison
uv run python -m crop_classifier.cli perennial calibrate runs/perennial/<selected>

# the locked test is SPENT (twice — §4.4 and §4.5). Do not run --eval-test again.

# §4.6 ablation: the panel model must carry no trend-manufacturing inputs
uv run python -m crop_classifier.cli train --model lightgbm \
    --run-name lightgbm_nometa --drop-features meta
uv run python -m crop_classifier.cli perennial pool-cv runs/perennial/lightgbm_nometa
uv run python -m crop_classifier.cli perennial calibrate runs/perennial/lightgbm_nometa

uv run python -m crop_classifier.cli perennial panel build
uv run python -m crop_classifier.cli perennial panel probe
uv run python -m crop_classifier.cli perennial panel extract      # 20-30 h, resumable
uv run python -m crop_classifier.cli perennial panel assemble     # ~2 h, 28 years
uv run python -m crop_classifier.cli perennial panel infer --run runs/perennial/lightgbm_nometa

# THE GATE — S4, S5, sensor drift, El Nino confound. Exits non-zero if S4 or S5 fails.
uv run python -m crop_classifier.cli perennial diagnostics

# the §7.0.3 cross-check: a second architecture over the same panel. SEPARATE PROCESSES —
# never import torch and lightgbm together on macOS (libomp clash -> segfault).
uv run python -m crop_classifier.cli perennial calibrate runs/perennial/ltae_3c_tuned_test
uv run python -m crop_classifier.cli perennial panel infer \
    --run runs/perennial/ltae_3c_tuned_test \
    --out data/processed/perennial/panel_predictions_ltae.parquet
uv run python -m crop_classifier.cli perennial diagnostics \
    --preds data/processed/perennial/panel_predictions_ltae.parquet --tag ltae \
    --elnino-model ltae

uv run python -m crop_classifier.cli perennial mapbiomas

# §4.1 the rules BSI_max variant. `model_kw` is deliberately NOT on the CLI — a model-variant
# experiment should be explicit in a script, not a flag someone can flip by accident.
uv run python -c "
from crop_classifier.train import train
train(model_name='rules', run_name='rules_3c_bsi',
      model_kw={'features': ('NDVI_p25', 'NDVI_amp', 'BSI_max')})"
uv run python -m crop_classifier.cli perennial pool-cv runs/perennial/rules_3c_bsi

# §8.2 mechanism + the report figures (writes docs/figures/, incl.
# elnino_signature_collapse.csv, which is where §8.2's numbers come from)
uv run python -m crop_classifier.perennial.report_figures

# §8.3 the >=1999 (no-El-Nino) retrain. --train-years restricts the TRAIN side only, so CV
# stays comparable with lightgbm_nometa. Do NOT add --eval-test.
uv run python -m crop_classifier.cli train --model lightgbm \
    --run-name lightgbm_nometa_from1999 --drop-features meta --train-years 1999-2023
uv run python -m crop_classifier.cli perennial pool-cv   runs/perennial/lightgbm_nometa_from1999
uv run python -m crop_classifier.cli perennial calibrate runs/perennial/lightgbm_nometa_from1999
uv run python -m crop_classifier.cli perennial panel infer \
    --run runs/perennial/lightgbm_nometa_from1999 --years 1999-2023 \
    --out data/processed/perennial/panel_predictions_from1999.parquet
uv run python -m crop_classifier.cli perennial diagnostics \
    --preds data/processed/perennial/panel_predictions_from1999.parquet --tag from1999

# ...and the control WITHOUT which "S4 now passes" cannot be attributed: the BASELINE
# model's own predictions truncated to the same years.
uv run python -c "
import pandas as pd
d = pd.read_parquet('data/processed/perennial/panel_predictions.parquet')
d[d.year >= 1999].to_parquet(
    'data/processed/perennial/panel_predictions_trunc1999.parquet', index=False)"
uv run python -m crop_classifier.cli perennial diagnostics \
    --preds data/processed/perennial/panel_predictions_trunc1999.parquet --tag trunc1999
```

`--drop-features` accepts column names or the aliases `meta` (the five acquisition
columns) and `location` (`centroid_lat`). The model's surviving column list is saved with
it and re-pinned at inference, so an ablated model cannot silently regain a feature that
still exists in the panel store.

Figures: `notebooks/05_perennial_trends.ipynb`.
