# Crop-classifier modelling pipeline — code guide & project state

> Companion to `plan.md` (the modelling blueprint, decisions A1–A6) and `docs/DATASETS.md`
> (the raw data). This document explains every module under `src/crop_classifier/`, the
> artifacts they produce, how to run the pipeline at three scales (smoke → medium → full),
> and what is verified vs still outstanding. Last updated: 2026-08-07 (feature exclusion —
> `--drop-features` + the inference-time `feature_names` pin — documented under `data.py`;
> §1 records how the 3-class strand ended and what carries back here). Earlier: 2026-07-22
> (full run complete — milestone 3 passed, see §1; label policy now v3: no `other` bucket,
> 24-entry merge map, GIRASOL dropped, MANGO/LIMON/MIXED_ORCHARD → MANGO_LIMON — see
> `labels.py` §3 below).
>
> **For the project-wide narrative — both strands, all the gates, with figures — read
> [`SUMMARY_FULL.md`](SUMMARY_FULL.md).**

## 1. Where the project stands

> **➡ 2026-08-11 (latest) — the tenure DiD was REOPENED with a different gate** (v3:
> measure the pre-trend and subtract it, rather than prove it negligible;
> `all_peru/tenure_did_plan.md` §9, `all_peru/RESULTS.md` §11). New code and interface
> changes:
>
> * **`allperu/tenure_did.py` gained the correction**: `amplification_factor` (**M**, derived
>   from window midpoints — never a constant), `corrected_effect`, `sensitivity_curve`,
>   `decision` (REPORTED / NOT-SEPARABLE), `bootstrap_covariance` (a diagnostic that never
>   enters the registered number), `registration` / `write_registration` (refuses to
>   overwrite), `run_corrected` / `print_corrected`. **`did()` gained `extra_post_fe`**, a
>   second interaction with POST, default `None` — R5's cohort time effects as an opt-in
>   robustness arm, never silently in the primary.
> * **New `allperu/did_sample.py`** — the DiD extraction draw: R1–R4 over the *full* national
>   table, whole 5 km regions, controls taken region-first so both arms share regions.
> * **`perennial/panel.py` gained `rebuild_year_stores` and `verify_years`**, and
>   `timing_probe` now takes `out=`. `verify_years` counts each year's pixel store against
>   that year's gate survivors and is the **only** acceptable completeness check;
>   `timing_probe`'s default destination is still the Piura panel's budget CSV, so a second
>   panel must pass its own path rather than overwrite the record of the first.
> * **`cross_sectional_contrast`** — the ⚠️ *descriptive* INSCRITO vs NO INSCRITO companion.
>   Returns a per-department table **always** (its sign is department-specific) and an
>   `IS_NOT_CAUSAL` field in the artifact. **`did()` now returns nan on a single cluster**
>   instead of raising a ZeroDivisionError from inside statsmodels' sandwich estimator.
> * New CLI: **`allperu did-sample`**, **`allperu tenure-register`**, **`allperu tenure-did2`**,
>   **`allperu tenure-xsec`**, and `perennial panel rebuild | verify`.
> * Tests: `tests/test_tenure_did.py` (31) and two new cases in `tests/test_coverage_years.py`.

> **⛔ 2026-08-11 (earlier) — the two-period tenure DiD ran and is NOT FUNDABLE (v2)**
> (`all_peru/tenure_did_plan.md` §8, `all_peru/RESULTS.md` §10). Stopped at N3, before any GEE
> spend; no estimate existed *at that point* and the locked test is untouched.
> **⚠️ Superseded by the v3 block above — the study was reopened and completed 2026-08-12.**
> New code, no existing interface changed:
>
> * **`allperu/tenure_did.py`** — `restrict` (R1–R4 + the N-D9 control), `window_means`,
>   `did` (parcel FE + window FE + department×post, region-clustered), `placebo` (the
>   equivalence gate, three outcomes), `integrity` (G2), `precision` (expected SE from measured
>   variance), `population_ceiling` (how many parcels could *ever* qualify),
>   `feasibility` (required vs available). New CLI: **`allperu tenure-did`** (exits 1 only on
>   G1 FAIL) and **`allperu tenure-ceiling`**.
> * ⚠️ **`did(..., dept_window_fe=True)` interacts department with the POST indicator**, not
>   with window. A dept×window dummy set is rank-deficient once the parcel FE is swept out and
>   returns **nan** standard errors rather than raising — pinned by a test.
> * **The method worth reusing:** `population_ceiling` + `precision` + `feasibility` decides
>   whether a study is possible *from the archive and measured variance*, in minutes, before
>   any extraction is funded.

> **2026-08-11 — the temporal-OOD plan ran in full (`all_peru/temporal_ood_plan.md` §6,
> `all_peru/RESULTS.md` §9). Interface changes that affect every strand:**
>
> * **`data.make_flat` / `make_dataset` take `augment_feat`** — a second feature table keyed
>   on the same `COD_PREDIO` whose rows are *appended* for the given index. In practice a
>   **degraded** copy of the store, assembled at endpoint observation density, so a parcel is
>   seen at two densities under one label. **Train side only**; `train.py`, `allperu/loyo.py`
>   and `allperu/lodo.py` all pass it through and never to val/test.
> * **`data.DROP_SETS` gained `order`** — the 33 order-statistic features
>   (`{channel}_{min,max,amp}`). ⚠️ Withholding them is what the density audit implies and
>   what LOYO **rejects**; the alias exists so the experiment is repeatable, not because it
>   should be used.
> * **`perennial/panel.py`: `panel_dirs(year, bundle_suffix="")` and
>   `infer_panel(..., bundle_suffix="")`** — score an *alternative* assembly of the same year
>   (`<panel>/<year>_qmap/`, `_oli/`, `_oliraw/`) without touching the audited bundle.
> * **New in `allperu/`:** `density.py` (the 1a audit, degradation, density-conditional
>   temperature, quantile alignment), `yearleak.py` (which features identify the label year),
>   `oli_overlap.py` (OLI extraction + the 3a overlap test + the split-half control),
>   `loyo.lodo_by_cohort` (**LODYO**). New CLI: `allperu density-audit | degrade |
>   density-calibrate | year-leak | lodyo | oli`.
> * **`allperu/loyo.py` now also reports `worst_cohort_macro_f1_min_support`** (cohorts with
>   ≥1000 parcels) alongside the registered ≥300 statistic — the registered minimum is decided
>   by a 464-parcel cohort where macro-F1 moves ±0.03 on resampling.
> * **⚠️ Selection rule extended: LOYO alone cannot select.** It holds region approximately
>   fixed, so a time-invariant feature is as exploitable there as in spatial CV
>   (`centroid_lat`: +0.0476 CV, +0.0474 LOYO, −0.0596 LODO). Use **LODYO** — free, it
>   re-scores existing LODO predictions per cohort. New primary:
>   `lightgbm_nometa_nolat_aug_yleak10`.
> * **⚠️ `perennial/harmonization.py` is measured and it is HARMFUL** — first ever run, and
>   worse than no correction on every axis (`all_peru/RESULTS.md` §9.3.2). It remains
>   implemented and unused, now with a number attached.
> * Tests: `tests/test_density.py`, `tests/test_temporal_ood.py`.

> **2026-08-10 — the window pivot ran and its three pre-gates FAILED; new modules and one
> change to `splits.py` that affects every strand.** Full account: `all_peru/RESULTS.md` §8,
> `all_peru/window_plan.md` §9. What matters for the code here:
>
> * **`splits.pick_test_units` now takes `balance_cols` + `n_candidates`.** It draws N
>   candidate whole-region test sets and keeps the one minimising total-variation distance
>   between test and trainval on the joint `year x label` histogram. `n_candidates=1` (the
>   default) reproduces the old draw exactly — pinned by a test — so no existing split moves.
>   The achieved TV distance is now written to `splits_meta.json` **always**, optimised or
>   not: an accidental temporal split is invisible unless someone records the number. New
>   configs `split_window.yaml` / `split_window_b3000.yaml`.
> * **New in `allperu/`:** `windows.py` (5-year window aggregation + the T3 diagnostic + the
>   tenure-error test), `loyo.py` (**leave-one-YEAR-out** — the temporal analogue of LODO,
>   and the evaluation this pipeline never had), `tenure.py` (`ESTADO en RRPP` on
>   `COD_PREDIO` for 1.78 M parcels, plus a second dated tenure observation from the
>   cadastre), `window_sample.py`, `estimate.py` (refuses to run while a gate is failed),
>   `external.py`, `export_crops.py`. Tests: `tests/test_window_pivot.py`.
> * **`cli.py`'s `--years` now accepts multi-range specs** (`1999-2003,2019-2023`) via
>   `data.parse_year_spec`, instead of a single `lo-hi`.
> * **Selection lesson, extended:** LODO showed spatial CV cannot see spatial memorisation;
>   LOYO now shows neither can see *temporal* transfer. Worst non-1998 cohort scores 0.406
>   against CV 0.581. Nothing in this pipeline's design ever held out a year — every parcel
>   carries one label year and train/val always span the same range.
> * **Next:** `all_peru/temporal_ood_plan.md` — **now RUN; see the 2026-08-11 block above
>   and RESULTS.md §9.** Step 1's acceptance failed, step 3 failed, and no 2019–23 estimate
>   is licensed.


> **2026-08-08 — a THIRD strand: all of Peru** (`src/crop_classifier/allperu/`, docs in
> [`all_peru/`](all_peru/)). The raw data grew from one department to 14 linkable ones.
> It reuses this pipeline through the same workspace switch plus the new `CC_FEAT`, and it
> produced the finding that most affects how models here should be *selected*:
> **`centroid_lat` is worth +0.047 macro-F1 on spatial CV and −0.060 on leave-one-
> department-out** — its contribution reverses sign when the held-out unit is a *place*.
> Spatial CV holds out 5 km cells inside departments the model has already seen, so it
> cannot separate memorisation from signal. See `all_peru/RESULTS.md` §6.2.
>
> Shared-code changes made there that apply here: the GEE hang fix, locality-packed chunking,
> `--train-years`, `metric_crs` in the split config, and `labels3.source_tables()`.

> **2026-08-04 — a second strand now shares this pipeline.** The 3-class
> perennial/annual/pasture classifier (`src/crop_classifier/perennial/`, plan +
> results in [`perennial/`](perennial/)) reuses every module here through an
> env-var workspace switch, `crop_classifier.paths` (`CC_PROC` / `CC_RUNS`). Consequences
> for anyone working on the 12-class pipeline:
>
> * `PROC`/`RUNS` module constants are **gone** — call `paths.proc()` / `paths.runs()` at
>   call time. The feature store is now **also switchable**: call `paths.feat()`, which reads
>   **`CC_FEAT`** and defaults to the shared Piura store. The 12-class and 3-class workspaces
>   deliberately share it (same parcels, different label column) and never set `CC_FEAT`;
>   the all-Peru strand (below) has different parcels and sets it. `landsat_gee.F_COVERAGE`
>   and `perennial.panel.PANEL_FEAT` became `f_coverage()` / `panel_feat()` for the same
>   reason — an import-time constant resolves before the env var can matter.
> * The **feature store has grown** to 5.65 M pixel-obs over 53,693 parcels (was 4.53 M /
>   47,851): the 3-class map admitted 6,113 parcels that had never been extracted. It is a
>   superset keyed on `COD_PREDIO`, so the 12-class runs are unaffected — but re-assembling
>   must pass statics for the **union** of both workspaces' parcel tables, or parcels
>   present only in the 12-class table get blank statics.
> * Two real bugs were fixed in shared code: `run_coverage` deduped on `COD_PREDIO` alone
>   (cross-year contamination — harmless for the single-year training store, fatal for a
>   panel) and `coverage_chunk` crashed on a year with no acquisitions. A **fourth model**
>   (`rules`, a phenology decision rule) is in the registry.
> * `missions_for_year` now takes an optional restriction and honours a module-level
>   `MISSION_FILTER`; the perennial panel sets it to `{"L5","L7"}`. Default behaviour
>   (all missions, incl. the newly added L9) is unchanged when the filter is `None`.
> * The **12-class locked test set is still unspent.** Never run `--eval-test` with
>   `CC_PROC` unset unless that is the intent. (The *3-class* test set has now been spent
>   **twice** — LightGBM, then tuned LTAE for interest — but that is a separate workspace and
>   does not touch the 12-class one.)
>
> **2026-08-05 — two findings from the 3-class strand that apply to this pipeline too:**
>
> * **`_retry` did not protect against hangs — NOW FIXED (2026-08-07).** The perennial panel
>   extraction stalled for 13.4 h with the process alive, zero output and no exception.
>   `socket.setdefaulttimeout(120)` and the 5-try backoff both failed to fire because
>   `_retry` only ever handled errors that *raise*, and **a hang is not an error**. Each GEE
>   call now runs on a worker thread under a 900 s wall-clock deadline
>   (`CHUNK_DEADLINE_S`); a blown deadline raises `ChunkTimeout`, which is
>   transient-classified and retried. Regression:
>   `tests/test_pipeline.py::TestGeeFaultTolerance`.
>   **Validated in production**: the all-Peru panel extraction hit these timeouts **25 times**
>   across five workers in one run (8/8/1/2/6, plus 96 ordinary transient network errors) and
>   lost nothing — every one would previously have been a permanent stall. (Supersedes the
>   "15 hangs / 110 errors" figure recorded mid-run.) ⚠️ **The first version of the fix had a
>   defect of its own**: it used a `ThreadPoolExecutor`, whose threads are non-daemon and are
>   *joined* by an `atexit` hook, so every timeout orphaned an unjoinable thread and all five
>   workers then sat at 0 % CPU for up to **3 h after their work was complete**
>   (`cancel_futures=True` does not help — it only drops futures still *queued*). Now
>   `_call_with_deadline` uses an explicit `threading.Thread(daemon=True)`; tests pin it. The
>   lesson: **the fix for a silent failure introduced a second silent failure one layer
>   down — verify a finished job by counting its output, never by "the process ended".**
> * **CV rank does not always match test rank.** Tuned LTAE beat LightGBM on all five spatial
>   CV folds (0.658 vs 0.647) yet lost the locked test (0.661 vs 0.681). `test − CV` was
>   +0.034 for LightGBM against +0.003 for LTAE — the higher-capacity model fit the trainval
>   regions better and transferred worse, and **spatial CV alone did not detect it**. Relevant
>   to any model selection made on CV in this pipeline.
>   **2026-08-09 — a third measurement of the same thing, and the sign flipped again**:
>   nationally LTAE *loses* CV to LightGBM (0.602 vs 0.628) **and** loses
>   leave-one-department-out (mean 0.442 vs 0.477), while the CV-best model
>   (`lightgbm_nometa`) is the LODO-*worst* (0.417). **CV rank, LODO rank and test rank are
>   three different orderings**; pick the estimator that matches the intended use before
>   comparing models, not after (`all_peru/RESULTS.md` §6.2c/§6.3).
>
> **⛔ 2026-08-06/07 — how the 3-class strand ended, and what carries back here.** Its 28-year
> panel was extracted, assembled and inferred, then **failed its validation gate** (temporal
> transfer and per-parcel flicker), so no trend was produced. Three lessons apply to *this*
> pipeline regardless:
>
> * **Audit what is in the feature matrix.** `make_flat` handed the model *everything* in the
>   store; acquisition metadata came along, and `frac_l7` — 2nd by gain — tracks the L5→L7
>   transition closely enough to manufacture a time trend from satellite availability alone.
>   Removing it cost **+0.0013** macro-F1. See `data.py` below for the `--drop-features` fix.
> * **Gain ≠ contribution, now measured twice** (29.8 % of gain worth 0.007 in the 12-class
>   mission audit; 10.89 % worth +0.0013 here). Do not read LightGBM gain as importance.
> * **1997–98 is a catastrophic El Niño, and the 12-class labels are ~80 % 1998–99 too.** The
>   flood collapses the spectral contrast the classes rest on (`perennial/RESULTS.md` §8.2).
>   Any conclusion here that leans on 1998 imagery inherits that exposure.

**FULL RUN COMPLETE (2026-07-22): all three models trained on the full extraction;
milestone 3 passed.** The full extraction ran clean (50,002 parcels' coverage, 1,219 pixel
chunks, 4.53M pixel-obs over 47,851 parcels, ~2.5 h, zero errors; gate pass 96.9%), and all
three models were trained under the buffered spatial-CV protocol. Pooled spatial-CV results
(38,535 val parcels; majority-class accuracy baseline 0.424, majority macro-F1 0.040):

| model | CV macro-F1 (folds) | pooled macro-F1 | bal. acc | acc | kappa |
|---|---|---|---|---|---|
| LightGBM | 0.311 ± 0.034 | 0.307 | 0.310 | 0.545 | 0.414 |
| LTAE | 0.347 ± 0.018 | 0.358 | 0.408 | 0.527 | 0.404 |
| PSE-LTAE | 0.348 ± 0.009 | 0.358 | 0.400 | 0.523 | 0.401 |

Every model clears the majority baseline. **LTAE/PSE-LTAE beat LightGBM on macro-F1 by
+0.05** — mostly by rescuing rare classes (ZARANDAJA F1 0.01→0.55, TRIGO 0.43→0.65,
CAÑA 0.10→0.18); LightGBM keeps a small accuracy edge on the majority classes. The
PSE→LTAE delta is ~0 (pixel-set encoder not yet earning its keep; it does halve fold
variance). Strong classes: ARROZ ~0.75, CAFE ~0.68–0.72, FALLOW ~0.50, PASTURE ~0.48.
Comparison figures: `runs/model_comparison_20260722/*.png`; per-run bundles
(confusion.png, reliability.png, per_class.csv, stratified.csv, torch fold curves) in
`runs/lightgbm_20260722_162635`, `runs/ltae_20260722_163303`, `runs/psetae_20260722_164933`
(pooled-CV evals via `preds_cv.parquet`). The **locked test set remains unspent**.

**Label space revised again after those runs (2026-07-22, v3):** GIRASOL dropped
(`drop_classes`) and MANGO/LIMON/MIXED_ORCHARD collapsed into **MANGO_LIMON** (`relabel`)
— the table is now **12 classes / 49,648 parcels** (§3 `labels.py`). The three 15-class
runs above are historical for the affected classes.

**LightGBM retrained on the 12-class space + Optuna sweep (2026-07-22 evening):**
default params → CV macro-F1 **0.368 ± 0.047**, pooled 0.374 / acc 0.569 / κ 0.44
(`runs/lightgbm_20260722_202427`). 30-trial sweep (3-fold objective,
`runs/sweep_lightgbm_20260722*`) → best `num_leaves=98, learning_rate=0.077,
min_child_samples=44`; refit on the full 5-fold protocol → **CV 0.379 ± 0.049, pooled
macro-F1 0.383 / acc 0.576** (`runs/lightgbm_12c_tuned`) — tuning is worth ~+0.01, the
label-space revision ~+0.06 (MANGO_LIMON F1 0.55 vs 0.29/0.14/0.33 for its parts).
Fold 0 is a persistent outlier (~0.29 vs ~0.40) with normal class/year mix — regional
difficulty, the honest spatial-CV error bar.
Torch training-loop correctness was audited: rising val loss alongside rising val
macro-F1 is confidence miscalibration under weighted CE; selection/checkpointing is on
val macro-F1, so saved models are unaffected (consider temperature scaling before
relying on confidence-based abstention).

**LTAE/PSE-LTAE retrained on the 12-class space + their Optuna sweeps (2026-07-30):**
defaults → LTAE CV **0.402 ± 0.036** (pooled 0.420, `runs/ltae_12c`), PSE-LTAE CV
**0.397 ± 0.030** (pooled 0.402, `runs/psetae_12c`). 30-trial sweeps (3-fold objective):
LTAE best `d_model=128, dropout=0.45, lr=4.5e-4` (0.420); PSE-LTAE best `d_model=128,
d_pix=32, dropout=0.30, lr=3.7e-3` (0.418) — `runs/sweep_ltae_20260730*`,
`runs/sweep_psetae_20260730*`. Tuned 5-fold refits — the current state of the art:

| model (12-class, tuned) | CV macro-F1 | pooled macro-F1 | pooled acc | κ | run |
|---|---|---|---|---|---|
| LightGBM | 0.379 ± 0.049 | 0.383 | 0.576 | 0.445 | `runs/lightgbm_12c_tuned` |
| **LTAE** | **0.409 ± 0.033** | **0.427** | 0.559 | 0.437 | `runs/ltae_12c_tuned` |
| PSE-LTAE | 0.409 ± 0.044 | 0.421 | 0.540 | 0.421 | `runs/psetae_12c_tuned` |

Same shape as the 15-class result: attention wins macro-F1 (+0.04) by rescuing rare
classes (ZARANDAJA 0.09→0.46/0.48, TRIGO 0.48→0.67); LightGBM keeps the accuracy edge
and is better on MAIZ (0.27 vs 0.23) and PASTURE (0.51 vs 0.45). PSE-LTAE ≈ LTAE still.
Sweep gains are small (~+0.007–0.02 pooled); both sweeps chose `d_model=128`.
Comparison figures: `runs/model_comparison_12c/*.png`. **Locked test set still unspent**
— LTAE is the leading candidate for the single test evaluation.

The upstream data work (dataset forensics, crop-label normalisation, the PETT
crop→polygon linkage) is done — see `CLAUDE.md` §§3–6. This pipeline starts from its
output, `data/processed/training_crop_polygon.parquet`.

## 2. The pipeline at a glance

```
training_crop_polygon.parquet (66k polygons, crops list, year)
        │
        ▼  labels.py ──────────── label policy (A2): keep crops, PASTURE/FALLOW as
        │                         land-cover, drop land_prep/unspecified, merge map for
        │                         intercrops, area 0.09–50 ha, year 1990–2020
        ▼
modeling_parcels.parquet (49,948 parcels, 15 classes)
        │
        ▼  splits.py ──────────── 1 km blocks + 5 km contiguous test/fold regions,
        │                         autocorrelation audit, locked 15% test, 5 CV folds,
        │                         1.5 km buffered dead-zones
        │
        ▼  features/landsat_gee.py (needs GEE auth)
        │     stage 1: coverage counts per parcel-year  → n_valid_obs, max_gap
        │     abstain gate (§7): n_valid_obs >= 4       → quality_ok
        │     stage 2: raw dated pixels, survivors only → pixels_<year>.parquet
        │
        ▼  features/assemble.py ─ offline, re-runnable without GEE
        │     features_lightgbm.parquet  (whole-year summaries + harmonics, no binning)
        │     tensor_perdate.npz         (LTAE:      X[N,64,11] + doy + mask)
        │     tensor_pixelset.npz        (PSE-LTAE:  X[N,64,8,11] + pixmask)
        │
        ▼  train.py ───────────── spatial CV → final refit → (optional) locked test, per
        │                         model: lightgbm | ltae | psetae
        ▼
evaluate.py (metrics/confusion/reliability/strata)   infer.py (predict + abstain)
```

Everything is driven from one CLI: `uv run python -m crop_classifier.cli --help`.

## 3. Module reference

### `labels.py` — label table (plan §7, decision A2)
`build()` reads the normalised polygon table and produces
`data/processed/modeling_parcels.parquet` + `label_map.json` + `label_exclusions.csv`.
Per parcel, `assign_raw_label(row, cfg)` returns `(label, reason)`:
single named crop → that crop; single pasture/fallow token → `PASTURE`/`FALLOW`
land-cover classes; `land_prep`/`unspecified` → dropped; multi-crop parcels → looked up in
the config `merge` map (key = sorted `+`-joined crop set, e.g. `CAFE+PLATANO: CAFE`),
otherwise dropped as `multicrop_unmerged`. **Label policy v2 (2026-07-22, user decision):**
`rare_policy: drop` — there is **no `other` bucket** any more; crops with fewer than
`min_class_parcels` (300) parcels are dropped (1,787 parcels). The merge map was extended
from 2 to **24 entries** (coffee-shade systems → CAFE, cacao systems → CACAO, orchard+annual
→ the perennial, `LIMON+MANGO` (±MAIZ) → a new **MIXED_ORCHARD** class, annual+annual → the
declared main crop, crop+fallow/land-prep token → the crop), keeping **4,936** intercropped
parcels (was ~1.6k). Note `CACAO` still fell below 300 (~103 merged + singles) and is
therefore dropped — lower `min_class_parcels` or extend the map if CACAO is wanted.
**Class surgery (v3, 2026-07-22):** after the merge map and before rare-class filtering,
`build()` applies two further config keys — `relabel` (class → class rename map; currently
`MANGO`/`LIMON`/`MIXED_ORCHARD` → **MANGO_LIMON**) and `drop_classes` (whole classes
excluded, tallied as `dropped_class`; currently **GIRASOL**, which had unusable spatial
support). Hard gates: area in [0.09, 50] ha (A5), year in [1990, 2020] (A1 — the titling
year is trusted as-is). `apply_coverage_gate(coverage)` is called after stage-1
extraction: joins `n_valid_obs`/`max_gap` and sets `quality_ok = n_valid_obs >= 4`
(NA = not yet measured). Config: `config/data.yaml`.

**Current output: 49,648 parcels, 12 classes** (ARROZ 21,564 · FALLOW 8,734 · MAIZ 5,809 ·
ALGODON 3,235 · CAFE 2,859 · MANGO_LIMON 2,298 · PASTURE 2,210 · TRIGO 1,162 · FRIJOL 617
· PLATANO 493 · CAÑA DE AZUCAR 364 · ZARANDAJA 303). Splits re-assigned: test = 9,145
parcels in 58 regions; coverage gate 48,106 pass / 1,542 fail / 0 unmeasured (the full
extraction already covers every parcel — label changes need **no** new GEE work and no
re-assembly, only retraining). MANGO_LIMON spans 119 regions (849 test parcels) — much
healthier spatial support than its components had. **The 2026-07-22 model runs predate
this 12-class map** — see §1.

### `splits.py` — spatial split (plan §5, decision A2)
`assign()` adds to the parcel table: `block_id` (1 km grid, EPSG:32717), `region_id`
(5 km grid), `split` (`test`/`trainval`), `fold` (0–4), and the buffer-exclusion flags.
Design points:

* **Test and CV folds are held out as contiguous 5 km regions**, not scattered 1 km
  blocks. The buffered dead-zone costs training data in proportion to held-out
  *perimeter*; with 329 scattered 1 km test blocks the 1.5 km buffer excluded 33,245 of
  48,289 parcels (69%) from training. With 51 contiguous regions the same buffer excludes
  **8,015** — final training pool ≈ 32.2k parcels.
* **Autocorrelation audit** (`autocorrelation_audit`): pairwise crop-agreement vs
  distance. Current result: 0.75 agreement under 100 m, still 0.37 at 4–5 km vs a 0.24
  random baseline → the decorrelation range exceeds 5 km. Since `buffer_m` = 1500 < that
  range, the code prints a WARNING: **residual neighbour leakage near region borders is a
  known, accepted limitation** (a wider buffer was considered and deliberately not taken —
  it trades training data for purity; revisit if test ≫ CV scores).
* Locked test ≈ 15% of parcels (whole regions; every class forced into test where
  possible without emptying it from trainval). `StratifiedGroupKFold` over regions gives
  5 folds; per-fold buffer flags (`buffer_excl_fold{k}`) keep each fold's training set
  away from its validation regions.
* Writes `splits_meta.json` (audit + settings + exclusion counts) and
  `class_block_counts.csv` (per-class evaluability report).

* **`metric_crs` is configurable** (default 32717, UTM 17S). Piura fits inside one UTM zone;
  Peru spans 17S–19S and block/region ids must come from a *single continuous grid*, so
  `config/split_allperu.yaml` projects the whole country to UTM 18S (scale error < 0.7 %,
  i.e. < 11 m on the 1.5 km buffer — fine for gridding, not for area estimation).

Config: `config/split.yaml` (or `split_allperu.yaml`). Re-running is cheap and safe (rewrites
the split columns, preserves coverage/quality columns).

### `features/landsat_gee.py` — two-stage GEE extraction (plan §6, decision A4)
Landsat Collection-2 L2, missions by year (L5 ≤2013, L7 ≥1999, L8 ≥2013), bands
harmonised to `B,G,R,NIR,SWIR1,SWIR2`; QA_PIXEL bits 1–4 + QA_RADSAT masked.

* **Stage 1 `run_coverage()`** — cheap `reduceRegions` pass per chunk: per-parcel count
  of clear acquisitions in its crop year (`n_valid_obs`) and longest run of empty months
  (`max_gap`). Feeds the abstain gate.
* **Stage 2 `run_pixels()`** — for gate survivors only: every clear pixel observation
  (raw DN + pixel lon/lat + DOY + mission) → `pixels_<year>.parquet`. This raw store is
  extracted **once**; both model representations are re-assembled from it offline.

Robustness (all unit-tested without GEE):
* chunks are **resumable** and **content-addressed** (file name = hash of the chunk's
  parcel IDs, so resuming stays correct if the parcel set/order/chunk size changes);
* `_retry()` backs off exponentially on transient errors (rate limits, 5xx, socket
  timeouts); deterministic "computation too big" errors skip retries and `_run_chunk()`
  recursively halves the chunk;
* **`_retry()` also enforces a wall-clock deadline** (`CHUNK_DEADLINE_S`, 900 s) by running
  each call on a worker thread. This is the durable fix for the **silent GEE hang** measured
  repeatedly in the Piura panel work (once for 13.4 h): a hang raises nothing, so the
  backoff above never fired because it only ever saw *exceptions*. A blown deadline now
  raises `ChunkTimeout`, which is transient-classified and retried like any other error.
  It fired **25 times** in the national panel extraction with zero workers lost.
  ⚠️ **The worker thread must be a daemon**, which is why `_call_with_deadline()` hand-rolls
  a `threading.Thread(daemon=True)` instead of using a `ThreadPoolExecutor`. Executor threads
  are non-daemon and `concurrent.futures.thread` *joins* them via `atexit`; a thread parked in
  a hung GEE read is never joinable, so the first version of this fix left every national
  panel worker alive at 0 % CPU for up to **3 h after its extraction was complete**.
  `shutdown(wait=False, cancel_futures=True)` does **not** avoid this — `cancel_futures` only
  drops futures still *queued*, never one already running. Pinned by
  `TestGeeFaultTolerance::test_abandoned_thread_is_daemon_so_the_process_can_exit`;
* **`_chunk_todo()` packs chunks for locality.** Grouping by year alone was fine for one
  compact department but spans **up to 110 deg²** nationally, because a single label year
  draws parcels from 14 departments. Departments are now visited in longitude order and
  packed greedily, cutting a chunk whenever the next would push its bbox past
  `MAX_CHUNK_BBOX_DEG2` (4 deg²) — 240 chunks at a 0.53 deg² median for all-Peru, and a
  no-op for a single-department table;
* the per-year combine dedupes on `(COD_PREDIO, doy, lon, lat, mission)`.

**Concurrency.** Because chunk names are content-addressed and a worker skips any chunk that
already exists, extraction can be split across several processes on **disjoint year ranges**
(`--years`), which is latency-bound rather than quota-bound: the all-Peru run went from 3.5
to a peak of 12 chunks/min on five workers, with zero errors and no torn files. The year
ranges must genuinely be disjoint — a process covering *all* years duplicates the others'
work rather than dividing it.

Requires `uv run earthengine authenticate` and GCP project `peru-crop-classifier`.

### `features/indices.py` — spectral channels
`scale_sr()` (Collection-2 scale factors), `add_indices()` → NDVI, EVI, NDWI, NDMI, BSI.
`CHANNELS` = 6 bands + 5 indices = 11; the channel order is shared by every model input.

### `features/assemble.py` — raw pixels → model inputs (plan §6, decision A3)
No binning, no interpolation, per A3. From the pixel store it builds:

* `features_lightgbm.parquet` — one row per parcel, 141 columns: per channel
  median/mean/std/min/max/p25/p75/amplitude + linear slope + **order-1 harmonic fit**
  (`h_mean, h_cos, h_sin` — encodes phenology peak timing without a time grid; NaN when
  under 4 observations, LightGBM handles NaN natively) + statics (area, lat,
  `n_valid_obs`, `max_gap`, L7 fraction).
* `tensor_perdate.npz` — LTAE input: per-date parcel-median sequences `X[N, T=64, 11]`
  with `doy` and validity `mask` (real irregular dates, padded).
* `tensor_pixelset.npz` — PSE-LTAE input: `X[N, 64, P=8, 11]` keeping the 8
  most-observed 30 m pixels per parcel, plus `pixmask`.
* `feature_meta.parquet` — per-parcel `n_dates`, `n_valid_pixels`.

### `data.py` — datasets and split selection
The single source of truth for "who trains on what": `fold_split(df, k)` and
`final_split(df)` apply the buffer flags; `make_dataset(kind, …)` returns `FlatData`
(LightGBM) or `SeqDataset` (sequence/pixelset). Channel normalisation (`Normalizer`) is
fit on the **train subset only** and stored inside the saved model. Import discipline:
this module never imports torch at module level (see §6 gotcha).

**Feature exclusion (added 2026-08-06, flat models only).** `make_flat` used to drop
exactly `COD_PREDIO`/`label_id` and hand LightGBM *everything else in the store* — which
silently included acquisition metadata. Two knobs now:

* `drop_features=` — column names, or the group aliases in `DROP_SETS`: **`meta`**
  (`META_FEATURES` = `frac_l7, n_valid_obs, n_dates, max_gap, n_valid_pixels`) and
  **`location`** (`centroid_lat`). Resolved by `resolve_drop_features()`, which accepts a
  comma string or a list and de-duplicates.
* `feature_names=` — score on *exactly* this column list, in this order, raising `KeyError`
  if the store lacks one. `infer()` passes the model's saved `feature_names`, so an ablated
  model cannot silently regain a withheld column from a store that still contains it.

Why it exists: on the 3-class strand `frac_l7` was the 2nd-highest-gain feature and tracks
the L5→L7 mission transition almost perfectly, so it would have manufactured a time trend
out of satellite availability. See `perennial/RESULTS.md` §4.6. Removing it cost **+0.0013**
macro-F1 — the third measurement in this repo that **gain ≠ contribution**.

### `models/` — the three-model ladder (plan §8)
All models share one interface (`fit(train, val, class_weight, run_dir)`,
`predict_proba(ds)`, `save`/`load`) and a lazy registry (`models/base.py` maps names to
modules without importing them — `get_model("lightgbm")` never touches torch).

* `trees.py` — **LightGBM** baseline (rung 1, `input_kind="flat"`). Uses the native
  `lgb.train` API with explicit `num_class` over the *full* class space, so a spatial
  fold that lacks a class still yields full-width probability vectors.
* `torch_common.py` — shared torch loop: AdamW, class-weighted cross-entropy, early
  stopping on val macro-F1, curves.csv/png, device via `device.pick_device()`
  (cuda → mps → cpu, per A6 — same code runs on the M3 Air and JASMIN Orchid). Non-finite
  losses/gradients are skipped rather than allowed to poison the weights.
* `ltae.py` — **LTAE** (rung 2, `sequence`): masked master-query temporal attention over
  the per-date sequences, sinusoidal DOY encoding; fully-masked rows are nan-guarded.
* `psetae.py` — **PSE-LTAE** (rung 3, `pixelset`): a pixel-set encoder (masked mean+std
  pooling over clear pixels per date) feeding the *same* `LTAECore`, so the rung-2→3
  delta isolates the value of the pixel-set encoder. Note the eps-inside-sqrt in the
  variance pooling — removing it reintroduces NaN gradients (regression-tested).

### `train.py`, `evaluate.py`, `infer.py`
* `train(model_name, …)` — spatial CV over the folds → final refit on
  trainval-minus-buffer → optional **single** locked-test evaluation (`--eval-test`;
  plan §9 discipline: never used for tuning). `sweep()` runs an Optuna search on CV
  macro-F1 only. Runs land in `runs/<name>/` (model.bin, cv_metrics.json, label_map,
  predictions, curves). `drop_features=` is threaded to every `make_dataset` call site and
  recorded in `cv_metrics.json`, so a run always states what it was denied.
* **`train_years=` / `--train-years` (added 2026-08-07)** restricts the **training** cohort
  to given PETT label years (`"1999-2023"`, or a comma list; `data.parse_year_spec` +
  `restrict_years`), in both the CV folds and the final refit. **Validation and locked-test
  membership are deliberately untouched**, so CV stays directly comparable to a run without
  the flag — that comparability is the whole point. Recorded in `cv_metrics.json`.
  Motivation and results: `perennial/RESULTS.md` §8.3 (Piura, El Niño cohort) and
  `all_peru/RESULTS.md` §5.2 (nationally the same restriction *costs* accuracy — the years
  worth excluding are regional). **Always pair it with a truncation control**: re-gate the
  *baseline* model's predictions over the same years, or an improvement cannot be attributed
  to the retrain rather than to the shorter series.
  `model_kw=` reaches the constructor and is **not** exposed on the CLI — pass it from
  Python for model-variant experiments (e.g. the `rules` feature swap in
  `perennial/RESULTS.md` §4.1).
* `evaluate.full_report(run_dir)` — plan §11 bundle: macro-F1 (primary), weighted-F1,
  balanced accuracy, kappa, majority baseline, per-class table, row-normalised confusion
  matrix, reliability curve + Brier, and macro-F1 stratified by area / n_valid_obs /
  max_gap / year.
* `infer.infer(run_dir, …)` — batch prediction over any parcel table with **abstention**:
  reasons `coverage_unmeasured` (never extracted), `quality_gate` (failed n≥4),
  `no_features`, `low_confidence` (below `--tau`). Never silently drops a parcel. Pins the
  scored columns to the model's saved `feature_names` (see `data.py` above).

### `cli.py` — entry points
```
uv run python -m crop_classifier.cli labels build
uv run python -m crop_classifier.cli splits assign
uv run python -m crop_classifier.cli features extract [--stage coverage|pixels|all]
                                     [--max-chunks N] [--years 1998,1999]
uv run python -m crop_classifier.cli features assemble [--t-max 64] [--p-max 8]
uv run python -m crop_classifier.cli train --model lightgbm|ltae|psetae [--eval-test]
                                     [--run-name NAME] [--drop-features meta|location|<cols>]
uv run python -m crop_classifier.cli sweep <model> [--trials 30]
uv run python -m crop_classifier.cli eval <run_dir>
uv run python -m crop_classifier.cli infer <run_dir> [--polygons f.parquet] [--tau 0.5]
```

### `allperu/` — the national extension (docs/all_peru)
Added 2026-08-07/08. Same Chain-A join and the same modelling recipe, run over every
department that has all three sources, then sampled back to Piura scale so cost is unchanged.

* `sources.py` — department registry: `departments()` returns the 15 linkable departments
  (bridge ∩ polygons ∩ SSET); `unlinkable()` is the exclusion audit trail.
  **`shapefile_view()` is load-bearing**: 19 of 24 departments ship their attribute table
  under the wrong basename (`QGIS/ANCASH/ANCASH.dbf`), and GDAL then opens the `.shp` and
  returns *zero columns*, so `COD_PREDIO` vanishes with no error. It symlinks a consistent
  basename; raw data is never touched.
* `build_labels.py` — Chain A per department. **`canon_key()`** strips leading zeros because
  bridge keys are zero-padded to 9 chars while BD SSET's are not — a naive join returns
  *zero* rows and reads as "no data for this department" (Ancash: 0 → 369,089 keys).
  **`clean_geometry()`** forces 2D: La Libertad's cadastre is 3D and Earth Engine rejects 3D
  GeoJSON outright. **`build_dept_sset_caches()`** scans *every* workbook, because a
  department's rows are not confined to the one named after it (Lima's span three).
* `sample.py` — down-samples to Piura scale by whole 5 km regions, departments allocated
  sqrt-proportionally with a floor and a per-region cap. Writes **`population_weight`**
  (deliberately *not* `sample_weight`, which `perennial/panel.py` computes for a different
  stratification; `build_panel` multiplies the two so a panel weight expands the whole way
  to the national population).
* `lodo.py` — **leave-one-department-out**, the evaluation Piura could not run. Same 1.5 km
  buffer at department borders; the early-stopping validation set is carved from the
  *training* departments, never the held-out one. This is the estimator the national model
  is selected on (`all_peru/RESULTS.md` §6). **`--tag` (added 2026-08-09) is required when
  running a second architecture**: without it the command writes fixed filenames
  (`lodo_metrics.csv`, `lodo_predictions.parquet`, `lodo_summary.json`) and per-department
  fits to `runs/lodo/<DEPT>`, so a second run silently overwrites the first — and comparing
  two models' transfer is the only reason to run a second one. With a tag, artifacts are
  suffixed and fits nest under `runs/lodo/<tag>/`. Pinned by
  `tests/test_allperu.py::TestLodo::test_tag_keeps_two_architectures_from_overwriting_each_other`.

### `config/`
* `data.yaml` — label policy: kept/dropped categories, land-cover classes, the intercrop
  `merge` map (extend here to recover more multi-crop parcels), `min_class_parcels`,
  area/year gates, `min_valid_obs` (the abstain gate).
* `split.yaml` — `block_km`, `region_km`, `buffer_m`, `n_folds`, `test_frac`, seed, audit
  settings. `split_allperu.yaml` adds `metric_crs: 32718` (one grid for 3 UTM zones) and
  `test_frac: 0.20`.
* `perennial.yaml` / `perennial_allperu.yaml` — the 3-class lexicon. The national one is
  **additive only** (no Piura assignment changes) and adds three mechanisms without which
  19.23 % of national records fell to a blanket `ANNUAL` guess: sierra/selva tokens,
  `stage_words` (registrars wrote `MAIZ EN FLORACION` — a *productive* pattern that cannot
  be enumerated), and `word_match` (resolve a phrase's individual words, then combine by
  `group_priority`). Final: **1.92 %**, under the 2 % budget the build enforces.

### `device.py`
`pick_device(prefer=None)` (cuda → mps → cpu; raises if a preferred device is
unavailable) and `seed_everything()`. Sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so ops missing
on MPS fall back to CPU instead of crashing.

## 4. Artifacts inventory (`data/processed/`)

| file | producer | contents |
|---|---|---|
| `modeling_parcels.parquet` | labels + splits + gate | 49,648 rows: geometry, `label`/`label_id`, area, year, centroids, `block_id`/`region_id`/`split`/`fold`, buffer flags, `n_valid_obs`/`max_gap`/`quality_ok` |
| `label_map.json`, `label_exclusions.csv` | labels | class↔id map; per-reason exclusion counts |
| `splits_meta.json`, `class_block_counts.csv` | splits | audit curve + settings + exclusion counts; per-class evaluability |
| `features/coverage.parquet` (+`coverage_chunks/`) | stage 1 | per-parcel `n_valid_obs`, `max_gap` |
| `features/pixels_<year>.parquet` (+`pixels_chunks/`) | stage 2 | raw store: one row per clear pixel observation |
| `features/features_lightgbm.parquet` | assemble | 141-col summary features |
| `features/tensor_perdate.npz`, `tensor_pixelset.npz` | assemble | LTAE / PSE-LTAE tensors |
| `runs/<name>/` | train | model.bin, cv_metrics.json, preds, curves, report bundle |

`data/` is gitignored; all of the above regenerate from the raw data + GEE.

## 5. Running at three scales

1. **Smoke (minutes, no new GEE work)** — the current feature store already holds the
   ~383-parcel extraction. `features assemble` → `train --model lightgbm` → `infer`
   exercises everything offline. `uv run pytest tests/` (47 tests) covers the logic that
   doesn't need data.
2. **Medium (hours)** — a deliberate subset to de-risk the full run, e.g.:
   `features extract --years 1998 --max-chunks 20` (measures the L5-only 1998 attrition —
   see §7) plus `--years 1999 --max-chunks 20`, then `assemble`, then
   `train --model lightgbm` (no `--eval-test`). Partial runs are first-class: chunks are
   resumable, and the coverage gate + per-year combine always run on whatever is on disk.
   Use this to estimate full-run wall-clock and check per-year gate pass rates.
   **Done 2026-07-21** (run `runs/lightgbm_20260721_132715`): 80 chunks in 6m44s
   (~7 s/chunk both stages) → full extraction projected ≈ 2.5 h (stage 1 remainder ~80
   chunks ≈ 9 min; stage 2 ~1,150 chunks ≈ 2.4 h). Gate pass 95–96% every year (§7).
   1,921 parcels with features; pooled-CV LightGBM: macro-F1 0.102, accuracy 0.372 vs
   majority baseline 0.393, kappa 0.13 — real but sub-baseline signal at ~4% of data
   (FALLOW F1 0.58, ARROZ 0.34; rare classes 0). Evaluate CV without touching the test
   set by pooling `fold*/preds_val.parquet` into `preds_cv.parquet` and running
   `eval <run> --preds preds_cv.parquet`.
3. **Full** — `features extract` with no limits: stage 1 over all 48k (~120 chunks),
   review attrition, stage 2 over all survivors (~800 chunks, several hours), `assemble`,
   then `train --model lightgbm` first. Per plan §13, the LTAE/PSE-LTAE work is **gated
   on LightGBM beating the majority-class baseline** on spatial CV. `--eval-test` is
   spent once, at the very end, on the chosen model.

## 6. Gotchas

* **Never import torch and lightgbm in one process on macOS** — each bundles its own
  libomp and co-loading segfaults (exit 139). The codebase is structured around this
  (lazy model registry, torch imported only inside `SeqDataset`); keep it that way when
  adding code. The test suite runs LightGBM training in a subprocess for this reason.
* Always `uv run …` (project venv, Python 3.11). GEE needs
  `earthengine authenticate` + a GCP project id (hard-coded default
  `peru-crop-classifier` in `landsat_gee.py`).
* `quality_ok` is a **nullable** boolean (NA = unmeasured). Compare with `== True` /
  `.isna()`, never truthiness; `np.select` needs explicit numpy bool masks (see
  `infer.py`).
* Geometry CRS: parcels are stored EPSG:4326; metric operations (blocks, buffers) go
  through EPSG:32717.
* Chunk stores under `features/*_chunks/` are append-only caches; deleting a chunk file
  just causes recomputation. The old positional-name smoke chunks coexist with the new
  hash-named ones (combine dedupes).

## 7. Known limitations & open decisions

* **Residual spatial leakage (accepted):** crop agreement stays above baseline past 5 km
  while the buffer is 1.5 km. Contiguous 5 km test regions confine the leak to region
  borders, but CV/test scores are still slightly optimistic. Decision on record: keep
  `buffer_m=1500`; revisit if locked-test ≪ CV.
* **Unevaluable rare classes:** GIRASOL was **resolved 2026-07-22** — dropped via
  `drop_classes` (its CV F1 was ~0 for all three models). Still weak: ZARANDAJA has only
  5 test parcels (4 test regions) and CAFE's test lives in 4 regions — their per-class
  test numbers will be anecdotal; report them as "insufficient spatial support" (though
  note the attention models reached ZARANDAJA CV F1 ≈ 0.5, so it is learnable).
* **1998 attrition — measured (medium run, 2026-07-21):** the gate is NOT the problem —
  1998 passes at 95.4% (7,628/8,000 measured), same as 1999's 96.1%. But the L5-only year
  is data-thin: median 5 clear obs vs 8 for 1999, ~4× fewer pixel-obs per parcel, and CV
  accuracy on 1998 parcels was 0.13 vs 0.47 for 1999 (confounded with fold geography and
  the small 1998 feature count — recheck at full scale). Expect 1998 to contribute weak,
  not missing, signal. Parcels with a null year are never extracted and abstain as
  `coverage_unmeasured`.
* **Defaults chosen without tuning:** `T_MAX=64`, `P_MAX=8`, `MIN_HARMONIC_OBS=4`,
  LightGBM/LTAE hyper-parameters. All are config/CLI-adjustable; the Optuna `sweep`
  exists for when real data arrives.
* **The `merge` map was extended 2026-07-22** to 24 entries (see §3 `labels.py`), keeping
  4,936 intercropped parcels; 4,577 multi-crop parcels remain unmerged/excluded. Further
  extension in `data.yaml` is still the cheapest way to grow the training set. `CACAO` was
  an intended merge target but fell below `min_class_parcels` and is dropped.
* ~~No model has been trained on real data yet~~ **Milestone 3 passed 2026-07-22** (see
  §1). ~~Optuna sweeps on the leading model~~ **all three models retrained + swept on
  the 12-class space 2026-07-30** (§1 table; tuned LTAE leads at pooled macro-F1 0.427).
  Next: the `max_gap` training-filter ablation, optionally temperature scaling for the
  abstain gate, then the single locked-test evaluation of the chosen model (LTAE).
