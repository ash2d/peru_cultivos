# Pipeline — code guide

What each module does and how to run it. **What has been run and what it produced** is in
[`RESULTS.md`](RESULTS.md); **what to do next** is in [`STATUS.md`](STATUS.md); the data it
consumes is in [`DATA.md`](DATA.md).

---

## 1. At a glance

```
training_crop_polygon.parquet  (polygons + crops list + year)
        │
        ▼  labels.py ─────────── label policy: keep crops, PASTURE/FALLOW as land-cover,
        │                        drop land_prep/unspecified, merge map for intercrops,
        │                        area 0.09–50 ha, year 1990–2020
        ▼
modeling_parcels.parquet
        │
        ▼  splits.py ─────────── 1 km blocks + 5 km contiguous test/fold regions,
        │                        autocorrelation audit, locked test, 5 CV folds,
        │                        1.5 km buffered dead-zones
        │
        ▼  features/landsat_gee.py            (needs GEE auth)
        │     stage 1  coverage counts per parcel-year → n_valid_obs, max_gap
        │     abstain gate  n_valid_obs >= 4           → quality_ok
        │     stage 2  raw dated pixels, survivors only → pixels_<year>.parquet
        │
        ▼  features/assemble.py ─ offline, re-runnable without GEE
        │     features_lightgbm.parquet   whole-year summaries + harmonics
        │     tensor_perdate.npz          LTAE      X[N,64,11] + doy + mask
        │     tensor_pixelset.npz         PSE-LTAE  X[N,64,8,11] + pixmask
        │
        ▼  train.py ──────────── spatial CV → final refit → (optional) locked test
        ▼
evaluate.py (metrics/confusion/reliability)     infer.py (predict + abstain)
```

Everything runs from one CLI: `uv run python -m crop_classifier.cli --help`.

---

## 2. Workspaces — set these before anything else

The same code serves four datasets through three environment variables (`paths.py`). Getting
them wrong silently reads or writes the wrong store.

| workspace | `CC_PROC` | `CC_FEAT` | `CC_RUNS` |
|---|---|---|---|
| 12-class Piura (closed) | *unset* | *unset* | *unset* |
| 3-class Piura | `data/processed/perennial` | *shared* | `runs/perennial` |
| **all-Peru** ⭐ | `data/processed/all_peru` | `data/processed/all_peru/features` | `runs/all_peru` |
| all-Peru population | `data/processed/all_peru_full` | — | — |
| tenure DiD | `data/processed/all_peru_did` | `…/features` | `runs/all_peru` |
| S2 labelling | `data/processed/all_peru` | `data/processed/all_peru/features_s2` | `runs/all_peru` |

⚠️ `CC_FEAT` exists because the Piura pixel store is **no longer implicitly shared** with the
national one.

---

## 3. Command reference

`uv run python -m crop_classifier.cli <group> <command>`. ⛔ marks commands whose estimand was
abandoned — the code is built and unit-tested, and it stays unrun (`RESULTS.md` §9).

### Core

| command | what it does |
|---|---|
| `labels build` | label table: category policy + merge map + area/year gates |
| `splits assign` | 1 km blocks, autocorrelation audit, locked test, CV folds, buffers |
| `features extract [--stage coverage\|pixels\|all] [--years A-B] [--max-chunks N]` | the two-stage GEE extraction |
| `features assemble [--t-max 64] [--p-max 8]` | pixel store → LightGBM features + tensors |
| `train --model lightgbm\|ltae\|psetae [--drop-features meta,location] [--train-years A-B] [--eval-test]` | spatial CV + final refit |
| `sweep <model> [--trials 30]` | Optuna on CV macro-F1; never touches the locked test |
| `eval <run_dir> [--preds preds_cv.parquet]` | the full report bundle |
| `infer <run_dir> [--polygons f.parquet] [--tau 0.5]` | batch prediction with abstention |

### `perennial` — the 3-class strand

| command | what it does |
|---|---|
| `perennial labels` | 3-class label table (needs `CC_PROC`) |
| `perennial panel build\|probe\|extract\|rebuild\|verify\|assemble\|infer` | the multi-year panel; `extract` is a 20+ h resumable job |
| `perennial diagnostics` | **the Phase-7 gate**: S4, S5, sensor drift, El Niño. Exits non-zero on failure |
| `perennial mapbiomas` | extract the MapBiomas benchmark panel |
| `perennial compare <runs> --out` | pooled CV comparison table + figures |
| `perennial calibrate <run>` | temperature scaling on held-out validation probabilities |
| `perennial pool-cv <run>` | pool `fold*/preds_val.parquet` → `preds_cv.parquet` |

### `allperu` — the national extension

| command | what it does |
|---|---|
| `allperu inventory` | what links to what, and why 9 departments are excluded |
| `allperu labels [--only DEPT]` | Chain-A build over every linkable department |
| `allperu sample --source` | down-sample to Piura scale by whole regions |
| `allperu lodo [--tag T] [--model M] [--drop-features …]` | **leave-one-department-out** — the selection estimator. Runs LODYO automatically at the end |
| `allperu loyo` | leave-one-**year**-out cohort transfer |
| `allperu lodyo --tag T` | re-score an existing LODO run per year cohort — department **and** year out at once. Free |
| `allperu tenure` | `ESTADO en RRPP` per parcel → `tenure_by_predio.parquet` |
| `allperu tenure-ceiling` | **feasibility**: how many parcels could ever enter a DiD. Run before funding anything |
| `allperu did-sample` / `tenure-register` / `tenure-did2` | the v3 DiD: draw → register → placebo-then-headline |
| `allperu tenure-xsec` | ⚠️ descriptive cross-sectional companion, **not an estimate** |
| `allperu tenure-error` | is classifier error non-differential in tenure? |
| `allperu density-audit` / `degrade` / `density-calibrate` | observation-density robustness |
| `allperu year-leak` | which features identify the label year? |
| `allperu export-status` | how much of `PERENNIAL` is really export; woody non-crop bound |
| `allperu esri-dates` / `label-budget` | imagery-date probe; campaign learning curve |
| **`allperu s2-labels <step>`** ⭐ | the live campaign — steps below |
| ⛔ `allperu windows` | the 5-year window diagnostic (T3) — pivot closed |
| ⛔ `allperu window-sample` | stratified window sample — pivot closed |
| ⛔ `allperu estimate` | the window-estimand tenure contrast — refuses to run unless T1–T3 passed |
| ⛔ `allperu external` | SIEA rank comparison |
| ⛔ `allperu oli` / `oli-refit` | OLI harmonisation — closed twice, `RESULTS.md` §6.4 |
| ⛔ `allperu tenure-did` | the **v2** DiD; superseded by `tenure-did2` |

`allperu s2-labels` steps: `universe | pool | probe | draw | split | chips | extract |
harmonisation | html | ingest | transitions`.

---

## 4. Module reference

### `labels.py`
`build()` → `modeling_parcels.parquet` + `label_map.json` + `label_exclusions.csv`. Per parcel,
`assign_raw_label(row, cfg)` returns `(label, reason)`: a single named crop → that crop; a
single pasture/fallow token → the land-cover class; `land_prep`/`unspecified` → dropped;
multi-crop → looked up in the config `merge` map (key = sorted `+`-joined set, e.g.
`CAFE+PLATANO: CAFE`), otherwise dropped as `multicrop_unmerged`.

`rare_policy: drop` — there is **no `other` bucket**; classes under `min_class_parcels` are
dropped. `relabel` renames classes, `drop_classes` excludes them wholesale. Hard gates: area
in [0.09, 50] ha, year in [1990, 2020]. `apply_coverage_gate(coverage)` joins
`n_valid_obs`/`max_gap` after stage 1 and sets `quality_ok = n_valid_obs >= 4`.
Config: `config/data.yaml`.

### `splits.py`
`assign()` adds `block_id` (1 km), `region_id` (5 km), `split`, `fold`, and buffer-exclusion
flags; writes `splits_meta.json` and `class_block_counts.csv`.

* **Test and CV folds are contiguous 5 km regions, not scattered blocks.** The buffered
  dead-zone costs training data in proportion to held-out *perimeter*: 329 scattered 1 km test
  blocks excluded 69 % of parcels from training; 51 contiguous regions exclude 8,015.
* **`autocorrelation_audit`** measures label agreement against distance and **prints a warning**
  because the decorrelation range exceeds the 1.5 km buffer (`DATA.md` §5). Residual neighbour
  leakage near region borders is a known, accepted limitation.
* **`metric_crs` is configurable** — Piura fits one UTM zone; the national config uses 32718.

Re-running is cheap and safe (rewrites split columns, preserves coverage/quality columns).

### `features/landsat_gee.py` — two-stage GEE extraction
Landsat Collection-2 L2, missions by year, bands harmonised to `B,G,R,NIR,SWIR1,SWIR2`,
QA_PIXEL bits 1–4 + QA_RADSAT masked.

* **Stage 1 `run_coverage()`** — cheap per-chunk `reduceRegions`: clear-acquisition count in the
  parcel's crop year (`n_valid_obs`) and longest empty-month run (`max_gap`). Feeds the gate.
* **Stage 2 `run_pixels()`** — for gate survivors only: every clear pixel observation → 
  `pixels_<year>.parquet`. Extracted **once**; both model representations re-assemble from it
  offline.

Robustness (all unit-tested without GEE):

* Chunks are **resumable** and **content-addressed** (filename = hash of the chunk's parcel
  IDs), so resuming stays correct if the parcel set, order or chunk size changes.
  ⚠️ **That also means a change to what a chunk *computes* is invisible to the cache** — adding
  columns requires invalidating it by hand.
* `_retry()` backs off on transient errors; deterministic "computation too big" errors skip
  retries and `_run_chunk()` recursively halves the chunk. **GEE throttle messages are
  transient-classified** — they arrive both as error strings *and* as silent hangs.
* **`_retry()` enforces a 900 s wall-clock deadline** (`CHUNK_DEADLINE_S`) on a worker thread.
  This is the durable fix for the **silent GEE hang**: a hang raises nothing, so plain backoff
  never fired. It fired 25 times in the national panel extraction with zero workers lost.
  ⚠️ **The worker thread must be a daemon** — `_call_with_deadline()` hand-rolls
  `threading.Thread(daemon=True)` rather than using a `ThreadPoolExecutor`, whose non-daemon
  threads are joined by an `atexit` hook and left every worker alive at 0 % CPU for hours after
  its work finished. `cancel_futures=True` does **not** help.
* **`_chunk_todo()` packs chunks for locality.** Grouping by year alone spans up to 110 deg²
  nationally; departments are visited in longitude order and packed greedily under
  `MAX_CHUNK_BBOX_DEG2` (4 deg²) — 240 chunks at a 0.53 deg² median, a no-op for one department.
* The per-year combine dedupes on `(COD_PREDIO, doy, lon, lat, mission)`.

**Concurrency.** Content-addressed names + skip-if-exists means extraction splits across
processes on **disjoint year ranges** (`--years`): 3.5 → 12 chunks/min on five workers. The
ranges must genuinely be disjoint — a process covering *all* years duplicates the others' work.
⚠️ 5 workers triggers GEE Restricted Mode; **2 is stable** for the S2 store.

### `features/indices.py`
`scale_sr()` (C2 scale factors), `add_indices()` → NDVI, EVI, NDWI, NDMI, BSI. `CHANNELS` =
6 bands + 5 indices = 11, in an order shared by every model input.

### `features/assemble.py`
No binning, no interpolation. Produces `features_lightgbm.parquet` (per channel
median/mean/std/min/max/p25/p75/amplitude + linear slope + order-1 harmonic fit
`h_mean,h_cos,h_sin`, NaN under 4 observations — LightGBM handles NaN natively — plus statics),
`tensor_perdate.npz`, `tensor_pixelset.npz`, `feature_meta.parquet`.

### `features/s2_gee.py`, `features/s2_assemble.py`
Sentinel-2 equivalents for the endpoint campaign. ⚠️ NDVI is formed **per pixel server-side**
(`S2_NDVI_BAND`) before reduction, because a quantile of a ratio is not the ratio of the
quantiles; percentile columns are named `NDVI_px_*` so they cannot collide with `add_indices`.

### `data.py`
The single source of truth for "who trains on what": `fold_split(df, k)`, `final_split(df)`,
`make_dataset(kind, …)`. Channel normalisation is fit on the **train subset only** and stored
inside the saved model. **Never imports torch at module level** (see §6).

**Feature exclusion.** `make_flat` used to hand LightGBM *everything else in the store*, which
silently included acquisition metadata. Two knobs:

* `drop_features=` — column names or the `DROP_SETS` aliases **`meta`**
  (`frac_l7, n_valid_obs, n_dates, max_gap, n_valid_pixels`) and **`location`**
  (`centroid_lat`).
* `feature_names=` — score on *exactly* this list. `infer()` passes the model's saved
  `feature_names`, so an ablated model **cannot silently regain** a withheld column from a store
  that still contains it.

### `models/`
One interface (`fit`, `predict_proba`, `save`/`load`) and a **lazy registry** (`models/base.py`
maps names to modules without importing them — `get_model("lightgbm")` never touches torch).

* `trees.py` — LightGBM, native `lgb.train` with explicit `num_class` over the *full* class
  space, so a fold missing a class still yields full-width probability vectors.
* `torch_common.py` — AdamW, class-weighted CE, early stopping on val macro-F1, curves,
  `device.pick_device()` (cuda → mps → cpu). Non-finite losses/gradients are skipped.
* `ltae.py` — masked master-query temporal attention, sinusoidal DOY encoding.
* `psetae.py` — pixel-set encoder feeding the *same* `LTAECore`, so the rung-2→3 delta isolates
  the encoder. ⚠️ The eps-inside-sqrt in variance pooling is load-bearing (regression-tested).
* `perennial/rules.py` — the registered phenology control. ⚠️ It reads only its 3 configured
  columns (`RESULTS.md` §2.2).

### `train.py`, `evaluate.py`, `infer.py`
* `train(model_name, …)` — CV → final refit → **optional single** locked-test evaluation
  (`--eval-test`). `drop_features=` is threaded to every `make_dataset` call and recorded in
  `cv_metrics.json`, so a run always states what it was denied.
* **`--train-years`** restricts the **training** cohort only; validation and locked-test
  membership are untouched so CV stays comparable. ⚠️ **Always pair it with a truncation
  control** — re-gate the *baseline* model over the same years, or an improvement cannot be
  attributed to the retrain (`RESULTS.md` §3.3).
* `model_kw=` reaches the constructor and is deliberately **not** on the CLI — a model-variant
  experiment should be explicit in a script, not a flag someone can flip by accident.
* `evaluate.full_report(run_dir)` — macro-F1 (primary), weighted F1, balanced accuracy, kappa,
  majority baseline, per-class table, row-normalised confusion, reliability + Brier, and
  macro-F1 stratified by area / `n_valid_obs` / `max_gap` / year.
* `infer.infer(run_dir, …)` — batch prediction with **abstention**: `coverage_unmeasured`,
  `quality_gate`, `no_features`, `low_confidence`. Never silently drops a parcel.

### `allperu/`
* `sources.py` — department registry. **`shapefile_view()` is load-bearing** (`DATA.md` §4.2).
* `build_labels.py` — Chain A per department. `canon_key()`, `clean_geometry()`,
  `build_dept_sset_caches()` each fix one silent trap (`DATA.md` §4).
* `sample.py` — down-sample by whole 5 km regions. Writes **`population_weight`** (deliberately
  *not* `sample_weight`, which `perennial/panel.py` computes for a different stratification;
  `build_panel` multiplies the two).
* `lodo.py` — leave-one-department-out, with the same 1.5 km buffer at department borders; the
  early-stopping validation set is carved from the *training* departments. ⚠️ **`--tag` is
  required when running a second architecture** or the second run silently overwrites the first.
* `loyo.py` — leave-one-year-out, and `lodo_by_cohort` (LODYO).
* `tenure.py`, `tenure_did.py`, `did_sample.py` — the DiD. `amplification_factor` derives M from
  window midpoints **in code**; `write_registration` refuses to overwrite.

### `labelling/`
`chips.py` (Esri chip rendering — zoom derived from probed resolution, with a placeholder
detector), `build_html.py` (self-contained labelling shards; **blindness is asserted on the raw
HTML string**), `ingest.py` (CSV → `labelled_parcels.parquet`, κ over parcels both labellers
*called*).

### `config/`
| file | contents |
|---|---|
| `data.yaml` | label policy: categories, the intercrop `merge` map, `min_class_parcels`, area/year gates, `min_valid_obs` |
| `split.yaml` / `split_allperu.yaml` | `block_km`, `region_km`, `buffer_m`, `n_folds`, `test_frac`, seed, `metric_crs` |
| `perennial*.yaml` | the 3-class lexicon. The national one is **additive only** and adds sierra/selva tokens, `stage_words` (`MAIZ EN FLORACION`) and `word_match` — without them 19.2 % of national records fell to a blanket `ANNUAL` guess; final 1.92 %, under the 2 % budget the build enforces |
| `split_s2labels.yaml` | the frozen S2 labelling split |

---

## 5. Artifacts — `data/processed/`

| file | producer | contents |
|---|---|---|
| `modeling_parcels.parquet` | labels + splits + gate | geometry, `label`/`label_id`, area, year, centroids, `block_id`/`region_id`/`split`/`fold`, buffer flags, `n_valid_obs`/`max_gap`/`quality_ok` |
| `label_map.json`, `label_exclusions.csv` | labels | class↔id map; per-reason exclusion counts |
| `splits_meta.json`, `class_block_counts.csv` | splits | audit curve + settings; per-class evaluability |
| `features/coverage.parquet` (+ `coverage_chunks/`) | stage 1 | per-parcel `n_valid_obs`, `max_gap` |
| `features/pixels_<year>.parquet` (+ `pixels_chunks/`) | stage 2 | raw store, one row per clear pixel observation |
| `features/features_lightgbm.parquet` | assemble | summary features |
| `features/tensor_perdate.npz`, `tensor_pixelset.npz` | assemble | LTAE / PSE-LTAE tensors |
| `runs/<name>/` | train | `model.bin`, `cv_metrics.json`, `label_map.json`, predictions, curves, report bundle |

`data/` and `runs/` are gitignored; everything regenerates from the raw data + GEE.

---

## 6. ⚠️ Gotchas

* **Never import torch and lightgbm in one process on macOS.** Each bundles its own libomp and
  co-loading segfaults (exit 139). The lazy model registry exists for this; keep it that way.
  The test suite runs LightGBM training in a subprocess.
* Always `uv run …`. GEE needs `uv run earthengine authenticate` plus a GCP project id
  (default `peru-crop-classifier`).
* **`quality_ok` is a nullable boolean** (NA = unmeasured). Compare with `== True` / `.isna()`,
  never truthiness; `np.select` needs explicit numpy bool masks.
* Parcels are stored EPSG:4326; metric operations go through the configured `metric_crs`.
* Chunk stores under `features/*_chunks/` are append-only caches — deleting a file just causes
  recomputation.
* **Verify a finished extraction by counting rows** (`perennial panel verify`), never by "the
  process exited". Use `deficit_ratio_to_median`, not an absolute tolerance — there is a
  structural sub-pixel floor (`LESSONS.md`).
* Never launch a second `nbconvert --inplace` on a notebook while one is running — two
  concurrent runs share the Jupyter runtime and deadlock.

---

## 7. Cookbook

Set the workspace first (§2). These are the commands of record.

```bash
# ── national single-year model ───────────────────────────────────────────────
export CC_PROC=data/processed/all_peru CC_FEAT=data/processed/all_peru/features CC_RUNS=runs/all_peru

uv run python -m crop_classifier.cli allperu labels
uv run python -m crop_classifier.cli allperu sample --source data/processed/all_peru_full
uv run python -m crop_classifier.cli splits assign --config src/crop_classifier/config/split_allperu.yaml
uv run python -m crop_classifier.cli features extract --stage all
uv run python -m crop_classifier.cli features assemble

# the selected arm. --drop-features is not optional: centroid_lat is memorisation.
uv run python -m crop_classifier.cli train --model lightgbm --drop-features meta,location \
    --run-name lightgbm_nometa_nolat

# selection is made HERE, not on CV. LODYO runs automatically at the end.
uv run python -m crop_classifier.cli allperu lodo --tag nolat --drop-features meta,location
uv run python -m crop_classifier.cli allperu loyo --drop-features meta,location
```

```bash
# ── the panel + its gate  (⛔ the gate FAILS; kept for reproduction only) ─────
uv run python -m crop_classifier.cli perennial panel extract --years 1999-2023   # 20+ h, resumable
uv run python -m crop_classifier.cli perennial panel rebuild                     # after ALL workers exit
uv run python -m crop_classifier.cli perennial panel verify
uv run python -m crop_classifier.cli perennial panel assemble
uv run python -m crop_classifier.cli perennial panel infer --run runs/all_peru/lightgbm_nometa_nolat
uv run python -m crop_classifier.cli perennial diagnostics                       # exits 1 on failure
```

⚠️ **The LTAE arm must run in a separate process from any LightGBM arm** (libomp, §6).

```bash
# ── the tenure DiD (complete — see RESULTS.md §7; do not re-run for a bigger sample) ──
export CC_PROC=data/processed/all_peru_did CC_FEAT=data/processed/all_peru_did/features

uv run python -m crop_classifier.cli allperu tenure-ceiling      # feasibility FIRST, always
uv run python -m crop_classifier.cli allperu did-sample
uv run python -m crop_classifier.cli allperu tenure-register     # refuses to overwrite
uv run python -m crop_classifier.cli allperu tenure-did2         # placebo, then headline
```

```bash
# ── the live S2 labelling campaign — see docs/s2_labelling/plan.md ───────────
export CC_PROC=data/processed/all_peru CC_FEAT=data/processed/all_peru/features_s2

uv run python -m crop_classifier.cli allperu s2-labels ingest --csv-dir <returned CSVs>
uv run python -m crop_classifier.cli allperu s2-labels transitions
```

```bash
# ── figures and tests ────────────────────────────────────────────────────────
uv run python -m crop_classifier.perennial.report_figures     # → docs/figures/
uv run pytest -q
uv run ruff check .
```
