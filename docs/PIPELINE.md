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
        │  (the S2 campaign forks here: features/s2_assemble.py writes the same two
        │   product shapes off s2_perdate.parquet — summaries + tensor_perdate.npz —
        │   over the agricultural year, with positions in days since Aug 1, and no
        │   pixel-set tensor, which that store cannot support)
        │
        ▼  train.py ──────────── spatial CV → final refit → (optional) locked test
        ▼
evaluate.py (metrics/confusion/reliability)     infer.py (predict + abstain)
```

Everything runs from one CLI: `uv run cc --help`.

---

## 2. Workspaces

The same code serves several datasets. Which one a command touches is a **named workspace**,
defined in [`workspaces.yaml`](../workspaces.yaml) and selected with `-w`:

```bash
uv run cc workspaces                    # what is configured, and what exists locally
uv run cc -w national train ...
```

| workspace | what it is |
|---|---|
| `demo` | the committed 1,302-parcel sample; runs with no raw data |
| `piura` | the original single-department build (12-class, closed) |
| `perennial` | 3-class labels over the **same** Piura pixel store |
| `national` | 14 departments, Landsat |
| `national_s2` | the same parcels, Sentinel-2 |
| `tenure_did` | the two-period difference-in-differences sample |

`-w` sets `CC_PROC` / `CC_FEAT` / `CC_RUNS` for that process. Those still work if set by hand,
but a workspace names the triple so they cannot disagree, and every `-w` run prints what it
resolved. `piura` and `perennial` deliberately share a feature store — same parcels, same
pixels, different label column — and `cc workspaces` prints the sharing.

The S2 **training** workspaces are the one set you never name yourself: `cc labelling train`
creates and selects `…/labels_s2/ws_<target>[_pilot][_clim_<arm>]` per arm, so label targets
cannot contaminate one another's tables.

Adding your own workspace, to score parcels of your own:
[`howto/04_predict_new_parcels.md`](howto/04_predict_new_parcels.md).

---

## 3. Command reference

`uv run cc <group> <command>`. The task-shaped groups (`data`, `satellite`, `analysis`, `labelling`, `advanced`) are the current names; the historical `features` / `perennial` / `allperu` groups still work and are hidden from `--help`. marks commands whose estimand was
abandoned — the code is built and unit-tested, and it stays unrun (`RESULTS.md` §9).

### Core

| command | what it does |
|---|---|
| `reproduce [name] [--list] [-v]` | re-derive every published headline number from the committed data, each printed beside its published value. ~90 s, no raw archive, no GEE |
| `data verify [name] [-v]` | what this clone can do; every gap named with the command or the licence that fills it |
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
| `allperu tenure-xsec` | descriptive cross-sectional companion, **not an estimate** |
| `allperu tenure-error` | is classifier error non-differential in tenure? |
| `allperu density-audit` / `degrade` / `density-calibrate` | observation-density robustness |
| `allperu year-leak` | which features identify the label year? |
| `allperu export-status` | how much of `PERENNIAL` is really export; woody non-crop bound |
| `allperu esri-dates` / `label-budget` | imagery-date probe; campaign learning curve |
| **`allperu climate <step>`** | per-parcel mean temperature + rainfall — `normals` (WorldClim, static) / `rainfall` (CHIRPS, per year) / `both`. `DATA.md` §7.4 |
| **`allperu cenagro-extract [--dept D] [--overwrite] [--verify]`** | slim the 25 OneDrive CENAGRO 2012 `.dta` files (409 cols, ~18 GB) to `data/raw/Cenagro_IV/<Dept>.parquet` (76 cols, ~0.3 GB). `--verify` runs the audit. `DATA.md` §1.5 |
| **`allperu cenagro-link`** | the NATIONAL census⇄PETT crosswalk by farmer name, 14 depts → `cenagro_pett_link.parquet`. farmer-level, not parcel-level |
| **`allperu cenagro-shift`** | PETT → CENAGRO 2012 → photo-interpreted 2019+, nationally, split by tenure. Perennial share of parcels and of **cadastral** area. `RESULTS.md` §8.6 |
| **`analysis parcel-table [--out F] [--run R] [--preds P]`** | ⭐ the export: one row per parcel, one column per observation of it — tenure, declared class + year, CENAGRO 2012 class, the human 2019+ label and the classifier's. 95,941 parcels, no analysis |
| **`allperu cenagro`** | CENAGRO 2012 as the 'before' instead of the PETT declaration — paired change, area-weighted, by link quality. **Piura only**, farmer-level link. `RESULTS.md` §8.5 |
| **`allperu s2-labels <step>`** | the live campaign — steps below |
| **`allperu s2-train <step>`** | train/compare models on the returned labels — steps below |
| ⛔ **`allperu closed <cmd>`** | the seven abandoned-estimand commands, one level down — see below |

**`allperu closed`** holds every command whose *estimand* failed a pre-registered gate.
The code is built, unit-tested and correct; it stays unrun (`RESULTS.md` §9) because "we
tried this and measured why it does not work" is a result. It is a sub-group so that
`allperu --help` lists the ~25 commands someone might want rather than 32 of which 7 are
closed. **Nothing about how they run changed** — `allperu windows …` is now
`allperu closed windows …`.

| command | what it does | why closed |
|---|---|---|
| `closed windows` | the 5-year window diagnostic (T3) | pivot closed, `RESULTS.md` §5 |
| `closed window-sample` | stratified window sample | same |
| `closed estimate` | the window-estimand tenure contrast — refuses to run unless T1–T3 passed | same |
| `closed external` | SIEA rank comparison | same |
| `closed oli` / `oli-refit` | OLI harmonisation | closed twice, §6.4 |
| `closed tenure-did` | the **v2** DiD | superseded by `tenure-did2`, §7.3 |

`allperu s2-labels` steps: `universe | pool | probe | draw | split | chips | extract |
harmonisation | html | assemble | ingest | combine | transitions`.

**`--round <name>` is how a second campaign is run.** It moves both the labels
(`labels_s2_<name>/`) and the feature store (`features_s2_<name>/`) off the campaign of
record, because `draw` replaces a sample and `assemble` replaces a feature table — either
would make the committed 865 labels unreadable. `combine` then writes the union as
`labels_s2_<name>_all`, which `labelling train --round <name>_all` trains on. The
step-by-step version, written for a non-programmer, is
[`howto/06_label_more_parcels.md`](howto/06_label_more_parcels.md). **`html` takes `--lang en|es`**; anything but
English writes to `labels_s2/html_<lang>/`, and the delivered set is `html_es/`. Only the
interface and codebook are translated — the label values written to the CSV stay the canonical
English constants, because `ingest.py` compares against them.

`allperu s2-train` steps: `prep | fit | lodo | baseline | report`, with
`--target t5|t4|t3|t3w|t2|t2w`, `--model lightgbm|ltae|rules`, `--pilot`, `--model-kw '{…}'`,
`--climate none|temp|rain|both|latlon`.

**`--climate` adds the WorldClim normals as model inputs** (`labelling/climate_arms.py`):
`temp` = `tmean_c`, `rain` = `precip_mm_yr`, `both` = the two, `latlon` = the *control*
(centroid lat/lon in their place). `prep --climate both` builds all the variant workspaces at
once, by **symlinking** the base workspace's parcels and label map so the folds are identical
across arms; `report --climate both` prints the whole comparison plus the paired per-department
Wilcoxon, the per-class F1s and the LightGBM gain ranks. Each model class takes it differently:
flat columns for LightGBM, a 16-d static embedding concatenated into the head for LTAE, and a
median split of the training set with its own thresholds either side for `rules`.
**Result: use LightGBM with `--climate temp`.** Mean temperature is the only climate column
that helps at both ends of the `t4`/`t3w` bracket (LODO +0.029 and +0.027). `both` was the
earlier recommendation and is superseded: the rainfall half does not replicate on `t3w`
(−0.001). Do not add climate to LTAE, where the lat/lon control matches the gain.
`RESULTS.md` §8.8 and §8.8b.

**`t2`/`t2w` are the two-class collapse (`PERENNIAL` vs `NON_PERENNIAL`) and `--model rules`
is refused on them** — the rule maps three semantic names onto label ids and in a two-class
space its `PASTURE_FALLOW` fallback resolves to `PERENNIAL`. It would return a number, not an
error. **And `t2` should not be adopted**: normalised against its own (much higher) majority
floor it is the lowest-skill arm in the study, and it gains +0.004 `PERENNIAL` F1 over `t3w`.
Take a binary output by summing probabilities from a multi-class model instead. `RESULTS.md`
§8.2c.

* `prep` turns `labelled_parcels.parquet` into one `CC_PROC` workspace per label target under
  `labels_s2/ws_<target>[_pilot]/`, each with its own `modeling_parcels.parquet`,
  `label_map.json` and a `features/` directory symlinked to the S2 store. `train.py`,
  `evaluate.py` and the model registry then run against it **unmodified**.
* `fit` is spatial CV + final refit, and by default stops there. **`--eval-test` also scores
  the locked test** — 161 usable parcels, all 14 departments, trained on the 633 trainval
  parcels outside the 3 km dead-zone. It was spent on 2026-09-01 (`RESULTS.md` §8.9,
  LightGBM/`t3w`/`temp`, 0.774 macro-F1) and **must not be used again**: the test can no longer
  independently confirm anything scored on it. The flag refuses `--model rules`.
* `lodo` holds out whole departments (default: those with ≥35 usable labels — **all 14**, as
  of the 2026-08-28 return). Required, not optional: every CV number on this data is the kind
  of number `centroid_lat` inflated, and the CV→LODO drop here is **−0.13 to −0.20 macro-F1**.
  The locked test is excluded from **both** sides, and `--pilot` is the default for it.
* `report` tabulates every fitted arm. Richer tables — per-class precision/recall, pooled
  out-of-fold confusion, paired per-fold tests, per-department LODO — are in
  `labelling/report_s2.py`, which also never reads the locked test.
* `baseline` scores the saved Landsat primary on the labelled parcels' **S2** features. A
  cross-sensor **lower bound**, not a measurement — `RESULTS.md` §8.3.
* `--pilot` folds the held-out 120-parcel pilot into trainval. **Non-canonical**: the frozen
  split holds it out. Frozen fold assignments are preserved and a pilot parcel inherits its
  region's fold, so no 5 km region is ever split across folds.

**One arm per process.** `fit --model ltae` must not share a process with anything that
imports LightGBM (§6, libomp). The steps are separate commands for exactly that reason — drive
them from a shell loop, never a Python one.

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

Robustness (all unit-tested without GEE). The three ways Earth Engine fails quietly are in
[`howto/05_get_satellite_data.md`](howto/05_get_satellite_data.md); what the code does about
them:

* Chunks are **resumable** and **content-addressed** (filename = hash of the chunk's parcel
  IDs), so resuming stays correct if the parcel set, order or chunk size changes. That also
  means a change to what a chunk *computes* is invisible to the cache — invalidate it by hand.
* `_retry()` backs off on transient errors, including throttle messages; deterministic
  "computation too big" errors skip retries and `_run_chunk()` recursively halves the chunk.
* `_retry()` enforces a **900 s wall-clock deadline** (`CHUNK_DEADLINE_S`) on a worker thread —
  the fix for the silent hang, which raises nothing so backoff never fired. It fired 25 times
  in the national panel extraction with no workers lost. **The thread must be a daemon**:
  `ThreadPoolExecutor`'s non-daemon threads are joined by an `atexit` hook, which left every
  worker alive at 0 % CPU for hours after its work finished. `cancel_futures=True` does not
  help.
* **`_chunk_todo()` packs chunks for locality.** Grouping by year alone spans up to 110 deg²
  nationally; departments are visited in longitude order and packed under
  `MAX_CHUNK_BBOX_DEG2` (4 deg²) — 240 chunks at a 0.53 deg² median, a no-op for one department.
* The per-year combine dedupes on `(COD_PREDIO, doy, lon, lat, mission)`.

**Concurrency.** Content-addressed names plus skip-if-exists means extraction splits across
processes on **disjoint year ranges** (`--years`): 3.5 → 12 chunks/min on five workers. Five
workers trips GEE Restricted Mode; **two is stable** for the S2 store.

### `features/indices.py`
`scale_sr()` (C2 scale factors), `add_indices()` → NDVI, EVI, NDWI, NDMI, BSI. `CHANNELS` =
6 bands + 5 indices = 11, in an order shared by every model input.

### `features/assemble.py`
No binning, no interpolation. Produces `features_lightgbm.parquet` (per channel
median/mean/std/min/max/p25/p75/amplitude + linear slope + order-1 harmonic fit
`h_mean,h_cos,h_sin`, NaN under 4 observations — LightGBM handles NaN natively — plus statics),
`tensor_perdate.npz`, `tensor_pixelset.npz`, `feature_meta.parquet`.

### `features/s2_gee.py`, `features/s2_assemble.py`
Sentinel-2 equivalents for the endpoint campaign. NDVI is formed **per pixel server-side**
(`S2_NDVI_BAND`) before reduction, because a quantile of a ratio is not the ratio of the
quantiles; percentile columns are named `NDVI_px_*` so they cannot collide with `add_indices`.

`assemble()` writes the LightGBM summary block over each parcel's **agricultural year
(Aug 1 – Jul 31)**; `assemble_tensor()` writes `tensor_perdate.npz` for LTAE.
**The sequence tensor's positions are days since Aug 1, not day-of-year.** The window
straddles the New Year, so `doy` would run 365 → 1 mid-series and the sinusoidal encoder would
place midwinter next to the first week of August. Nothing raises; the model just learns worse.
Pinned by `tests/test_s2_train.py`. There is **no** pixel-set tensor and there cannot be —
`s2_perdate.parquet` holds per-date medians and quantiles, never the pixels, so PSE-LTAE is not
buildable from this store.
`assemble()` also asserts that no `centroid_*`, `frac_l7`, `mission` or `doy` column reaches the
store, so the two settled negative results cannot be silently re-admitted.

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
  the encoder. The eps-inside-sqrt in variance pooling is load-bearing (regression-tested).
* `perennial/rules.py` — the registered phenology control. It reads only its 3 configured
  columns (`RESULTS.md` §2.2).

### `train.py`, `evaluate.py`, `infer.py`
* `train(model_name, …)` — CV → final refit → **optional single** locked-test evaluation
  (`--eval-test`). `drop_features=` is threaded to every `make_dataset` call and recorded in
  `cv_metrics.json`, so a run always states what it was denied.
* **`--train-years`** restricts the **training** cohort only; validation and locked-test
  membership are untouched so CV stays comparable. **Always pair it with a truncation
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
  early-stopping validation set is carved from the *training* departments. **`--tag` is
  required when running a second architecture** or the second run silently overwrites the first.
* `loyo.py` — leave-one-year-out, and `lodo_by_cohort` (LODYO).
* `tenure.py`, `tenure_did.py`, `did_sample.py` — the DiD. `amplification_factor` derives M from
  window midpoints **in code**; `write_registration` refuses to overwrite.
The national CENAGRO route is pinned by **`tests/test_cenagro_national.py`** (27 tests, no
`data/` needed): the declared schema, the over-common-name cap, Wilson-on-Kish intervals,
post-stratification restoring the population's cell counts, `WOODY_NON_CROP` staying
unmapped, and the figure script rendering. `tests/test_cenagro.py` covers the Piura module.

* `cenagro_extract.py` — the 25-department CENAGRO slim-down (`DATA.md` §1.5). Reads each
  `.dta` **once**, chunked, with an explicit `usecols`, and writes one Parquet per
  department; the output stays **long** (one row per parcel × crop-order) because each
  consumer aggregates differently. The Parquet schema is **declared, not inferred** — a
  chunked write needs one schema for the whole file and pandas will hand chunk 1 an object
  column and chunk 2 a float one. `verify()` is the acceptance check and reports rows
  in/out, the `P009_01` non-blank rate, whether the column set is identical across all 25,
  whether `P024_03` resolves against the crop table, and the posesionario rate that tests
  what the source folder's "sin posesionario" actually filtered (answer: nothing).
* `cenagro_link.py` — the national Chain-B name link. Three routes (full / token-set /
  paterno|materno|1º), matched **within department**, scored by district agreement then
  candidate uniqueness then route strictness. **A name key that pulls more than
  `MAX_CANDIDATES` (25) distinct parcels is dropped, not resolved** — at that multiplicity
  the match carries no information, and keeping it would load the sample with whichever
  departments have the most repeated surnames. Validated against notebook 02's independent
  Piura crosswalk: the `high` tier agrees on 95.5 % of shared producers.
* `cenagro_shift.py` — the three-observation comparison (`RESULTS.md` §8.6–8.7). Carries
  `poststratify()`, which reweights the linked panel to the national population on
  department × declared class — **not optional**, the link over-selects perennial parcels
  16.6 % vs 9.9 %. Every S2 share is design-weighted; `token_audit()` prints the unmapped
  tail sorted by frequency, which is how `MELOCOTONERO` was caught. `_level()` returns a
  **Wilson** interval on **Kish's effective n** — the imagery arm's design weights are uneven
  enough that n_eff is ~⅓ of the row count, and a symmetric Wald interval on a share that
  small runs below zero. **`figure()` holds no drawing code**: it recomputes the levels,
  diffs them against the values baked into `docs/figures/perennial_over_time_by_tenure.py`,
  prints a paste-ready block if they have drifted, and calls that script's `draw()` — so the
  standalone reproduction and the pipeline figure cannot diverge silently.
* `parcel_table.py` — the export (`analysis parcel-table`, pinned by
  `tests/test_parcel_table.py`). The same four observations `cenagro_shift.py` compares, one
  row per parcel and no analysis. The universe is `national_panel.parquet` — 95,941 parcels,
  the largest set a clone can build — and every later observation is a **left join**, so a
  column is null where that instrument never looked. Two decisions carry the module. The 2025
  prediction is read from `fold*/preds_val.parquet` + `preds_test.parquet`, never from
  `cc predict` on the campaign's own parcels: **the model was fitted on them**, so an applied
  prediction there is the model grading its training rows, and `s2_pred_source` records which
  it is. And a locked-test parcel legitimately appears in **both** files — it carries a fold
  id, and `data.fold_split` takes every row of that fold as validation while the train side
  is `split == "trainval"` only — so the refit's read wins and only a *fold–fold* duplicate
  raises. `PRED_TO_DECLARED` renames the imagery `OTHER` to `PASTURE_FALLOW` so all four
  columns are one vocabulary and can be crosstabbed.
* `cenagro.py` — CENAGRO 2012 as the "before" observation. Classifies **both** sides with the
  same lexicon machinery, because otherwise part of the measured change is a change of
  definition; `token_audit()` reports the unmapped share and **must be run before trusting a
  number** — 4.09 % of census tokens fell to `ANNUAL` and 80 % of that was one token,
  `VERGEL FRUTICOLA` ("fruit orchard"), which moved the headline by 10 pp.
  Piura only, and the census↔parcel link is **farmer-level**.
* `climate.py` — per-parcel mean temperature + rainfall (`DATA.md` §7.4). Samples at the parcel
  **centroid**, which is exact here because no parcel (max 50 ha) is larger than one climate
  pixel (WorldClim ~86 ha, CHIRPS ~3,000 ha); `build_normals` prints that check rather than
  assuming it. **A masked cell returns NaN, not an error** — climate rasters mask the ocean
  and the cadastre runs to the shoreline, so `_sample_points` falls back to the nearest valid
  cell and reports the count. CHIRPS is read **remotely** with a `/vsicurl` window over Peru;
  nothing global is stored.

### `labelling/`
`chips.py` (Esri chip rendering — zoom derived from probed resolution, with a placeholder
detector), `build_html.py` (self-contained labelling shards; **blindness is asserted on the raw
HTML string**), `ingest.py` (CSV → `labelled_parcels.parquet`, κ over parcels both labellers
*called*), `train_prep.py` (`labelled_parcels.parquet` → a trainable `CC_PROC` workspace per
label target, plus LODO and the transferred Landsat baseline), `report_s2.py` (the readable
results: pooled out-of-fold per-class tables, confusion in parcel counts, paired per-fold
tests, per-department LODO — **none of which reads the locked test**).

`item_centroid.py` is a small operator tool, not part of any pipeline: give it an `item_id`
from a labelling shard and it returns that parcel's lat/lon, which is what you need to open the
parcel in an external viewer when a label looks wrong. Run it directly
(`uv run python -m crop_classifier.labelling.item_centroid <item_id>`); it is deliberately not a
CLI command because it answers a question during labelling, not a step in a run.

`ingest.py` stores the annotator's label **verbatim** over six values and takes no modelling
decision; `train_prep.TARGETS` takes them, one workspace per reading, so an arm is a file on
disk rather than a flag. `train_prep` sets `quality_ok = True` on every row: it is a
*Landsat* extraction-quality flag, NA throughout this campaign, and `load_parcels` filters on
`== True` — forwarding it untouched trains on **zero rows and reports an empty dataset, not an
error**. The S2 equivalents (`no_s2_observations`, `sub_pixel_parcel`) are applied by `ingest`
upstream.

### `config/`

**Configs are diffs, not copies.** `crop_classifier/config_loader.py` resolves three keys:

* **`extends: <file>`** — load that file first (relative, recursive), then let this file's own
  top-level keys **replace** the parent's.
* **`add: {key: [...]}`** — **append** to the parent's list under `key`, deduped, order
  preserved. This is the additive-only discipline: a derived lexicon may add tokens and may not
  reassign one, because otherwise part of a measured "change in the land" is a change of
  definition (`RESULTS.md` §8.5). `tests/test_config_loader.py` **checks** it, and
  `labels3.build_resolver` raises anyway if a token ends up in two groups.
* **`drop: [key, ...]`** — delete an inherited key. Needed only where a variant genuinely
  renames a group (`perennial_binary.yaml` has `non_perennial` where the base has `annual`).

The chain, pinned by `EXPECTED_CHAIN` in that test so a silently-repointed parent is a failure:

```
data.yaml                       split.yaml ──► split_b3000.yaml
perennial.yaml                  split_allperu.yaml ─┬─► split_window.yaml ──► split_window_b3000.yaml
  ├─► perennial_4c.yaml                             └─► split_s2labels.yaml
  ├─► perennial_binary.yaml
  └─► perennial_allperu.yaml ──► perennial_cenagro.yaml
```

**It was 2,050 lines of copy-forward before this.** `perennial_cenagro.yaml` was
`perennial_allperu.yaml` pasted whole, which was `perennial.yaml` pasted whole — 559 lines with
two "Original header follows" markers in it, of which ~30 tokens were the actual content. You
cannot see an additive-only claim in a full copy; you have to diff it.

| file | contents |
|---|---|
| `data.yaml` | label policy: categories, the intercrop `merge` map, `min_class_parcels`, area/year gates, `min_valid_obs` |
| `split.yaml` / `split_allperu.yaml` | `block_km`, `region_km`, `buffer_m`, `n_folds`, `test_frac`, seed, `metric_crs` |
| `perennial*.yaml` | the 3-class lexicon. The national one is **additive only** and adds sierra/selva tokens, `stage_words` (`MAIZ EN FLORACION`) and `word_match` — without them 19.2 % of national records fell to a blanket `ANNUAL` guess; final 1.92 %, under the 2 % budget the build enforces |
| `split_s2labels.yaml` | the frozen S2 labelling split — also read by `train_prep` for `buffer_m`/`seed` when the pilot is folded in |
| `perennial_cenagro.yaml` | the 3-class lexicon for the **census** vocabulary — `perennial_allperu.yaml` **plus tokens only**. census side only; running it over PETT would change labels of record |

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
| `climate/parcel_climate_normals.parquet` | `allperu climate normals` | 726,808 parcels × 34: `tmean_c` (°C), `precip_mm_yr` (mm/yr), monthly profile, seasonality. **Static** (1970–2000 normal) |
| `climate/parcel_rainfall_annual.parquet` | `allperu climate rainfall` | 726,808 parcels × 33: `precip_mm_<year>` (mm) for 1996–2024. **Year-resolved** — the one safe across years |
| `cenagro/*.csv` | `allperu cenagro` | PETT→CENAGRO 2012 paired change: raw, like-for-like, area-weighted, by link quality, plus the token audit |
| `cenagro/parcel_table.parquet` | `analysis parcel-table` | ⭐ one row per parcel × 23 columns: tenure, `pett_year`/`pett_class`, `cen_class`, `s2_label`/`s2_class`, `s2_pred_class` + `s2_pred_source`, `weight`, `n_observations`. 95,941 rows, the imagery columns null on all but 157 |
| `labels_s2/skill_by_target.csv` | `report_s2.skill_table` | every target's macro-F1 **against its own majority floor**, plus `PERENNIAL` F1 — the only cross-target-comparable columns |

The **only** NaN in either climate table is `precip_seasonality_cv` for 1,275 Ica parcels
that receive exactly 0 mm/year — 0/0, left undefined rather than imputed. `DATA.md` §7.4.

**The S2 campaign, under `all_peru/labels_s2/`:**

| file | producer | contents |
|---|---|---|
| `labelled/*.csv` | the annotator | one row per labelling; `item_id`, `label`, `boundary_mismatch`, `seconds_spent` |
| `labelled_parcels.parquet` | `s2-labels ingest` | one row per parcel, label **verbatim**, `exclude_reason`, `usable` |
| `kappa_report.json`, `stratum_counts.csv` | `s2-labels ingest` | κ + G1/G2/G3 readings; per (dept × declared × observed) counts |
| `ws_<target>[_pilot]/` | `s2-train prep` | a full `CC_PROC` workspace: `modeling_parcels.parquet`, `label_map.json`, `features/` symlinks |
| `ws_*/lodo_<model>.csv` | `s2-train lodo` | per-department held-out metrics |
| `ws_<target>[_pilot]__clim_<arm>/` | `s2-train prep --climate` | one workspace per climate arm; parcels + label map **symlinked** from the base so the folds match exactly |
| `climate_arms_<target>.csv` | `s2-train report --climate` | the 15-row comparison: CV, LODO, accuracy, floor-normalised skill, per-class F1, deltas |
| `climate_arms_<target>_{consistency,per_class,gain}.csv` | `s2-train report --climate` | the three companion tables §8.8 quotes: per-fold/per-department deltas + paired Wilcoxon, out-of-fold per-class F1, LightGBM gain ranks |
| `ws_<target>[_pilot]/climate_location_audit.json` | `climate_arms.location_proxy_audit` | how much department identity the climate columns carry (0.676 accuracy vs a 0.091 prior) |
| `ws_*/landsat_baseline*.{parquet,json}` | `s2-train baseline` | the transferred Landsat read + its caveat, recorded in the file |
| `model_comparison.csv` | `s2-train report` | every arm: CV mean/sd/min/max fold, LODO mean/sd |
| `declared_to_observed_transitions.csv` | `s2-labels transitions` | weighted declared (1996–2006) → observed (2019+) shares + 95 % CIs, **no classifier in it** |
| `features_s2/tensor_perdate.npz` | `s2_assemble.assemble_tensor` | LTAE input; **positions are days since Aug 1** |
| `runs/s2_labels/ws_<target>[_pilot]/<model>/` | `s2-train fit` | the usual run bundle, per arm |

`data/` and `runs/` are gitignored; everything regenerates from the raw data + GEE.

---

## 6. Gotchas

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

## 7. Commands of record

**If you are trying to *do* something, read
[`howto/`](howto/01_setup.md) instead** — those are start-to-finish and explain the choices.
What follows is the reference list: the exact invocations that produced the results on file.

```bash
# ── national single-year model ───────────────────────────────────────────────
uv run cc -w national labels build
uv run cc -w national data sample --source data/processed/all_peru_full
uv run cc -w national splits assign --config src/crop_classifier/config/split_allperu.yaml
uv run cc -w national satellite extract --stage all
uv run cc -w national satellite assemble

# the selected arm. --drop-features is not optional: centroid_lat is memorisation.
uv run cc -w national train --model lightgbm --drop-features meta,location \
    --run-name lightgbm_nometa_nolat

# selection is made HERE, not on CV. LODYO runs automatically at the end of LODO.
uv run cc -w national advanced lodo --tag nolat --drop-features meta,location
uv run cc -w national advanced loyo --drop-features meta,location
uv run cc -w national evaluate runs/all_peru/lightgbm_nometa_nolat --tag nolat
```

```bash
# ── the panel + its gate (the gate FAILS; kept for reproduction only) ────
uv run cc -w perennial advanced panel extract --years 1999-2023   # 20+ h, resumable
uv run cc -w perennial advanced panel rebuild                     # after ALL workers exit
uv run cc -w perennial advanced panel verify
uv run cc -w perennial advanced panel assemble
uv run cc -w perennial advanced panel infer --run runs/all_peru/lightgbm_nometa_nolat
uv run cc -w perennial advanced panel-gate                        # exits 1 on failure
```

**The LTAE arm must run in a separate process from any LightGBM arm** (libomp, §6).

```bash
# ── the tenure DiD (complete — see RESULTS.md §7; do not re-run for a bigger sample) ──
uv run cc -w tenure_did analysis did-feasibility      # feasibility FIRST, always
uv run cc -w tenure_did analysis did-sample
uv run cc -w tenure_did analysis did-register         # refuses to overwrite
uv run cc -w tenure_did analysis did                  # placebo, then headline
```

```bash
# ── the live S2 labelling campaign — see docs/s2_labelling/plan.md ───────────
uv run cc -w national_s2 labelling campaign html --lang es
uv run cc -w national_s2 labelling campaign ingest --csv-dir <returned CSVs>
uv run cc -w national_s2 labelling campaign transitions
```

```bash
# ── regenerate the slimmed CENAGRO 2012 department files ─────────────────────
# data/ is gitignored, so DATA_ACCESS.md §4 is the record of how data/raw/Cenagro_IV/ is
# made. Set `cenagro_source_dir:` in workspaces.yaml first. ~50 min for all 25.
uv run cc data cenagro-extract                          # all 25
uv run cc data cenagro-extract --dept Piura --overwrite
uv run cc data cenagro-extract --verify                 # the audit
```

```bash
# ── the national PETT → CENAGRO 2012 → 2019+ comparison (RESULTS.md §8.6) ────
# needs data/raw/Cenagro_IV/ (see above) and the national PETT build.
uv run cc -w national data cenagro-link             # ~6 min, 14 departments
uv run cc -w national analysis perennial-shift      # ~3 min, draws the figure too
uv run cc -w national analysis perennial-shift --figure          # redraw only
uv run python docs/figures/perennial_over_time_by_tenure.py      # standalone, no project
# -> data/processed/cenagro/national_*.csv + national_panel.parquet
# -> docs/figures/perennial_over_time_by_tenure.png
```

```bash
# ── the per-parcel export: the same observations, one row each, no analysis ──
# runs off the committed panel — no licensed archive, no GEE.
uv run cc -w national analysis parcel-table                          # ~10 s
uv run cc -w national analysis parcel-table --out parcels.csv        # CSV instead
uv run cc -w national analysis parcel-table --preds preds2025.parquet   # howto/04 §C
# -> data/processed/cenagro/parcel_table.parquet
```

```bash
# ── figures and tests ────────────────────────────────────────────────────────
uv run python -m crop_classifier.perennial.report_figures     # → docs/figures/
uv run python -m crop_classifier.allperu.report_figures       # → docs/figures/
uv run pytest -q
uv run ruff check .
```

Both figure commands read only persisted artefacts under `data/processed/` — no satellite
calls, no training.

**Every committed figure and who makes it.** `docs/figures/` is checked in, so a file in it that
no doc cites is weight with no reader; `tests/test_docs.py` fails on one. 24 of 37 had
accumulated that way and were removed on 2026-08-31.

| figure | generator |
|---|---|
| `profiles_12class_ndvi.png`, `profiles_3class.png`, `elnino_signature.png`, `elnino_mechanism.png`, `elnino_signature_collapse.csv`, `flicker_vs_statics.png` | `perennial.report_figures` |
| `window_control_drift.png`, `did_result.png` | `allperu.report_figures` |
| `perennial_over_time_by_tenure.png` | `perennial_over_time_by_tenure.py` beside it — **self-contained, matplotlib only**; `allperu cenagro-shift --figure` recomputes the numbers, checks them against the ones baked into that script and calls its `draw()` |
| `per_class_f1.png`, `pooled_cv_metrics.png` | `perennial compare` |
| `panel_budget.csv` | `perennial panel probe` (its default `--out`) |
| `l7_coverage.csv` | **no generator in the tree** — a one-off probe, kept because §7.1's archive-limit numbers are read off it |
| `big_parcels_grid.png` | `notebooks/04_inspect_parcel_basemaps.ipynb` (see `notebooks/README.md`) |
| `data_overview_flowchart.svg`, `validation_gates_flowchart.svg` | hand-drawn; **no generator, do not delete** |
