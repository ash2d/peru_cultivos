# Plan — Landsat crop classifier for Piura (train / val / test / inference)

> Companion to `CLAUDE.md`. Read `CLAUDE.md` first for data provenance and the forensic
> history; this file is the **modelling blueprint**. It (1) assesses whether the `src/`
> pipeline is ML-ready and lists the gaps to close, (2) critically reviews the proposed
> feature approach against the *actual* parcel/pixel statistics, and (3) specifies a
> modular, config-driven train→eval→infer system an agent can implement.
>
> **Prime directive for the implementing agent:** everything is keyed on `COD_PREDIO`,
> every table joins on it, and **no random row splits are ever allowed** — spatial
> autocorrelation makes that a leakage trap (see §5). Build the split first, wire it
> through every experiment.

---

## 0. Decisions locked in (with the user, 2026-07-20)

These **supersede** any older either/or wording elsewhere in this file. Read this table first.

| area | decision | choice |
|---|---|---|
| **Models** | the only three built | **LightGBM** (baseline) · **LTAE** · **PSE-LTAE** — a controlled ladder (§8) |
| **A1 · Year** | trust the titling `year` as the crop-year | **Yes, as-is** — no cohort restriction; accept the label↔season ceiling (§14) |
| **A2 · Labels** | which categories become classes | keep `crop` as crop classes; keep **pasture** + **fallow** as land-cover classes; **drop** `land_prep` + `unspecified` (§7) |
| **A2 · Labels** | intercrop / merge policy | config `merge` map — whitelisted crop-sets collapse to one class (e.g. `CAFE+PLATANO → CAFE`); other multi-crop parcels excluded (§7) |
| **A2 · Splits** | block size | **1 km** blocks (was 10 km) for more folds — **with a buffered dead-zone + autocorrelation-range audit** to stop leakage (§5) |
| **A3 · Features** | binning? | **No binning, no interpolation.** LightGBM = whole-year summary + harmonic features over all clear obs; LTAE/PSE = raw per-date + mask (§4/§6) |
| **A4 · Extraction** | scope + storage | cheap coverage pass first, then **export raw dated pixels once, only for gate survivors** (`n_valid_obs ≥ 4`) (§6) |
| **A5 · Area** | oversized / sub-pixel | hard-exclude `area_ha > 50` **and** `< 0.09` via the gate; no special-casing for now (§7) |
| **A6 · Compute** | hardware | must run on **Apple-Silicon MPS (M3 Air)** *and* **CUDA (JASMIN Orchid)** — auto device-select, CPU fallback (§8/§10) |
| **Gate** | abstain threshold | `n_valid_obs ≥ 4` (+ area + valid year + ≥1 pixel); `max_gap` is a separate training-side guard (§7) |

---

## 0b. TL;DR of the recommendation

- The `src/` label pipeline is **sound but not yet ML-ready**: it emits clean
  `(polygon, crops[], year)` but has **no label target, no features, no splits, no feature
  store**. Add four modules (`labels.py`, `splits.py`, `features/`, `data.py`) before any model.
- **The proposed 1st/50th/99th-percentile-pixel idea is right in spirit but wrong in the
  specifics for this data.** Median parcel = **0.36 ha ≈ 4 Landsat pixels**; **75 % of
  parcels have *zero* pure interior pixels**. At ~4 pixels, "p1" and "p99" are just the
  min/max of 4 noisy, edge-contaminated values — that's noise, not within-field signal.
  **Keep the median (p50) time series; demote p1/p99 to optional, pixel-count-gated
  dispersion features; and test the learned generalisation of the idea — a Pixel-Set
  Encoder (PSE-LTAE) — which is the SOTA parcel-based method and is purpose-built for
  low/variable pixel counts.** Full argument in §4.
- **Three models (config-driven), a controlled ladder** — each rung adds exactly one capability:
  **LightGBM** (whole-year summary features, NaN-native — tabular baseline) → **LTAE** (temporal-attention
  on per-date pixel medians; no binning, no interpolation) → **PSE-LTAE** (adds the pixel-set
  encoder on top — SOTA for tiny/variable-pixel parcels). Choose the winner by **spatially-blocked
  CV macro-F1**, not accuracy (ARROZ is 61 % of the usable set). MiniRocket/TempCNN are dropped:
  both need a dense NaN-free grid, i.e. exactly the interpolation this data forbids (next bullet).
- **Landsat availability is the binding constraint (measured — notebook 05 §5/§5b, GEE).**
  Parcel-weighted **~5.5 valid months/parcel**; only 47 % of parcels reach ≥6 months, and the
  biggest cohort (**1998 ≈ 46 % of labels**) is the weakest (L5-only, El Niño). Crucially the gaps
  are **contiguous, not scattered**: 62 % of parcels have a ≥3-month missing block, 23 % ≥5 months.
  ⇒ **Do not interpolate to a dense monthly grid** — it fabricates unobserved phenology. **No binning,
  no interpolation** (A3): **whole-year summary features** for the tree (NaN, not fill),
  **mask-native attention** for the DL models, and an **abstain gate** (§7, `n_valid_obs ≥ 4`) for
  near-empty parcels.
- **Usable labelled set:** ~**42.5k** single-crop *crop* parcels (12 classes with ≥ 300 parcels) plus
  optional **pasture/fallow** land-cover classes; the `n_valid_obs ≥ 4` gate keeps ~**33k** before the
  area gate (notebook 05 §5c). **Severe imbalance** (ARROZ ~60 %) and — because crops grow in blocks —
  a much smaller *effective* independent-sample count. Design for that.

---

## 1. What we are predicting, and from what

**Task.** Per parcel: given its polygon geometry and its crop `year`, pull a year-long
Landsat pixel time series inside the polygon, and predict the **declared crop class**.

**Inputs (already built by `src/`):**
| file | grain | key columns used |
|---|---|---|
| `data/processed/training_crop_polygon.parquet` | 1 row / `COD_PREDIO` | `geometry` (EPSG:4326), `crops` (list), `crop_categories` (list), `n_labels`, `is_vegetated`, `year`, `years`, `area_ha`, `codigo_sset` |
| `data/processed/training_crop_records.parquet` | 1 row / (parcel,crop,year) | provenance: `crop_raw`, `owner`, `fecha`, `area_m2` |

**Label reality (from the data, not assumptions):**
- **84 %** of parcels are single-crop; **16 %** carry an intercrop list.
- **~1 crop-year per parcel** — effectively one static label per parcel, *not* a time
  series of labels. So there is **no temporal train/test axis** — only spatial.
- The `year` is a **titling/registration date**, unreliable as a growing season (batch/stub
  values; `2001-01-01` ≈ 35 % of rows). Consequence: we model the **full 12-month
  phenological profile** of the crop-year and must **not** trust an exact planting date.
  **Decision A1:** we still use the titling `year` as the crop-year window **as-is** (no cohort
  filtering); the resulting label↔season noise is an accepted accuracy ceiling (§14).

---

## 2. Current `src/` pipeline: readiness assessment

**What exists and is good** (`build_training_data.py`, `crop_normalization.py`):
- Reliable real-key join `SSET → grafica_tabular_Piura → polygons`; clean, deduped,
  `COD_PREDIO`-keyed outputs (§5b of `CLAUDE.md`).
- `normalize_label()` turns dirty free-text into `(crop, category)` lists with categories
  `crop / pasture / fallow / land_prep / unspecified`; nothing dropped by type.
- Year cleaned (impossible years nulled), area cleaned, geometry deduped.

**Gaps that block ML (must add):**
1. **No modelling label.** `crops` is a list with mixed categories; there is no single `y`,
   no class vocabulary, no rare-class/`other` policy, no single-vs-multi-crop decision.
2. **No features.** Nothing pulls Landsat pixels. This is the largest missing piece.
3. **No splits.** No spatial blocks / CV folds / locked test set. Random splitting would
   leak massively (§5).
4. **No feature store.** GEE extraction is slow and rate-limited; features must be
   extracted **once**, cached to disk, and reused by every model and by inference.
5. **Minor hardening:** add parcel **centroid lon/lat** + a projected geometry (for spatial ops and
   area). **Area gate (A5):** hard-exclude `area_ha < 0.09` (**14.3 %** sub-pixel, < 1 Landsat pixel)
   and `area_ha > 50` (43 parcels, up to 1499 ha; the notebook-04 check shows most are *real*
   perennial/forestry, but they're excluded for now to stay safe).

**Acceptance for "ML-ready":** a single command produces `modeling_parcels.parquet`
(one row/parcel: `COD_PREDIO, label, label_id, block_id, fold, split, area_ha,
n_pixels_est, quality_ok, geometry`) plus a cached **feature tensor store** joinable on
`COD_PREDIO`. Nothing downstream re-reads the raw `.dta/.xlsx`.

---

## 3. The hard data constraints that drive every design choice

Measured on `training_crop_polygon.parquet` (reproduce with the snippet in §14):

| constraint | number | design consequence |
|---|---|---|
| Median parcel area | **0.36 ha** | ~**4** Landsat (30 m, 900 m²) pixels/parcel |
| Parcels with ≥ 10 px | **22.6 %** | percentile tails are unstable for most parcels (§4) |
| Parcels with **0 pure** interior px (1-px erosion) | **74.9 %** | can't afford edge erosion; mixed pixels are the norm |
| Sub-pixel parcels (< 0.09 ha) | **14.3 %** | hard-excluded by the area gate (A5) |
| Class imbalance (ARROZ) | **61 %** of usable | macro-F1, class weights, stratified sampling |
| Spatial autocorrelation | ~**86 %** of neighbours share crop | **spatial block CV mandatory**; effective N ≪ parcel N |
| Distinct crop-years/parcel | ~**1** | no temporal split; label is static |
| Study extent | ~**190 × 130 km** | **1 km** blocks (A2) → thousands of populated blocks; use a **buffered dead-zone** to stop leakage (§5) |
| Sensors covering crop years (bulk 1998–99, span 1996–2019) | L5 (’84–’13), L7 (’99–), L8 (’13–) | extraction must pick mission(s) by `year` |

---

## 4. Reassessing the proposed feature approach (challenge + refinement)

**Your proposal:** monthly Landsat pixels inside each polygon → per-parcel-per-month
**1st / 50th / 99th percentile** NDVI (+ other bands) → 3 time-series curves → classify.

**What's right:** compress the pixel *distribution* per parcel per month (not just the
mean), and represent the parcel as a **multi-month phenological curve**. Both are correct
instincts and align with best practice for parcel-based SITS classification.

**What breaks on this data:**
1. **Not enough pixels for percentiles.** With a median of **~4 pixels**, the 1st and 99th
   percentiles are effectively `min` and `max` of 4 samples — dominated by **residual
   cloud, edge/mixed pixels, and sensor noise**, not agronomic within-field variation. The
   p1/p99 *spread* mostly encodes noise at 30 m over 0.5 ha fields.
2. **Edge contamination is unavoidable.** 75 % of parcels have **no** pure interior pixel,
   so even p50 is partially mixed with neighbours. (Mitigated by the fact that neighbours
   are usually the *same* crop — §3 — but not always.)
3. **NDVI-only throws away separability.** Rice/cotton/maize/mango separate better with
   **multiple bands + indices** (NDWI for flooded rice, NDMI/SWIR for cotton senescence,
   red-edge n/a for Landsat) than with a single NDVI triplet.

**Recommended refinement (do all three; they're config options, not forks):**
- **Default recipe — `median_multiband`:** per parcel per **clear observation**, the **median**
  across valid pixels of **6 SR bands + 5 indices** (§6). This per-date median is the robust core
  signal that LTAE consumes directly and that LightGBM summarises (§6). Include `n_valid_obs` and
  `max_gap` as static features so models can discount low-support parcels.
- **Your idea, salvaged — `percentile` recipe:** keep **p50** plus a **robust dispersion**
  (IQR or MAD, not p1/p99) computed **only** where `n_valid_pixels ≥ 6`, else set to 0 and
  flag. This preserves within-field spread where it's real and suppresses it where it's noise.
- **The learned version — `pixelset` recipe (feed PSE-LTAE):** instead of hand-picking
  percentiles, pass the **raw set of up to `P` pixels** per parcel per **observation** to a **Pixel-Set
  Encoder**, which *learns* the summary statistics (mean, std, learned pooling). This is the
  principled generalisation of "use percentiles," is robust to variable pixel counts, and is
  SOTA for parcel classification (Garnot & Landrieu, PSE + L-TAE). Strongly recommended to test.

**Bottom line:** ship `median_multiband` as the default, offer `percentile` as an ablation
to test your hypothesis directly, and benchmark `pixelset`/PSE-LTAE as the model most
likely to extract signal from tiny parcels.

**Temporal aggregation — no binning, and why we do not interpolate (A3; measured, notebook 05 §5b).**
All three models consume the **same raw per-date pixel store**, summarised differently — **none bins
onto a fixed grid, and none interpolates** across the contiguous gaps (62 % of parcels have a ≥3-month
block, so interpolation fabricates unobserved phenology):
- **LightGBM** needs a fixed-length vector, so it takes **whole-year summary features** over *all*
  clear observations: per band/index the median/mean/std/min/max + a few percentiles, plus
  **phenology metrics** (amplitude, integral, per-channel slopes) and **low-order harmonic-fit
  coefficients**. The harmonic phase/amplitude encode *when* greenness peaks **without a time grid** —
  timing is kept, no binning, no empty cells to fill. A sparse parcel just yields noisier summaries
  (flagged by the `n_valid_obs` feature).
- **LTAE / PSE-LTAE** take the **raw per-date observations** with day-of-year positions + a padding
  mask. Attention weights the real acquisitions directly — ideal for long *contiguous* gaps (better
  than an RNN carrying state across a 4–5 month void).
- **Interpolation is off by design.** It would only ever be needed to feed a dense-grid model, and we
  ship none (MiniRocket/TempCNN dropped). Parcels too sparse to summarise are **abstained** via the
  `n_valid_obs ≥ 4` gate (§7), never filled.

---

## 5. Splits, cross-validation, and leakage (do this before modelling)

**Why random splits are forbidden.** ~86 % of adjacent parcels share a crop and Piura grows
in single-crop blocks; a random split puts a parcel's spectral twin (its neighbour) in both
train and test → wildly optimistic scores that collapse in the real world.

**Spatial blocking scheme (`splits.py`) — 1 km blocks with a buffered dead-zone (A2):**
1. Reproject centroids to a metric CRS (**EPSG:32717**).
2. Assign each parcel a **`block_id`** = its cell in a **1 km grid** (`floor(x/1000)`, `floor(y/1000)`),
   configurable via `--block-km`. 1 km (vs 10 km) gives far more blocks → more/finer CV folds and a
   larger effective test set — **but** it risks re-introducing leakage, so it is paired with items 3 & 6.
3. **Autocorrelation-range audit (do this first — it sets 4 & 6):** measure crop agreement vs
   inter-parcel distance (join-count / empirical variogram). The distance at which agreement decays to
   the study baseline is the **decorrelation range `r`**. Block size + buffer must respect `r`: if
   `r ≫ 1 km`, 1 km blocks *alone* leak and the **buffer (item 6) is mandatory**.
4. **Locked test set:** hold out whole blocks totalling ~**15 %** of parcels, chosen to keep all
   classes present (greedy/stratified block selection). Touch it **once**, at the end.
5. **Spatial CV on the remaining blocks:** `StratifiedGroupKFold` with **`groups=block_id`**,
   **k = 5** folds, stratifying by class as far as group constraints allow. Report **mean ± std**
   across folds — that spread is your honest error bar.
6. **Buffered dead-zone (the leakage guard that makes 1 km safe):** when forming each fold and the
   test set, drop from *training* any parcel within **`buffer` metres** (config, default ≈ `r`, e.g.
   1000–2000 m) of a val/test block. Neighbours of a held-out parcel are thus excluded from train
   rather than leaking in. Log how many parcels the buffer removes.
7. Optionally also group by **`owner`** (from the records table) so one farmer's parcels never
   straddle a fold (they cluster spatially and by crop); add if leakage audits show inflation.

**Leakage checklist (the implementing agent must satisfy every item):**
- [ ] All splitting is by **`block_id`** groups; no `train_test_split(shuffle=True)` anywhere.
- [ ] **Feature scaling / imputation / PSE-LTAE normalisation stats / PCA** are `fit` on **train folds
      only**, applied to val/test (wrap in an sklearn `Pipeline` or fit inside the CV loop).
- [ ] **Class vocabulary, `other`-bucketing thresholds, and normalisation constants** are
      derived from **train only**.
- [ ] **1 km blocks are buffered:** the dead-zone (§5 item 6, ≈ decorrelation range `r`) removes
      train parcels adjacent to val/test blocks; the audit (§5 item 3) justifies `block-km` + `buffer`.
- [ ] **One row per `COD_PREDIO`** (pipeline dedups; assert it).
- [ ] Test set is evaluated **once**; all tuning uses CV folds (nested where feasible).
- [ ] No label-derived feature (obvious, but assert `year`/`owner`/`area` aren't proxies you
      accidentally over-trust — `area` *is* allowed as a feature but log its importance).

---

## 6. Feature engineering & the GEE extraction (`features/`)

**Two-stage extraction, one raw store, two assemblies (A3/A4).**
- *Stage 1 — cheap coverage pass:* over the label-filtered candidate parcels, `reduceRegions`-count
  the clear (QA-masked) acquisitions per parcel-year → **`n_valid_obs`**, and the monthly presence →
  **`max_gap`** (exactly what notebook 05 §5 does, cheaply). Apply the **abstain gate** →
  **survivor list** (`n_valid_obs ≥ 4`).
- *Stage 2 — raw pixel export, survivors only:* export the **raw per-parcel dated pixels once** (every
  clear observation + its date) **for the survivors** — do not spend export budget on parcels that
  fail the gate.

Then assemble the raw store offline into the two representations the three models need — never hit
GEE twice:
- **LightGBM feature table** (no binning): per parcel, **whole-year summary statistics** over all
  clear obs — per band/index median/mean/std/min/max + percentiles, plus **phenology metrics**
  (amplitude, integral, slopes) and **low-order harmonic-fit coefficients** (encode peak *timing*
  without a time grid) + static feats. A fixed-length row; sparse simply means fewer obs (flagged by
  `n_valid_obs`).
- **Per-date sequence store** (for LTAE / PSE-LTAE): the ordered clear observations with **day-of-year
  positions** and a **padding mask** — no binning, no interpolation.

**Sensor selection by year** (`landsat_gee.py`):
- 1996–1998 → **L5** (`LANDSAT/LT05/C02/T1_L2`).
- 1999–2011 → **L5 ∪ L7** (merge; L7 fills L5 gaps; L7 SLC-off after 2003 adds striping but
  more obs). 2012–2013 → L7 (∪ L5 through 2013). 2013+ → **L8** (`LC08/C02/T1_L2`).
- Harmonise band names to a common set; apply Collection-2 SR scale
  (`*0.0000275 - 0.2`). (Reuse `_mask_l5` from notebook 01 §8 / notebook 04.)

**Cloud / quality management (critical):**
- Per-pixel mask from **`QA_PIXEL`** (C2): drop **cloud (bit 3), cloud shadow (bit 4),
  dilated cloud (bit 1), cirrus (bit 2)**; keep clear/water as configured. Also drop
  `QA_RADSAT`-flagged saturation.
- Record per parcel **`n_valid_obs`** (clear acquisitions across the year — this drives the abstain
  gate), **`n_valid_pixels`**, and **`max_gap`** (longest run of consecutive missing months — a
  *label-quality* signal used in §7, not part of the hard gate).
- **Missing data — do NOT interpolate across the contiguous gaps** (measured: 62 % of parcels have
  a ≥3-month block, notebook 05 §5b). Instead:
  - **LightGBM** → whole-year summary features over whatever clear obs exist; **no binning, no
    interpolation** — a sparse parcel just gives noisier summaries (flagged by `n_valid_obs`).
  - **LTAE / PSE-LTAE** → pass the **padding mask + day-of-year positions**; these consume irregular
    series natively (attention weights the real acquisitions regardless of gap length).
- **Abstain gate (shared by all three models, §7):** require **`n_valid_obs` ≥ 4** and
  **≥ 1 valid pixel**; else `quality_ok=False` — excluded from training, abstained at inference.
  Applying the *same* gate to every model keeps their metrics comparable. Note `max_gap` is
  **not** in the hard gate — since nothing interpolates and all three models are missingness-tolerant,
  a contiguous gap is not a computational problem; `max_gap` instead acts as a separate,
  perennial-aware **training-side label-quality** filter (§7).

**Bands & indices (per parcel per observation):**
- SR bands: **Blue, Green, Red, NIR, SWIR1, SWIR2** (harmonised across missions).
- Indices: **NDVI** (veg), **EVI** (veg, less saturation), **NDWI/**flood (rice),
  **NDMI** (NIR-SWIR1 moisture), **NDWI-green (McFeeters)** for open water, **SAVI/BSI**
  (soil/fallow). → **~11 channels**.
- Aggregation per the assembly (§4): whole-year `summary` features (LightGBM), or `per_date` raw obs
  (LTAE median vectors / PSE-LTAE pixel sets). **No fixed-grid binning in any path.**

**Feature store layout** (extract once, reuse forever):
```
data/processed/features/
  coverage.parquet              # stage 1: COD_PREDIO, n_valid_obs, max_gap, quality_ok (the gate)
  pixels_<year>.parquet         # stage 2 raw store (survivors): COD_PREDIO, date, pixel_id, B,G,R,NIR,SWIR1,SWIR2, qa
  features_lightgbm.parquet     # assembly: COD_PREDIO + whole-year summary/harmonic/static columns
  tensor_perdate.npz            # assembly: X [N, Tmax, C or C×P], doy [N, Tmax], mask [N, Tmax]
  feature_meta.parquet          # COD_PREDIO, n_valid_obs, n_valid_pixels, max_gap, quality_ok
```
Run **stage 1 → gate → stage 2** (survivors only), then assemble both offline
(`features/assemble.py`) without re-hitting GEE. Chunk GEE calls by block, cache per-year, be
idempotent/resumable (skip parcels already in the store), respect rate limits.

**Static features** appended for the LightGBM model: `area_ha`, `n_valid_obs`, `n_valid_pixels`,
`max_gap`, `centroid_lat` (climate proxy), sensor id. (Keep `area_ha`; log its importance so you
know how much the model leans on size rather than spectra.)

---

## 7. Label space (`labels.py`)

Config-driven, train-derived, versioned. Default policy:
- **Category policy (A2):** keep the **`crop`** category as crop classes; keep **`pasture`** and
  **`fallow`** as their own **land-cover classes**; **drop `land_prep` and `unspecified`** (ARADO,
  GRADEO, MECANIZADO, HABILITADO, REFORESTACION, blanks — not observable crops).
- **Crop classes:** the **12 crops with ≥ ~300 single-crop parcels** — `ARROZ, MAIZ, ALGODON, TRIGO,
  MANGO, FRIJOL, LIMON, PLATANO, CAÑA DE AZUCAR, GIRASOL, CAFE, ZARANDAJA` (config `min_class_parcels`,
  default 300). Rarer crops → `other` or dropped (config `rare_policy`).
- **Grain + `merge` map (A2, config-driven — the user-editable section of `config/data.yaml`):** a
  parcel is eligible if it is **single-crop** (`n_labels==1`) **OR** its exact crop-set matches a
  whitelisted **`merge`** key that collapses it to one class (e.g. `CAFE+PLATANO → CAFE`). Multi-crop
  parcels **not** in `merge` are excluded (a full multi-label head is deferred, §11). This lets you
  fold specific intercrops into a class instead of losing them.
  ```yaml
  # config/data.yaml — label section
  min_class_parcels: 300
  rare_policy: other                     # other | drop
  keep_categories: [crop]
  landcover_classes: [pasture, fallow]   # kept as their own classes; [] to exclude
  drop_categories: [land_prep, unspecified]
  merge:                                 # sorted, '+'-joined crop-set -> output class
    "CAFE+PLATANO": CAFE
    "CACAO+CAFE+PLATANO": CAFE
  ```
- **Abstain gate → `quality_ok`** (the single quality threshold, shared by all three models and by
  inference): `0.09 ≤ area_ha ≤ 50`, valid `year`, **`n_valid_obs` ≥ 4**, `≥ 1 valid pixel`
  (from §6). It is a **fixed** threshold (no train-derived cutoff → no leakage). Parcels failing it
  are **excluded from training and abstained at inference**, not imputed — keep them in a separate
  bucket so §11 can report coverage-vs-accuracy honestly. `n_valid_obs ≥ 4` retains ~**33k** parcels
  (~78 % of the probed base, before the area gate; notebook 05 §5c); tightening to ≥ 6 would drop to
  ~23k for marginal quality gain, so 4 is the default. Expect a larger abstain share in the weak
  **1998** cohort (notebook 05 §5b); that is the point, not a bug.
- **`max_gap` is a label-quality guard, NOT part of the hard gate** (reasoning: since nothing
  interpolates and all three models are missingness-tolerant, a *contiguous* gap is not a
  computational problem — `n_valid_obs` already measures how much signal exists). Its real risk is
  **label validity**: a big contiguous gap can mean the crop's *discriminative phenology was never
  observed* (e.g. an annual whose growing season is the El Niño cloud blackout), so the parcel is
  effectively **mislabelled relative to what's observable** and pollutes *training*. Therefore:
  - Use `max_gap` as an **ablatable, training-side** filter (default off; e.g. exclude
    `max_gap ≥ 6` **annuals** from training only), **perennial-aware** — perennials (MANGO, LIMON,
    NARANJA, CAFE, PLATANO, CACAO) are ~flat evergreen, so a long gap barely hurts them; do **not**
    penalise them. Decide it by **ablation**: train with/without the filter, compare macro-F1 on the
    well-observed test parcels; keep it only if it helps.
  - Regardless, **feed `max_gap` (and DOY positions, which LTAE uses) as a feature** so a model can
    learn to lower its own confidence when sampling is clustered — the model-native version of the
    guard, backed at inference by the confidence-abstain (`max proba < τ`, §12).
- **Output:** `modeling_parcels.parquet` = `COD_PREDIO, label, label_id, block_id, fold,
  split, area_ha, n_pixels_est, n_valid_obs, max_gap, quality_ok, geometry` + a saved
  `label_map.json`.
- **Report the class histogram and the *effective* block count per class** (how many distinct
  **1 km** blocks each class spans) — a class in a handful of blocks cannot be evaluated honestly no
  matter how many parcels it has; fold it into `other` (§14).

---

## 8. The three models (a controlled ladder)

Three models, deliberately — not a zoo. Each rung adds **exactly one** capability over the one
below, so the comparison is a clean ablation: if a rung wins you know *which* capability earned it.
All implement one interface (§10) and declare the feature **shape** they consume.

| # | model | `input_kind` | data assembly (§6) | adds over previous |
|---|---|---|---|---|
| 1 | **LightGBM** (baseline) | `flat` | whole-year summary + harmonic features (no binning), NaN-native | the bar to beat |
| 2 | **LTAE** (middle) | `sequence` | per-date medians `[Tmax, C]` + DOY + mask | temporal attention over irregular dates |
| 3 | **PSE-LTAE** (SOTA) | `pixelset` | per-date **pixel sets** `[Tmax, C, P]` + DOY + mask | the pixel-set encoder |

1. **LightGBM** — the **whole-year summary/harmonic feature row** (§6, no binning) + static feats
   (`area_ha`, `n_valid_obs`, `max_gap`, `centroid_lat`, sensor). Class weights, native early
   stopping, **native NaN handling**. Fast, interpretable (SHAP). Runs on **CPU** (its CUDA build is
   optional and not needed). XGBoost is a drop-in alternative behind the same `flat` interface.
2. **LTAE** — a **Lightweight Temporal Attention Encoder** (Garnot & Landrieu) over the per-date
   **median** vectors, using **day-of-year positions + padding mask**. No binning, no interpolation.
   Attention (not recurrence) is the right backbone here because the gaps are long and *contiguous*
   (notebook 05 §5b) — it weights the few real acquisitions directly rather than propagating state
   across a 4–5 month void, which is exactly where an LSTM/GRU degrades. (A time-aware GRU with Δt
   is an acceptable substitute behind the same `sequence` interface if you want a recurrent point of
   comparison, but it is expected to be weaker for the gap reason above.)
3. **PSE-LTAE** — the **Pixel-Set Encoder** learns the per-parcel-per-date pixel-distribution
   summary (the learned "percentiles" from §4), feeding the **same LTAE** temporal core. Because it
   shares rung 2's temporal core and differs only in the pixel handling, it is a direct test of
   *"does the pixel-set encoder earn its keep over a plain median?"* — SOTA for tiny/variable-pixel
   parcels, purpose-built for exactly this data (§3/§4).

**Dropped on purpose:** MiniRocket, TempCNN, InceptionTime, plain LSTM, Presto. The first three
require a dense **NaN-free** grid → they force the interpolation this data forbids (§4); the LSTM is
dominated by the LTAE here; Presto is a possible later extension, not a core model.

**Hardware portability (A6).** The two Torch models must run on both **Apple-Silicon MPS** (the M3
Air) and **CUDA** (JASMIN Orchid). A single `pick_device()` helper selects `cuda → mps → cpu`; keep
tensors **fp32** (MPS has no fp64), set `PYTORCH_ENABLE_MPS_FALLBACK=1` for any op MPS lacks, and
**never hard-code `.cuda()`**. Seed and results should match (within fp tolerance) across the two.
LightGBM is CPU on both.

**Expectation:** LightGBM will be a *very* strong, hard-to-beat baseline. The two attention models
earn their keep only via irregular-date handling (LTAE) and set encoding (PSE). Let spatial-CV
macro-F1 decide; **do not assume DL wins** — on ~4-pixel parcels with sparse series it may not.

---

## 9. Training, loss curves, imbalance, sweeps

**Config-driven training (`train.py`).** One YAML/dataclass config per run selects: data
recipe, split params, model + hyperparameters, training params, sweep. `--model <name>` on
the CLI overrides. Every run writes to `runs/<model>_<recipe>_<timestamp>/` with the
resolved config, metrics, curves, and the model artifact.

**Imbalance handling (ARROZ = 61 %):**
- **Class weights** (`compute_class_weight("balanced")`) for trees and DL losses.
- Optional **weighted/`balanced` mini-batch sampling** for DL.
- Optional **focal loss** for DL to focus on hard/rare classes.
- **Never** oversample before the split; only within train folds.

**Loss curves & monitoring (DL):**
- Log **train & val loss** and **train & val macro-F1** every epoch → save `curves.csv`
  and a `curves.png` (loss + macro-F1 panels). Watch for the train/val gap (overfitting) —
  likely with rare classes and spatial blocks.
- **Early stopping** on **val macro-F1** (patience ~10–15); checkpoint best.
- Trees: log the boosting metric vs `n_estimators` (early stopping) as the analogue.

**Hyperparameter sweeps:**
- **Optuna** (TPE), objective = **mean spatial-CV macro-F1** (nested: sweep on
  train-CV, never on the locked test set). Per-model search spaces in
  `config/sweeps/<model>.yaml` (e.g. LightGBM: `num_leaves, lr, min_child_samples,
  feature_fraction, lambda_*`; LTAE/PSE-LTAE: `d_model, n_head, mlp, dropout, lr, weight_decay,
  batch`). Cap trials (e.g. 50–100), log all trials to a `sweep.csv` + parallel-coordinates
  plot. Fix seeds; record them.
- Also sweep **data-assembly** choices as first-class (LightGBM summary-feature set incl. harmonic
  order; PSE pixel-set size `P`; window shift) — the feature assembly may matter more than model
  hyperparameters here. **No binning or gap-fill switches to sweep** — both are off by design (A3/§4).

---

## 10. Modular code architecture (so you can pick a model, train, then infer)

```
src/crop_classifier/
  crop_normalization.py     # DONE
  build_training_data.py    # DONE
  labels.py                 # NEW  §7  -> modeling_parcels.parquet + label_map.json
  splits.py                 # NEW  §5  -> block_id, fold, split (locked test)
  features/
    landsat_gee.py          # NEW  §6  GEE per-parcel dated pixel export (raw store, extract once)
    indices.py              # NEW      spectral indices
    assemble.py             # NEW      raw store -> features_lightgbm.parquet + tensor_perdate.npz
  data.py                   # NEW      CropDataset/DataModule: serves flat|sequence|pixelset
  device.py                 # NEW      pick_device() cuda|mps|cpu, seeding (A6)
  models/
    base.py                 # NEW      CropModel interface + registry (@register("name"))
    trees.py                # NEW      LightGBM (+ XGBoost drop-in)            (flat)
    ltae.py                 # NEW      LTAE temporal-attention encoder        (sequence)
    psetae.py               # NEW      PSE + LTAE (shares ltae.py core)       (pixelset)
  train.py                  # NEW  §9  config-driven, CV, curves, sweeps
  evaluate.py               # NEW  §11 metrics, confusion matrix, stratified reports
  infer.py                  # NEW  §12 batch inference
  cli.py                    # NEW      entrypoints (typer/argparse)
  config/                   # NEW      data.yaml, split.yaml, models/*.yaml, sweeps/*.yaml
```

**Common model interface (`models/base.py`):**
```python
class CropModel(Protocol):
    name: str
    input_kind: Literal["flat", "sequence", "pixelset"]   # what data.py must serve it
    def fit(self, train_ds, val_ds, class_weight) -> "CropModel": ...
    def predict_proba(self, ds) -> np.ndarray:            ...   # [N, n_classes]
    def predict(self, ds) -> np.ndarray:                  ...   # [N]
    def save(self, path: Path) -> None:                   ...
    @classmethod
    def load(cls, path: Path) -> "CropModel":             ...

MODEL_REGISTRY: dict[str, type[CropModel]] = {}   # @register("lightgbm") etc.
```
`data.py` reads the assembly (`features_lightgbm.parquet` for `flat`, `tensor_perdate.npz` for
`sequence`/`pixelset`) + `modeling_parcels.parquet` and yields the shape the chosen model's
`input_kind` requires (summary-feature row for `flat`, `[Tmax,C]`+DOY+mask for `sequence`, pixel sets
for `pixelset`). **Trainer, evaluator, and inference are model-agnostic** — they call only the
interface, so the three models are three files + three registry entries.

**CLI (the "pick a model" UX):**
```bash
uv run python -m crop_classifier.cli features extract --years all           # stage1 coverage+gate, then stage2 raw pixels (survivors)
uv run python -m crop_classifier.cli features assemble                      # -> features_lightgbm.parquet + tensor_perdate.npz
uv run python -m crop_classifier.cli labels build --config config/data.yaml # category policy + merge map + gate (§7)
uv run python -m crop_classifier.cli splits assign --block-km 1 --buffer-m 1500 --folds 5 --test-frac 0.15
uv run python -m crop_classifier.cli train  --model lightgbm                # baseline (summary feats, CPU)
uv run python -m crop_classifier.cli train  --model ltae                    # middle (per-date, masked; mps|cuda)
uv run python -m crop_classifier.cli sweep  --model psetae --trials 60      # SOTA (pixel sets; mps|cuda)
uv run python -m crop_classifier.cli eval   --run runs/lightgbm_2026xxxx
uv run python -m crop_classifier.cli infer  --run <run_dir> --polygons path.parquet --out preds.parquet
```

**Dependencies to add:** `uv add lightgbm torch optuna typer` (+ `xgboost` optional). `torch` ships
both MPS (macOS) and CUDA (Linux) backends from the same package, so one lockfile serves the M3 Air
and JASMIN Orchid; `pick_device()` (`device.py`) picks the backend at runtime.

---

## 11. Evaluation metrics (what to report, and why)

Accuracy is misleading (predict-ARROZ = 61 %). Report, per **spatial-CV fold (mean ± std)**
and once on the **locked test set**:
- **Macro-F1 (primary)** — unweighted over classes; the number to optimise/compare.
- **Balanced accuracy**, **Cohen's κ**, **weighted-F1**, overall accuracy (context only).
- **Per-class precision / recall / F1 + support**; a **row-normalised confusion matrix**
  (expect MAIZ↔ARROZ, MANGO↔LIMON, CAFE↔PLATANO confusions — agronomically sensible).
- **Stratified breakdowns** (key insight generators):
  - by **parcel pixel-count / area** — quantify small-parcel degradation.
  - by **`n_valid_obs` / `max_gap`** (and perennial vs annual) — show performance vs coverage;
    validates the §7 `n_valid_obs` gate and whether the `max_gap` label-quality filter earns its keep.
  - by **region/block** and by **sensor/year** (L5 vs L7-SLC-off vs L8; flag the 1998 cohort).
- **Abstain / coverage accounting (report alongside every metric):** the abstain gate excludes
  parcels, so state **coverage** (share of parcels scored) next to macro-F1 — a model that abstains
  on 40 % and scores 0.7 is not comparable to one that scores 0.7 on all. Report a
  **coverage–accuracy curve** (macro-F1 vs abstain threshold τ) and confirm all three models use the
  **same fixed gate** (§7) so their headline numbers are comparable.
- **Probability quality:** reliability/calibration curve + Brier/NLL (needed because inference uses
  confidences for the abstain option).
- **Baseline deltas:** always report vs (a) majority-class and (b) the LightGBM baseline, so the
  LTAE/PSE-LTAE gains are contextualised.
- **`evaluate.py`** writes `metrics.json`, `confusion.png`, `per_class.csv`,
  `stratified.csv`, `reliability.png` into the run dir.

---

## 12. Inference (`infer.py`)

- **Same feature path as training** — reuse `features/` so train/infer can't diverge.
  Given `polygons.parquet` (needs `COD_PREDIO`, `geometry`, `year`): extract → assemble
  (the assembly for the run's model) → `predict_proba` → argmax.
- **Output** `preds.parquet`: `COD_PREDIO, pred_label, pred_proba, top3, n_valid_obs,
  max_gap, quality_ok, abstained`.
- **Abstain** when `quality_ok=False` (the §7 gate: `n_valid_obs < 4`) or
  `max proba < τ` (config) — do not emit confident crops for parcels observed a couple of times.
- **Two modes:** (a) reuse the cached feature store for parcels already extracted (no GEE);
  (b) cold-start a new parcel+year via on-the-fly GEE extraction.
- Model artifact + `label_map.json` + resolved config are loaded from the **run dir** so
  inference is fully reproducible and model-agnostic.

---

## 13. Phased milestones (suggested order for the agent)

1. **Harden pipeline** — `labels.py` (incl. the §7 abstain gate) + `splits.py` →
   `modeling_parcels.parquet`. *Acceptance:* class histogram, per-class block counts, abstain-share
   report (esp. the 1998 cohort), locked test set that contains all classes.
2. **GEE extraction** — `features/landsat_gee.py` (stage-1 coverage+gate, then stage-2 raw dated
   pixels for **survivors only**) + `assemble.py` (→ `features_lightgbm.parquet` + `tensor_perdate.npz`).
   Do a **1–2 block pilot** first, sanity-check NDVI curves for known ARROZ/MANGO parcels (rice shows
   a strong flooded→green cycle; mango is evergreen/flat), then scale out. *Acceptance:* both
   assemblies for the pilot; gate/abstain share in line with notebook 05 §5b/§5c.
3. **Baseline** — **LightGBM** (whole-year summary features) under **buffered 1 km** spatial CV.
   *Acceptance:* macro-F1 ≫ majority; confusion matrix agronomically plausible; small-parcel &
   coverage degradation quantified. **Gate the DL work on this**: if LightGBM ≈ majority, the problem
   is data/label noise, not model — revisit §14 before building the attention models.
4. **Attention models** — **LTAE** (per-date medians) then **PSE-LTAE** (pixel sets); loss curves +
   early stopping. Compare to the LightGBM bar; the LTAE→PSE-LTAE delta measures the pixel-set value.
5. **Sweeps** — Optuna per model **and** over data-assembly choices (LightGBM summary-feature set +
   harmonic order, PSE pixel-set size `P`, window shift). Lock the winner.
6. **Final test-set eval + inference** — evaluate winner once on the locked test set (report coverage
   alongside macro-F1); wire `infer.py`.
7. **Extensions** — multi-label head for intercropped parcels; add PASTURE/FALLOW classes; a
   time-aware GRU as a recurrent comparator; revisit the census (2012) name-linked labels
   (notebook 02) as a weak second time point.

---

## 14. Open questions, risks, and reproducibility notes

- **Label↔season mismatch is the ceiling on accuracy.** The `year` is a titling date; if a
  parcel was fallow/rotated in that calendar year the "label" won't match the imagery.
  Perennials (mango/lime/coffee/banana) are robust to this; annuals (rice/maize/cotton) are
  exposed. Consider a **confidence-weighted** or **noise-robust** loss, and expect a real
  performance ceiling — report it honestly rather than tuning past it. **Decision A1:** the `year` is
  trusted **as-is** (no cohort filtering); this ceiling is accepted and must be reported, not tuned past.
- **Effective sample size ≪ 34k.** Rare classes may live in a handful of blocks; a class in
  < ~8 blocks probably can't be learned/evaluated reliably — consider folding it into `other`.
- **L7 SLC-off striping** (2003+) injects gaps/artifacts; the QA mask + monthly median
  mitigates, but flag SLC-off-heavy parcel-years.
- **Availability is now measured, not assumed (notebook 05 §5/§5b):** ~5.5 valid months/parcel,
  contiguous gaps, weak 1998 cohort. This is *why* the design is seasonal-composite + mask-native +
  abstain rather than a dense 12-month series — do not re-add monthly interpolation.
- **Extraction cost/limits (A4):** the raw pixel export is the bottleneck — so run the **cheap
  coverage pass first and export raw pixels only for gate survivors** (`n_valid_obs ≥ 4`). Chunk by
  block, cache per-year, make it resumable; extract **once** so both assemblies rebuild offline.
- **Block size is a leakage knob (A2):** 1 km blocks need the §5 autocorrelation audit + buffered
  dead-zone; if the audit shows a long decorrelation range, raise `buffer` (or `block-km`) until
  neighbour leakage is controlled. Validate on the locked test set — once.
- **Reproducibility & hardware (A6):** pin seeds; save resolved config + `label_map.json` + git SHA
  per run; version the assemblies + gate/split params; keep `data/` gitignored (local-only). Torch
  models run on **MPS (M3 Air)** and **CUDA (JASMIN Orchid)** via `pick_device()` — fp32,
  `PYTORCH_ENABLE_MPS_FALLBACK=1`; expect parity within fp tolerance across the two.
- **Sanity snippet** (reproduce §3 numbers):
  ```python
  import geopandas as gpd
  p = gpd.read_parquet("data/processed/training_crop_polygon.parquet")
  px = p.area_ha*1e4/900                        # 30 m pixels/parcel
  print(px.median(), (px>=10).mean(), (p.area_ha<0.09).mean())
  ```
