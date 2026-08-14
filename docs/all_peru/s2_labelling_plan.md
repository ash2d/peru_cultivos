# S2 endpoint labelling — build plan

> **For an agent.** Everything below is buildable from the existing repo. Where a number is
> given it is either measured (cited) or a proposal to be checked at the stated gate.
>
> **Goal:** a photo-interpreted training set for a **3-class (PERENNIAL / ANNUAL / OTHER)
> Sentinel-2 parcel classifier for 2019 onwards**, covering **all 14 linkable departments**,
> with a frozen train/test split.
>
> **⚠️ ONE ROUND, 1,000 PARCELS. There is no second pass.** The allocation therefore has to be
> defensible before it is drawn rather than corrected afterwards (§2.1). The label space is
> four values (§8.1) and everything about how they are used downstream is decided later,
> against the stored labels.
>
> **Supersedes** [`endpoint_labels_plan.md`](endpoint_labels_plan.md) §§2–8. Three deliberate
> departures from it, all at the user's direction:
> 1. **All 14 linkable departments, not the 4 DiD departments.** A classifier for 2020+ Peru
>    trained on four adjacent coastal agro-export departments would reproduce the ~0.09
>    macro-F1 spatial-generalisation gap the project has already measured.
> 2. **No model output enters the design.** No confusion-cell strata, no assessment add-on.
>    Every stratifier is model-independent, so these labels can train a new model without
>    distilling the old one's errors.
> 3. **Labelling is a self-contained HTML file**, not a hosted service.
>
> Out of scope: the causal DiD and any model-based area estimate. In scope, and the campaign's
> second deliverable: the **weighted declared→observed transition matrix** (§11 step 10) —
> a descriptive conversion estimate read straight off the labels, with no classifier in it.

---

## 1. Universe and eligibility

**The 14 linkable departments only** — those with a bridge (`Grafica_Tabular/<Dept>.dta`) *and*
polygons *and* non-zero linked records. Source of truth is `allperu.sources`: **derive the
list from the registry, do not hard-code it.** It is the same universe as
`modeling_parcels.parquet`, so every parcel carries a declared PETT class.

Excluded, and why it matters: the 8 polygon-only departments (Amazonas, Apurímac, Cusco,
Huánuco, Junín, Madre de Dios, Puno, Ucayali, + Callao) have cadastral geometry but no
bridge, and Loreto/San Martín have no polygons at all.

⚠️ **Record this as a permanent scope limit, not a silent omission.** Requiring a declared
class is what makes the class-stratified draw possible (§2), and it is worth ~8x the
labelling budget — but it means **the campaign produces no labels from the sierra or selva,
and there is no later round to add them**. The resulting model is licensed for the 14
linkable departments; applying it to Cusco, Puno, Junín or Amazonas would be extrapolation
with nothing to check it against. State that in the model card as a limit, not a to-do.

**Eligibility filter**, applied before drawing:

1. valid non-null geometry, forced 2D (`build_labels.clean_geometry` — La Libertad's cadastre
   is 3D and Earth Engine rejects it outright);
2. non-null `COD_PREDIO`;
3. `area_ha ≥ 0.15` — below that a 10 m inward buffer leaves too few S2 pixels to summarise;
4. Esri imagery at the parcel centroid is **≤ 1.2 m and dated ≥ 2019** (§4).

Record `n_eligible` and `n_population` per stratum: the weights refer to the **eligible**
population, and the eligible fraction is itself a reportable number.

## 2. Sample design

Three stratifiers, chosen for "simplest that is defensible":

| variable | levels | how it enters | why |
|---|---|---|---|
| **department** | 14 | stratum, sqrt-proportional allocation with a floor | matches `allperu.sample.allocate`; makes leave-one-department-out evaluable from the same labels |
| **declared PETT class** | 3 | stratum, **2 : 1 : 1** allocation within department, `PERENNIAL` over-weighted | class balance is worth ~8x the labelling budget (`label_budget_curve.csv`: 2,000 stratified = 0.548 vs 8,000 region-drawn = 0.529, SD 0.007 vs 0.035); the tilt is §2.1 |
| **geography** | 5 km region | **constraint, not stratum** — cap **2** parcels per region | converts budget into places rather than neighbours, and gives the block split enough units; the cap tightens from 3 to 2 because at 1,000 parcels places are scarcer than parcels |
| **declared crop** | — | **cap, not stratum** — no single crop > 40 % of a (dept × class) cell where the population allows | stops a `PERENNIAL` cell being all mango without fragmenting the design |

⚠️ The declared class is a **~1998 declaration**, used here only as a *proxy* to balance the
draw. It is not the label, it is never shown to the labeller, and the resulting labels are
independent of it.

### 2.1 Allocation — 1,000 parcels, `PERENNIAL` over-weighted

```
14 departments, sqrt-proportional with a floor
  floor 55/dept                            770
  remainder, sqrt-proportional by parcels  230   -> 1,000   (range ~55 to ~110/dept)

within each department, declared class 2 : 1 : 1
  PERENNIAL : ANNUAL : PASTURE_FALLOW  =  50 % : 25 % : 25 %
  e.g. a 55-parcel dept -> 28 / 14 / 13 ;  a 110-parcel dept -> 55 / 28 / 27

 TOTAL 1,000 drawn  + 100 double-labelled for kappa  =  1,100 parcel-labellings
```

At ~2 min/parcel that is **~37 h**, ~18 h each across two labellers. Every parcel gets a
label (§8.1), so attrition is only `boundary_mismatch` and sub-pixel parcels — expect
**~900 usable** for 3-class training. The learning curve
puts 1,000 stratified labels at **91 % of ceiling** against 2,000's 94 % — a real budget, not
a crippled one, and the 3 points it gives up are the price of the campaign being finite.

**Why `PERENNIAL` is over-weighted — two reasons, and they compound.**

1. **The strata are 1996–2006 declarations and the labels are 2019+.** Conversion is one-way
   in practice: a declared `PERENNIAL` parcel is very likely still perennial, while a
   declared `ANNUAL` one may have converted. Stratifying on a 20-year-old declaration
   therefore *under-counts* the current perennial population, and over-weighting the one
   stratum that reliably contains perennials corrects for it.
2. **`PERENNIAL` is the minority class and the binding constraint on the metric.** In the
   learning curve `PERENNIAL` F1 trails macro-F1 at every budget (0.529 vs 0.519 at n=500,
   0.561 vs 0.548 at n=2,000, and it is still climbing at 16,000 where macro-F1 has
   flattened). It is also the class the research question is about.

**With no correction round, the tilt has to be defensible before it is drawn — so check it
against the range of transition rates rather than a single guess.** The realised class mix
depends on rates nobody has measured; the question is not what they are but whether 2:1:1
survives the plausible range. Sweeping declared-`ANNUAL`→perennial conversion from 5 % to
30 %:

| declared→observed conversion | realised PERENNIAL | ANNUAL | OTHER |
|---|---:|---:|---:|
| low (5 %) | 43 % | 26 % | 31 % |
| central (15 %) | 46 % | 25 % | 29 % |
| high (30 %) | 54 % | 20 % | 26 % |

**`PERENNIAL` lands at 43–54 % across the whole range and no class ever drops below ~20 %.**
That robustness is the reason to keep 2:1:1 rather than hedge: the tilt comes mostly from the
declared-`PERENNIAL` stratum, which is stable because conversion runs one way — perennials
rarely revert. Almost none of it rests on the rate that is unknown. Reproduce this sweep in
`label_sample.py` and persist it, so the allocation choice is on the record with its
sensitivity rather than as a bare number.

⚠️ **What 1,000 buys and what it does not.** Enough for a national model and a national test
number. **Not** enough for per-`(department × class)` accuracy — 42 cells at ~24 drawn each.
Per-department LODO lands at n ≈ 40–80 and must be read as a spread, not a ranking (§10, G3).
This is the final budget, so these are the limits of what the campaign can ever report.

### 2.2 Weights

`weight = N_h / n_h` (eligible stratum population / stratum sample). Written as a column at
draw time. Any population share is `Σ_h (N_h/N) × rate_h`, never a raw sample mean.

⚠️ **And prior-correct the model, not just the estimates — this matters more now.** The
declared-`PERENNIAL` stratum is **9.9 % of the population and 50 % of the draw, a ~5x
over-sample**. A model trained on that learns a prior nowhere near the population's, so any
probability read as a share is wrong by roughly that factor. Record the training prior in the
model card and apply class weights or an explicit prior shift before reporting a share. The
old plan weighted the sample and forgot the model.

## 3. Train/test split — frozen before labelling

Reuse `splits.assign` with the all-Peru config (`config/split_allperu.yaml`, `metric_crs:
32718`):

* **unit = 5 km `region_id`**, never individual parcels;
* **test = 20 % of regions, stratified by department**, so every department appears on both
  sides and the national test number is not one region's luck;
* **buffer 3,000 m** (measured free: CV 0.580 ± 0.024 vs 0.5805 ± 0.002);
* trainval split into 5 folds by `StratifiedGroupKFold(groups=region_id)`.

**Freeze and commit the assignment before a single parcel is labelled**, and do not show it
in the labeller. Also report **LODO across the 14 departments** — it is free from the same
labels and it is the evaluation that has overturned selection twice in this project.

At 1,000 parcels capped 2/region that is **≥ 500 regions → ~100 test regions ≈ 200 test
parcels.** Enough for a national headline, thin per class — report the test number with a
binomial CI and lean on 5-fold CV for anything finer.

⚠️ Check the buffer dead-zone after assignment. With ≤2 parcels/region the sample is
scattered, so losses should be small; if >10 % of trainval is sterilised, drop to 1,500 m.

## 4. Esri date probe — extended to the national draw

`allperu/esri_dates.py` already probes the ArcGIS World Imagery metadata layers (30 cm / 60 cm
/ 1.2 m) at real parcel centroids and returns `{res, year, date}` per `COD_PREDIO`. It
hard-codes the at-risk pool and 4 departments — **refactor `probe()` to take an arbitrary
parcel frame** and leave the existing CLI behaviour intact.

Order matters: **draw a candidate pool ~2.5x the target per stratum, probe it, then take the
first N eligible in random order.** Filtering after a draw would silently distort inclusion
probabilities; drawing from the eligible sub-population does not. Persist the full probe
result (including ineligible parcels) — the eligible fraction per department is gate G0.

`imagery_date` becomes a first-class column and defines the parcel's label year.

## 5. Sentinel-2 extraction

New module `features/s2_gee.py`, mirroring `features/landsat_gee.py`'s structure (chunking,
caching, the `_retry` wall-clock deadline). **Reuse `_call_with_deadline` verbatim** — silent
GEE hangs are recurring in this project and cost 13.4 h once.

### 5.1 Collection and harmonisation

```python
S2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")   # L2A surface reflectance
```

* Use **`_HARMONIZED`**, never `COPERNICUS/S2_SR`. Processing baseline 04.00 (from
  2022-01-25) shifted L2A reflectance by a +1000 DN offset; the harmonised collection removes
  it so one scale factor applies to the whole archive. Without this there is a radiometric
  step in the middle of the target window — the OLI lesson repeated.
* L2A begins **2017-03-28**, which the ≥2019 eligibility filter already clears.
* Scale: `/10000`, then clip to `[0, 1]`.

⚠️ **Verify the harmonisation rather than trusting it** (§9.5 of RESULTS.md is what happens
when a correction is assumed to work): take ~200 spectrally stable parcels, plot monthly
median NDVI and SWIR1 across 2021-07 → 2022-07, and confirm no step at 2022-01-25. Persist
the figure. This is a 20-minute check that guards the whole feature store.

### 5.2 Masking

```python
CSP = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
img = img.linkCollection(CSP, ["cs_cdf"]).updateMask(lambda i: i.select("cs_cdf").gte(0.60))
```

Cloud Score+ (`cs_cdf`, threshold 0.60) rather than QA60 or s2cloudless: it is per-pixel,
handles haze and thin cirrus, and covers 2015-06 onward. Additionally drop SCL class 1
(saturated/defective). Do **not** stack more masks than that — over-masking thins the
observation series, which is the exact mechanism §9.2 measured as harmful.

### 5.3 Bands, geometry, reduction

* Bands `B2,B3,B4,B8` (10 m) + `B11,B12` (20 m) → renamed `B,G,R,NIR,SWIR1,SWIR2`, which is
  precisely `features/indices.BANDS`. **`indices.add_indices` then works unchanged**, giving
  the same 11 channels (+ NDVI, EVI, NDWI, NDMI, BSI) as the Landsat store. Replace
  `scale_sr` with the S2 `/10000`; write `scale_sr_s2` beside it rather than branching inside
  it.
* **Erode the parcel by 10 m** before reduction to kill edge/mixed pixels. If the eroded
  geometry is empty, fall back to the original and set `eroded=False`.
* Reduce with `reduceRegions(reducer=median().combine(count(), sharedInputs=True), scale=10,
  tileScale=4)` → **one row per parcel × date**, not raw pixels. LTAE/PSE-LTAE are dropped
  (they lose on LODO 0.442 vs 0.477, LOYO 0.4784, W2 −0.1025), and per-date medians are
  ~50x cheaper to export.
* Require `n_px ≥ 5` for a date to be kept; store `n_px` as a column.

### 5.4 Windows

| purpose | window |
|---|---|
| **model features** | the **agricultural year Aug 1 – Jul 31** containing `imagery_date` |
| **annotator NDVI trace** | **24 months centred on `imagery_date`** |

Aug–Jul contains one complete sierra season (sow Sep–Nov, harvest Apr–Jun) instead of
splitting it, which a calendar year does. This deviates from the Landsat pipeline's calendar
year deliberately: nothing here needs to be comparable to that store, and the harmonic
features are largely insensitive to window phase anyway.

The 24-month trace is the single most valuable thing you can give the labeller: **a fallow
parcel stays flat across two seasons, an annual peaks twice, a young orchard shows
low-amplitude persistent green.** That is what makes ANNUAL-vs-OTHER callable at all — the
old plan retreated to a binary target because it assumed single-date interpretation.

### 5.5 Features

New `features/s2_assemble.py`, or a `channels`/`scale` parameter on the existing
`features/assemble.py`. Emit the **LightGBM summary block only**: per channel
`median/mean/std/min/max/p25/p75/amp/slope/h_mean/h_cos/h_sin` (`assemble._summaries`),
plus `n_dates`, `n_px` stats, `eroded`, `area_ha`. **No `centroid_lat`, no acquisition
metadata** — both are settled negative results.

### 5.6 Cost

1,000 parcels × 1 agricultural year, server-side medians. Expect well under an hour of GEE.
Also **measure and report clear-observation counts per parcel-year by department** — S2's
5-day revisit should give far more than Landsat's 13–24, but the selva will be cloud-limited
and that number bounds what the model can do there.

## 6. Chips

New `labelling/chips.py`. Per parcel, two Esri panels — no S2 true-colour composite, it is
worse than Esri and would only crowd the page:

1. **detail** — extent = parcel bbox padded to a minimum of 120 m, north-up, ~320 px;
2. **context** — ~600 m extent, same centre.

Both drawn with `contextily` on `providers.Esri.WorldImagery`, at the highest zoom Esri
serves (z18–19), resampled to output size. On both: **the target parcel outlined in a bright
colour, neighbouring parcels in a thin contrasting outline.** Perennial detection is a
*contrast* with neighbours (RESULTS.md §8.2) and this is the cheapest way to supply it. Add a
scale bar to the detail panel.

Encode as JPEG q72. Budget ~20 KB per panel.

⚠️ **Rate limits and terms.** ~2,000 chips is still a lot of tile requests: use a persistent
on-disk tile cache, 2–4 workers, and expect ~1 h. Check Esri's terms of use for
programmatic basemap access in a research context before the full run.

## 7. The labelling HTML

New `labelling/build_html.py`. One self-contained file per **shard of 250 parcels** (2 JPEGs
≈ 53 KB base64 + a ~2 KB inline SVG ≈ 55 KB/parcel → ~14 MB/shard, inside the 16 MB artifact
cap). 1,000 parcels → **4 shards**, i.e. two each if the labellers split them.

**Layout, per item:** detail chip and context chip side by side, **NDVI phenology to the
right as an inline SVG** — 24 months, observations as points on a light line, a vertical rule
at `imagery_date`, **y-axis fixed at −0.1 to 1.0** so parcels are visually comparable across
the whole set. Inline SVG rather than a rendered PNG: sharper, a third of the size, and
readable at any zoom.

**Controls** — keyboard first, auto-advance on a class key:

| key | value (§8.1) |
|---|---|
| `1` `2` `3` `4` | PERENNIAL / ANNUAL / OTHER / WOODY_NON_CROP |
| `c` | cycle confidence 1–3 |
| `b` | toggle "boundary no longer matches visible field" |
| `←` `→` | navigate |
| free text | crop guess, if distinguishable |

**Mechanics:** progress bar; autosave every action to `localStorage` keyed by shard id so a
closed tab loses nothing; a resume banner; **Download CSV** button. Implement the download as
a plain `Blob` + `<a download>` so the file works opened from disk, with
`window.claude?.downloads?.save` used when present — a published artifact has no shared
storage, so the CSV coming back by email or upload is the collection mechanism either way.

**Blindness:** item order randomised with a per-labeller seed; **the declared PETT class and
the train/test assignment are never rendered, and must not appear anywhere in the embedded
JSON** — a labeller anchored on the 1998 declaration manufactures agreement between
declaration and endpoint, which is the human form of the `centroid_lat` failure. Department
may be shown; it is inferable from the imagery anyway.

Emitted CSV columns: `item_id, cod_predio, labeller, label, confidence, crop_guess,
boundary_mismatch, seconds_spent, timestamp`.

## 8. Label scheme and codebook — write and freeze before the pilot

### 8.1 The four values

**The annotator picks exactly one of four for every parcel:**

| value | covers |
|---|---|
| `PERENNIAL` | woody or multi-year crop — mango, lime, avocado, olive, coffee, cacao, banana/plantain |
| `ANNUAL` | sown and harvested within a cycle — rice, maize, cotton, potato, beans, wheat |
| `OTHER` | everything else: pasture, fallow, bare, scrub, natural vegetation, water, built |
| `WOODY_NON_CROP` | trees that are not a crop — windbreaks, riparian strips, invaded parcels |

That is the whole label space. **How each value is treated downstream — whether
`WOODY_NON_CROP` trains as `OTHER` or is dropped, how the 3-class target is formed — is a
modelling decision made later, against the stored labels.** Nothing here needs to anticipate
it.

Two fields carry the rest at no extra cost per parcel: **`confidence` (1–3)** and the
free-text **`crop_guess`**. There is no abstain option, so `confidence = 1` is what marks a
parcel the annotator could not really call — filter on it downstream rather than losing the
parcel now.

### 8.2 The codebook

The single biggest determinant of κ, and the old plan had none. One page, with 3–4 example
chips per value pulled from the pilot. **Align the definitions with
`config/perennial_allperu.yaml`** so photo-labels and declared labels mean the same thing:

* **PERENNIAL** — woody or multi-year crop expected to hold the parcel >3 years: mango,
  lime, avocado, olive, coffee, cacao, banana/plantain, oil palm. *Visual:* regular crown
  pattern or row structure, canopy texture, green in both seasons of the trace.
  ⚠️ **Sugarcane is ANNUAL here** (`cana_policy: annual`), matching MapBiomas.
* **ANNUAL** — sown and harvested within a cycle: rice, maize, cotton, potato, beans, wheat.
  *Visual:* uniform texture, no crowns, sharp field boundaries; one or two NDVI peaks with
  returns to bare.
* **OTHER** — everything that is neither a perennial crop nor an annual crop: pasture,
  fallow, prepared bare ground, scrub, natural vegetation, **and non-agricultural land**
  (water, built-up, road, riverbed). *Visual:* no crop geometry, or crop geometry with
  nothing growing across both seasons of the trace.
* **WOODY_NON_CROP** — trees that are not a crop (windbreaks, riparian strips,
  invaded/abandoned parcels). *Visual:* tree cover without rows or a planting grid, often
  following a watercourse or field edge rather than filling the parcel. Never folded into
  PERENNIAL by the labeller: declared woody non-crop is 2.79 % of the pool and that is a
  *lower* bound, since it counts only what was declared, not 20 years of invasion.
**Decision rules for the ambiguous cases** (write these down, they are where κ is won). Every
parcel gets one of the four, so each rule has to end somewhere:

* mixed parcels → the class covering >50 % of the parcel; if genuinely even, `OTHER` with
  `confidence = 1`;
* agroforestry with a closed tree canopy → `PERENNIAL`;
* **young plantings** → `PERENNIAL` if a regular planting grid is legible *or* the trace shows
  low-amplitude green persisting through both dry seasons; otherwise `OTHER` with
  `confidence = 1`. **Never `ANNUAL` by default** — an immature orchard read as annual is a
  false negative on exactly the transition this project is about;
* unreadable parcel (cloud, deep shadow, partial coverage) → `OTHER` with `confidence = 1`;
* a parcel whose visible field boundary disagrees with the cadastre → label what is inside the
  outline and set `boundary_mismatch`.

## 9. Ingest

New `labelling/ingest.py`: read the returned CSVs, join on `item_id` → `COD_PREDIO`, compute
**Cohen's κ on the 100 overlap parcels** (4-way, and perennial-vs-rest separately), resolve
disagreements by adjudication rather than majority (there are only two labellers), and write:

* `data/processed/all_peru/labels_s2/labelled_parcels.parquet` — `COD_PREDIO, geometry,
  label, confidence, crop_guess, boundary_mismatch, imagery_date, imagery_res, dept,
  declared_class, stratum, weight, region_id, split, fold`
* `labels_s2/kappa_report.json`, `labels_s2/stratum_counts.csv`

**Store `label` exactly as the annotator recorded it.** How `WOODY_NON_CROP` and low-confidence
parcels are treated is a modelling decision taken against this table, not baked into it. The
only rows to hold out of training regardless are `boundary_mismatch=True` and parcels with
fewer than ~5 usable S2 pixels — those are mixed-pixel label noise, which caps a model
outright.

## 10. Gates

| gate | criterion | if it fails |
|---|---|---|
| **G0** | ≥ 80 % of the probed candidate pool has ≤1.2 m imagery dated ≥2019 | re-scope which departments are in scope |
| **G1** | **κ ≥ 0.75** over the 4 values in the pilot (§11 step 3) | stop; revise the codebook, not the sample |
| **G2** | `confidence = 1` on < 25 % of drawn | imagery or class definitions are unfit |
| **G3** | ≥ **35 usable labels per department** and ≥ **150 per class nationally** | do not report that department / class |
| **G4** | the S2 model beats the **existing Landsat model scored on the same held-out labelled test fold** | the labels revert to validation-only use |

⚠️ **G3 is a per-department gate, not a per-cell one, and the budget forces that.** 1,000
parcels over 42 `(dept × declared class)` cells is ~24 each, so a 30-per-cell criterion cannot
be met at any allocation. Reporting per-cell accuracy here would be reporting noise.
Per-department LODO is the finest honest cut, and even that is n ≈ 40–80.

⚠️ **G4 replaces the old plan's "LODO mean > 0.477".** That compared an S2 model (1,000
endpoint labels) against a Landsat model (54k parcels, 14 departments, 1998 labels) and would
have lost for reasons unrelated to S2. Scoring both on the same parcels with the same labels
is free — Landsat predictions for these parcels already exist — and it is the first
apples-to-apples comparison this project could make.

⚠️ **Report κ for perennial-vs-rest alongside the 4-way number.** The gap between them is
essentially `WOODY_NON_CROP`/`PERENNIAL` confusion — the hardest call in the codebook, and
the one that tells a later modelling decision how much to trust that distinction.

⚠️ **G2 now keys on `confidence`, because there is no abstain value.** With four forced
choices, a parcel the annotator could not really call still gets a class, so
`confidence = 1` is the only signal that the imagery was not up to it. That makes the
confidence field load-bearing rather than decorative — say so in the codebook.

⚠️ **G1 assumes two labellers.** With one, κ becomes intra-rater — re-label 100 parcels after
a two-week gap — which is a weaker claim and must be reported as such.

⚠️ **G4 is the one gate with no recovery path.** Every other failure is answered by revising
the codebook or narrowing what gets reported. If the S2 model loses to the Landsat model on
the same fold, there is no second campaign to fix it — the labels become a validation set,
which is still the project's first endpoint accuracy measurement and still worth having.
Say that in advance so the outcome is not read as the campaign having failed.

## 11. Order of operations

Cheapest step that can kill the plan runs first.

| # | step | new code | output | ~cost |
|---|---|---|---|---|
| 1 | Build the eligible universe (14 linkable depts, §1) | `allperu/label_sample.py` | `label_universe.parquet` | 1 h |
| 2 | Probe Esri dates on a 2.5x candidate pool; **G0** | `esri_dates.probe` refactor | `esri_dates_national.csv` | ~1 h |
| 3 | **Pilot: 120 parcels, both labellers, κ; G1** | chips + html (§6–7) | `kappa_report.json`, frozen codebook | ~4 h each |
| 4 | Draw 1,000; assign split; freeze and commit | `label_sample.py`, `splits.assign` | `label_sample.parquet` + `splits_meta.json` | 1 h |
| 5 | Render 2,000 chips | `labelling/chips.py` | chip cache | ~1 h |
| 6 | S2 extraction + **harmonisation check** (§5.1) | `features/s2_gee.py` | `s2_pixels_*.parquet`, harmonisation figure | <1 h GEE |
| 7 | Build 4 HTML shards; send | `labelling/build_html.py` | `labelling/shard_XX.html` | 1 h |
| 8 | **Label**; ingest; **G2, G3** | `labelling/ingest.py` | `labelled_parcels.parquet` | **~37 h human** |
| 9 | Assemble features; train; CV / test / LODO; **G4** | `features/s2_assemble.py` | run dir + model card | hours |
| 10 | Report the declared→observed transition matrix and the realised class mix | extend `label_budget.py` | transition rates, learning-curve position | 1 h |

**Step 3 is where the next half-day belongs.** The imagery is 82.9 % 1.2 m, not sub-metre, so
inter-rater agreement at that resolution is the live unknown, and it decides whether the
other 37 hours are worth spending. With no second round it is also the only chance to fix the
codebook before it is frozen into 1,000 parcels.

**Step 10 is a deliverable, not a planning step.** It was sizing a round 2; with one round it
becomes a result in its own right. The weighted **declared (1996–2006) → observed (2019+)
transition matrix** is the descriptive conversion estimate this project has never been able
to produce — the question RESULTS.md §11 called "vacuous if conversion did not happen, a
finding if it did", answered from labels rather than from a classifier. Report it weighted
(§2.2), with CIs, and state where 1,000 labels sits on the learning curve as a limitation.

## 12. Tests to write

* `tests/test_label_sample.py` — allocation hits the floor and the totals; region cap holds;
  crop cap holds; weights reconstruct the eligible population exactly.
* `tests/test_s2_gee.py` — band renaming maps onto `indices.BANDS`; the S2 scale is applied
  once and only once; the Aug–Jul window boundary is correct either side of 1 January;
  `_call_with_deadline` raises on a hang.
* `tests/test_ingest.py` — every `label` in a returned CSV is one of the four values; κ
  matches a hand-worked example; every `item_id` resolves to exactly one `COD_PREDIO`.
* `tests/test_build_html.py` — no shard exceeds 16 MB; **no declared class, split, or fold
  value appears anywhere in the emitted HTML** (assert on the raw string — this is the
  blindness guarantee); every `item_id` round-trips through the CSV.
