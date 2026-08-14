# S2 endpoint labelling — build record

> What was built, what was measured, and every decision taken without asking.
> Plan: [`s2_labelling_plan.md`](s2_labelling_plan.md). Codebook (frozen):
> [`s2_labelling_codebook.md`](s2_labelling_codebook.md).
>
> **Status 2026-08-12: steps 1-2 and 4-7 of the plan's §11 are DONE, and step 9's feature
> assembly has run. The labelling HTML exists and is ready to send.** Step 3 (the pilot)
> and step 8 (labelling) are human work and are the only things standing between here and a
> trained model. Ingest and the transition matrix are built, unit-tested **and
> round-tripped on the real shards** (§5b) — they are unrun only because no human labels
> exist yet.
>
> **Revised 2026-08-13 (§11): three changes after reviewing the emitted HTML — the right
> panel is now a 100 m zoom instead of a 600 m context view, `UNSURE` is a fifth label key
> rather than an implicit use of confidence-1, and the labeller types their own name instead
> of inheriting `A`/`B` from the filename. Chips and shards were re-rendered; everything
> below reflects the rebuilt files.**
>
> **Revised again 2026-08-13 (§12): five further changes — the zoom panel is 200 m, the
> left panel is a genuine *context* view, `NON_AGRICULTURE` is a sixth label with an
> explicit boundary rule against a narrowed `OTHER`, the confidence control is gone, and the
> NDVI trace carries a p25-p75 ribbon that required a full S2 re-extract. §11's numbers for
> chip sizes, panel widths and the label vocabulary are superseded by §12; its
> *reasoning* still stands.**

---

## 1. What exists now

| artefact | where | what it is |
|---|---|---|
| eligible universe | `labels_s2/label_universe.parquet` | **614,876** parcels, 14 departments |
| Esri date probe | `labels_s2/esri_dates_national{,_supp}.csv` | **4,519** probed centroids |
| eligible candidates | `labels_s2/label_candidates_eligible.parquet` | **3,957** parcels |
| the frozen draw | `labels_s2/label_sample.parquet` | **992 main + 120 pilot**, split assigned |
| Esri chips | `labels_s2/chips/` | **2,224 JPEGs, 0 failures**, 41.6 MB (context + 200 m zoom) — §12.1-2 |
| S2 per-date store | `features_s2/s2_perdate.parquet` | **117,755 parcel-dates**, 1,106 parcels |
| S2 features | `features_s2/s2_features_lightgbm.parquet` | 12 summaries x 11 channels |
| **the labelling HTML** | **`labels_s2/html/*.html`** | **9 self-contained files, 1,112 parcels** |
| gates | `labels_s2/gates.json` | G0 and the draw size, recorded as they ran |

**The shards, as emitted** — `pilot_A/B` (120 each), `shard01_A … shard05_A` (200/200/200/200/92,
alternating between labellers so each gets ~half), `overlap_A/B` (the 100 double-labelled
parcels, both labellers). Largest file **10.3 MB**, inside the 16 MB cap. Total
**1,092 parcel-labellings** for the main round plus 240 for the pilot. The `_A`/`_B` suffix
is now only a *suggestion of who should open the file* — see §11.3.

**All 1,112 drawn parcels are labellable.** Six carry no S2 trace at all (Huancavelica 3,
Arequipa/La Libertad/Piura 1 each) — they are **kept**, because the imagery alone is still
callable, and listed in `html/excluded_from_shards.csv` rather than dropped silently.

New code: `allperu/{label_sample,s2_campaign}.py`, `features/{s2_gee,s2_assemble}.py`,
`labelling/{chips,build_html,ingest}.py`, `config/split_s2labels.yaml`, the
`esri_dates.probe_frame` refactor, `indices.scale_sr_s2`. New tests:
`tests/test_{label_sample,s2_gee,ingest,build_html}.py` — **67 tests; the full suite is
348 and passes.**

One command runs the whole thing:

```bash
export CC_PROC=data/processed/all_peru CC_FEAT=data/processed/all_peru/features_s2
uv run python -m crop_classifier.cli allperu s2-labels <step>
# universe -> pool -> probe -> draw -> split -> chips -> extract -> html
#          -> [humans label] -> ingest -> transitions
```

---

## 2. Gates that have run

### G0 — Esri imagery eligibility: **PASS**

**87.6 %** of 4,519 probed parcel centroids have imagery ≤1.2 m dated ≥2019, against an
≥80 % criterion. But the national number hides the thing that actually shaped the campaign:

| department | eligible | | department | eligible |
|---|---:|---|---|---:|
| AREQUIPA, AYACUCHO, HUANCAVELICA, ICA, LAMBAYEQUE, LIMA, MOQUEGUA, TACNA | 1.000 | | LA_LIBERTAD | 0.965 |
| TUMBES | 0.997 | | ANCASH | 0.960 |
| **PIURA** | **0.612** | | **CAJAMARCA** | **0.598** |
| **PASCO** | **0.390** | | | |

⚠️ **Piura is one of the three worst-covered departments in the country for recent
high-resolution imagery.** Every previous strand of this project was built on Piura. If the
campaign had been scoped to Piura, as the earlier plan's DiD departments were, more than a
third of the draw would have been ineligible and the shortfall would have looked like a
sampling bug rather than a coverage fact.

The imagery is also **much more recent than the plan assumed**: 1,587 of 4,519 parcels are
2025 and only 82.9 %-is-1.2 m held — the median department is 2024-25. The consequence is
good (a genuinely current endpoint) and one to watch (the 24-month trace for a 2025 parcel
runs to 2026, so it is shorter on the right).

### Draw size — **992 of 1,000**

Recorded rather than smoothed over. Three cells could not be filled and none of them is a
budget problem:

| cell | target | drawn | why |
|---|---:|---:|---|
| HUANCAVELICA PASTURE_FALLOW | 16 | 9 | the department has **44 populated 5 km regions in total**; at 2 parcels/region its whole ceiling is 88 |
| PASCO PASTURE_FALLOW | 14 | 10 | 325 eligible parcels nationally **and** 39 % imagery eligibility |
| HUANCAVELICA PERENNIAL | 33 | 32 | same region ceiling |

**The binding constraint in the sierra is places, not parcels.** That is the region cap
doing exactly what it was put there for; relaxing it to fill the quota would have bought 8
parcels by putting them next door to ones already drawn.

---

## 3. Decisions taken without asking

Every one of these was a judgement call the plan left open or did not anticipate. They are
listed with what they cost, so a later reader can overturn one on evidence rather than
taste.

**D1 — the pilot is drawn disjointly from the 1,000, not as a subset.** 120 parcels drawn
first, from the same design, then removed from the universe before the main draw. The
codebook has to be frozen against parcels that are not part of the measured sample, or the
pilot's own labels are contaminated by the codebook they produced. Cost: 120 extra
labellings, which the plan already budgeted as step 3.

**D2 — the 100 double-labelled parcels are a fifth shard both labellers receive**, not 50
each swapped between the two. Total labellings 992 + 100 = 1,092, the plan's structure
exactly. It keeps κ from depending on how the labellers happened to divide the work, and it
makes the overlap set a single file that can be sent to a third person if adjudication is
needed.

**D3 — the class allocation rounds *up* on `PERENNIAL`.** Plain largest-remainder gives
55 → 27/14/14, sharing the odd parcel away from the class the 2:1:1 tilt exists for. The
implementation gives the over-weighted class the rounding, reproducing the plan's worked
examples 55 → 28/14/13 and 110 → 55/28/27. Pinned by a test.

**D4 — a department short in one class is topped up from its own other classes**, never
from another department. The department is the stratum LODO needs; the class ratio is a
tilt. Topping up across departments would silently undo the sqrt-proportional allocation.

**D5 — the eligible population is *estimated*, not assumed.** Only the candidate pool was
probed, so `N_h(eligible) = N_h(universe) × eligible_rate_h(pool)`. The weights expand to
that, and they reconstruct it exactly (checked in the draw report and in a test). Using the
raw universe would have over-stated Pasco and Piura by 2.5x.

**D6 — the short departments got a second simple random sample, not a targeted top-up.**
1,704 further parcels probed in 7 departments. Drawing preferentially from unused regions
would have been faster and would have changed the inclusion probability by region, breaking
the weights. Two SRS draws without replacement from one stratum are still an SRS of it, so
pooling them is exact.

**D7 — `split_s2labels.yaml` balances the test draw on `dept × declared_class`, not on
`year`.** Every label in this campaign is a 2019+ photo-interpretation, so there is no
label-year composition to balance — the mechanism that made Piura's locked test 48.3 %
year-1998 cannot arise here. 200 candidate draws are scored and the closest kept.

**D8 — the buffer stays at 3,000 m.** It sterilises **78 of 791** trainval parcels
(**9.9 %**), just inside the plan's 10 % trigger. Left alone, but it is close enough that
anyone re-drawing the sample should re-check it rather than assume.

**D9 — the pilot is excluded from both split arms** (`split = "pilot"`), rather than
landing in trainval by default. A parcel used to write the codebook is not a clean training
example for the model the codebook produced.

**D10 — one extraction serves both the features and the trace.** The agricultural year
Aug 1 – Jul 31 containing `imagery_date` is always a sub-interval of the 24 months centred
on it (proved by a test over 150 dates), so the 24-month window is extracted once and
`s2_assemble.restrict_to_ag_year` cuts the feature window out of it. Halves the GEE cost.

**D11 — same-date granule splits are combined pixel-count-weighted, not dropped.** A parcel
in an S2 tile overlap is reduced once per granule; both halves are real data over their own
part of the parcel. Keeping the larger would throw away observations at exactly the parcels
that straddle a tile edge.

**D12 — chip zoom is chosen from the probed source resolution, with a placeholder check.**
See §4.

**D12b — where `imagery_date` comes from, and what it is not.** It is **queried, not
chosen**: the Esri basemap cannot be asked for a year, so `esri_dates.probe` hits the
ArcGIS World_Imagery metadata layers (9 = 30 cm, 10 = 60 cm, 11 = 1.2 m) at the parcel's
representative point, **stops at the first layer that returns a dated feature**, and reads
`SRC_DATE2`. Resolution is therefore decided by layer order and never by date. Where
several dated footprints overlap one point at that resolution, the **most recent** wins —
the code previously kept the oldest while the comment beside it claimed otherwise; measured
on 200 sampled parcels, **0 have more than one dated feature**, so the branch never fired
and no drawn parcel's date changes. Fixed and pinned by
`tests/test_label_sample.py::TestEsriProbeSelection`.

⚠️ Because chips are fetched live from the basemap, `imagery_date` describes what the
metadata layer reports at that point rather than provably what was rendered. Probe and chip
rendering both ran on 2026-08-12, so the exposure is a day — but a re-render months later
should re-probe rather than trust the stored date.

**D13 — the labelling shards use `A`/`B` as labeller ids** and are named
`shard01_A.html … overlap_B.html`. Renaming a file does not change what it contains; the
labeller id is written into the CSV from the filename's constant, so the returned files are
self-describing.
⚠️ **Superseded 2026-08-13 by §11.3** — the id now comes from a name the labeller types,
and the filename's letter is only a suggestion of who should open it.

---

## 3b. The harmonisation check, and the check's own two failure modes

The plan asks for a 20-minute confirmation that `S2_SR_HARMONIZED` really removes the
+1000 DN offset that processing baseline 04.00 introduced on 2022-01-25. Getting a *useful*
answer took three attempts, and both of the wrong ones looked convincing.

**Attempt 1 — mean NDVI six months before vs six months after the cut.** Reported a
**+0.055 step**. It is entirely the growing season: "before" is Jul–Jan and "after" is
Jan–Jul, and the monthly medians rise 0.38 → 0.57 across exactly that span. The comparison
measured Peruvian phenology and called it a sensor step.

**Attempt 2 — within-parcel, within-month-of-year pairing.** Principled-looking and still
wrong: a parcel's observations do not land on the same days of the month in two different
years, so within a month the two sides sample different parts of a steep seasonal curve.
On a synthetic pure sine with **no step at all** it returns +0.018 at one cut date and
−0.014 at another, purely from where the sampling grid happens to sit.

**What is used — and it is three separate corrections, none optional:**

1. the seasonal cycle is **fitted out per parcel**, not binned out:
   `value ~ a + b·cos(2πt) + c·sin(2πt) + step·1[date ≥ cut]`, and `step` is the
   coefficient. On the synthetic series it returns ~0 by construction;
2. a **placebo cut one year earlier**, where nothing happened, gives the estimator's own
   noise floor; the reported quantity is `step − placebo`;
3. the verdict is read off **raw bands, never off an index** — and the judgement is against
   what the failure would look like (**+0.10 reflectance**, an unremoved +1000 DN offset at
   the 1/10000 scale), not against zero.

**Result — the store verifies.** Placebo-corrected step: **SWIR1 +0.0107, R +0.0087,
NIR +0.0025**, against **+0.10** for an unremoved offset. An order of magnitude too small
to be the baseline change.

⚠️ **NDVI reads −0.031 and that is not a contradiction — it is why an index cannot decide
this.** All three bands move together by ~+0.007, which is what a mild year-to-year
illumination/atmosphere difference looks like; NDVI is a ratio and additionally absorbs
every real change in how green the country was between 2021 and 2022. The bands answer the
question about the sensor. NDVI answers a question about the weather.

---

## 4. Two bugs that would have failed silently, and what they cost

**Esri returns a *valid image* above the zoom it serves.** `bounds2img(zoom="auto")` on a
120 m extent asks for z20+, and Esri answers with a flat grey "Map data not yet available"
tile. No error, no exception — the first chip batch rendered perfectly and was blank. The
placeholder is exactly RGB (205,205,205) with chroma ≈0.03 against >15 for any real scene,
so `chips.is_placeholder` detects it and `_fetch` steps the zoom down until the imagery is
real. Starting zoom now comes from the probed resolution: **30cm→z19, 60cm→z18, 1.2m→z17**.

⚠️ **z17 is ~1.07 m/pixel, so a 120 m detail chip is ~112 native pixels upsampled to 320.**
That is the honest ceiling for most of the sample and it is exactly what makes the pilot's
κ the live unknown the plan says it is. Realised over the whole draw: **z17 for 837
parcels, z18 for 82, z19 for 23** — so **89 % of the campaign is interpreted at ~1 m**.
2,224 chips rendered, **0 failures**, median 19 KB, 43 MB total.

**Chunking by extraction window alone spans the whole country.** Parcels sharing an
`imagery_date` month are scattered across all 14 departments, so a 25-parcel chunk's
bounding box covered most of Peru — and `filterBounds` then reduces *every S2 granule in
that rectangle, per image*. Cost scales with extent, not parcel count. The first extraction
run stalled at 3 of 89 chunks with the workers at 0 % CPU. Fixed by packing chunks under a
**1 deg² bbox cap** (`s2_gee.MAX_CHUNK_BBOX_DEG2`): 215 chunks, median bbox **0.004 deg²**.

⚠️ **`landsat_gee._chunk_todo` already carried this exact lesson, with the number
(4 deg²) in a comment.** It was learned twice because the new module mirrored the Landsat
module's *structure* and not its *packing*. If a third extraction module is ever written,
take the chunker, not the shape.

**A third: GEE throttles were being classified as failures.** The extraction died outright
at 144 of 215 chunks on `EEException: Too many concurrent aggregations`. That is not an
error in the request — it is a refusal to serve *right now*, and the identical request
succeeds a minute later. `_TRANSIENT_MSGS` did not list it, so `_retry` re-raised instead
of backing off. Nothing was lost (chunks are content-addressed and the job resumed), but a
long unattended run would have stopped for a condition it should have waited out.

Fixed in the **shared** `landsat_gee._TRANSIENT_MSGS`, so every extraction in the project
gets it: `too many concurrent`, `concurrent aggregations`, `restricted mode`,
`concurrency limit`. Pinned by `tests/test_s2_gee.py::TestTransientClassification`. The
other half of the fix is workers: **4 was too many for this project's quota, 2 is stable.**
Note RESULTS.md §11 recorded the *opposite* presentation of the same throttle — 5 workers
producing eight silent 900 s hangs rather than an exception. **It can arrive either way, so
handle both.**

---

## 4b. Sentinel-2 gives 2–4x the observations Landsat did

Measured over the extraction, per parcel *agricultural year* (`s2_coverage_report.csv`):

| department | median clear dates | | department | median clear dates |
|---|---:|---|---|---:|
| MOQUEGUA | 113.0 | | LAMBAYEQUE | 47.0 |
| AREQUIPA | 64.0 | | LA_LIBERTAD | 44.0 |
| ICA | 63.0 | | LIMA | 41.0 |
| TACNA | 58.0 | | AYACUCHO | 39.5 |
| TUMBES | 54.0 | | HUANCAVELICA | 39.5 |
| ANCASH | 52.5 | | CAJAMARCA | 33.5 |
| PIURA | 51.0 | | **PASCO** | **19.0** |

Against the Landsat store's **13–24**. Even Pasco — the cloudiest, on the selva edge — is at
the top of the Landsat range, and under 2 % of parcels anywhere fall below 10 clear dates.

⚠️ This matters beyond convenience. **Observation density is the one mechanism this project
has *measured* driving the panel failures**: probability compresses toward the base rate as
evidence thins (+0.052 per log-observation on true perennials, −0.022 on true annuals,
RESULTS.md §8), and the Landsat archive thinned from ~24 to ~13 obs/parcel-year exactly
across the window of interest. S2 removes the thinning and roughly triples the density. It
does **not** follow that the S2 model will be free of the artefact — that has to be
measured, not assumed — but the input that caused it is materially better.

---

## 5. What the campaign can and cannot report

Fixed by the budget, not by anything that can be improved later:

* **Enough** for a national model and a single national test number. The locked test is
  **201 parcels in 150 regions**, every department present on both sides. Report it with a
  binomial CI and lean on 5-fold CV for anything finer.
* **Not enough** for per-`(department × class)` accuracy — 42 cells at ~24 drawn each.
  Per-department LODO lands at n ≈ 43–85 and must be read as a spread, not a ranking.
* **No sierra or selva labels ever.** The 8 polygon-only departments (Amazonas, Apurímac,
  Cusco, Huánuco, Junín, Madre de Dios, Puno, Ucayali, + Callao) have no bridge, so no
  declared class, so no stratified draw. **This is a permanent scope limit of the model,
  not a to-do**: applying it to Cusco or Puno is extrapolation with nothing to check it
  against. It belongs in the model card as a limit.
* **The training prior is not the population prior.** Declared `PERENNIAL` is 9.9 % of the
  population and 51 % of the draw — a ~5x over-sample. Any probability read as a share is
  wrong by roughly that factor unless class weights or an explicit prior shift are applied
  first. The weights are in the table; the model has to be corrected too.

### The realised class mix, and why 2:1:1 was kept

The strata are 1996–2006 declarations and the labels are 2019+, so the mix depends on
transition rates nobody has measured. Persisted as `label_class_mix_sweep.csv`:

| declared→observed conversion | PERENNIAL | ANNUAL | OTHER |
|---|---:|---:|---:|
| low (5 %) | 44.2 % | 26.0 % | 29.8 % |
| central (15 %) | 47.5 % | 23.5 % | 29.0 % |
| high (30 %) | 52.5 % | 19.8 % | 27.8 % |

`PERENNIAL` lands at **44–53 %** across the whole plausible range and no class drops below
**~20 %**. That robustness — not the point estimate — is the reason to keep the tilt with
no correction round available. Almost all of it comes from the declared-`PERENNIAL`
stratum, which is stable because conversion runs one way.

---

## 5b. The pipeline was round-tripped end to end before handover

Not "the code has unit tests" — the **actual emitted shards** were put through the ingest:

* **blindness re-asserted on the raw text of all 9 files**: no `trainval`, no
  `PASTURE_FALLOW`, no `declared_class`/`split`/`fold`/`region_id` anywhere;
* synthetic CSVs from two labellers → `allperu s2-labels ingest` → κ computed on the 100
  overlap parcels, G1/G2/G3 evaluated, `labelled_parcels.parquet` written;
* → `allperu s2-labels transitions` → the weighted declared→observed matrix with CIs.

It found one real bug: the sample and the item key both carry `item_id`, so merging them on
`COD_PREDIO` produced `item_id_x`/`item_id_y` and the next merge died. **That would have
surfaced only when the real labels came back**, which is the worst possible moment. Fixed;
the key is now authoritative and the sample's copy is dropped.

The synthetic outputs were then **deleted** — nothing in `labels_s2/` is a label.

---

## 6. What is next, in order

1. **The pilot (plan §11 step 3) — the next half-day, and the only thing that can still
   kill this.** Both labellers do `pilot_A.html` and `pilot_B.html` (120 parcels each,
   ~4 h), the CSVs come back, `allperu s2-labels ingest` reports κ. **G1 needs κ ≥ 0.75.**
   If it fails, revise the codebook — *not* the sample. The imagery is 1.2 m for most of the
   country, so agreement at that resolution is the live unknown and it decides whether the
   other ~37 hours are worth spending.
2. **Label the four main shards + the overlap shard.** ~1,092 labellings, ~2 min each.
3. `ingest` → **G2** (`UNSURE` share < 25 %) and **G3** (≥35 usable per department,
   ≥150 per class over the **five** real classes — see §12.3, `NON_AGRICULTURE` has no
   stratum of its own and is expected to fall short).
4. `features/s2_assemble.py` → train → CV / locked test / LODO → **G4**: the S2 model must
   beat the **existing Landsat model scored on the same held-out labelled parcels**. Those
   Landsat predictions already exist, so it is the first apples-to-apples comparison this
   project can make.
5. **The transition matrix** (`allperu s2-labels transitions`) — weighted declared
   (1996–2006) → observed (2019+), with CIs and no classifier in it. A deliverable, not a
   planning step.

⚠️ **G4 is the one gate with no recovery path**, and that was decided in advance: if the S2
model loses, the labels become a validation set — still this project's first endpoint
accuracy measurement, and still worth having. Say so before the number arrives, not after.

---

## 11. Three changes after reviewing the emitted HTML (2026-08-13)

All three came from opening the actual files rather than from the tests. Chips and shards
were re-rendered; the suite is **375 tests, all passing**, ruff clean.

### 11.1 The right panel is a 100 m zoom, not a 600 m context view

`chips.CONTEXT_M = 600.0` became **`chips.ZOOM_M = 100.0`**, and the files it writes are
`<item_id>_zoom.jpg` instead of `<item_id>_context.jpg`. The left panel is unchanged: the
whole parcel, outlined yellow, neighbours cyan, at ≥120 m across.

**Why the old panel was the wrong second view.** Both panels were centred on the same point,
so at 600 m the context view showed the same landscape the detail view already showed, only
smaller — for a large parcel, *neither* panel resolved canopy. The discriminator the codebook
actually asks for is **texture**: regular crowns on a grid (PERENNIAL) against irregular
blobs (WOODY_NON_CROP) against smooth uniform tone (ANNUAL / bare). That is a property of a
~100 m patch, and it was the one thing the page did not show.

Checked on the largest drawn parcel (`M00891`, 48.97 ha): the detail panel shows the whole
field with a 500 m bar, the zoom panel shows individual crowns with a 25 m bar. The scale bar
is computed per panel (`_nice_bar`, a round number ≈ ¼ of panel width), so the two are not
silently comparable.

⚠️ **Most imagery is 1.2 m, so the zoom panel enlarges more than it resolves** (994 of 1,112
parcels render at z17). That is stated in the codebook: if it is not clear at 100 m across,
the information is not in the pixels, and the answer is `UNSURE`.

Re-render cost: 1,112 zoom chips, 6 workers, **0 failures**. Detail chips were cached and
reused. Total store **30.5 MB** across 2,224 files (down from 43 MB — a 100 m tile at the
same 320 px carries less detail than a 600 m one), largest shard **10.3 MB**.

### 11.2 `UNSURE` is a fifth label, not a low confidence

Added `UNSURE` on key `5` to `build_html.LABELS` and `ingest.LABELS`, with
`ingest.CLASSES` (the four real ones) and `ingest.ABSTAIN = "UNSURE"` kept separate.

**Why.** The original design had no abstain: a labeller who could not call a parcel was
expected to guess and set confidence to 1. In practice nobody moves a confidence control,
and a guess-at-confidence-1 is **indistinguishable from a real label downstream** — the
training set silently absorbs it. An explicit abstain is separable: `UNSURE` parcels are
excluded from training with `exclude_reason = "unsure"` and counted.

What changed in the gates:

* **G1 is now `kappa_called`** — κ over parcels *both* labellers actually called. κ over all
  five values conflates "we disagree about the land" with "one of us abstained", and those
  need different fixes (codebook vs imagery). Both are reported, plus
  `one_abstained_other_did_not` and a per-labeller abstain rate.
* **G2 is now `unsure_share < 0.25`**, replacing the confidence-1 share, which is also still
  reported as `low_confidence_share_among_called`.
* **G3 counts only the four real classes**, so abstains cannot pad a class toward its floor.
* `resolve()` gained a `"one_abstained"` outcome: the parcel takes the label the other
  labeller gave, flagged.

Confidence is kept, and now means what it says — it grades parcels that *were* called.

⚠️ `kappa_called` returns `nan` when the called parcels are all one class. That is correct
(κ is undefined there) and it **fails** G1 rather than passing it; pinned by
`tests/test_ingest.py::TestDegenerateKappa`.

### 11.3 The labeller types their own name

The page opens with a name box (`id="who"`). Nothing is downloadable until it is filled, and
the name is what goes into the CSV's `labeller` column and the filename
(`shard01_ana.csv`). Progress autosaves under `s2label_<shard>_<name>`, so two people can
label on one machine without clobbering each other, and work done before a name was typed
migrates to the name when it is set.

**Why.** The old design took the id from a constant baked in at build time, which forced the
build to know who the labellers would be — and if a file was forwarded to someone else, the
CSV would quietly claim the wrong person did the work. κ between two people is only
meaningful if the file says who they were.

The `_A`/`_B` in the filename survives as a **routing suggestion**: it is how the work is
divided and how the overlap shard reaches both people. It no longer determines anything in
the data.

Item order is seeded per (shard, labeller-slot) via md5, so `overlap_A.html` and
`overlap_B.html` hold the **same 100 parcels in different orders** — verified on the emitted
files — which keeps order effects from inflating κ. md5 rather than `hash()`, which is
salted by `PYTHONHASHSEED` and would not reproduce across runs.

### 11.4 Re-verified on the rebuilt files

* **Blindness**, on the raw text of all 9 shards: no `trainval`, `PASTURE_FALLOW`,
  `declared_class`, `region_id`, `"fold"` or `"split"`. Every shard carries the `UNSURE` key
  and the name box.
* `overlap_A` / `overlap_B`: same set of 100 item_ids, **different order**.
* **Full ingest round-trip re-run** on the rebuilt shards with synthetic CSVs from two named
  labellers (`ana`, `ben`): 1,112 parcels ingested, 1,073 usable, 33 excluded as `unsure`,
  6 as `no_s2_observations`; per-labeller abstain rates and `one_abstained_other_did_not`
  reported; all four gates evaluated. κ ≈ 0 as expected — the synthetic labels are
  independent draws. **Synthetic outputs deleted afterwards**; nothing in `labels_s2/` is a
  label.

---

## 12. Five changes after the second review (2026-08-13)

Like §11's, these came from looking at the emitted artefacts rather than from the tests.
Four are cheap and local. The fifth — the NDVI ribbon — turned out to need data the store
did not contain, and is the only one that cost a re-extraction.

**Where §11 and §12 disagree, §12 wins.** §11.1's 100 m zoom, its 30.5 MB chip store and
its five-value vocabulary are all superseded below; the *reasoning* in §11 is unchanged and
is why these are the changes they are.

### 12.1 The zoom panel is 200 m across, not 100 m

`chips.ZOOM_M` 100 -> 200. One constant, but it forced the panel below it to move too: at
200 m the old left panel — parcel bbox floored at **120 m** — would have been *narrower*
than the "zoom" for every parcel under ~150 m across, which is **half the draw**. Two panels
where the context view is the more magnified one is worse than one panel, so §12.2 is not an
independent change; it is the other half of this one.

### 12.2 The left panel is a real context view

`DETAIL_MIN_M = 120.0` became **`CONTEXT_MIN_M = 400.0`** with proportional padding, and the
files it writes are `<item_id>_context.jpg`. The extent is now

```
context_extent_m(span) = max(span + 2 * min(0.5 * span, 300 m), 400 m)
```

— padding of **half the parcel's own extent on each side**, capped at 300 m absolute,
floored at 400 m. Chosen, not defaulted:

| knob | why | realised |
|---|---|---|
| **proportional** (0.5) | a big field is not swamped by margin, a small one is not starved of it | the old panel was a flat 1.24x the parcel — 12 % margin |
| **capped** (300 m) | the largest parcel in the draw is 2,127 m across; uncapped padding would demand a ~4 km tile mosaic for one chip | max **2,727 m** |
| **floored** (400 m) | the 5th-percentile parcel is 66 m across; at 1.24x it filled the frame and showed no setting at all | **400 m for everything up to the median** |

**Why the old panel was the wrong first view.** §11 fixed the *second* panel and left the
first one showing the parcel almost edge to edge. But the same crop reads differently in
different geographies — rice beside a river, orchard on a town edge, pasture against forest
— and the campaign spans 14 departments from Tumbes to Tacna. Checked on the emitted files:
the median parcel (`M00237`, 0.72 ha, Arequipa) now shows the **dry riverbed** running past
it and the terraced fields above; at the old 195 m it showed the parcel and a rim of dirt.
The largest (`M00891`, 48.97 ha, Pasco) shows the **river and forest edge** it sits against
— exactly the discriminator between a selva perennial and woody non-crop.

⚠️ **The stale-chip trap was real and was avoided deliberately.** An earlier revision of
this campaign wrote `*_context.jpg` for a fixed 600 m view. A leftover from it would have
loaded into the page **silently and wrongly** rather than failing, so the whole chip
directory was deleted before re-rendering rather than overwritten. Verified after: 2,224
files, `context` + `zoom` only, **zero `_detail.jpg` survivors**.

The neighbour lookup was made to track the panel. It was a fixed +/-600 m box; under a
2.7 km panel that leaves the outer fields unoutlined, which reads to a labeller as *"this
parcel has no neighbours"* rather than as a missing lookup. It is now derived from
`context_extent_m` itself, so widening the panel can never outrun it again.

### 12.3 `NON_AGRICULTURE` is a sixth label, and `OTHER` was narrowed to make room

Added on key **6** to `build_html.LABELS` and `ingest.LABELS`/`CLASSES`. `UNSURE` stays on
**5**: it was added in §11 and renumbering the abstain to make the list read tidily would
change what a briefed labeller's fingers do.

**The class overlap was the whole risk.** `OTHER` was defined as *"everything else: pasture,
fallow, prepared bare ground, scrub, natural vegetation, water, built-up, road, riverbed"* —
the last four of which are exactly `NON_AGRICULTURE`. Adding the class without narrowing the
old one would have left two labellers guessing at the same parcel, and κ is the gate that
decides whether the other ~37 hours get spent. So `OTHER` is now **farmable land that is not
currently a crop**, and the codebook carries a separating test stated as one question:

> **Could this ground be sown next season exactly as it stands?**
> Yes -> `OTHER`. No — something would have to be demolished, dug up or drained first, or it
> is permanently water, rock or pavement -> `NON_AGRICULTURE`.

With a worked table in both directions (fallow and ploughed soil are `OTHER`; riverbed sand
and quarry floor are `NON_AGRICULTURE`) and a **priority ladder**, because the boundary that
actually bites is three-way: `WOODY_NON_CROP` is tested *before* `NON_AGRICULTURE`, so
riparian trees along a river are woody non-crop while the water and gravel beside them are
not. `tests/test_build_html.py` asserts the emitted page carries the rule and that `OTHER`
no longer claims `water` / `built-up` / `riverbed`.

**Why it is worth a class at all:** a fallow field can convert to a perennial and a road
cannot. Pooling them puts a structurally impossible outcome in the same class as the one
this project exists to measure.

⚠️ **G3 gets harder and that is intended.** The per-class floor is >=150 over what are now
**five** real classes, and `NON_AGRICULTURE` was carved out *after* the draw was fixed, so it
has no stratum of its own and may well land below the floor. That is a decision to pool at
training time, not a reason to re-draw — the gate now says so in its own `note`, and
`classes_below` names which.

### 12.4 The confidence control is gone

Removed from the page (`c`, the button, `cycleConf`), from the CSV, from `resolve()`, from
the persisted table and from `_gates`.

§11.2 added `UNSURE` because *"a confidence control that has to be actively set does not get
used"*, and then kept confidence anyway "as a secondary graded signal". That left **two ways
to record doubt** — one hard and read by a gate, one graded and never moved off its default
— and a 1-of-3 guess is still indistinguishable from a real label once it is in the training
set. One abstain that works beats one that works plus one that does not.

**What G2 reports afterwards:** it is now a **single number**, `unsure_share < 0.25`. It
previously carried `low_confidence_share_among_called` beside it as a softer reading of the
same quantity; that key is gone, and a test asserts the gate dict is exactly
`{value, criterion, pass}` so it cannot creep back. `read_csvs` still *accepts* a CSV
carrying a `confidence` column — any file downloaded before the change must ingest rather
than strand real work — it is simply surplus.

### 12.5 The NDVI ribbon — and the data it needed did not exist

The trace now draws a shaded **p25-p75 band** behind the median line: the spread of NDVI
**across the parcel's own pixels on each date**.

⚠️ **`s2_perdate.parquet` could not support it.** The store was one row per parcel-date
holding *band medians and a pixel count* — 117,755 x 11 columns, no within-parcel
quantiles. The ribbon the request most naturally means is a property of the pixel
distribution, and no arithmetic on medians recovers it. Two things had to change:

1. **`ee.Reducer.percentile([25, 75])` joined the combine** in `s2_gee.perdate_chunk`,
   `sharedInputs=True`, so median + count + quartiles come out of **one** pass over the
   pixels. A second `reduceRegions` would have re-read the imagery, which is the expensive
   part.
2. **NDVI is now formed per pixel server-side** (`S2_NDVI_BAND`), because *a quantile of a
   ratio is not the ratio of the quantiles*. `(NIR_p25 - R_p25)/(NIR_p25 + R_p25)` is not
   the 25th percentile of NDVI and would have drawn a band of the wrong width.

**And that forced the line to move too.** The per-pixel median and the NDVI recomputed from
band medians are different quantities — measured at **0.0025 NDVI median absolute
difference** on this store, small but not zero — so plotting one inside the other's band
would let the median sit visibly *outside* its own ribbon on some dates, which reads as a
bug. `build_traces` therefore draws the line from `NDVI_px_p50` whenever the quartiles are
present. **The model features are untouched**: they still use the band-median NDVI, and the
new columns are deliberately named `NDVI_px_*` so `indices.add_indices`'s `NDVI` cannot
collide with them.

**What the ribbon is for, stated in the codebook so it is not misread:** it is **not an
error bar**. Narrow = the parcel is doing one thing everywhere; persistently wide = it is
internally varied — crowns against bare inter-row, or genuinely half one thing and half
another, which is a prompt to apply the >50 % rule or press `5`.

The renderer degrades rather than fails: a store without the quartile columns still builds a
shard with the plain median line, pinned by a test.

**The re-extract, and what it verified.** 215 chunks, 2 workers, ~2 h — GEE reported
*"exceeded the compute quota of its noncommercial tier … restricted mode"* on every call
throughout, which throttled it to ~2.5 chunks/min but produced **no errors, no timeouts and
no split retries**. The 215 cached chunks had to be **invalidated by hand**: they are
content-addressed on parcel-set + window, so a re-run would have found every filename
present and skipped it, and the new columns would simply never have appeared.

The rebuilt store is **117,768 parcel-dates over 1,106 parcels** against the old
117,755/1,106, and that near-equality is the check worth having:

| | |
|---|---|
| old parcel-dates reproduced | **117,755 of 117,755 (100 %)** |
| bands moving >1 DN | **93 rows, 0.079 %** (20 % of them granule splits, the rest GEE re-run non-determinism) |
| quartiles non-null | **1.0000** |
| `p25 <= p50 <= p75` violations | **0** |
| median IQR width | **0.0835 NDVI** |
| line inside its own band | **100 %** of dates |

Coverage reproduces §4b exactly (Moquegua 113.0, Piura 51.0, Pasco 19.0 median clear dates
per parcel-agricultural-year) and the same **6** parcels have no S2 date at all. The
harmonisation check was re-run against the new store and still verifies: placebo-corrected
**SWIR1 +0.0092, R +0.0090, NIR +0.0015** against §3b's +0.0107/+0.0087/+0.0025, an order of
magnitude below the +0.10 an unremoved offset would give.

⚠️ **The clip was added because of what the ribbon does when it escapes the axes.** `Y` was
always clamped; `X` never was. A stray circle outside the plot rect is a dot; a *filled*
band is a smear across the whole card. Measured on a real parcel, the band reaches x=338.2
against a plot edge at x=334 — small, but real — so the data marks are now inside a
`clipPath`. The dashed imagery-date rule stays outside it, being an axis annotation.

### 12.6 Re-verified on the rebuilt files

* **Blindness, on the raw text of all 9 shards**: no `trainval`, `PASTURE_FALLOW`,
  `declared_class`, `region_id`, `"fold"`, `"split"`, `"stratum"`, `"weight"`, `"batch"`,
  `"crop_set"`, `"label_id"`, and no declared-class value in any payload.
* ⚠️ **One real gap found, and it is pre-existing.** `build_html.FORBIDDEN_FIELDS` contains
  `overlap`, and `overlap_A/B.html` do contain the string once — as
  `const SHARD_ID="overlap"`, the file's own name, also shown in the visible header. **The
  blindness test had never seen it**: its fixture sets `overlap=False` for every row, so no
  build under test ever produced an overlap shard. A test now asserts blindness *on an
  overlap shard*, with the shard-id occurrence as the single documented exception and
  `count == 1` pinned. **Not renamed** — the filenames carry the routing (§D2) — but the
  consequence is on the record: a labeller can tell which shard is double-labelled, and κ is
  measured on exactly that shard, so it may be labelled more carefully than the ones that
  train the model.
* **The six-value vocabulary is on every shard** (`'6':'NON_AGRICULTURE'`, `'5':'UNSURE'`),
  the CSV header is `item_id, labeller, label, crop_guess, boundary_mismatch,
  seconds_spent, timestamp` — **no `confidence`** — and `cycleConf` / `id="conf"` /
  `e.key==='c'` are absent from all 9.
* **The ribbon is on 1,105 of 1,112 items** (the 6 with no S2 trace, plus one parcel with a
  single observation — a band needs two points).
* `overlap_A` / `overlap_B`: **same 100 parcels, different order**.
* **Full ingest round-trip re-run** on the rebuilt shards with synthetic CSVs from two named
  labellers, drawing all six values: 1,112 parcels ingested, 981 usable, 98 `unsure`, 26
  `boundary_mismatch`, 7 `no_s2_observations`. G2's gate dict is exactly
  `{value, criterion, pass}`. G1 fails at κ = 0.015, which is correct — the synthetic labels
  are independent draws — and G3's per-class floor fails at 84, which is the §12.3
  consequence showing up exactly where predicted. Transition matrix rebuilt with
  `NON_AGRICULTURE` as a column. **Synthetic outputs deleted afterwards; nothing in
  `labels_s2/` is a label.**
* Shards are **7.7–13.5 MB** (was 10.3 MB max), inside the 16 MB cap with less headroom —
  the chip store grew 30.5 → 41.6 MB because both panels now cover more ground. The
  superseded median-only store was **deleted**, not kept beside the new one: a stale
  `s2_perdate_medianonly.parquet` is the same trap as a stale `*_context.jpg`.

**Suite: 409 tests, all passing, ruff clean** (was 375 at §11).

