# S2 endpoint labelling — the live campaign

**Goal:** a photo-interpreted training set for a **3-class Sentinel-2 parcel classifier for
2019 onwards**, covering all 14 linkable departments, with a frozen train/test split.

**Why it is the only remaining route:** every accuracy figure in [`RESULTS.md`](../RESULTS.md) is
measured in ~1997–2006. **Not one is measured in 2019–2023**, which is the only period the
research question is about. Four estimands failed and every technical mitigation has been tried
and measured. This is the one plan that *verifies* the classifier instead of working around it.

Codebook (frozen, embedded in the HTML): [`codebook.md`](codebook.md).
One entry point: `uv run python -m crop_classifier.cli allperu s2-labels <step>`.

---

## 1. State — everything except the labelling is built

| | |
|---|---|
| Eligible universe | **614,876 parcels** (14 depts, `area_ha ≥ 0.15`) |
| Esri centroids probed | 4,519 — **G0 PASSES at 87.6 %** ≤1.2 m and ≥2019 |
| Chips rendered | **2,224** (0 failures) |
| S2 parcel-dates extracted | **117,768** |
| Labelling shards | **9 HTML files covering all 1,112 parcels**, largest 13.5 MB |
| Draw | **992 main + 120 pilot**, split frozen in `config/split_s2labels.yaml` |
| Test fold | 201 parcels in 150 regions, every department on both sides |

⭐ **Sentinel-2 gives a median 19–113 clear dates per parcel-agricultural-year by department,
against the Landsat store's 13–24.** Even Pasco, the cloudiest, sits at the top of the Landsat
range. Observation density is the one mechanism this project has *measured* driving the panel
failures ([`RESULTS.md`](../RESULTS.md) §5), and S2 roughly triples it. **That does not mean the
artefact is gone — measure it, do not assume.**

The pipeline was **round-tripped end to end on the real shards** before handover: blindness
re-asserted on the raw text of all 9 files, synthetic CSVs through `ingest` → κ → G1/G2/G3 →
`labelled_parcels.parquet` → `transitions`. It found one real bug (an `item_id` collision
producing `item_id_x`/`item_id_y`) that would otherwise have surfaced the moment the real
labels came back. The synthetic outputs were then deleted — **nothing in `labels_s2/` is a
label.**

---

## 2. ⭐ What is left

This is the only outstanding work in the project. Steps 1–2 are human.

### 1. The pilot — the next half-day, and the only thing that can still kill this

Both labellers do `pilot_A.html` and `pilot_B.html` (120 parcels each, ~4 h), the CSVs come
back, then:

```bash
uv run python -m crop_classifier.cli allperu s2-labels ingest --csv-dir <returned CSVs>
```

**G1: `kappa_called` ≥ 0.75** — κ over parcels *both* labellers actually called, so
"we disagree" and "one abstained" stay separable.

⚠️ **If G1 fails, revise the codebook — not the sample.** The imagery is 1.2 m for most of the
country, so agreement at that resolution is the live unknown, and it decides whether the other
~37 hours are worth spending. With only one labeller available, κ becomes intra-rater: re-label
100 parcels after a gap.

### 2. Label the main shards

Four main shards plus the overlap shard: ~1,092 labellings at ~2 min each, **~37 h human**.
Then `ingest` again:

* **G2:** `UNSURE` share < 25 %. Failure means the imagery or the class definitions are unfit.
* **G3:** ≥35 usable labels per department and ≥150 per class nationally.

⚠️ **G3 will probably fail for `NON_AGRICULTURE`.** Its ≥150 floor now spans **five** real
classes and `NON_AGRICULTURE` has no stratum of its own. **Expect it to fall short and pool at
training time rather than re-draw** — there is no second round.

### 3. Train and compare

```bash
uv run python -m crop_classifier.cli features assemble    # via features/s2_assemble.py
uv run python -m crop_classifier.cli train --model lightgbm ...
uv run python -m crop_classifier.cli allperu lodo --tag s2
```

**G4: the S2 model must beat the existing Landsat model scored on the same held-out labelled
parcels.** Those Landsat predictions already exist, so this is the first apples-to-apples
comparison the project can make.

⚠️ **G4 is the one gate with no recovery path, and that was decided in advance.** If the S2
model loses, the labels become a **validation set** — still the project's first endpoint
accuracy measurement, and still worth having. This is on the record *before* the number
arrives, not after.

### 4. The transition matrix — a deliverable, not a planning step

```bash
uv run python -m crop_classifier.cli allperu s2-labels transitions
```

The weighted **declared (1996–2006) → observed (2019+)** transition matrix, with CIs and **no
classifier in it**. This is the descriptive conversion estimate the project has never been able
to produce. Report it weighted, with CIs, and state where 1,000 labels sits on the learning
curve as a limitation.

---

## 3. Decisions already frozen

* **Six label values:** `PERENNIAL` / `ANNUAL` / `OTHER` / `WOODY_NON_CROP` / **`UNSURE`** /
  `NON_AGRICULTURE`, plus a boundary-mismatch flag.
* **`UNSURE` is a label, not a low confidence.** The original design had no abstain, and a guess
  recorded at confidence 1 is *indistinguishable from a real label downstream*. **Confidence was
  then removed entirely** — two ways to record doubt, one of which corrupts the data, is worse
  than one.
* **`OTHER` is narrowed to "farmable land not currently a crop"**, because the old `OTHER`
  explicitly listed water/built-up/road/riverbed; adding `NON_AGRICULTURE` without narrowing it
  would have put two labellers on opposite sides of the same parcel. The separating test, in the
  codebook and asserted on the emitted HTML: ***could this ground be sown next season exactly as
  it stands?*** Plus a **priority ladder** — `WOODY_NON_CROP` before `NON_AGRICULTURE`, so
  riparian trees are woody and the gravel beside them is not.
* **Two Esri panels plus a 24-month S2 NDVI trace.** The trace is what makes ANNUAL-vs-OTHER
  callable at all. The right panel is a **200 m zoom** (texture is the actual
  PERENNIAL-vs-WOODY_NON_CROP discriminator); the left is a real **context** view sized
  `max(span + 2·min(0.5·span, 300 m), 400 m)` — proportional, capped, floored, because the same
  crop reads differently in different geographies. Scale bars are per panel and the neighbour
  box derives from the panel extent.
* **The labeller types their own name.** It goes into the CSV and the filename, progress is keyed
  to it, and `_A`/`_B` in a filename is only a routing suggestion. Item order is md5-seeded per
  (shard, slot), so the two overlap shards hold the same parcels in different orders.
* **Blindness is asserted on the raw HTML string** — no declared class, split or fold appears
  anywhere (`tests/test_build_html.py`). ⚠️ One documented exception: `overlap_A/B.html` contain
  the string `overlap` once, as their own `SHARD_ID`. So **a labeller can tell which shard is
  double-labelled, and κ is measured on exactly that shard.** Pinned by a test.

---

## 4. ⚠️ Known limits — state these in the model card

* **One round, 1,000 parcels. There is no second pass.** The allocation had to be defensible
  before it was drawn rather than corrected afterwards.
* **What 1,000 buys:** a national model and **one national test number**. **Not**
  per-(department × class) accuracy — that is 42 cells at ~24 each.
* **No sierra/selva labels, ever.** The 8 polygon-only departments have no bridge, so no declared
  class, so no stratified draw. A permanent limit, not a to-do.
* **The training prior is a ~5× `PERENNIAL` over-sample.** Prior-correct the *model*, not just
  the estimates.
* ⚠️ **Piura is one of the three worst-covered departments for recent high-res imagery** (0.61
  eligible, with Cajamarca 0.60 and Pasco 0.39; eight departments are 1.00). Every earlier
  strand was built on Piura — a campaign scoped to Piura would have lost a third of its draw,
  and it would have looked like a sampling bug.
* The 8-parcel shortfall against 1,000 is **places, not parcels**: Huancavelica has 44 populated
  5 km regions in the entire universe, against a 2-per-region cap.

---

## 5. Build record — what a future agent needs to know

### The harmonisation was verified, and verifying it took three attempts

The obvious before/after-the-cut comparison reports a **+0.055 NDVI "step" that is entirely the
growing season**. A within-month-of-year pairing looks principled and still returns ±0.018 on a
pure sine. What works: **fit the seasonal cycle out per parcel, subtract a placebo cut one year
earlier as the noise floor, and read the verdict off raw bands, never an index.** Result:
SWIR1 / R / NIR steps **+0.009 / +0.009 / +0.002**, against **+0.10** for an unremoved +1000 DN
offset. NDVI reads −0.034 and that is Peruvian weather, not Sentinel-2.

### ⚠️ The NDVI ribbon needed data that did not exist

`s2_perdate.parquet` held band **medians** and a pixel count — no within-parcel quantiles — and
**no arithmetic on medians recovers a quantile**. Fixed by adding
`ee.Reducer.percentile([25,75])` to the `perdate_chunk` combine (`sharedInputs=True`, so it is
one pass over the pixels, not a second `reduceRegions`) **and by forming NDVI per pixel
server-side**, because a quantile of a ratio is not the ratio of the quantiles. That forced the
plotted line onto `NDVI_px_p50`: it differs from band-median NDVI by 0.0025, enough to put the
median outside its own band on some dates. Model features are untouched and the columns are
named `NDVI_px_*` so they cannot collide with `add_indices`.

⚠️ **The 215 cached chunks were content-addressed on parcel-set + window, so the new columns
would have been silently skipped — the cache had to be invalidated by hand.** Re-extraction:
215 chunks / 2 workers / ~2 h, no errors, with GEE in restricted mode throughout. **100 % of the
117,755 old parcel-dates reproduced**, 0.079 % of bands moving >1 DN, 0 quartile-ordering
violations, median IQR 0.0835 NDVI.

### Two silent failures, both caught only by looking at the output

1. **Esri returns a valid flat-grey image above the zoom it serves** — `zoom="auto"` on a 120 m
   extent rendered blank chips with no error. Zoom now comes from the probed resolution
   (30 cm→z19 / 60 cm→z18 / **1.2 m→z17**) with a placeholder detector.
2. **Chunking by extraction window alone spans the whole country**, because parcels sharing an
   imagery month sit in all 14 departments; `filterBounds` then reduces every granule in that
   rectangle and the first run stalled at 3 of 89 chunks at 0 % CPU. Fixed with a 1 deg² bbox
   cap. ⚠️ `landsat_gee._chunk_todo` **already carried that exact lesson in a comment** — it was
   learned twice because the new module copied the old one's structure, not its chunker.

### A third GEE throttle shape

`Too many concurrent aggregations` was not in `landsat_gee._TRANSIENT_MSGS`, so `_retry`
re-raised instead of backing off and killed a running extraction at 144 of 215 chunks. Now
classified transient — **and note the same throttle also arrives as silent 900 s hangs. It comes
both ways; handle both.** 4 workers is over this project's quota; **2 is stable**.
