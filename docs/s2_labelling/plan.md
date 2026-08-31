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

## 1. State — labelling essentially complete, models trained, one shard outstanding

**Updated 2026-08-28.** `shard01`–`shard03` came back on 2026-08-27, joining `pilot`,
`shard04` and `shard05`. **1,012 of 1,112 labellings returned, 865 usable, one annotator.**
Every model arm has been fitted. Numbers: [`RESULTS.md`](../RESULTS.md) §8.1–8.2b.

| | returned | |
|---|---|---|
| labellings | **1,012** of 1,112 | 865 usable (139 `UNSURE`, 6 with no S2 observation, 2 boundary mismatch) |
| **G1** κ_called ≥ 0.75 | **unmeasured** | the overlap shard is still not in the return — an *absence*, not a failure. **The only blocker left** |
| **G2** `UNSURE` < 25 % | **0.139 ✅ PASS** | at 1.2 m the annotator could call 86 % of parcels |
| **G3** ≥35/dept | **48 ✅ PASS** | ⭐ first time. All **14** departments clear it, against 4 at the last read — which is what makes LODO a real estimate |
| **G3** ≥150/class | **54 ⛔ FAIL** | `NON_AGRICULTURE`, as pre-registered → pool into `OTHER` (`--target t4`). `PERENNIAL` (115) and `ANNUAL` (141) are also under the floor |
| locked test | **UNSPENT** | 201 parcels, **161 labelled**. `allperu s2-train` exposes no `--eval-test`, and nothing in §8 reads it |

⭐ **Step 3 has now been executed, not rehearsed, and it reversed its own earlier reading.**
LTAE wins CV in **8 of 8** arms and loses leave-one-department-out in **4 of 4** targets. See
§2 step 3 below — the recommendation there has flipped.

---

## 1b. What was built to run step 3

| | |
|---|---|
| Eligible universe | **614,876 parcels** (14 depts, `area_ha ≥ 0.15`) |
| Esri centroids probed | 4,519 — **G0 PASSES at 87.6 %** ≤1.2 m and ≥2019 |
| Chips rendered | **2,224** (0 failures) |
| S2 parcel-dates extracted | **117,768** |
| Labelling shards | **9 HTML files covering all 1,112 parcels**, in Spanish (`labels_s2/html_es/`), largest 13.5 MB |
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

Steps 1–2 are human and are the only blockers.

### 1. ▶ The overlap shard — G1 is still unmeasured

**G1: `kappa_called` ≥ 0.75** — κ over parcels *both* labellers actually called, so
"we disagree" and "one abstained" stay separable. The return so far is **one annotator and no
overlap shard**, so κ does not exist yet and the campaign currently has **no measure of its own
label noise**. Every accuracy figure in `RESULTS.md` §8.2 inherits that.

With one labeller κ becomes **intra-rater: re-label 100 parcels after a gap.**

⭐ **It is now the only labelling worth doing.** The learning curve has flattened — the last
+14 % of training data moved LightGBM +0.021 and LTAE −0.001 — so 100 more *training* labels
buy almost nothing, while 100 *re-labels* buy the one number the campaign has never had.

⚠️ On the record beside it: the annotator's **median time per parcel is 2–4 seconds**
(76 % of `shard01` under 5 s), against this plan's ~2 min budget. That does not make the labels
wrong. It means the noise floor is unquantified, and G1 is what would quantify it.

⚠️ **If G1 fails, revise the codebook — not the sample.**

### 2. ✅ The remaining shards are in

`shard01`–`shard05` + `pilot` returned. **G2 passes at 0.139. G3-per-department passes at 48 —
all 14 departments.** G3-per-class fails for `NON_AGRICULTURE` (54 against 150), exactly as
pre-registered: **pool it into `OTHER` via `--target t4`. There is no second round; do not
re-draw.**

⭐ **The learning curve has flattened.** At 354 labels, folding in the 120-parcel pilot
(+48 % training data) moved LightGBM **+0.050** macro-F1 on `t4`. At 865 labels the same pilot
(+14 %) moves LightGBM **+0.021** and LTAE **−0.001**. **More labels are no longer the
constraint — label *noise* is, and that is G1.**

### 3. ✅ Train and compare — done, and the default flipped back

```bash
uv run python -m crop_classifier.cli allperu s2-labels ingest --csv-dir data/processed/labels_s2
uv run python -m crop_classifier.cli allperu s2-train prep            # 8 workspaces
uv run python -m crop_classifier.cli allperu s2-train fit  --model lightgbm --target t4 --pilot
uv run python -m crop_classifier.cli allperu s2-train lodo --model lightgbm --target t4
uv run python -m crop_classifier.cli allperu s2-train report
```

⚠️ **Start from LightGBM, not LTAE.** This reverses what this section said before, and the
reversal is the interesting part.

* On **cross-validation** LTAE wins **8 of 8** arms (+0.013 to +0.068, p < 0.05 in 5 of 8).
* On **leave-one-department-out** LightGBM wins **4 of 4** targets: `t5` 0.543 vs 0.496, `t4`
  0.539 vs 0.515, `t3` 0.623 vs 0.620, `t3w` 0.697 vs 0.633.
* From CV to LODO, LightGBM falls 0.05–0.13 and **LTAE falls 0.14–0.19** — 1.4–3.4× more, every
  target.

**LTAE's extra CV skill is skill that does not leave the training departments.** Given 47 dates
per parcel and ~500 training parcels, the attention encoder finds structure that identifies the
*department*; that pays inside it and is worth nothing outside. It is what `centroid_lat` does,
produced by a model class instead of a feature.

⚠️ **What actually changed is the evidence, not the store.** The earlier "LTAE wins LODO in
both arms" was a mean over **4** departments (SD 0.15) — the only four that then had ≥40 labels.
There are now **14**. *Always report a LODO figure as "0.539 over 14 departments".*

⚠️ **LODO is not optional**, unchanged and now demonstrated: the CV→LODO drop here is
**−0.05 to −0.19 macro-F1**, and it is the axis on which the model ranking inverts.

⛔ **Two extra targets were run and one of them should not be adopted.** `t2`/`t2w` collapse the
label space to `PERENNIAL` vs `NON_PERENNIAL`. Raw macro-F1 rises (`t2` 0.769 CV), but so does
the majority-class floor — **0.467 at two classes against 0.171 at four** — and normalised
against it `t2` is the **lowest-skill arm in the study** (LODO skill 0.180). Collapsing 3
classes to 2 moves `PERENNIAL` F1 by **+0.004**. ⭐ What it does settle: moving
`WOODY_NON_CROP` from one side to the other moves `PERENNIAL` F1 by **+0.190** — the binding
constraint is that one codebook boundary, not the number of classes, **and G1 is the thing that
would measure it.** `RESULTS.md` §8.2c.

⚠️ **One arm per process** — LightGBM and torch cannot share one on macOS. The CLI steps are
separate commands for that reason; drive them from a shell loop.

### 3b. ⚠️ G4 must be restated before it can be adjudicated

The gate as written says *"the S2 model must beat the existing Landsat model scored on the same
held-out labelled parcels — those Landsat predictions already exist"*. **They do not.** The
national panel covers **22 of the 1,112 drawn parcels**: the campaign drew from the
614,876-parcel eligible universe, not from the 4,565-parcel panel sample. That was an error in
this plan, found by running the step rather than by reading it.

The available substitute — loading the saved Landsat booster and scoring it on the labelled
parcels' **S2** features — is a **cross-sensor lower bound, not a measurement**, for exactly the
reason §6.4 closed the OLI route: the sensor difference is cover-type dependent and no global
linear map removes it. The distortion is visible (ANNUAL recall 0.932 at precision 0.25–0.31).

**Pick one, before the labelling finishes:**

* **(a)** extract genuine Landsat features for the labelled parcels' imagery years and score the
  existing model on those — a real apples-to-apples comparison, at the cost of one more GEE job;
* **(b)** drop the comparison and let G4 become the S2 model's own locked-test number against a
  stated floor.

⚠️ **G4 still has no recovery path.** If the S2 model loses, the labels become a **validation
set** — still the project's first endpoint accuracy measurement, and still worth having. That
was on the record before any number arrived and it stays there.

### 4. The transition matrix — a deliverable, not a planning step

```bash
uv run python -m crop_classifier.cli allperu s2-labels transitions
```

The weighted **declared (1996–2006) → observed (2019+)** transition matrix, with CIs and **no
classifier in it**. This is the descriptive conversion estimate the project has never been able
to produce. Report it weighted, with CIs, and state where 1,000 labels sits on the learning
curve as a limitation.

✅ **Run, on all 865 usable labels** —
`labels_s2/declared_to_observed_transitions.csv`, tabulated in
[`RESULTS.md`](../RESULTS.md) §8.2b. The headline: of parcels declared `ANNUAL` in 1996–2006,
**2.9 % [0, 5.9] read as perennial in 2019+** and **57.9 %** read as farmable ground not
currently cropped.

⚠️ And the declared-`PERENNIAL` row is why the `WOODY_NON_CROP` mapping is a *reported target*
rather than a preprocessing step: only **21.1 %** still read `PERENNIAL`, **32.9 %** read
`WOODY_NON_CROP` and 32.4 % read `OTHER`. A third of the perennial sample turns on the
codebook's hardest call, and G1 — the measurement that would say how reliably it is made —
still does not exist.

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
* **The delivered set is Spanish** (`labels_s2/html_es/`, built with `s2-labels html --lang es`);
  the English build stays as a reference in `labels_s2/html/`. ⚠️ **Translation touches the
  display name only** — the value written to the CSV, and every value `ingest.py` compares
  against, stays the canonical English constant. Localising the stored value would have made the
  ingest match nothing and report an empty label distribution *rather than an error*, which is
  the same class of silent failure as the four data traps.
* **The label is stored verbatim; the class mapping is a modelling decision downstream.**
  `ingest` writes the annotator's six values untouched. `labelling/train_prep.TARGETS` holds the
  four readings — `t5` (all five real classes), `t4` (`NON_AGRICULTURE` → `OTHER`, the
  pre-registered pooling), `t3` (woody + non-ag dropped, the clean Landsat head-to-head) and
  `t3w` (woody → `PERENNIAL`, every parcel kept) — **each as its own workspace on disk**. The
  gap between `t3` and `t3w` is then a measured quantity, not an assumption. The Landsat panel
  independently calls **5 of 5** woody parcels `PERENNIAL`, which is what `t3w` had assumed.
* **Blindness is asserted on the raw HTML string** — no declared class, split or fold appears
  anywhere (`tests/test_build_html.py`). ⚠️ One documented exception: `overlap_A/B.html` contain
  the string `overlap` once, as their own `SHARD_ID`. So **a labeller can tell which shard is
  double-labelled, and κ is measured on exactly that shard.** Pinned by a test.

---

## 3b. Build record — running step 3 on the first third

Three things were found by running the pipeline that reading it would not have surfaced.

1. **G4's premise was false** (§2 step 3b) — the "existing Landsat predictions" cover 22 of
   1,112 parcels, because the campaign and the panel drew from different populations.
2. **`quality_ok` would have trained on zero rows.** It is a *Landsat* extraction-quality flag,
   NA for every parcel in this campaign, and `load_parcels` filters on `== True`. Forwarding it
   untouched produces an empty dataset **and no error** — the same failure shape as the four
   data traps. `train_prep` sets it explicitly; pinned by a test.
3. **The LTAE position axis had to change.** `ag_year` is Aug 1 – Jul 31, so a day-of-year
   encoding runs 365 → 1 in the middle of every parcel's series and the sinusoidal encoder puts
   midwinter next to the first week of August. Positions are **days since Aug 1** — monotone
   across the window, in the same [0, 365) range, and phase-aligned across parcels precisely
   because every window starts on the same date. Nothing would have raised.

There is **no pixel-set tensor and cannot be**: `s2_perdate.parquet` holds per-date medians and
quantiles over a parcel's pixels, never the pixels, so PSE-LTAE is not buildable from this store.

---

## 4. ⚠️ Known limits — state these in the model card

* **One round, 1,000 parcels. There is no second pass.** The allocation had to be defensible
  before it was drawn rather than corrected afterwards.
* ⚠️ **One annotator so far.** Every figure in `RESULTS.md` §8.2 rests on a single person's
  reading with **no κ**, so it carries an unmeasured label-noise term. Two of the six labels
  (`WOODY_NON_CROP`, `OTHER`) sit on the codebook boundary the design flagged as hardest and
  together account for 62 % of the usable sample.
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
