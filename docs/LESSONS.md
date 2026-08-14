# Lessons

Findings that are about **method, not Peru**. Each cost days to establish. The evidence is in
[`RESULTS.md`](RESULTS.md); this page is the transferable claim.

---

## Evaluation

### An OOD evaluation is only blind along the axis it holds fixed

`centroid_lat` gains **+0.0476 on spatial CV** and **+0.0474 on leave-one-year-out** — the same
number — because LOYO holds region approximately fixed so a time-invariant feature is exactly as
exploitable there. On leave-one-department-out it *loses* **0.0596**. On LODYO (department
**and** year out at once) its advantage is **−0.005**.

**Do:** report CV / LODO / LOYO / LODYO for every candidate. LODYO is free — it re-scores
existing LODO predictions per year cohort, no new fits.

### An evaluation nobody remembers to run does not exist

LODYO overturned the model selection, and it was the evaluation most likely to be skipped. It
now runs automatically at the end of every `allperu lodo`. Wire the decisive check into the
pipeline, not into a checklist.

### Gain ≠ contribution

Measured twice. `frac_l7` was the 2nd-highest-gain feature in the panel model and ablating it
cost **+0.0013 macro-F1** — i.e. nothing. Acquisition metadata was 29.8 % of total gain and
worth 0.007. **Never justify keeping a feature by its importance score**; ablate and measure.

### CV rank and test rank disagree, and spatial CV will not warn you

Tuned LTAE beat LightGBM on **all 5 CV folds** (0.658 vs 0.647) and **lost the locked test**
(0.661 vs 0.681). `test − CV` was +0.034 for one model and +0.003 for the other.

### A quantity measured inside one department is not a quantity

"Piura is one of the hardest departments" was a property of `centroid_lat`, not of Piura —
0.301 under the rejected model, 0.479 under the selected one, 0.512 under an architecture with
no location feature. Separately, a cross-sectional tenure gap reported for Piura **reversed
sign** when computed nationally. Pooled numbers whose sign flips by region describe a quantity
that does not exist.

---

## Time series from a single-year model

### Time-invariant features manufacture stability; sensor metadata manufactures change

Per-parcel class flicker over a 25-year panel was **monotone in how many statics the model
carries** — 0.517 (3 statics) / 0.744 (2) / 0.980 (0) — while k = 0 accuracy barely moved
(0.586 / 0.544 / 0.581). Statics contribute almost nothing to being *right* and almost
everything to *not changing your mind*. This was **pre-registered and confirmed exactly**.

Symmetrically, `frac_l7` ramps 0.000 → 1.000 across the study period and would have produced
the headline trend from satellite availability alone.

**Do:** if a stability metric is your gate, report it for the static-free arm too — that is the
honest end of the range.

### Per-year errors compound: accuracy that is fine for one year is not fine for a trajectory

A 3-class classifier at ~0.55–0.59 accuracy produces a 25-year per-parcel series dominated by
classification noise, because per-year errors are near-independent. **Aggregate before you
difference:** window-mean *probability* over 5 years collapsed flicker 0.75 → 0.088. It fixed
the noise and not the estimand.

### Observation density moves probabilities toward the base rate

As clear observations fall ~24 → ~13 per parcel-year, within-parcel probability moves **+0.052
per log-observation on true perennials and −0.022 on true annuals** (p < 1e-6). Both classes
revert to the base rate as evidence thins, which at the level of a share is **indistinguishable
from real conversion**.

⚠️ **This is not a calibration artefact and recalibration will not fix it.** Fitted temperature
*falls* with density (T = 1.448 − 0.160·log n), so at low density the model is **over**-confident
and the correct recalibration pushes probabilities *further* toward the base rate. Applying it
moved the target metric by 0.0001.

### Always pair a restriction with its truncation control

Retraining without the El Niño years turned a temporal-transfer gate from FAIL to PASS — but
the *baseline* model's own predictions truncated to the same years also passed. Most of the gain
was the shorter panel, not the retrain. Without the control the result is unattributable.

---

## Study design

### A pre-trend gate should ask whether the nuisance could overturn the conclusion

Three gates were written for one study:

* **v1** — `|coef| < 0.005` **and** "CI contains 0". Passed by an imprecise estimate.
  *A gate that rewards imprecision is not a gate.*
* **v2** — equivalence: the whole CI inside ±0.0025. The measured point estimate landed
  **outside** the band, so the SE needed to fit inside was **negative**. **Unpassable at any
  sample size, in any country** — more precision returns FAIL, never PASS.
* **v3** — *measure* the pre-trend and subtract it, carrying its uncertainty:
  `corrected = headline − M·placebo`, `se = sqrt(se_h² + M²·se_p²)`. Report the sensitivity
  curve over M. **This returned an answer.**

**Do:** derive the amplification factor M from window midpoints **in code**, never hard-code it.
And never read a *direction* from a corrected point estimate — here its sign flipped between
outcomes, which is what a noise term does.

### Check feasibility before funding anything

`required n from measured variance` vs `available n from the archive` takes minutes and answers
"can this study exist?". One study needed 25,202 parcels per arm; all of Peru holds **6,559**.
Run this before every extraction. Size for the gate that *decides*, not for the headline.

### Define a pre-period per unit, not per calendar

Two "dated observations" turned out to be snapshots of a rolling programme: declarations spread
over 13 years, per-record transaction dates, 24.6 % undated, and a registration rate
**non-monotone** in the gap between observations. A calendar pre-period is contaminated by early
treatment. The pre-window must end before **each unit's own** treatment-eligibility date.

### Register the analysis, then let the registration bind

The placebo was estimated *first* and the code path enforces that order. The measured
headline/placebo covariance was computed, reported, and **deliberately not used**, because a
formula that moves after the data are seen is not registered.

---

## Engineering

### Verify a finished job by counting its output

Never by "the process ended" or "the file exists". A fix for a silent GEE hang introduced a
*second* silent failure one layer down: it used a `ThreadPoolExecutor`, whose non-daemon threads
are joined by an `atexit` hook, so every timeout orphaned a thread and all five workers sat at
0 % CPU for up to 3 h **after their work was complete**, never exiting. (`cancel_futures=True`
does not help — it only drops futures still queued.)

### A guessed completeness threshold will cry wolf

A 0.995 tolerance failed 13 of 15 years on a *complete* extraction. The pattern was the
diagnosis: a truncated year is an **outlier**; a uniform 99.4 % across 13 years extracted by 5
independent workers is a **floor**. Measured: the same ~85 parcels are missing every year
(Jaccard 0.95, median 0.11 ha) because a parcel smaller than a Landsat pixel passes the
*coverage* gate and returns no pixel rows.

**Do:** add a statistic that measures **consistency across units** (`deficit_ratio_to_median`)
rather than an absolute ratio.

### The silent failures this project actually hit

Every one returned success. Budget for them.

| failure | symptom |
|---|---|
| GEE hang | process alive, no output, no exception — `_retry` only catches things that *raise* |
| GEE throttle | arrives **both** as an error string *and* as 900 s hangs — handle both |
| Esri above its max zoom | a valid **flat-grey image**, not an error |
| `.dbf` under the wrong basename | GDAL opens the shapefile and returns **zero columns** |
| zero-padded join keys | a string join returns **zero rows**, reads as "no data for this department" |
| content-addressed cache | new columns **silently skipped**; the cache had to be invalidated by hand |
| per-year store re-globbing | `pixels_2021.parquet` written 5 times, every snapshot well-formed |

**The durable fix for the hang:** run each call under a wall-clock deadline on a **daemon**
thread, so a hang raises and is retried. It fired 25 times in one national extraction with zero
workers lost.

### An unused correction is an untested correction

`harmonization.py` was implemented months before it was first run and documented as the safe
path. When finally run it was **harmful on every axis** — worse than no correction at all.

### Cap the geographic extent of a work chunk

Grouping extraction chunks by time alone produced chunks spanning up to 110 deg²; `filterBounds`
then reduces every granule in that rectangle and the job stalls at 0 % CPU with **no error**.
⚠️ This was learned **twice** — the second module copied the first one's structure but not its
chunker, and the comment carrying the lesson was in the part not copied.

### Verify a sensor step against its own failure mode

Comparing before/after a sensor cut reported a +0.055 NDVI "step" that was **entirely the
growing season**. A within-month-of-year pairing looks principled and still returns ±0.018 on a
pure sine. What works: fit the seasonal cycle out per unit, subtract a **placebo cut one year
earlier** as the noise floor, and read the verdict off **raw bands, never an index**.

### A global linear correction cannot remove a class-dependent difference

Two sensors differed by **+0.070 NDVI on perennial parcels and +0.045 on pasture**, and that
0.025 between-class spread survived every correction arm. A band-level linear map moves all
classes *together*. The only map that would work is conditioned on the very thing the model is
trying to predict.

---

## Human labelling

### An abstain must be its own label, not a low confidence

A guess recorded at confidence 1 is **indistinguishable from a real label downstream**. Make
`UNSURE` a value, compute κ only over items *both* raters actually called, and report the
abstain share as its own number. Then remove the confidence control — two ways to record doubt,
one of which corrupts the data, is worse than one.

### Adding a class means narrowing its neighbours in the same edit

Adding `NON_AGRICULTURE` without narrowing `OTHER` — whose definition explicitly listed
water/built-up/road — would have put two labellers on opposite sides of the same parcel. Ship a
**separating test** ("could this ground be sown next season exactly as it stands?") and a
**priority ladder** for overlaps, in the same change.

### Check the store before designing a visual

An interquartile ribbon needed within-parcel quantiles that did not exist — and **no arithmetic
on medians recovers a quantile**. It forced a full re-extraction. Relatedly, **a quantile of a
ratio is not the ratio of the quantiles**: an index must be formed per pixel *before* reduction.

### Assert blindness on the artefact, not on the generator

The test asserted no class/split/fold string appears in the emitted HTML — but its fixture never
built the double-labelled shard, so one leak went unnoticed. **Test the file you actually ship,
in every configuration you ship it in.**
