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

### An architecture verdict is conditional on the store it was measured on

"LTAE loses" was measured three times on the Landsat store and treated as settled: worse on
LODO (0.442 vs 0.477), worse on LOYO, worst flicker, so *"try another architecture"* was written
into the closed list. Re-asked on the Sentinel-2 store — the same code, the same protocol, the
same parcels — LTAE wins CV in **8 of 8** arms.

Nothing about the architecture changed. What changed is that it is now fed a **median 47 clear
dates per parcel-year instead of 13–24**. An attention encoder over an irregular date axis was
being starved of the axis, and a starved model's verdict is a verdict about the diet.

**The rule:** record what a negative architecture result was measured *on*, and re-ask it when
the input distribution moves — not when someone merely dislikes the answer. The distinguishing
question is whether a **mechanism** changed (here, observation density, which this project had
already measured as the dominant driver of its other failures), not whether a new run might get
lucky.

### ⭐ …but re-ask it on the axis that decides, because an architecture can memorise too

That CV win does not survive holding out a department. On the same S2 store LTAE loses
leave-one-department-out in **4 of 4** label targets, and the reason is visible in the size of
the fall: from CV to LODO, LightGBM drops 0.05–0.13 macro-F1 and **LTAE drops 0.14–0.19** — 1.4
to 3.4× more, every time.

So the architecture is doing exactly what `centroid_lat` did. Given 47 dates and 500 parcels it
finds structure that identifies *which department a parcel is in*, that structure pays inside
the training departments, and it is worth nothing outside them. **A feature is not the only
thing that can manufacture accuracy that does not travel; a model class can.** Any "architecture
X beats Y" claim needs the same CV *and* OOD pair that a feature-selection claim needs.

### ⭐ An OOD estimate over 4 units is 4 numbers

The reading immediately above was, for two days, the opposite: "LTAE wins LODO in both arms."
Nothing was wrong with the code, the data or the protocol. That LODO mean averaged **4
departments** — the only four that then had ≥40 labels — with an SD of 0.15. At **14**
departments the sign flips and holds across four independent label targets.

The generalisable point is not "n was small", which everyone already knows. It is that a
leave-one-*group*-out mean has **two** sample sizes — parcels and groups — and the second one is
the one that governs, is usually an order of magnitude smaller, and is almost never quoted.
**Report a LODO figure as "0.539 over 14 departments", never as "0.539".** Ours would have been
caught earlier by that one habit.

### ⭐ macro-F1 is not comparable across label spaces — the floor moves with it

Collapsing the 4-class problem to 2 (perennial vs everything) raised macro-F1 from 0.672 to
0.715, which reads as a large gain. It is not one. Macro-F1 averages per-class F1 with equal
weight, so a model that never predicts the minority still collects a full score on the majority
and divides by the class count: **always guessing the largest class scores 0.124 at 5 classes,
0.171 at 4, 0.228 at 3 and 0.467 at 2.** Normalised against its own floor, the 2-class arm was
the **lowest-skill** arm in the study, and on an unseen department it scored 0.180 — barely
above guessing.

Two habits fix it. **Report the majority-class score beside every macro-F1**, and when comparing
label spaces, report **the F1 of the one class you actually care about**, which is unaffected by
how many others exist. Here that was decisive: collapsing 3 classes to 2 moved `PERENNIAL` F1
by **+0.004**, while moving one ambiguous class (`WOODY_NON_CROP`) from one side to the other
moved it by **+0.190**. The label space was never the constraint; one boundary inside it was.

### ⭐ Classify both sides of a before/after with the same lexicon — then audit the tail

Comparing a 1999 land declaration with a 2012 census means mapping two vocabularies onto one
class space, and any slippage between them appears as change in the land. The census uses fuller
crop names than the titling registry ("LIMON ACIDO", not "LIMON"), and **4.09 % of its crop
tokens fell through to the `ANNUAL` catch-all — of which 80 % was the single token
`VERGEL FRUTICOLA`, literally "fruit orchard".** The largest unmapped token in the file was
being counted as the opposite of what it is, in the direction of the hypothesis.

Correcting it moved the headline from **+2.4 pp to +12.5 pp**. Nothing raised, and a 4 %
fallback rate looks tolerable in aggregate — the danger is that an unmapped tail is almost never
uniformly distributed, so its *concentration* matters far more than its size. **Print the
unmapped tokens sorted by frequency and read the top ten by hand**; the budget check alone would
have passed this.

### ⭐ Score a new covariate against a same-shaped control, not against nothing

Adding mean temperature and rainfall to the S2 endpoint classifier raised cross-validation
**and** leave-one-department-out (`RESULTS.md` §8.8). That is the right shape for a real
covariate, and it is not enough on its own: a 1 km climate surface is a smooth function of
location, and location is this project's canonical memorisation feature. So the arm was run a
fourth time with **centroid latitude + longitude in place of the two climate columns** — same
count, same time-invariance, same smoothness, no agro-climatic content.

The control is what produced the finding. LightGBM got **+0.038** LODO from climate against
**+0.014** from coordinates: 2.7×, so most of it is content. LTAE got **+0.027** from climate
against **+0.037** from coordinates: the gain is real and it is *geography*. The rule got
**+0.028** from climate and **−0.021** from coordinates. **Three model classes, one covariate,
three different answers to "is this signal or memorisation" — and none of them is visible
without the control.**

The general form: when a proposed feature is a smooth function of something the model must not
memorise, the comparison that decides is not *feature vs nothing*, it is *feature vs a
same-shaped surrogate carrying only the thing you are worried about*. Build the surrogate in the
same run, with the same folds, and put it in the table labelled as a control so nobody later
reads it as a proposal.

Corollary, learned in the same table: **verify the "nothing" arm reproduces its recorded
numbers.** Plumbing statics into the dataset and the network touched code every arm shares. The
no-climate arms coming back at 0.6717 / 0.5387, 0.7084 / 0.5154 and 0.3923 / 0.3605 — the
figures already in §8.2 — is what licensed reading the deltas as deltas.

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
| an NA gate column forwarded into a new workspace | `quality_ok` is NA for the S2 campaign and `load_parcels` filters `== True` → **trains on zero rows, reports an empty dataset** |
| day-of-year on a window that straddles the New Year | positions run 365 → 1 mid-series; the encoder places midwinter beside August. Nothing raises — the model just learns worse |
| a climate/DEM raster sampled at a coastal centroid | rasters mask the ocean and the cadastre runs to the shoreline → **NaN, not an error**, and the parcel silently leaves any model using the column |
| a per-unit output directory that is never created | `pandas.to_parquet` raises only at the *end* of a long leave-one-out loop, after every fit has been paid for |
| a rule/heuristic model given a label space it was not written for | `RuleModel` maps three semantic class *names* onto ids; in a 2-class space the missing name falls back to a position that is the wrong class. It runs and returns a plausible number |
| an unmapped-token catch-all in a crop lexicon | 4 % of tokens, 80 % of them one token pointing one way — moved a headline by 10 pp with no warning |

**The durable fix for the hang:** run each call under a wall-clock deadline on a **daemon**
thread, so a hang raises and is retried. It fired 25 times in one national extraction with zero
workers lost.

### A budget check cannot see a concentrated error inside a small tail

Twice now, mapping an external vocabulary onto this project's crop classes left a small
unmapped tail that **passed its budget check** while containing a single token big enough to
move the headline. Piura: 4.09 % fell through, **80 % of it `VERGEL FRUTICOLA`** ("fruit
orchard"), and the answer moved +2.4 → +12.5 pp. Nationally: 1.35 % fell through, **45.8 % of
it `MELOCOTONERO`** — the peach *tree*, when `MELOCOTON` and `DURAZNO` were already classified
PERENNIAL — and the answer moved +7.8 → +8.4 pp on parcels.

A share budget answers "is the tail small?" The question that decides the number is "**is the
tail concentrated, and on what?**" Those are different questions and the first cannot answer
the second: a 1 % tail that is 46 % one perennial token is worse than a 4 % tail spread evenly
over vegetables.

**Print the unmapped tail sorted by frequency and read the top ten, every time.** Both failures
were visible in one line of output and invisible in the budget. Note the shape of both misses:
the external source used the *tree* name for a crop already classified under its *fruit* name.
That is a vocabulary gap, not a policy question, and it is resolved by looking rather than by
deciding.

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

### Store the label verbatim; take the modelling decision downstream and in the open

The campaign has six label values and the model it must be compared against has three. There is
no mapping that is simply correct: `WOODY_NON_CROP` is either dropped (clean comparison, a third
of the data gone) or folded into `PERENNIAL` (every parcel kept, the baseline flattered).

Ingest therefore stores what the annotator said and nothing else, and **each reading is a
separate workspace on disk** rather than a flag. The gap between two readings then becomes a
measurable quantity — here 0.06–0.11 macro-F1 — instead of an assumption buried in a
preprocessing step. It also made the assumption checkable: the Landsat panel independently
called **5 of 5** woody parcels `PERENNIAL`, which is the thing the `t3w` mapping had asserted.

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
