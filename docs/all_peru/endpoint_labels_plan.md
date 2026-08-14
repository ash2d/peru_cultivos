# Endpoint labels by photo-interpretation — plan

> **Status: gate zero RUN 2026-08-12; labelling not started.** Repeatedly scoped out of every
> previous plan "because it needs human labelling effort, not because it ranks below" the
> alternatives ([`temporal_ood_plan.md`](temporal_ood_plan.md) §5b). Four estimands failed
> outright and every technical mitigation has been tried and measured
> ([`RESULTS.md`](RESULTS.md) §9), so the ranking argument no longer holds either.
>
> **⚖️ 2026-08-12 (a).** The fifth estimand — the two-period tenure DiD — did *not* fail: it
> returned a **bounded null** (RESULTS.md §11). That raises this plan's value rather than
> lowering it. Endpoint labels would (a) give the DiD a **validated** outcome instead of an
> unverified prediction, and (b) supply the endpoint error matrix **by tenure group** that
> §8.5 showed is differential — the missing ingredient for the far larger **cross-sectional**
> comparison (363,529 at-risk parcels against the DiD's 6,559).
>
> **⚠️ 2026-08-12 (b) — THE PLAN CHANGED SHAPE. Two premises of the original §2 were
> assumptions and both are now measured, and both were wrong.** The imagery is **not
> sub-metre** (82.9 % is 1.2 m) and it is **not contemporaneous with 2019–23** (64.2 % of
> parcels sit on 2024+ imagery). A third premise — that a proportional draw over whole 5 km
> regions is the right sample — is measured here as **~8x more expensive** than the
> alternative. The revised design below supersedes the original §2 entirely.
>
> **The scope also grew:** the labels are now for **training a new Sentinel-2 classifier**, not
> only for validating the Landsat one. That is the change that removes the failure mode
> underlying all four dead estimands. See §1b.
>
> **Read first:** `RESULTS.md` §9.4 (why mitigation ran out), §8.4 (LOYO), §8.5 (T1's
> unverifiable endpoint assumption), and CLAUDE.md §8.

---

## 0. Why this is now the top-ranked task

**Every accuracy number this project has ever produced is measured at the label year
(~1997–2006). Not one is measured at the endpoint, which is the only period the research
question is about.** Everything said about the endpoint is extrapolation:

* leave-one-year-out can only probe *inside* 1996–2009, and even there the worst non-1998
  cohort scores 0.406 against CV 0.581 (§8.4);
* T1's non-differential-error assumption is **verified at ~1999 and assumed at the endpoint**
  (§8.5) — stated as an unresolved caveat every time the tenure contrast is mentioned;
* the window diagnostic's control pool drifts on every arm and every model, and after three
  steps of mitigation the best control slope is still −0.0382/decade (§9.2.4);
* the two-period DiD ([`tenure_did_plan.md`](tenure_did_plan.md)) has an outcome that is
  *still a prediction*: if the classifier is blind to the transition, the DiD is unbiased and
  empty, and nothing in that plan can detect it.

This task is the only one that **verifies** rather than mitigates.

## 1. What it produces

1. **An endpoint test set** — the project's first, and the first honest answer to "how good is
   this model *now*?"
2. **An endpoint error matrix**, which is what an Olofsson-style bias-corrected area estimate
   needs. `perennial/area.py` is built and has never been usable because this input did not
   exist.
3. **A direct check of T1's non-differential-error assumption at the endpoint** — stratifying
   the sample on tenure makes this free.
4. **A validated outcome for the DiD** — the same labels turn `tenure_did_plan.md`'s
   prediction-based outcome into a measured one, at least on the labelled subset.

## 1b. And now a fifth: training supervision for a Sentinel-2 classifier

The original plan used these labels only to *validate* the existing Landsat model. That leaves
the model trained on 1997–2006 labels and applied to years LOYO says it was never licensed for.
**Labelling at the endpoint makes it possible to train there too, which deletes the failure
mode rather than measuring it.**

Sentinel-2 rather than Landsat, for three measured reasons:

| | Landsat 30 m | Sentinel-2 10 m |
|---|---:|---:|
| median parcel (0.566 ha) | 6.3 px | **56.6 px** |
| 10th-percentile parcel (0.138 ha) | 1.5 px | 13.8 px |

The 10th-percentile parcel gets more S2 pixels than the *median* parcel gets from Landsat, and
47 % of parcels are under 0.5 ha. Secondly, S2 over the target years is the dense, stable end
of the observation-density curve whose thinning drove the drift (§9.2). Thirdly there is no
mission boundary inside the window.

**The deeper point: every previous design needed the classifier to be right at both ends of a
20-year gap and to drift equally. This design needs it right at one end, and takes the other
end from the written record.** That is strictly less demanding.

⚠️ **This means new code**: an S2 sibling to `features/landsat_gee.py`. Same shape (coverage
gate → export → assemble), but export **parcel × date medians rather than raw pixels**. The
raw-pixel export exists only to feed LTAE/PSE-LTAE, which have now lost on every axis that
decides (LODO 0.442 vs 0.477, panel flicker 0.980, W2 −0.1025, LOYO 0.4784). Dropping them
makes S2 roughly 50x cheaper to extract despite 9x the pixels.

---

## 2. Gate zero — the imagery, measured

`allperu/esri_dates.py` probes the Esri World Imagery metadata service at **real parcel
centroids** (240 at-risk parcels across the four target departments) rather than counting
tiles in a bounding box — one huge old tile and one small new tile count the same, but only
the parcel-weighted number decides anything. Output: `docs/figures/esri_imagery_dates.csv`.

**Result 1 — the imagery is recent, but later than the plan assumed. Median year 2025.**

| imagery year | LA_LIBERTAD | CAJAMARCA | PIURA | LAMBAYEQUE | all |
|---|---:|---:|---:|---:|---:|
| pre-2015 (unusable) | 2 % | 3 % | 0 % | 0 % | 1.2 % |
| 2015–2018 | 7 % | 5 % | 20 % | 0 % | **7.9 %** |
| 2019–2023 | 48 % | 17 % | 8 % | 33 % | **26.7 %** |
| **2024+** | 43 % | 75 % | 72 % | 67 % | **64.2 %** |

**Result 2 — it is not sub-metre. 82.9 % is 1.2 m**, 12.1 % 60 cm, 5.0 % 30 cm. La Libertad is
100 % 1.2 m. The original §2 said "sub-metre" throughout and that was never checked.

### 2.1 What follows — move the window to the imagery

**Do not fix the target window at 2019–2023.** That came from the DiD's windows, not from any
constraint; S2 runs to the present. Instead:

* **label each parcel against its own imagery date**, and
* **extract S2 features for that parcel's own year.**

The extraction pipeline is already per-parcel-per-year, so this costs nothing and it uses the
64 % of parcels carrying the freshest imagery instead of discarding them. Record the imagery
date per parcel as a first-class column; it becomes the parcel's label year.

### 2.2 What follows — 1.2 m changes what is answerable

At 1.2 m a mature orchard's row structure and crown texture are readable; a **young planting
(under ~3 years) is not**. Those are exactly the recent conversions of most interest, so
**expect systematic under-detection of recent conversion**. Do not attempt to suppress it —
let `UNSURE` absorb it and report the rate, because it bounds a bias nothing else can see.

The same resolution finding is the strongest single argument for the binary target in §4.

### 2.3 ⛔ A second labelled time point is NOT available from Esri

The idea of labelling two dates (e.g. 2016 and 2023) to get three time points per parcel is
sound in principle and **not supported by the imagery**: only 7.9 % of parcels have 2015–2018
coverage. Esri Wayback stores historical *published* versions, but a 2016 Wayback tile over
Peru shows whatever was current then — often 2010–2013 acquisition — so it is not "2016
imagery". Google Earth Pro has genuine dated historical coverage but is manual-only.

**It should be re-scoped rather than abandoned, for two reasons.** Producing the trend is the
classifier's job — hand-labelling both ends does that work twice. And it would yield no
placebo anyway: tenure is last observed ~2011, so both dates are post-treatment.

➡ **Re-scoped as a change-validation set (§7, optional):** ~400–600 parcels on Google Earth,
**stratified on predicted change**. At a ~3 %-per-7-years conversion rate a random 500 would
contain ~15 changers and measure nothing; oversampling predicted-changers is what makes it
estimable.

---

## 3. Sample size — measured, not assumed

`allperu/label_budget.py` runs a learning curve on the **existing** national Landsat store and
PETT labels: a conservative proxy for an S2 campaign, since S2 should raise the ceiling. What
transfers is the *shape* — where the curve flattens — which is what a budget needs.
Output: `docs/figures/label_budget.png`, `label_budget_curve.csv`.

| labels | 3-class macro-F1 | % of ceiling | `PERENNIAL` F1 | binary macro-F1 | % of ceiling |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.519 | 89 % | 0.529 | 0.693 | 95 % |
| 1,000 | 0.531 | 91 % | 0.546 | 0.704 | 96 % |
| **2,000** | **0.548** | **94 %** | **0.561** | **0.715** | **98 %** |
| 4,000 | 0.558 | 95 % | 0.571 | 0.722 | 99 % |
| 8,000 | 0.576 | 98 % | 0.587 | 0.730 | 100 % |

**~2,000 stratified labels reach 94 % of what unlimited labels would give.** 4,000 buys one
further point; past that it is flat.

**⭐ The protocol matters more than the count. Drawing whole 5 km regions — as every other
sample in this project is drawn — is worth roughly one-eighth of drawing stratified by class.**
2,000 stratified (0.548) beats 8,000 region-drawn (0.529). Variance collapses too: SD 0.007
against 0.040, so a small unstratified campaign is a lottery — at n = 2,000 region-drawn, fold
results ranged 0.373 to 0.498 on which blocks came up.

⚠️ **This reverses the original §2's "draw over whole 5 km regions".** Region draws exist to
stop neighbour leakage between *train and test*; here the test fold is spatially separated
anyway, so scattered training parcels cannot inflate it. The two purposes were conflated.

**Convergent sizing.** The original §2 derived 1,500–3,000 from *validation* precision (~380
correctly-classified parcels per class for ±0.05 user's accuracy at 95 %). The learning curve
derives ~2,000 from *training* saturation, independently. Both land in the same place, so one
sample serves both purposes — split it by 5 km region into train and test rather than buying
two.

---

## 4. Labels — binary primary, three-way recorded

**Primary target: `PERENNIAL` vs everything else.** Three reasons, all measured:

1. It is what the research question asks — export-oriented perennial versus not.
2. It reaches 95 % of its ceiling at 500 labels against 89 % for 3-class (§3), and it keeps
   the stratification arithmetic feasible (§5).
3. ⚠️ **Annual-versus-fallow is the pair a photo-interpreter cannot reliably call from a
   single date** — a harvested annual field and a fallow field look identical, and at 1.2 m
   there is no texture cue to separate them. Making it a headline class imports label noise
   directly into the training set.

**Record anyway, at no extra cost:** the three-way call (`PERENNIAL` / `ANNUAL` /
`PASTURE_FALLOW`), plus **`UNSURE`**, **`NOT_AGRICULTURE`**, and **`WOODY_NON_CROP`**.
`UNSURE` is mandatory: forcing a call on an ambiguous parcel is how a photo-interpreted set
acquires a bias that looks like ground truth.

**Also record, as attributes rather than strata:** the declared crop, the crop type if
distinguishable (mango / lime / coffee / banana …), a per-parcel confidence, the imagery date,
and a "boundary no longer matches visible field" flag.

---

## 5. Stratification — the design

Stratification does two jobs here: **make the cells that carry the result big enough to
measure**, and **make error measurable separately for each group being compared**. Everything
is then weighted back to the population.

### 5.1 The variable the original plan omitted: the confusion cell

The headline number is one cell of a 2×2 table:

| | predicted NOT perennial | predicted `PERENNIAL` |
|---|---|---|
| **declared NOT perennial (PETT, ~1998)** | stayed annual | ⭐ **apparent conversion** |
| **declared `PERENNIAL`** | apparent reversion | stayed perennial |

The whole project reduces to the size of the starred cell, and whether it is real depends on
its **user's accuracy** — of the parcels the model calls converted, how many actually did.
Nothing else in the design measures that.

**The arithmetic that decides it.** In the four target departments, **313,245 parcels have a
tenure observation**, of which **6.9 % are declared `PERENNIAL`**; the endpoint perennial rate
in the at-risk pool is ~7 %. A **proportional** sample of 2,000 therefore yields:

| cell | proportional | stratified |
|---|---:|---:|
| stayed annual | ~1,732 | 500 |
| ⭐ **apparent conversion** | **~130** | **500** |
| apparent reversion | ~69 | 500 |
| stayed perennial | ~69 | 500 |

Split that ~130 across tenure (2) × department (4) and it is **~16 per cell** — below this
plan's own E3 gate of 30, in the only cell that matters. Equal allocation gives **62**.

### 5.2 The four variables

| variable | levels | what it buys | why not otherwise |
|---|---:|---|---|
| **confusion cell** (declared × predicted) | 4 | user's accuracy of the conversion cell — the result itself | proportional starves it (§5.1) |
| **tenure at declaration** | 2 | tests T1's differential-error assumption at the endpoint | currently *assumed*; already false at the label year (−1.75 pp, §8.5) |
| **department** | 4 | spatial-generalisation gap is ~0.09 macro-F1; a pooled matrix averages things that differ | enables per-department correction |
| **crop type** | — | **record, do not stratify** | fine crop fragments the sample; declared class already carries what the design needs |

**4 × 2 × 4 = 32 cells × 62 = 1,984.** That is the 2,000.

### 5.3 Departments: four, holding 90 % of the treated parcels

La Libertad (43.6 % of qualifying treated), Cajamarca (74.9 % cumulative), Piura (83.2 %),
Lambayeque (90.0 %). This is the coastal agro-export zone and it is where the tenure variation
actually is. National coverage at useful per-department density costs ~4x for coverage the
current questions do not need; extend later if the first round works.

### 5.4 ⚠️ Tenure: two levels, not three — measured

A three-level split (already registered / became registered / never registered) would align
with the DiD arms, and **cannot be equally allocated**:

```
LAMBAYEQUE  x became_registered x PERENNIAL  ->    11 parcels exist
LA_LIBERTAD x became_registered x PERENNIAL  ->    90
CAJAMARCA   x became_registered x PERENNIAL  ->   349
```

You cannot draw 62 from 11. So **stratify on two levels (registered vs not, at declaration)
and record the three-level version as an attribute**. For thin cells take a census — all 11 —
exactly as the DiD took all 6,559 treated. Do not force equal allocation across departments
inside a thin stratum; use the sqrt-proportional-with-floor allocation `allperu sample`
already implements.

⚠️ Tenure is last observed at the ~2011 cadastre cut and the control arm is contaminated after
that (RESULTS.md §11.7). The strata are **"tenure as of 2011"** and must be named that way.

### 5.5 Weights

Each parcel's weight is `N_h / n_h` — stratum population over stratum sample. Every reported
share is `Σ_h (N_h/N) × (rate within h)`, **never a raw sample average**. The conversion cell
is over-sampled roughly **12x**, so an unweighted sample mean would overstate the conversion
rate by about an order of magnitude. Same `sample_weight` discipline as the national sample,
for the same reason.

### 5.6 Two operational traps

* **Freeze the stratification map before drawing and keep it as an artifact.** Stratifying on
  predicted class means the sample is defined relative to *some* model. That is standard for
  accuracy assessment — but if the model is later retrained, the weights still refer to the
  frozen map. Storing it is what keeps the weights valid.
* **Label blind to the cell.** The labeller must not know a parcel came from the "apparent
  conversion" stratum. That cell carries the result, so it is where confirmation bias would do
  the most damage.

---

## 6. The traps, from this project's own history

* **Perennial detection is a *contrast*, not a level** (§8.2). A labeller shown one parcel in
  isolation will do worse than one shown its neighbours — which is exactly what
  `notebooks/04_inspect_parcel_basemaps.ipynb` already draws. **Keep the neighbour context**;
  it matters more at 1.2 m than it would have at 30 cm.
* **Woody non-crop is 2.79 % of the pool and is a *lower* bound** (§6.3): it counts only
  *declared* woody non-crop, not vegetation that invaded an abandoned parcel over 20 years.
  Photo-interpretation is the only thing in this project that can see the difference — give it
  its own label value rather than folding it into `PERENNIAL`.
* **Perennial ≠ export** (57 % / 20 % / 23 %, §6.4). Record crop type where distinguishable; it
  converts a §6.4 assumption into a measurement.
* **Parcel boundaries are a PETT cadastre from ~2011** (CLAUDE.md §3) and may not match current
  land use. Record the mismatch flag.
* ⚠️ **The baseline is a declaration, not a measurement.** The change estimate is
  `S2-classified now − PETT-declared 1998`: two different instruments, and the 1998 error may
  be **differential by tenure**, which is the treatment. §11.13 found the classifier's own
  error differential at −0.044, large enough to be the entire effect. Two mitigations:
  restrict to PETT-`ANNUAL` parcels so the baseline is constant by construction (R1, as the
  DiD does), and photo-interpret a subsample against **year-matched Landsat** for
  perennial-vs-not on parcels over ~2 ha (20+ pixels, feasible at 30 m). Stratifying on tenure
  detects differential *endpoint* error but cannot touch differential *declaration* error.

---

## 7. Order of operations

Sequenced so the cheapest step that can kill the plan runs first.

| # | step | cost | kills the plan if |
|---|---|---|---|
| 0 | ✅ **Esri imagery dates** (`allperu esri-dates`) | done | — (it reshaped it instead) |
| 1 | **Pilot: 100 parcels, two labellers, κ on perennial-vs-not at 1.2 m** | ~4 h | κ < 0.75 — needs different imagery, not more parcels |
| 2 | Freeze the stratification map; draw ~2,000 over the 32 cells | ~1 h compute | a cell cannot be filled |
| 3 | Build the S2 feature pipeline; extract for the 2,000 labelled parcels at each parcel's own imagery year | ~1 day + modest GEE | — |
| 4 | Label; train; LODO | ~35 h + minutes | ceiling below Landsat's 0.58 |
| 5 | Weighted error matrix, by tenure and by department; bias-corrected area (`perennial/area.py`) | hours | — |
| 6 | *(optional)* change-validation set, ~500 parcels on Google Earth, stratified on predicted change (§2.3) | ~10 h | — |

**Step 1 is where the next hour belongs.** Sub-metre imagery turned out to be rare, so
inter-rater agreement at 1.2 m is now the real unknown — and a day of pilot labelling tells you
whether the other 35 hours are worth spending.

## 8. Gates

* **E0** ✅ **passed with amendments** — high-resolution imagery exists at 98.8 % of probed
  parcels, but at 1.2 m and mostly 2024+, which is what §2.1/§2.2 respond to.
* **E1** inter-rater **κ ≥ 0.75** on perennial-vs-not, else stop and revise the protocol.
  ⚠️ **Raised from 0.6.** Noisy *validation* labels add variance; noisy *training* labels cap
  the model outright, and §1b makes these training labels.
* **E2** `UNSURE` + `NOT_AGRICULTURE` < 25 % of the drawn sample, else the imagery or the class
  definitions are not fit for purpose.
* **E3** every one of the 32 strata has ≥ 30 labelled parcels before any per-cell accuracy is
  reported.
* **E4** the S2 model's LODO mean must exceed the Landsat model's **0.477**, else the new
  strand buys nothing and the labels revert to validation-only use.

## 9. What it costs

Human time, and that is the whole reason it keeps being deferred. At ~1 min/parcel with the
existing notebook-04 viewer, 2,000 parcels is **~35 hours of labelling**, plus 200 overlap
parcels for κ. New code: an S2 feature pipeline (§1b). GEE: modest — 2,000 parcels × 1 year
each with server-side medians, not the multi-hour panel extractions.

**It is the cheapest task in the project in compute and the most expensive in attention, and
it is the only one that would let anybody state an endpoint number and defend it.**

## 10. What it still cannot do

* **No placebo, so no causal claim on titling from this alone.** One outcome time point means
  the v3 pre-trend discipline is unavailable; the DiD remains the causal instrument and this
  remains the descriptive one.
* **It answers the descriptive question, which has never been answered** — has land shifted
  annual → perennial? — and that is what gives the DiD's bounded null its meaning. "Titling did
  not cause conversion" is a finding if conversion happened and vacuous if it did not. Nobody
  currently knows which.
