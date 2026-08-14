# The two-period tenure difference-in-differences — plan

> **⚖️ Status: REOPENED as v3 and RUN IN FULL 2026-08-11/12. An estimate exists and it is a
> bounded null. See [§9](#9-v3--reopened-with-a-different-gate-2026-08-11) and
> [`RESULTS.md`](RESULTS.md) §11.** Headline **−0.0011, CI [−0.0126, +0.0104]**; placebo
> **−0.0029** (no anticipation); corrected at M = 7 **+0.0195, CI [−0.0330, +0.0720]** ⇒
> **NOT-SEPARABLE** under the registered rule. ⭐ Now that the placebo is measured, the **v2
> gate turns out to have been unpassable at any sample size** — its point estimate lies outside
> the band, so more precision returns FAIL, never PASS.
>
> **⛔ Superseded status (v2): RAN 2026-08-11 and STOPPED AT N3 — NOT FUNDABLE. See
> [§8](#8-what-happened) and [`RESULTS.md`](RESULTS.md) §10.** N1/N2 completed, N3 returned
> infeasible (the placebo gate needs 25,202 parcels per arm; **6,559 exist in all of Peru**),
> so N4/N5 never ran. ⭐ The pilot's headline **−0.0383 reverses to +0.0338** once the
> pre-period is required to precede each parcel's own declaration.
>
> **Gates revised 2026-08-11 (v2), before the run.** Five gates became three;
> G1 became an equivalence test with a registered INCONCLUSIVE outcome; the design moved from
> calendar pre/post to **parcel-specific** pre/post after the two tenure observations were
> re-measured and found to be snapshots of a rolling titling programme, not two shared dates
> (§1.1). Deleted gates and their reasons are kept in §4, not removed. The successor to
> [`temporal_ood_plan.md`](temporal_ood_plan.md), whose steps 1–3 all ran and none of which
> fixed the underlying problem ([`RESULTS.md`](RESULTS.md) §9).
>
> **Read first:** `RESULTS.md` **§9.6** (the free pilot of exactly this design — including the
> placebo that does *not* clear), §8.2/§8.3 (why the previous control pool failed), §8.5 (T1's
> differential false-positive rate), §8.7 (where the second tenure observation came from),
> and CLAUDE.md §8.

---

## 0. Why this plan exists

Four estimands have failed, and every one of them failed the same way: the classifier's
behaviour changes over time for reasons that have nothing to do with the land, and every
design so far needed a *level* or a *trend* that the classifier could not deliver.

This plan stops trying to fix that. It changes the **comparison** so the artefact appears on
both sides and subtracts out.

**The one thing to understand before reading further:** the two dated tenure observations
supply the **treatment**. The classifier still supplies the **outcome**, at both dates, with
all of its drift intact. Nothing here makes the classifier better. What it does is make the
classifier's error *common to the two groups being compared*, which is a far weaker
requirement than accuracy.

Why that works here and did not work for the window pivot: M2 compared annual-declared
parcels against **perennial**-declared parcels, and the drift is class-specific — the
compression pushes true perennials down and true annuals up (§8.3). Two different kinds of
land, two different biases, nothing cancels. Here **both groups are annual-declared parcels
starting at the same ~6 % predicted-perennial baseline**, differing only in a paperwork event.
Measured in the §9.6 pilot: they drift at +0.0127 and +0.0115 per decade and their difference
is flat at −0.0013.

---

## 1. The data that makes this possible

Found on disk 2026-08-10 (§8.7), built as `allperu.tenure.tenure_two_period`, written to
`data/processed/all_peru/tenure_two_period.parquet`:

| | |
|---|---|
| BD SSET `ESTADO en RRPP` | registration status at **declaration** (parcel-specific, 1996–2009) |
| bridge `.dta` `estado` + `fech_tran` | cadastre's own status at its **transaction date** (mostly 2011–12) |
| parcels with both | **1,780,580** |
| moving NO INSCRITO → REGISTERED | **8.6 %** (152,742); Piura 25.5 %, Tumbes 31.8 %, Lima 20.7 % |

### 1.1 ⚠️ These are two SNAPSHOTS OF A ROLLING PROCESS, not two titling opportunities

The first version of this plan read the two observations as two fixed dates that every parcel
shared, which would make "pre" and "post" calendar concepts. **They are not.** Titling ran as a
continuous programme of departmental campaigns across the whole panel period, and what we hold
are two *observations* of each parcel's position in it, each taken on its own date. Measured
on the built table (2026-08-11):

| fact | measured |
|---|---|
| declaration year `reg_year` | spread **1996–2009** (bulk 1997–2003), tail to 2019 |
| `fech_tran` is **not one national cut** | 3–112 distinct dates *per department*; LA_LIBERTAD spans **1999–2015**, ICA from **2000**; most departments 2011–12 |
| **parcels with a status but NO date** | **437,340 (24.6 %)** — all of AYACUCHO, plus others |
| parcels whose cadastre date precedes their declaration | 420 |
| gap between the two observations | mean **10.1 y**, range 0–15 |
| registration rate by that gap | **non-monotone**: 0.28 at 0–3 y, 0.13 at 6–9 y, 0.31 at 12–15 y |

Four consequences, and every one of them binds the design:

1. **"Pre" and "post" are parcel-specific, not calendar.** A parcel declared in 2007 has no
   clean pre-period inside a panel that starts in 1999 with treatment possible from 1996.
   §2 therefore defines the windows **relative to each parcel's own two dates** and keeps only
   parcels for which a calendar window falls entirely on the right side of both.
2. **The placebo window is not automatically pre-treatment.** W04 (2004–08) is pre-treatment
   only for parcels whose registration cannot have happened before 2009. The original plan
   asserted W04 was "entirely pre-treatment"; that was an assumption about timing the data
   does not support, and it is a live candidate explanation for the pilot's −0.0169 placebo.
3. **The exposure interval is a confounder, not a nuisance.** Because the rate is non-monotone
   in the gap, exposure is not a duration effect — it tracks **which departmental campaign a
   parcel sat in**. Treated and control parcels must therefore be compared **within
   declaration-year cohort and department**, or the contrast is partly a campaign contrast.
4. **A quarter of the treatment variable is undated and must be dropped**, not defaulted. An
   undated status change cannot be placed on either side of a window.

**On the 2012 CENAGRO census — a note so this is not re-litigated.** The second observation used
here is the **bridge `.dta`, keyed to `COD_PREDIO`**, i.e. genuinely parcel-level. The 2012
agricultural census is a *different* source, reachable only through the farmer-**name** link
(Chain B, CLAUDE.md §4), which pins the person and not the parcel. It is **not** used as the
treatment here and must not be substituted for it: a name-level treatment assigned to parcels
would be measurement error in the treatment, and ~56 % of farmers hold more than one parcel.

Restricted to the pool this design uses — parcels whose PETT declaration was an **annual**
crop, in the 14 linkable departments (`data/processed/all_peru_full/`):

| | at-risk parcels | became registered | rate |
|---|---:|---:|---:|
| **national total** | **363,531** | **44,957** | 12.4 % |
| PIURA | 36,087 | 14,147 | 0.392 |
| LA_LIBERTAD | 46,749 | 9,774 | 0.209 |
| LIMA | 19,679 | 6,911 | 0.351 |
| CAJAMARCA | 91,814 | 3,790 | 0.041 |
| ICA | 13,287 | 3,227 | 0.243 |
| AREQUIPA | 28,803 | 2,485 | 0.086 |
| ANCASH | 69,226 | 2,240 | 0.032 |
| (7 more) | 57,886 | 2,383 | — |

⚠️ `tenure_two_period.parquet` contains a small number of duplicate `COD_PREDIO` rows
(726,812 join hits against 726,808 parcels). **Always `drop_duplicates("COD_PREDIO")` before
joining**, or a handful of parcels are silently double-weighted.

---

## 2. The design

**Unit** parcel. **Panel** two or more 5-year windows (`allperu/windows.py`).

**Treatment** `became_registered` — NO INSCRITO at the parcel's declaration, REGISTERED at its
cadastre transaction date. The change happened *somewhere inside that parcel's own interval*;
the timing within it is unobserved (§1.1).

**Control (stated explicitly — it was not, and it is not obvious).** The primary control is
**NO INSCRITO at both observations**: same starting state, same programme, differing only in
whether the paperwork completed. Parcels that were **already INSCRITO at declaration** are
*not* the primary control — they are a different kind of parcel, carrying T1's 4.4 pp raw
false-positive gap (§8.5), and using them rebuilds M2's "two different kinds of land" failure
one level down. They are a **separate tagged sensitivity arm**.

**Restrictions (none of them optional).**

| # | restriction | why |
|---|---|---|
| R1 | PETT label == `ANNUAL` (the at-risk pool) | both arms then start at a true perennial share ≈0, so the class-specific drift applies equally to both. **Dropping this rebuilds the design that already failed.** |
| R2 | the parcel's two observations are **both dated** | 24.6 % are undated (§1.1); an undated change cannot be placed in time |
| R3 | cadastre date **after** declaration date | 420 parcels violate it |
| R4 | the pre-window ends **before** the parcel's declaration year, and the post-window begins **after** its cadastre date | otherwise "pre" contains treatment |
| R5 | treated and control compared **within department × declaration-year cohort** | the registration rate is non-monotone in exposure, so it tracks campaign waves, not duration (§1.1) |

**Windows.** The calendar grid stays `allperu/windows.py`'s W99/W04/W09/W14/W19, but R4 makes
window *eligibility* parcel-specific. In practice:

* **post** = W14 (2014–18) and W19 (2019–23) — after every plausible cadastre date;
* **pre** = the latest window entirely before the parcel's declaration year;
* **placebo** = a second pre-window, also entirely before declaration.

⚠️ **R4 is expensive and the plan must be honest about it.** The panel starts in 1999 and most
declarations are 1997–2003, so for the bulk of parcels *no* window is entirely pre-declaration
and **the strict placebo is only available for late-declaring parcels** (roughly `reg_year`
≥ 2009, for whom W99 and W04 are both clean). N1 measures how many parcels survive R4 before
anything is drawn. If that pool is too small to gate on, **say so and stop** — an unidentified
placebo is the reason this design exists, and running without one repeats the pilot's error at
greater cost.

**Outcome** window-mean `prob_PERENNIAL` (W-D9: the probability, not the thresholded class —
thresholding turns a parcel sitting stably at p ≈ 0.45 into a coin flip). The thresholded
share is reported alongside as a secondary.

**Estimator** parcel FE + window FE + **department × window FE** (R5), `treat × post`, SEs
clustered by 5 km `region_id`. Parcel FE absorbs every time-invariant parcel property —
**including T1's differential false-positive rate** (§8.5), which is a level difference and
therefore drops out. Department × window FE absorbs any region-specific drift, which is the
artefact class that killed M2. What does *not* drop out is a bias gap that **widens over time
differently in the two arms**; that is what G1 tests.

⚠️ **Both arms must live inside the same regions.** Regions are drawn whole (the
autocorrelation range), but a region must contain **treated and control parcels**, never be
assigned to one arm. An arm-pure region makes treatment collinear with the clustering unit and
turns every local shock into a confound.

---

## 3. Decisions that bind this work

| # | decision | rationale |
|---|---|---|
| **N-D1** | **The placebo is the primary gate, and it has THREE outcomes: PASS / FAIL / INCONCLUSIVE.** If it does not clear, no effect is reported, whatever the post-period coefficient says | a non-zero value means the groups were already diverging and the design is not identified. The third outcome is registered *now* because at the sample sizes in play a null placebo fails on noise about a third of the time (§4), and "we didn't have the precision" is a different finding from "there is a pre-trend" |
| **N-D8** | **Pre/post are parcel-specific (R4), not calendar.** Any window used as "pre" must lie entirely before that parcel's own declaration | §1.1 — the two observations are snapshots of a rolling programme, so a fixed calendar placebo is contaminated by early treatment |
| **N-D9** | **The control is NO INSCRITO at both observations.** Already-INSCRITO parcels are a sensitivity arm, never the primary control | §2 — they are a different kind of parcel, which is exactly how M2 failed |
| **N-D2** | Register the gates and their thresholds **before** the extraction, and write them into code that exits non-zero | the project's whole track record is pre-registered predictions being confirmed (plan.md §7b.1) |
| **N-D3** | The estimand is the **difference in differences on the at-risk pool only**. No level, no trend, no per-parcel conversion date | levels and trends are exactly what four estimands have failed to deliver |
| **N-D4** | Every experiment gets its own tagged run dir and tagged tables. **Nothing overwrites** | four models, four panels and three workspaces coexist |
| **N-D5** | The national **locked test stays unspent** | unchanged since §6.2b |
| **N-D6** | Negative results are written up as fully as positive ones | the project's most valuable findings are all negative |
| **N-D7** | **Report the pilot's numbers alongside the new ones.** If the properly-powered study reverses the pilot, that is a finding about pilots and must be said | §9.6 is on the record and cannot be quietly dropped |

---

## 4. The gates — THREE, not five

The first version had five gates. Two of them could not do their job and are removed for
stated reasons, not quietly dropped:

* **old G2 (tenure bias per window) is deleted — it is not identified.** T1 works because the
  at-risk pool's *true* perennial share is ≈0 at the label year, so the predicted share **is**
  the false-positive rate. By W14/W19 the true share is no longer ≈0 — that is the quantity
  being estimated. So "test the `tenure × window` interaction" cannot separate bias drift from
  real conversion in exactly the windows it was meant to police. Restricted to the pre-windows
  where it *is* identified, it is the same regression as G1 with a different grouping variable,
  and it is reported there as a companion, not as a gate.
* **old G5 (arm slopes agree within 0.005/decade) is deleted — it was un-passable.** The gap
  between the two arms' slopes *is* the estimate, in slope form. A real 2 pp effect over
  ~15 years is ≈0.013/decade, 2.6× the threshold, so G5 could only be passed by finding
  nothing. Its legitimate content — *do both arms drift together before treatment?* — is the
  same assumption G1 tests and is folded into G1 as a reported companion series.

What remains runs in this order. **G1 is fatal on its own.**

---

### G1 — the pre-trend. **Fatal. Equivalence, not significance.**

DiD across two windows that are **both entirely before the parcel's own declaration** (R4).

| | |
|---|---|
| **PASS** | the **whole 95 % CI lies inside ±0.005** |
| **FAIL** | the CI excludes 0 and the point estimate is outside ±0.005 |
| **INCONCLUSIVE** | anything else — the CI is wider than the equivalence band |

**Why it is built this way.** The old criterion paired an equivalence bound (|coef| < 0.005)
with a significance test (CI contains 0). Those pull in opposite directions: the first gets
easier as precision improves, the second gets *harder* — it rewards a noisy estimate and
punishes a precise one. A pre-trend of 0.004 ± 0.001 would have failed the old gate while
passing its own first clause. An equivalence test is the correct instrument for "show me this
is near zero", and it makes the precision requirement explicit instead of hidden.

**What a pass does and does not buy.** The placebo spans one 5-year step; the headline spans
~15 years. A pre-trend sitting just inside ±0.005 extrapolates to ~0.015 of contamination on
the headline — **about 40 % of the pilot's −0.038**. A pass bounds the contamination; it does
not eliminate it, and the write-up must say so.

⚠️ The §9.6 pilot gives **−0.0169 (se 0.0132, p 0.20)** and would be **INCONCLUSIVE** under
this gate, which is the honest verdict — 334 treated parcels cannot resolve a pre-trend the
size of the effect. **Resolving that is what the study is for.**

**Reported alongside (not gates):** the two arms' own pre-period window series (old G5's
content), and the pre-window `tenure × window` interaction (old G2's identifiable part).

### G2 — sample integrity. One gate, four cheap checks, all computable before extraction.

| check | criterion |
|---|---|
| **common support** | standardised difference **< 0.25** between arms on department, parcel size, declaration-year cohort and exposure gap |
| **no differential attrition** | the rate at which a window *qualifies* (`MIN_YEARS = 3` of 5) differs between arms by **< 0.02** in every window |
| **no training contamination** | the share of each arm that was in the model's training set differs by **< 0.02** |
| **arms share regions** | **≥ 80 %** of sampled parcels sit in a region containing both arms |

Rationale for each: 0.25 is the conventional ceiling and is *appropriately* loose because
parcel FE means balance is not required for identification — this is a common-support check,
not a matching exercise. Attrition and training membership are new: window qualification
depends on observation density, which depends on parcel size, which correlates with titling;
and the model was trained on some of these parcels **labelled `ANNUAL`**, which holds their
predicted perennial probability down in every window. Either one, if it differs by arm, biases
the DiD directly. Both are two-line computations on the existing panel and neither was checked.

⚠️ **Baseline `p_mean` is reported but is NOT a balance criterion, and the sample is never
trimmed on it.** Selecting on the pre-period outcome induces regression to the mean, and mean
reversion looks exactly like the pilot's placebo: a smooth divergence beginning before
treatment. The old plan's "report the trimmed sample too" would have risked manufacturing the
artefact G1 exists to detect.

Dropped from the old G3: **"every stratum has both arms"**. Treatment rates run from Piura
0.392 to Huancavelica 0.004, so enforcing it literally deletes the low-treatment sierra and
keeps the coastal departments — a non-random deletion in the direction of maximum El Niño and
latitude exposure. The region-sharing check above is what that clause was really after.

### G3 — precision, sized for the gate that decides. **Computed and reported before extraction.**

| | criterion |
|---|---|
| **G3a — placebo resolution (binding)** | expected SE on the G1 contrast **≤ 0.0025**, so a true null passes the ±0.005 band with ≥95 % probability |
| **G3b — effect power** | ≥ 80 % power at **2 pp** on the headline contrast |

**G3a is new and it is the one that sizes the study.** The old G4 powered the *headline* while
N-D1 made the *placebo* decisive, so it would have reported "PASS, 92 % power" on a sample that
then coin-flips the gate that actually ends the project. Scaling the pilot's placebo SE
(0.0132 on 334 treated):

| treated parcels | expected placebo SE | P(fail G1 when the truth is zero) |
|---:|---:|---:|
| 334 (pilot) | 0.0132 | 71 % |
| 2,000 (the old target) | 0.0054 | 35 % |
| 4,000 | 0.0038 | 19 % |
| **6,500** | **0.0030** | **9 %** |
| 10,000 | 0.0024 | 4 % |

G3b's arithmetic is unchanged and correct (~1,300 treated for 2 pp, ~5,200 for 1 pp), and 2 pp
remains the right target: baseline is ~6 %, and T1's known artefact is 1.75 pp, so anything
smaller is uninterpretable regardless of power.

Two notes that travel with G3. The pilot SE is an **assumption**, not a measurement — it comes
from a different sample with a different department mix and a measured `deff` of 2.36, so
report it as such (this project forbids exactly this kind of extrapolation for GEE timing).
And for a fixed parcel budget, **balanced arms beat 1:2**: 3,000/3,000 gives a larger effective
sample than 2,000/4,000 at identical cost.

---

## 5. Task sequence

### N1 — Reproduce the pilot, cleanly and as a module *(no new data; hours)*
Move the §9.6 scratch analysis into `allperu/tenure_did.py` with `run_did(...)`,
`placebo(...)`, `integrity(...)` and a `gate(...)` that returns PASS/FAIL/INCONCLUSIVE per
G1–G3 and exits non-zero on FAIL. Reproduce **−0.0383 (se 0.0141)** headline and **−0.0169
(se 0.0132)** placebo exactly, and pin both in a test. Tests: `tests/test_tenure_did.py`,
following `tests/test_window_pivot.py`.

**Three things N1 must settle before anything else runs:**

1. **⚠️ Reconcile the pilot's table with its own regression.** §9.6's series gives arm
   differences of −0.0020 (W99) and +0.0054 (W19) — a raw W99→W19 DiD of **+0.0074**. The
   regression reports **−0.0569** on the same contrast. **Opposite sign, 7× magnitude.**
   Probably composition (raw means over whoever qualifies each window vs. within-parcel FE on
   the balanced subset) — but if composition can flip the sign, the headline is a composition
   artefact until shown otherwise. Reproduce **both**, explain the gap, and add a
   balanced-panel version of the raw series. This, not matching two coefficients, is N1's
   acceptance criterion.
2. **Measure how many parcels survive R2–R4** (dated, ordered, with a genuinely
   pre-declaration window). §2 warns this may be small. Report it before N3 is designed.
3. **Report the pilot under the new gate definitions**, so the pilot and the study are scored
   on the same instrument (N-D7). Expected: G1 INCONCLUSIVE.

### N2 — Run G2 (sample integrity) on the existing panel *(no new data; hours)*
All four checks are computable now. Two of them — differential attrition and training-set
membership — have never been run on anything in this project.

### N3 — Draw the sample *(no GEE; hours)*
Extend `allperu/window_sample.py` (or a sibling) to stratify on
**department × declaration-year cohort × `became_registered`** (R5), over **whole 5 km
regions**, with **both arms inside each region** — draw the region, then take treated and
control from within it. Never assign a region to an arm.

Target from **G3a**, not G3b: **≥6,500 treated and ≥6,500 control** for a 9 % false-fail rate
on the placebo. If the budget will not carry that, the fallback is explicit — draw the smaller
sample, and accept in advance that G1's likely verdict is INCONCLUSIVE rather than PASS. Keep
a per-region cap so no region dominates. Write `sample_weight` and **use it in every reported
share** — the allocation is not proportional and an unweighted number is a statement about
the sample, not about Peru.

⚠️ Departments differ enormously in treatment rate (Piura 0.392, Huancavelica 0.004). A
proportional draw would be almost entirely Piura and La Libertad, which is the department the
El Niño and the latitude findings both single out. **Cap Piura's share and report the
estimate with and without it.**

### N4 — Extract *(the only GEE spend)*
Only the windows the design uses: **1999–2003, 2004–2008, 2014–2018, 2019–2023**. `--years`
takes multi-range specs, so this is one command. Follow the §3a hygiene that worked: run
`timing_probe` first and size from the measured rate (the last panel measured ~1.2–1.5 h per
year over 4,565 parcels), then **verify every year by counting output parcels** — never by
"the process ended". Mission policy is unchanged: **L5+L7 only, OLI stays out** (§9.5).

### N5 — Assemble, infer, gate, estimate
Assemble per year, infer with the **selected** model
(`runs/all_peru/selected_model.json` → `lightgbm_nometa_nolat_aug_yleak10`), then run G1–G3.
**Only if G1 PASSES and G2 passes** produce the DiD estimate. G1 INCONCLUSIVE is **not** a
pass: report it as such and stop. With:
* the coefficient on the probability outcome **and** on the thresholded share;
* W14 and W19 reported **separately** — W19 is where the density artefact is worst, and the
  pilot's coefficient tripled there (−0.0197 → −0.0569), which is itself suspicious;
* a sensitivity arm on a second model (`lightgbm_nometa_nolat` and `ltae`) — if the sign
  depends on the architecture, it is not a finding;
* the estimate discounted for §6.4's export share (PERENNIAL is 57 % export / 20 % mixed /
  23 % domestic) before any sentence about export crops.

---

## 6. Risks this design does *not* solve

* **Parallel trends is an assumption, and G1 can only ever fail to reject it.** Farmers who
  register may already be planning to invest — that is reverse causality, and it produces
  exactly the pre-trend G1 looks for. The pilot's −0.0169 is a live warning, not a formality.
* **Treatment timing is coarse, and it is coarse in a specific direction.** `fech_tran` dates
  a cadastre *transaction*, not the inscription event, and the two observations are snapshots
  of a rolling programme (§1.1). So this is a two-period DiD and **not** a staggered event
  study; no event-time plot can be produced, and anyone asking for one should be pointed here.
  The live risk is that registration happened *early* in a parcel's interval, inside what a
  calendar design would call the pre-period — which R4 exists to prevent and which is a
  candidate explanation for the pilot's placebo.
* **Exposure is confounded with campaign.** The registration rate is non-monotone in the gap
  between observations (§1.1), so the gap is not a duration — it marks which departmental
  titling wave a parcel sat in. R5 conditions on it; nothing removes it entirely.
* **`estado` is a pipeline state, not a binary.** 73.6 % of NO INSCRITO parcels sit in
  `IN_PROCESS`. The treatment definition must be stated explicitly and a sensitivity arm run
  with `IN_PROCESS` treated as control, then as excluded.
* **Woody non-crop (2.79 % of the pool, §6.3) and perennial ≠ export (§6.4)** are unchanged
  by any of this and still bound the interpretation.
* **⭐ The control arm is contaminated after 2011, and nothing here can see it** (added
  2026-08-11 with §9). The last observation of tenure is the cadastre cut (~2011–12); the
  outcome runs to **2023**. Peru's titling programme did not stop, so an unknown share of the
  control arm was titled in 2012–23 while still being counted as untreated. That is treatment
  in the control group, and it drags any real effect **toward zero**. Consequences, both of
  which must be stated wherever a number from this design is quoted: a positive finding is an
  **underestimate**, and a NOT-SEPARABLE finding is partly attributable to this contamination
  and is **not** evidence of no effect. Bounding it needs a third dated observation, which
  this project does not have.
* **The outcome is still a prediction.** If the classifier is simply blind to the transition
  being studied, the DiD is unbiased and empty. Endpoint labels
  ([`endpoint_labels_plan.md`](endpoint_labels_plan.md)) are the only way to know, and they
  would also give this design a validated outcome measure. **The two plans are complements,
  not alternatives.**

---

## 7. What this plan does not do

No per-parcel trajectories. No annual series. No national area estimate. No conversion dates.
No use of the locked test. If the gates pass, the deliverable is **one number with a
confidence interval, plus the gates that licensed it**.

---

## 8. What happened

**Ran 2026-08-11 against this plan as registered (v2). N1 and N2 completed; N3 returned
infeasible; N4 and N5 were not run. No estimate exists. Full account:
[`RESULTS.md`](RESULTS.md) §10.**

### 8.1 Scoreboard

| task | outcome |
|---|---|
| **N1** reproduce the pilot as a module | ✅ done — all five §9.6 coefficients reproduce to 4 dp, pinned by tests |
| **N2** G2 on the existing panel | ⛔ **3 of 4 checks FAIL** |
| **N3** draw the sample | ⛔ **INFEASIBLE — the required sample does not exist in Peru** |
| **N4** extract | **NOT RUN** (stop rule) |
| **N5** assemble, infer, estimate | **NOT RUN** (stop rule) |

| gate | registered criterion | measured | verdict |
|---|---|---|---|
| **G1** placebo | whole 95 % CI inside ±0.005 / 5 y | +0.0049, CI [−0.026, +0.036] | **INCONCLUSIVE** |
| **G2** integrity | 4 checks | common support 0.609, train-membership −0.062, shared regions 0.366 | ⛔ **FAIL** |
| **G3a** precision | expected placebo SE ≤ 0.0025 | requires 25,202/arm; **6,559 treated exist** | ⛔ **FAIL (2.41×)** |
| **G3b** power | ≥80 % at 2 pp | achievable MDE 6.9 pp | ⛔ FAIL |

### 8.2 What the plan got right

* **§1.1 was the decisive section, and adding it was what killed the design.** Re-measuring
  the two tenure observations as snapshots of a rolling programme produced R4; R4 produced the
  cohort restriction; the cohort restriction produced the infeasibility. A plan revision that
  cost an hour saved a multi-hour extraction and a wrong published number.
* **N-D1's three-outcome G1 was the right instrument.** Every valid arm returned
  INCONCLUSIVE, not FAIL. Under the v1 criterion (`|coef| < 0.005` **and** CI contains 0) the
  primary arm would have **PASSED** — +0.0049 is inside the band and its CI contains zero —
  and the study would have proceeded to extraction on a placebo that resolves nothing. *The
  gate that would have passed it is the one that rewarded imprecision.*
* **N-D9 mattered more than expected.** The pilot's control mixed already-`INSCRITO` parcels
  into the comparison, importing T1's differential false-positive rate as a 6.5 pp baseline
  gap — which is where its headline came from.
* **G2's two new checks earned their place.** Differential attrition passed cleanly everywhere;
  **training-set membership** and **shared regions** both failed, and neither had ever been
  measured in this project.

### 8.3 What the plan got wrong

* **N3's target (≥6,500 treated) was set from the G3a *ladder* without checking the
  *population*.** The ladder said 6,500 treated would give a 9 % false-fail rate; the archive
  contains exactly 6,559 treated parcels that survive R4, and the required number was 25,202.
  The plan sized the sample against the budget and never asked whether the parcels existed.
  **`population_ceiling` should have been in the plan as a gate, not discovered during N3.**
* **§2's warning about R4 ("R4 is expensive and the plan must be honest about it") understated
  it.** The plan anticipated a *small* strict-placebo pool; it did not anticipate that the
  powered cohort and the placebo-capable cohort are nearly disjoint (6,559 vs 380).
* **The plan assumed the pilot's headline would be tested, not reversed.** N-D7 was written
  for the case where the properly-powered study contradicts the pilot. In the event the
  reversal came from a *specification* fix at pilot sample size — cheaper and more decisive
  than the study it was hedging against.

### 8.4 The verdict

The design is not refuted — its central premise (drift common to both arms cancels) held, and
nothing here shows the estimand is uninteresting. It is **not identifiable at adequate
precision on this data**, because the interval in which treatment could have occurred overlaps
the only pre-period the panel has. That is a fact about the titling programme and the Landsat
record, not about the model, and no budget changes it.

**Do not reopen this by relaxing R4, widening the band, or dropping the at-risk restriction.**
Each of those makes the design feasible by removing the thing that made it credible, and
§10.6 records why. Reopening needs either a panel that starts materially before 1996 (it
cannot — 1992 has zero clear acquisitions over Piura) or a dated registration *event* rather
than two snapshots.

---

## 9. v3 — REOPENED with a different gate (2026-08-11)

> §8 stays exactly as written. This section records the reopening, what changed, and what
> came out. **Nothing in §1–§7 changes: R1–R5 and N-D1…N-D9 all still bind.**

### 9.0 Why this is not the reopening §8.4 forbids

§8.4 says: *do not reopen this by relaxing R4, widening the band, or dropping the at-risk
restriction.* None of those happens here. **R4 is applied unchanged** — it is what makes
"before" mean before, and removing it is what produced the pilot's since-reversed −0.038.
R1 and N-D9 are unchanged. What changes is **the gate, not the design**.

G1 v2 was an *equivalence* test: prove the pre-trend is inside ±0.005/5 y. That is a
demanding instrument, and the sample it needs — 25,202 parcels per arm from measured
variance — does not exist in Peru (6,559 do, §10.4). **The standard alternative in the DiD
literature is to measure the pre-trend and subtract it, carrying its uncertainty into the
final interval.** That is strictly less demanding and it *is* achievable with the parcels
that exist. It also makes the cost of not being able to prove the pre-trend away explicit
and quantitative, rather than fatal.

### 9.1 The arithmetic that governs everything

The placebo spans **2.5 years** (P1 1999–2000 → P2 2001–2003); the headline spans **17.5**
(W99 → the mean of W14 and W19). So a pre-trend that persists contaminates the headline by
**M = 7×** the placebo coefficient — and its standard error is amplified by 7 as well:

```
corrected = headline − M × placebo
se        = sqrt(se_headline² + M² × se_placebo²)
```

**M is derived from window midpoints in code** (`tenure_did.amplification_factor`), never a
constant: change the window grid and M must move with it, or the correction silently uses a
stale factor. Pinned by a test.

The consequence is registered in advance and is unflattering: at the achievable precision the
corrected SE is ~0.017, so **only an effect larger than about 3.3 pp can survive the
correction.** Anything smaller is genuinely not separable from pre-existing drift, and saying
so is the result.

### 9.2 What was registered, before any extraction

Written to `data/processed/all_peru_did/did2_registration.json` **before** N4 ran, and not
edited afterwards (`write_registration` refuses):

| | |
|---|---|
| **primary decision rule** | report an effect **only if** the 95 % CI of `headline − M × placebo` excludes zero; otherwise **NOT-SEPARABLE**, which is a complete published outcome |
| **primary outcome** | window-mean `prob_PERENNIAL` (W-D9 — the probability). Thresholded share is secondary |
| **primary contrast** | pre = W99 (1999–2003); post = W14 + W19, also reported separately |
| **placebo** | P1 (1999–2000) vs P2 (2001–2003), both entirely pre-declaration for the `reg_year ≥ 2004` cohort |
| **sensitivity curve, always reported** | corrected effect + CI at **M ∈ {0, 1, 3, 5, 7}**. M = 0 is the uncorrected headline; M = 7 assumes the pre-trend runs straight for 17.5 years and is the pessimistic end |
| **G2** | measured and reported as a **diagnostic, not a gate** — common support, training membership and shared regions are known to fail (§10.3) and the sample is not redesigned to fix them |

### 9.3 What changed in the task sequence

* **N3 (the draw) is sized to the population, not to a precision target** — take *every*
  qualifying treated parcel, since 6,559 is all there is. New module `allperu/did_sample.py`.
* **N4 extracts 15 years, not 20**: `1999-2003,2014-2023`. W04 and W09 are not needed —
  the placebo now comes from splitting W99.
* **N5 produces the placebo first, then the headline, then the correction and the curve.**
  A headline computed before its pre-trend is a number nobody can un-see; the code path
  enforces the order.

### 9.4 A risk the plan did not list, and must

**The last observation of tenure is ~2011, but the outcome runs to 2023.** Peru's titling
programme kept running, so an unknown share of the control arm was certainly titled between
2012 and 2023 and this design cannot see it. Those parcels sit in the control arm while
receiving the treatment, which drags any real effect **toward zero**. A positive finding is
therefore an underestimate; a NOT-SEPARABLE finding is *partly* attributable to this
contamination and cannot be read as evidence of no effect. Add it to §6.

### 9.5 What happened — scoreboard first

**Ran 2026-08-11/12 against §9 as registered. All four tasks completed; an estimate exists.
Full account: [`RESULTS.md`](RESULTS.md) §11.**

| task | outcome |
|---|---|
| **T1** the correction, in code + tests | ✅ done — M derived from midpoints, 31 tests |
| **T2** draw the sample | ✅ **14,625 parcels** (6,559 treated = a census, 8,066 control) |
| **T3** extract | ✅ **15 years, 211,567 parcel-years, 63.6 M pixel-obs**, ~13.5 h |
| **T4** assemble, infer, estimate | ✅ 219,375 parcel-years, three arms |

| quantity | measured |
|---|---|
| **placebo** P1→P2 (2.5 y) | **−0.0029** (se 0.0037, p 0.43) — *no anticipation* |
| **headline** W99→W14+W19 | **−0.0011** (se 0.0059, p 0.85), CI **[−0.0126, +0.0104]** |
| W14 / W19 separately | −0.0022 / −0.0006 |
| **corrected at M = 7** | **+0.0195** (se 0.0268), CI [−0.0330, +0.0720] |
| **decision (registered rule)** | ⚖️ **NOT-SEPARABLE** |
| sensitivity arm `nolat` | headline −0.0020, placebo −0.0009 — agrees |
| negative control `ltae` | headline −0.0154 (p 0.055) — same sign, **14× the magnitude** |

**The headline is a tight null, not a large effect hidden by a pre-trend.** The correction
widens ±1.3 pp to ±5.3 pp; it does not move the point estimate off zero.

### 9.6 What §9 got right

* **The reopening was justified, and by more than expected.** §10 stopped because the v2 gate
  needed 25,202 parcels per arm. Now that the placebo has actually been measured, **the v2 gate
  was unpassable at any sample size**: the point estimate (−0.00295) sits *outside* the ±0.0025
  band, so the SE needed for its CI to fit inside is negative. More precision would have
  returned FAIL, never PASS — and it would have failed on a pre-trend of −0.0118/decade that
  cannot change a null conclusion. **A pre-trend gate must ask whether the pre-trend could
  overturn the conclusion, not whether it is smaller than a pre-chosen number.**
* **R4 and N-D9 did their jobs.** The arms' baseline gap is **−0.0028**, against the pilot's
  **+0.065**. The regression-to-the-mean signature §10.1 identified in the pilot is absent.
* **Drawing controls region-first fixed G2's worst failure by design** — shared regions
  **0.366 → 0.849** — and drawing from the full national table fixed training-set
  contamination (**−0.062 → +0.0026**). 3 of 4 G2 checks now pass; the sample was not
  redesigned to achieve it.
* **Registering the sensitivity curve was right.** A single M = 7 number (+0.0195) reads as a
  positive effect; the curve shows it is the uncorrected −0.0011 pushed around by a noise term.

### 9.7 What §9 got wrong

* **The verify tolerance was set by taste and cried wolf on a complete extraction.** 13 of 15
  years "failed" at 0.9941–0.9963 against a guessed 0.995. The real floor is **sub-pixel
  parcels** — the same ~85 every year, median 0.11 ha (RESULTS.md §11.4.1). Fixed by measuring
  the floor and adding a floor-independent consistency statistic.
* **The 2:1 control target was not reachable and the plan should have checked first.** 1.23:1
  achieved; LA_LIBERTAD 2006 alone wants 5,698 controls and the department holds 2,249. The
  same failure mode as §8.3's — sizing against a target without asking whether the parcels
  exist — though here it cost nothing, because the missing controls would have been absorbed
  by the department × POST fixed effects.
* **The plan did not anticipate that the architecture would matter for magnitude.** `ltae`
  gives −0.0154 against LightGBM's −0.0011. The registered rule only tested the *sign*.

### 9.8 The verdict

The design is identified, the premise held, and the answer is a **bounded null**:
|effect| < 1.3 pp uncorrected, < 5.3 pp after the pessimistic pre-trend correction, against a
~7 % baseline. **NOT-SEPARABLE** under the registered rule.

⚠️ **It must be quoted with §6's new risk**: the tenure observations end ~2011 while the
outcome runs to 2023, so part of the control arm was certainly titled in between. That
attenuates toward zero, which means this null is **not** evidence that titling has no effect.

Reopening again does not need a bigger sample — the population is exhausted. It needs a
**dated registration event** rather than two snapshots, or endpoint labels
([`endpoint_labels_plan.md`](endpoint_labels_plan.md)) to validate the outcome the DiD is
measuring.
