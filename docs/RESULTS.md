# Results

Every number here was measured. One section per strand, in the order they were run. Each opens
with a verdict line.

**CLOSED means do not reopen** — the reason is stated, and it is a fact about the data or the
population, not a preference. **LIVE** means work is outstanding; the task list is in
[`STATUS.md`](STATUS.md).

For the readable narrative of the same material, see [`REPORT.md`](REPORT.md). For findings
that generalise beyond Peru, see [`LESSONS.md`](LESSONS.md).

---

## 0. Scoreboard

| # | strand | question | verdict | code |
|---|---|---|---|---|
| 1 | 12-class crop classifier | which of 12 crops? | ⛔ CLOSED — not separable | `models/`, `labels.py` |
| 2 | 3-class land-state (Piura) | perennial / annual / pasture? | ✅ **WORKS** — locked test 0.681 | `perennial/` |
| 3 | Piura 28-year panel | annual trajectory per parcel | ⛔ FAILED gate (S4, S5) | `perennial/panel.py` |
| 4 | All of Peru, single year | same, 14 departments | ✅ **WORKS** — CV 0.628 | `allperu/` |
| 4b | All-Peru 25-year panel | annual trajectory per parcel | ⛔ FAILED gate, all 3 arms | `perennial/panel.py` |
| 5 | Window pivot | 5-year window shares | ⛔ FAILED 3 pre-gates | `allperu/windows.py` |
| 6 | Temporal OOD fixes | make 2019–23 predictable | ⛔ FAILED; 4 routes closed | `allperu/density.py`, `oli_*.py` |
| 7 | Tenure DiD (v3) | does title cause conversion? | ⚖️ **COMPLETE — bounded null** | `allperu/tenure_did.py` |
| 8 | S2 endpoint labelling | validated 2020+ labels | 🔴 **LIVE** | `labelling/`, `features/s2_*` |

**The one result to know before doing any modelling:** `centroid_lat` is worth **+0.047
macro-F1 on spatial CV, +0.047 on leave-one-year-out, and −0.060 on
leave-one-department-out**. Spatial CV holds out 5 km cells *inside departments the model has
already seen*, so it cannot separate memorisation from signal. §4.2.

---

## 1. The 12-class crop classifier — ⛔ CLOSED, superseded

**Verdict: the annual crops are not separable from one another at 30 m, and no model choice
fixes it.**

49,648 parcels, 12 classes, pooled spatial-CV macro-F1:

| model | pooled macro-F1 | accuracy |
|---|---|---|
| **LTAE** (tuned) | **0.427** | 0.559 |
| PSE-LTAE | 0.421 | 0.540 |
| LightGBM | 0.383 | 0.576 |

Majority-class accuracy baseline is 0.425 — i.e. the best model barely clears predicting
"rice" for everything. Per-class F1 shows why:

| class | F1 | support | | class | F1 | support |
|---|---|---|---|---|---|---|
| ARROZ (rice) | 0.751 | 16,446 | | ALGODON (cotton) | 0.376 | 2,855 |
| CAFE (coffee) | 0.698 | 2,322 | | MAIZ (maize) | **0.228** | 4,175 |
| TRIGO (wheat) | 0.666 | 661 | | PLATANO (banana) | 0.197 | 448 |
| MANGO_LIMON | 0.529 | 1,435 | | CAÑA DE AZUCAR | 0.147 | 339 |
| FALLOW | 0.512 | 7,224 | | FRIJOL (beans) | **0.114** | 501 |

Only rice (flooded, spectrally distinctive) and coffee (evergreen canopy) do well. Cotton,
maize, beans, wheat and zarandaja produce **the same whole-year NDVI curve** — a season
peaking Mar–May, senescent by August (`figures/profiles_12class_ndvi.png`).

**What this bought:** the argument for strand 2. The distinction the imagery *can* support is
not "which crop" but "does this parcel hold canopy year-round" — which is also the distinction
the research question needs (perennial ≈ export, annual ≈ domestic).

⚠️ **The 12-class locked test is spent** (macro-F1 0.427).
⚠️ `CAÑA DE AZUCAR` looks perennial (high NDVI all year) but is classed `ANNUAL` by policy,
because it is replanted on a multi-year cycle. Deliberate, and a registered sensitivity arm.

---

## 2. The 3-class land-state classifier, Piura — ✅ WORKS

**Verdict: 0.681 macro-F1 on a locked test. This is the one component of the project that is
solid, and everything downstream assumes it.**

`PERENNIAL` / `ANNUAL` / `PASTURE_FALLOW`, 56,419 parcels, four models on identical
spatially-blocked folds (including `rules`, an explicit depth-2 phenology rule registered as a
scientific control).

| # | criterion | result |
|---|---|---|
| S1 | beat majority baseline by ≥ 0.15 macro-F1 | **PASS** (+0.393 on CV) |
| S2 | best ML beats the rule-based control | **PASS** (+0.257; read as ~+0.20, see below) |
| S3 | beat MapBiomas at predicting PETT labels | **PASS** (+0.296) — structurally, not fairly |
| S4 | temporal transfer degrades gracefully | ⛔ **FAIL** — §3 |
| S5 | trajectories are not noise | ⛔ **FAIL** — §3 |
| S6 | trend agrees with MapBiomas | not evaluable — MapBiomas has no perennial class |

**Locked test, spent once** (`runs/perennial/lightgbm_3c_final`, n = 9,671):

| metric | value |
|---|---|
| **macro-F1** | **0.681** |
| accuracy | 0.748 (majority baseline 0.671) |
| balanced accuracy | 0.681 · Cohen's κ 0.478 · Brier 0.178 |
| per class F1 | `ANNUAL` 0.833 · `PERENNIAL` 0.769 · `PASTURE_FALLOW` 0.440 |

**Test (0.681) > CV (0.647)** — the reassuring direction. The accepted residual spatial
leakage did not inflate CV. Temperature scaling gave T = 1.440, ECE 0.093 → 0.037, argmax
preserved.

⚠️ **The Piura locked test is SPENT TWICE.** It was used a second time on 2026-08-05 at the
user's explicit request, for tuned LTAE, out of interest. Nothing was selected on it, but it is
**no longer a clean held-out estimate for any future selection**. Tuned LTAE beats LightGBM on
all 5 CV folds (0.658 vs 0.647) and **loses the locked test** (0.661 vs 0.681) — CV rank and
test rank disagree, and spatial CV did not catch it.

**Model selection:** LightGBM, on the documented cheap-model tie-break (LTAE led by 0.0027,
inside the ~0.02 threshold, and is ~100× more expensive for a 220k-parcel-year panel).

### 2.1 The MapBiomas benchmark — a decisive negative finding

**MapBiomas Peru Collection 3 cannot separate perennial from annual cropland in Piura.** Codes
36/46/47/48 (perennial crop, coffee, citrus, other perennial) and 15 (pasture) **never appear
anywhere in Piura**. The only agricultural codes present are 21 (mosaic), 40 (rice), 72 (other
crops); 63 % of our parcels land in class 21, with mango/lime orchards (84 %), coffee (45 %)
and fallow (58 %) all in that same bucket.

| predictor | macro-F1 | accuracy | coverage |
|---|---|---|---|
| **ours** | **0.681** | 0.748 | 100 % |
| MapBiomas, best configuration (mosaic → `PASTURE_FALLOW`) | 0.385 | 0.580 | 90 % |

MapBiomas gets **worse** on large parcels (0.289 at 3–50 ha vs 0.387 at 0–0.5 ha),
contradicting the expectation that mixed pixels were its binding constraint. The missing class
is, and parcel size cannot fix that.

**Frame this honestly:** S3 passed for a structural reason, not because a fair contest was won.
The useful conclusion is the positive one — a parcel-level model is *necessary* here precisely
because the off-the-shelf product does not resolve the distinction the question depends on.
⚠️ MapBiomas is itself Landsat-derived at 30 m, so it shares sensors, cloud regimes and
mixed-pixel problems. It is a benchmark, not ground truth.

### 2.2 The `rules` control was under-specified

The registered control reads only its 3 configured columns, so `BSI_max` sat unused in the same
store. Swapping `NDVI_max` → `BSI_max` wins 5/5 folds (CV 0.388 → 0.437) and repairs a
pathological class mix — the registered rule predicts `PASTURE_FALLOW` (its fall-through
branch) for 69 % of parcels against a true 22 %, which is why its accuracy sits *below* the
majority baseline. **Not adopted**: `PERENNIAL` F1 *falls* 0.493 → 0.379 and it forfeits the
control's one clean win, threshold stability. Both are kept
(`runs/perennial/rules_3c_bsi/`). **Read S2's "+0.257" as ~+0.20.**

---

## 3. The Piura 28-year panel — ⛔ FAILED its gate

**Verdict: the classifier works; the trend does not. No trajectories, transitions or area
estimates exist, and none should be produced.** That code is built and unit-tested and stays
unrun.

Panel extracted, assembled and inferred: **215,320 parcel-years**, 1996–2023, L5+L7 only.

* **S4 temporal transfer: FAIL** — accuracy 0.746 at k = 0 but **0.600 at k = −3** (0.146 >
  0.10 tolerance). That bin is 66 % panel-year 1996 and contains 1997 at accuracy 0.326.
  Forward transfer (k = +1…+3) is fine, within 0.010.
* **S5 flicker: FAIL** — **0.428** of `PERENNIAL`-labelled parcels flicker against a 0.15
  criterion. **Not a coverage artefact** (0.421 with the four thin years dropped). Smoothing
  halves it (0.221), which is exactly the outcome the plan warned would *hide* the problem.
* **Sensor drift: PASSES** — 0 of 30 mission-boundary steps exceed 2× the within-era
  year-to-year movement, raw bands included. But cross-parcel *spread* grows 1.37× into the
  L7-only era.

### 3.1 It is the data, not the model

The panel was re-inferred with **LTAE** — a different architecture carrying **no statics at
all** — and the gate **fails harder**: flicker 0.780 vs 0.428. A third run with
`lightgbm_nometa_nolat` gives a clean monotone ordering:

| arm | statics | flicker | k = 0 accuracy |
|---|---|---|---|
| `lightgbm_nometa` | 3 | 0.428 | 0.746 |
| `lightgbm_nometa_nolat` | 2 | 0.547 | 0.752 |
| `ltae` | **0** | **0.780** | 0.712 |

**Time-invariant features manufacture *stability* exactly as `frac_l7` manufactured *change*.**
S5 is therefore gameable: LightGBM's 0.428 is the optimistic end and LTAE's 0.780 the honest
one. **"Try the other model" is a closed route** — any fix must change the data or the
estimand.

### 3.2 The El Niño confound — the smoking gun

Trained on 1999+2000, tested on 1998, with the required control arm: `PERENNIAL` recall
**0.361 → 0.000**, `PASTURE_FALLOW` 0.589 → 0.100, `ANNUAL` *up* 0.049. Robust across seeds,
region definitions, and with/without the metadata ablation. **Both architectures put
`PERENNIAL` recall on 1998 at ~0.02–0.03** against controls of 0.54/0.66. 1996–98 are the
panel's baseline years, so the perennial baseline is near-zero for artefactual reasons and any
rise from it is uninterpretable.

**The mechanism (§8.2 of the old doc, worth keeping):** the perennial signature is a
*contrast* with annual neighbours, and the flood collapsed it from both sides. The `NDVI_p25`
class gap falls **+1.47 SD → +0.46 SD** and `NDVI_amp` is annihilated (−0.60 → −0.10 SD),
because perennials lost canopy (floor 0.518 → 0.448) while flooded annual ground stayed green
(0.363 → 0.404). **`BSI_max` is the one feature that holds** (−0.76 → −0.90 SD; the raw gap
widens too, so it is not a normalisation artefact). Numbers in
`figures/elnino_signature_collapse.csv`.

⚠️ **Do not read that as a missing-feature problem.** LightGBM *has* `BSI_max` (rank 9/135) and
still lost `PERENNIAL` on 1998. The deficit is **weighting** (all `BSI_*` = 6.51 % of gain
against a dominant greenness family), and reweighting is a testable hypothesis, not a
demonstrated fix.

### 3.3 Restricting the baseline to ≥1999 — tested, not sufficient

`train --train-years 1999-2023` restricts the **train** side only, so CV stays comparable.
**S4 flips to PASS (0.146 → 0.063); S5 still FAILS (0.428 → 0.356).** CV cost is *zero* on the
years it still covers (pooled CV on ≥1999 validation parcels 0.6412 vs 0.6415).

**A control gate is what makes this readable:** the *baseline* model's own predictions
truncated to ≥1999 also pass S4 at 0.076, so most of the S4 gain is the shorter panel, not the
retrain. Always pair `--train-years` with a truncation control.

### 3.4 ✅ `frac_l7` — resolved, and the reason it matters

`data.py` fed LightGBM every column except `COD_PREDIO`/`label_id`, so acquisition metadata
were model inputs. `frac_l7` was the **2nd-highest-gain feature** and ramps 0.000 across the
1996–98 baseline → 1.000 from 2002 — it would have manufactured the headline trend from
satellite availability alone.

Fixed by `train --drop-features meta|location|<cols>`, and **`infer()` now pins the column list
to the model's saved `feature_names`** so an ablated model cannot silently regain a feature the
panel store still contains. **Ablation cost +0.0013 macro-F1 — i.e. nothing.** Second
independent measurement that **gain ≠ contribution**.

---

## 4. All of Peru, single year — ✅ WORKS

**Verdict: the national single-year classifier is sound, and it produced the project's most
important methodological finding.**

14 departments, **946,872 linked polygons** (14.3× Piura), 726,808 3-class-eligible parcels.
Working sample **56,419 parcels** — deliberately Piura-sized, so cost is unchanged.

| | Piura | all-Peru |
|---|---|---|
| linked polygons | 66,352 | **946,872** |
| label years with >5 % of mass | 2 (1998, 1999) | **8 (1997–2006)** |
| neighbour label agreement at 0–100 m | ~0.86 | 0.72 (less spatially trivial) |
| CV macro-F1 | 0.648 | 0.628 ± 0.013 |
| CV fold variance | ±0.036 | **±0.013** (3× steadier) |
| calibration out of the box | ECE 0.090, T = 1.41 | **ECE 0.026, T = 1.09** |

Per class the national profile is **much more balanced**: `PASTURE_FALLOW` 0.485 → 0.606 (the
sierra supplies grazing land Piura lacked), `ANNUAL` 0.781 → 0.627 (Piura's rice monoculture
made it nearly trivial), `PERENNIAL` 0.685 → 0.655.

⚠️ **The sample's class mix is not the population's.** Square-root department allocation
upweights small departments, which are the coastal perennial ones, so `PERENNIAL` runs at
**20.2 % of the sample against 9.9 % of the population**. Good for training; **any area or
share figure must use `sample_weight`**, which reconstructs the population exactly (total
weight = 726,808).

### 4.1 The El Niño exclusion is Piura-specific

`--train-years 1999-2023` cost −0.0003 in Piura but **−0.0048 nationally**, because 1996–98
also holds sierra and southern parcels that are perfectly readable. The primary national model
keeps all years.

### 4.2 ⭐ `centroid_lat` is spatial memorisation, not agro-climatic signal

With 14 departments the pipeline can finally hold out a *place* rather than a neighbouring
5 km cell.

| | CV (unseen cell) | LODO mean | LODO pooled | LODO std |
|---|---|---|---|---|
| `lightgbm_nometa` (has `centroid_lat`) | **0.628 ± 0.013** | 0.417 | 0.539 | 0.098 |
| **`lightgbm_nometa_nolat`** | 0.581 ± 0.002 | **0.477** | 0.537 | **0.082** |
| Δ | **−0.047** | **+0.060** | −0.002 | −0.016 |

**Latitude's contribution reverses sign the moment the held-out unit is a place**, and **12 of
14 departments improve without it** (Piura most, +0.178). In Peru latitude is nearly a
climate-zone label, so it looks like real information — but it transfers like a lookup table.

Two companions:

* **The spatial-generalisation gap is ~0.09 macro-F1** (0.629 unseen-cell → 0.539
  unseen-department), invisible to every evaluation a single-department project could run.
* ⚠️ **"Piura is one of the hardest departments to predict" was an artefact of
  `centroid_lat`, not a fact about Piura.** It scores 0.301 (13th of 14) under the *rejected*
  `nometa` model, but **0.479 (6th of 14)** under the selected `nolat` model and **0.512 (2nd
  of 14)** under LTAE, which has no location feature at all. Piura is the most distinctive
  latitude band, so it is where a latitude lookup table fails hardest.

**Selection therefore reverses Piura's** — same rule, different evidence.

### 4.3 LTAE nationally — a negative result

Trained on identical sample/folds/store: CV **0.602 ± 0.010**, and far the worst calibrated
before scaling (ECE 0.120, T = 2.08). On the deciding criterion it loses too — **LODO mean
0.442** against `nolat`'s 0.477, and worst of three on pooled LODO (0.508). It wins only on
across-department *spread* (0.056 vs 0.082/0.098) and lifts the two hardest departments.

**LTAE carries no statics at all, so the memorisation mechanism cannot apply to it — yet it
still transfers worse.** *"Fewer statics" is not a monotone recipe for generalisation; the
specific feature was the problem, not staticness.*

### 4.4 ⛔ The national panel failed the same gate, on all three arms

25 years, 2,949 chunks, 37.6 M pixel-obs, **114,125 parcel-years per arm** (4,565 × 25).

| arm | statics | S4 | S5 `PERENNIAL` flicker (crit. 0.15) | gate |
|---|---|---|---|---|
| `lightgbm_nometa` | 3 | PASS (dev 0.029) | **0.517** | ⛔ FAIL |
| `lightgbm_nometa_nolat` | 2 | PASS (dev 0.013) | **0.744** | ⛔ FAIL |
| `ltae` | **0** | FAIL (dev 0.107) | **0.980** | ⛔ FAIL |

**The pre-registered prediction is confirmed exactly** (registered before any of this ran):
flicker is **monotone in how many time-invariant features the model carries**, and every rung
is worse than Piura's (0.428 / 0.547 / 0.780). Meanwhile **k = 0 accuracy barely moves**
(0.586 / 0.544 / 0.581): statics contribute almost nothing to being *right* and almost
everything to *not changing your mind*.

**Three things make this a cleaner failure than Piura's, which is what makes it diagnostic:**

1. **S4 passes on both LightGBM arms** — the panel starts in 1999 and label years spread over
   1997–2006, so no k-bin is dominated by one cohort.
2. **There are no thin years** — Piura's flagged coverage failures (2009 at 49.2 %, 2011 at
   43.3 %) pass at 95.4 / 96.2 % nationally, minimum 93.4 % over 25 years. **So 0.98 flicker
   cannot be a coverage artefact.**
3. The El Niño arm shows no class-specific collapse — but ⚠️ **do not read that as a
   clearance**: the national 1998 arm is only 5.5 % Piura, so it never tested the flood.

**What that isolates:** not the baseline, not coverage, not the sensor boundary, not the
architecture. **A single-year 3-class classifier at ~0.55–0.59 accuracy is simply not accurate
enough for a per-parcel annual trajectory** — per-year errors are near-independent, so a
25-year series is dominated by classification noise.

---

## 5. The window pivot — ⛔ CLOSED, three pre-gates failed

**Verdict: aggregating to 5-year windows fixes flicker and does not fix the estimand. No
extraction was funded.**

The proposal: aggregate predictions to 5-year windows, take the baseline from the *observed*
PETT label instead of predicting it, and identify the tenure contrast within-year.

* **M1 works.** Aggregating window-mean *probability* (not modal class) collapses annual
  3-class flicker 0.75 → **8.8 % of parcels with ≥2 window-state changes**; 82 % never change
  state. The Phase-7 instability is genuinely fixed by aggregation.
* **⛔ T3 fails on W2, every arm.** The PETT-`PERENNIAL` control pool — which must be flat for
  the at-risk series to mean anything — drifts **−0.044 / −0.059 / −0.103 per decade**
  (`nometa` / `nolat` / `ltae`) while the at-risk pool rises only **+0.018**. The yardstick
  moves 7× further than the signal, the other way. Not composition: a balanced panel gives the
  same series.
* **⛔ T2 (leave-one-year-out) fails.** Worst non-1998 cohort **0.406** against CV 0.581
  (tolerance 0.10); `PERENNIAL` recall spans 0.370–0.882. 1998 is mid-pack nationally (0.523).
  **A 2019–23 prediction is not licensed.**
* **⛔ T1 fails on the leg that matters.** Accuracy is non-differential in tenure (coef +0.013,
  p 0.107) but the **at-risk false-positive rate is 0.111 `INSCRITO` vs 0.155 `NO INSCRITO`**
  (−1.75 pp conditional on region + size, p 0.016) — the size of the target effect, pointing
  the *opposite* way to the cross-sectional association.

**A measured partial mechanism — probability compression.** Landsat observation density falls
~24 → ~13 clear obs/parcel-year (L5 retired, L7 SLC-off). Within parcel, window probability
tracks it: **+0.052 per log-observation on true perennials, −0.022 on true annuals**
(p < 1e-6). Both classes revert to the base rate as evidence thins, which is indistinguishable
from conversion at the level of a share. Explains 0–29 % of the drift, and is **monotone in how
few statics the arm carries**.

**Also measured here, and still useful:** `buffer_m=3000` costs nothing (CV 0.580 ± 0.024 vs
0.5805 ± 0.002); the stratified window sample draws 13,002 parcels at a measured design effect
of **2.36**, not the assumed 1.4; `PERENNIAL` is **57 % export / 20 % mixed / 23 % domestic**;
declared woody non-crop is **2.79 %** of the pool — bigger than the effect being sought.

⚠️ **Piura's supporting evidence came from `lightgbm_nometa`, the arm LODO disqualified.** On
the selected `nolat` arm Piura already drifts −0.017/decade.

---

## 6. Temporal out-of-distribution fixes — ⛔ CLOSED, four routes measured shut

**Verdict: the acceptance test failed. A new primary model was adopted at zero cost, but no
2019–23 estimate is licensed.**

### 6.1 ⭐ LOYO inherits spatial CV's blind spot

Leave-one-year-out holds region approximately fixed so that year varies — so a *time-invariant*
feature is as exploitable there as in CV:

| evaluation | `centroid_lat` advantage |
|---|---|
| CV (unseen 5 km cell) | **+0.0476** |
| **LOYO** (unseen year) | **+0.0474** — the same number |
| LODO (unseen department) | **−0.0596** |
| **LODYO** (unseen department **and** year) | **−0.005** |

The free fix is **`allperu lodyo`**, which re-scores the *existing* LODO predictions per
label-year cohort — no new fits. **An OOD evaluation is only blind-sided along the axis it
holds fixed. Report CV / LODO / LOYO / LODYO for every candidate.**

**LODYO now runs automatically at the end of every `allperu lodo`** — an evaluation nobody
remembers to run is an evaluation that does not exist, and this is the one that overturned the
selection.

### 6.2 New primary model, adopted for consistency not for effect

**`lightgbm_nometa_nolat_aug_yleak10`** (`runs/all_peru/selected_model.json`; previous record
preserved at `selected_model_20260809.json`). Degradation augmentation plus the top-10
year-identifying features withheld. Beats the incumbent on **all five** criteria — CV +0.0006,
LODO +0.0024, LOYO +0.0025, LODYO +0.0040, **W2 control slope −0.0592 → −0.0382 (35 %
flatter)**.

⚠️ **No labelled difference is individually significant** (paired p 0.14–0.54). Adopted for
consistency in sign at zero cost, not for a demonstrated effect. The two changes are **not
additive** — `yleak10` alone makes W2 *worse* (−0.0679).

⛔ **Step 1 acceptance FAILED.** W2 target |0.01| (best 0.038); LOYO worst-cohort gap target
0.10 (best 0.151). Observation density is a real mechanism worth ~a third of the artefact,
**not** the artefact.

### 6.3 Two routes closed by measurement

* **Recalibration cannot undo probability compression.** Fitted temperature *falls* with
  density (T = 1.448 − 0.160·log n; ECE 0.076 at n≈5 vs 0.008 at n≈19), so at endpoint density
  the model is **over**-confident and the correct fix pushes probabilities *further* toward the
  base rate. Applying it moved W2 by **0.0001**.
* **Per-year quantile alignment is worse than doing nothing** — W2 **−0.1023**, nearly 2× the
  baseline.

### 6.4 ⛔ The OLI route is closed — twice, and the second time tells you why

OLI alone would give **21.8 clear observations per parcel-year against L7's 13.1** — the
largest lever on the density mechanism anyone found. It fails anyway.

1. **Roy's published harmonisation is harmful here.** Raw OLI shifts the PETT-`PERENNIAL`
   control pool by −0.042 (split-half self-noise −0.0035); `perennial/harmonization.py` —
   implemented months earlier, documented as the safe path, **never once run** — makes that
   **−0.107** and lowers agreement too. Worse than no correction on every axis. **An unused
   correction is an untested correction.**
2. **Refitting the coefficients on our own data also fails.** 27,573 **same-day** L7/OLI parcel
   pairs exist in WRS sidelaps, but they **cannot identify a slope**: the SD of the two
   sensors' difference (0.064–0.102) *exceeds* the SD of either sensor's own values
   (0.065–0.088), and blue correlates at 0.12. OLS returns a physically impossible 0.116
   (de-diluted by binning, still only 0.26). Only a per-band **offset** is estimable — and that
   offset is the **best arm on every axis** (agreement 0.6452, at-risk pool bias −0.0003) and
   **still fails**: the control pool overshoots −0.0423 → **+0.0360**.

**The reason is structural.** Raw OLI reads **+0.070 NDVI greener on perennial parcels and
+0.045 on pasture**, and that **0.025 between-class spread survives every arm**. A global
band-level linear map moves all classes *together* and cannot remove a difference *between*
them; the only map that would work is conditioned on cover type, which is what the model is
trying to predict.

⛔ **Do not propose refitting the coefficients — it is done.** Reopening needs a *cleaner
paired sample* (pixel-level co-registration, or larger parcels — not parcel medians in scene
sidelaps). ⚠️ The registered 0.95 agreement criterion was also **never achievable**: the
model's split-half self-agreement ceiling is **0.655**.

### 6.5 Two structural feature results

* **Order statistics are density-fragile.** `_min`/`_max`/`_amp` move **0.347** within-SD per
  e-fold of observation count against **0.080** for the harmonic/slope fits — a 4.3× gap, land
  held fixed. ⚠️ **Do not act on that ranking alone**: dropping all 33 of them is the arm LOYO
  *rejects* (mean −0.0026, worst `PERENNIAL` recall 0.344).
* **The features identify the label year at 0.508 accuracy against a 0.155 baseline**, held out
  by region. `frac_l7` was one symptom of a systemic property.
* A *parcel*'s L7 SLC-off loss is **11 pp, not the nominal scene-level 22 %**.
* **LTAE is the worst arm on the temporal axis too** (LOYO 0.4784, LODYO 0.4965, W2 −0.1025),
  extending §4.3's negative result to a second dimension.

---

## 7. ⚖️ The two-period tenure DiD — COMPLETE, a bounded null

**Verdict: the project's first actual estimate. Titling moves predicted perennial probability
by less than ±1.3 pp. Do not reopen for a bigger sample — the population is exhausted.**

The design that survived: two *dated* tenure observations already on disk — BD SSET's
declaration-time `ESTADO en RRPP` (~1997–2006) and the bridge `.dta`'s cadastre `estado` +
`fech_tran` (≈2011–12). **1,780,580 parcels with two dated observations, 8.6 % moving
NO INSCRITO → REGISTERED.** The classifier supplies the *outcome* at both dates, drift and all
— the point is that the drift is now **common to both arms and differences out**.

### 7.1 The sample, and why it is a census

| step | parcels |
|---|---:|
| R1 at-risk (PETT `ANNUAL`, tenure resolved) | 363,529 |
| R2 both observations dated | 254,209 |
| R3 cadastre after declaration, before post-windows | 254,033 |
| **R4 pre-window entirely before the parcel's own declaration** | **80,868** |
| after the control definition | 32,399 |
| ⇒ **drawn: 6,559 treated (every one that exists) + 8,066 control** | **14,625** |

Extraction: 15 years (1999–2003, 2014–2023), **211,567 parcel-years, 63.6 M pixel-obs, ~13.5 h
on 5 workers**. 219,375 parcel-years inferred.

### 7.2 ⭐ The result

Registered before extraction in `did2_registration.json`. Placebo estimated **first**, enforced
by the code path.

| arm | placebo (2.5 y) | headline W99→W14+W19 | corrected M = 7 | decision |
|---|---:|---:|---:|---|
| **`nolat_aug_yleak10`** ⭐ | **−0.0029** (0.0037) | **−0.0011** (0.0059, p 0.85) | **+0.0195** (0.0268) | **NOT-SEPARABLE** |
| sens: `nolat` | −0.0009 (0.0042) | −0.0020 (0.0056) | +0.0042 (0.0301) | NOT-SEPARABLE |
| neg. control: `ltae` | −0.0026 (0.0101) | −0.0154 (0.0080, p 0.055) | +0.0031 (0.0709) | NOT-SEPARABLE |

**Sensitivity curve (required reporting):** M = 0 → −0.0011 [−0.0126, +0.0104]; M = 1 →
+0.0018; M = 3 → +0.0077; M = 5 → +0.0136; **M = 7 → +0.0195 [−0.0330, +0.0720]**.
**NOT-SEPARABLE at every M.**

**But the headline is a *tight null*, and that is the substantive finding.** Titling moves
predicted perennial probability by less than **±1.3 pp**, on the entire national population of
qualifying parcels. The correction does not rescue a large effect from a pre-trend; it widens a
null from ±1.3 pp to ±5.3 pp. **Export-discounted** (`PERENNIAL` is 57.3 % export): corrected
**+0.0112 [−0.0189, +0.0413]**. No sentence about export crops may use the undiscounted number.

**No anticipation.** A story where crop change *leads* the paperwork would show a **positive**
pre-trend; it is absent, and the baseline gap is only −0.0028.

### 7.3 The methodology, which is the reusable part

`corrected = headline − M·placebo`, `se = sqrt(se_h² + M²·se_p²)`, with **M = 7 derived from
window midpoints in code**, never hard-coded (17.5 y headline / 2.5 y placebo). The price was
registered in advance and is unflattering: only an effect larger than **~3.3 pp** can survive
the correction.

**⭐ The v2 gate was unpassable at any sample size — measurable only after running.** v2
demanded *proof* the pre-trend was negligible (equivalence, 25,202 parcels/arm against 6,559
that exist). The measured placebo point estimate is **−0.00295, already outside the registered
±0.0025 band**, so the SE letting its CI fit inside is **negative**: more precision returns
FAIL, never PASS. And it would have failed on −0.0118/decade — a pre-trend that cannot change a
null. See [`LESSONS.md`](LESSONS.md).

⚠️ **Never read a direction from the corrected point estimate.** The correction's sign flips
between outcomes (−0.0029 on probability, +0.0029 on the thresholded share), which is what a
noise term does.

### 7.4 What the earlier versions established (do not redo)

* **The pilot's headline reverses sign.** −0.0383 (p 0.007) becomes **+0.0338 (p 0.173)** once
  the pre-period is required to precede *each parcel's own declaration*. The pilot was measuring
  a pre-existing divergence. **Do not quote −0.038.**
* **The two tenure observations are snapshots of a rolling programme, not two shared dates** —
  declarations spread 1996–2009, `fech_tran` is a per-record transaction date (3–112 distinct
  dates per department), **24.6 % of parcels carry a status with no date**, and the registration
  rate is **non-monotone** in the gap between observations (campaign waves, not duration).
  Hence R4.
* **The population is exhausted.** Cohort ≥2004 gives 6,559 treated but *one* clean pre-window;
  cohort ≥2009 gives two pre-windows but **380 treated**. Either the pre-period is clean or it
  is powered, never both.
* **Sample-design fixes, not tuning:** training-set membership between arms −0.062 → **+0.0026**
  (draw from the full national table, not the model's own sample); shared regions 0.366 →
  **0.849** (draw controls region-first).

### 7.5 ⚠️ The caveat that binds every number

**Tenure is last observed ~2011–12; the outcome runs to 2023, and Peru's titling programme kept
running.** An unknown share of the control arm was certainly titled and unobserved. That
attenuates any real effect **toward zero**. So a positive finding here would be an
underestimate, and **this null is partly attributable to that contamination and is NOT evidence
that titling has no effect.** Bounding it needs a third dated observation, which does not exist.

⚠️ **Common support still fails** (0.409 at department level) and is structural — 75 % of
treated parcels are La Libertad + Cajamarca, so the estimate applies to where titling actually
happened.

⚠️ **Magnitude is architecture-dependent even though sign is not.** `ltae` gives −0.0154
against LightGBM's −0.0011. Any future design needing a *level* rather than a null must treat
architecture as a first-order uncertainty.

### 7.6 The cross-sectional companion — descriptive, NOT an estimate

Computed on request (`allperu tenure-xsec`). Among at-risk parcels the
`INSCRITO` − `NO INSCRITO` gap runs **−0.0429 at W99 → −0.0216 at W19**. Three findings, none
of which is a treatment effect:

1. **The W99 gap IS the classifier's false-positive-rate gap.** True perennial share at the
   label year is ≈0 by construction, so the baseline gap is pure error — and it comes out at
   −0.0429 against T1's independently measured −0.044 (§5), from a completely different
   direction. **The baseline "association" is a property of the instrument, not of the land.**
2. **The sign is department-specific**, so the pooled number describes a quantity that does not
   exist. `INSCRITO` reads higher in 5 of 14 departments at W99 and 4 of 14 at W19; spread runs
   Lima +0.110 to Ayacucho −0.186.
3. ⚠️ **The Piura figure does not replicate and the direction reverses.** The old
   perennial-strand "24.9 % vs 10.9 %, `INSCRITO` higher" comes out as **−0.081, `INSCRITO`
   lower** here. **Do not carry that figure forward as a national fact.**

The artifact carries an `IS_NOT_CAUSAL` field so it cannot be lifted out of context.

---

## 8. S2 endpoint labelling — 🔴 LIVE

**Verdict: built up to the point where humans take over. This is the only outstanding work in
the project.**

Everything measured in this document is measured in ~1997–2006. **Not one number is measured in
2019–2023, which is the only period the research question is about.** Hand-labelling recent
high-resolution imagery is the one remaining route that *verifies* the classifier rather than
working around it.

Delivered: eligible universe **614,876 parcels**, **4,519 Esri centroids probed** (G0 passes at
87.6 % ≤1.2 m and ≥2019), **2,224 chips**, **117,768 S2 parcel-dates**, **9 HTML shards
covering 1,112 parcels**, split frozen before labelling.

⭐ **Sentinel-2 gives a median 19–113 clear dates per parcel-agricultural-year by department,
against the Landsat store's 13–24.** Observation density is the one mechanism this project has
*measured* driving the panel failures (§5), and S2 roughly triples it. That does not mean the
artefact is gone — measure it, do not assume.

**Plan, gates and remaining steps: [`s2_labelling/plan.md`](s2_labelling/plan.md).**

---

## 9. Things that were built and never run, deliberately

All unit-tested; all blocked by a gate that failed. They stay unrun.

| code | estimand | blocked by |
|---|---|---|
| `perennial/trajectories.py` | per-parcel transition series | §3, §4.4 |
| `perennial/area.py`, `allperu/estimate.py` | Olofsson-corrected area / share | §3, §4.4 |
| `allperu/external.py` | SIEA comparison | same |
| `perennial/harmonization.py` | Roy OLI→ETM+ correction | §6.4 — harmful when finally run |
| `allperu/oli_refit.py`, `oli_overlap.py` | locally refitted OLI correction | §6.4 — closed |

⚠️ **The national locked test is UNSPENT** and should stay that way until an estimand passes
its gate.
