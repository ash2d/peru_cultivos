# Results

Every number here was measured. One section per strand, in the order they were run. Each opens
with a verdict line.

**CLOSED means do not reopen** — the reason is stated, and it is a fact about the data or the
population, not a preference. **LIVE** means work is outstanding; the task list is in
[`STATUS.md`](STATUS.md).

For the readable narrative of the same material, see [`REPORT.md`](../reports/REPORT.md). For findings
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
| 8 | S2 endpoint labelling | validated 2020+ labels | 🔴 **LIVE** — 1,012/1,112 labelled, G2/G3-dept pass | `labelling/`, `features/s2_*` |

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
**separation from the annual class on the same feature**, not an absolute level — and the flood
closed that separation from both sides. The `NDVI_p25` class gap falls **+1.47 SD → +0.46 SD**
and `NDVI_amp` is annihilated (−0.60 → −0.10 SD), because perennials lost canopy (floor
0.518 → 0.448) while flooded annual ground stayed green (0.363 → 0.404). A threshold fitted on
normal years then sits inside both distributions, so recall goes to zero. **`BSI_max` is the one
feature that holds** (−0.76 → −0.90 SD; the raw gap widens too, so it is not a normalisation
artefact). Numbers in `figures/elnino_signature_collapse.csv`.

⚠️ **"Separation" here is a property of the dataset, not a computation the model performs.**
The quantity is `(median PERENNIAL − median ANNUAL) / pooled SD`, measured across parcels after
the fact (`perennial/report_figures.py::elnino_signature_collapse`, restricted to regions present
in both cohorts so place is held roughly fixed). **No model in this project uses any neighbour
feature** — every prediction comes from one parcel's own time series. Earlier revisions of this
paragraph said "contrast with annual neighbours", which wrongly implied a spatial computation.

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

⚠️ **This verdict is about LTAE *on the Landsat store*, and it carries over only in part.**
Re-asked on the Sentinel-2 store — median 47 clear dates per parcel-year against 13–24 here —
LTAE wins **CV in 8 of 8 arms** and **loses LODO in 4 of 4 targets**, dropping 1.4–3.4× more
than LightGBM when the department changes. So the *deciding* axis agrees with this section and
the CV axis does not: LTAE's extra S2 skill is skill that does not leave the training
departments. §8.2.

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

**Verdict: 1,012 of 1,112 labellings are in, all three models are trained and compared, and the
comparison produced a reversal. Only the 100-parcel overlap shard (gate G1) is outstanding.**

⭐ **The headline: LTAE wins cross-validation in all 8 arms and loses leave-one-department-out
in all 4 targets.** Its CV advantage is skill that does not leave the training departments —
the fourth instance of this project's central finding, and the first to catch an *architecture*
rather than a feature. §8.2.

⭐ **The campaign's own deliverable is now readable, and it does not show the shift the project
was built around**: of parcels declared `ANNUAL` in 1996–2006, **2.9 % [0, 5.9] read as
perennial today** — while **57.9 %** read as farmable ground not currently cropped. §8.2b.

⭐ **Collapsing to two classes (perennial vs not) makes macro-F1 rise and the model worse.**
The floor rises with it — always guessing the largest class scores 0.467 at two classes against
0.171 at four — and normalised against that floor, `t2` is the **lowest-skill arm in the study**.
`PERENNIAL` F1 gains +0.004 over the 3-class model. §8.2c.

⭐ **Using the 2012 census as the "before" instead of the PETT declaration** gives the project's
first paired two-declaration comparison: on 8,669 Piura parcels, perennial declarations go
**28.2 % → 40.7 % of parcels (+12.5 pp) and 51.4 % → 66.0 % of cadastral area (+14.6 pp)**
between ~1999 and 2012, with conversion concentrated on the larger parcels. §8.5.

⚠️ **The locked test (201 parcels, 161 labelled) is UNSPENT** and nothing in §8 reads it.

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

### 8.1 The returned labels — 1,012 of 1,112, one annotator

**Returned 2026-08-27/28:** `pilot` (120) and `shard01`–`shard05` (892). Only the 100-parcel
**overlap shard** is outstanding. **865 of the 1,012 are usable**; the 147 excluded are 139
`UNSURE` (the annotator's abstain), 6 with no usable Sentinel-2 observation, and 2 flagged as
a boundary mismatch between the polygon and the imagery.

| gate | what it asks | criterion | reading | |
|---|---|---|---|---|
| **G1** `kappa_called` | do two passes over the same parcels agree, counting only parcels *both* passes actually called? | ≥ 0.75 | **not measurable** | the overlap shard is not in the return. Not a failure — an *absence* |
| **G2** `UNSURE` share | what fraction of parcels could the annotator not call at all? | < 0.25 | **0.139** ✅ **PASS** | at 1.2 m imagery, 86 % of parcels are callable. This was the live unknown |
| **G3** labels per department | is any department too thin to hold out? | ≥ 35 usable | **48** (Moquegua) ✅ **PASS** | ⭐ first time. **All 14 departments** now clear it, against 4 at the last read |
| **G3** labels per class | is any class too thin to learn? | ≥ 150 usable | **54** (`NON_AGRICULTURE`) ⛔ **FAIL** | pre-registered as *pool into `OTHER`, do not re-draw* — that is the `t4` target below. `PERENNIAL` (115) and `ANNUAL` (141) also sit under the floor |

**Label counts, all 1,012 returned:** `OTHER` 395, `WOODY_NON_CROP` 166, `UNSURE` 141,
`ANNUAL` 141, `PERENNIAL` 115, `NON_AGRICULTURE` 54.

⚠️ **G1 is unmeasured and every number below inherits that.** The campaign still has no
measurement of its own label noise. Related and on the record: the annotator's **median time
per parcel is 2–4 seconds** (76 % of `shard01` under 5 s), against the plan's ~2 min budget.
That does not make the labels wrong; it means the noise floor is unquantified, and the intra-rater
re-label of 100 parcels is what would quantify it.

### 8.2 ⭐ The result: LTAE wins cross-validation and loses out-of-department

**This reverses the previous reading of §8.2, which was taken on 4 departments and 354 labels.**

**The four label targets.** The campaign records five classes; the Landsat model has three.
These are four defensible readings of the same labels, kept as separate arms rather than
resolved, because the gap between them *is* the size of the decision:

| target | what it is | classes | n usable |
|---|---|---|---|
| `t5` | the campaign's own scheme, untouched | `PERENNIAL` / `ANNUAL` / `OTHER` / `WOODY_NON_CROP` / `NON_AGRICULTURE` | 865 |
| **`t4`** ⭐ | `NON_AGRICULTURE` → `OTHER`. The pre-registered pooling G3 calls for | 4 | 865 |
| `t3` | drops `WOODY_NON_CROP` and `NON_AGRICULTURE` — the clean 3-class head-to-head, at the cost of a quarter of the data | 3 | 646 |
| `t3w` | `WOODY_NON_CROP` → `PERENNIAL`. Keeps every parcel; the *generous* reading, since a model with no woody class does this anyway | 3 | 865 |

**What the classes mean** (codebook, frozen): `PERENNIAL` = a standing tree/bush crop in rows;
`ANNUAL` = a field crop replanted each season; `OTHER` = farmable ground not currently cropped
(fallow, bare, pasture); `WOODY_NON_CROP` = trees that are not a crop (riparian scrub,
plantation edge, abandoned orchard gone wild); `NON_AGRICULTURE` = ground that could never be
sown as it stands (road, building, quarry).

**The three sets.** The split was frozen in `config/split_s2labels.yaml` **before any labelling**
and is not re-drawn.

| set | what it is | n usable |
|---|---|---|
| **train** | 4 of the 5 spatially-blocked folds, plus the 120-parcel pilot pool where `+pilot` is used | ~510 per fold (`t4`) |
| **validation** | the held-out 5th fold, rotated 5× so every trainval parcel is predicted exactly once out-of-fold. Folds are whole 5 km regions and a **3 km dead-zone** is stripped from each train side, so no training parcel sits within 3 km of a validation parcel | ~170 per fold (`t4`) |
| **test (locked)** | 201 parcels in 150 distinct regions, every department on both sides. 161 now labelled | **UNSPENT — read by nothing in this section** |

**The metric.** *Macro-F1*: the F1 score computed per class and averaged with equal weight, so
a rare class counts as much as a common one. On `t4`, always guessing the largest class
(`OTHER`, 51.8 % of trainval parcels) scores **0.171**; on `t3w` it scores **0.228**, on `t3`
**0.253** and on `t5` **0.124**. Those are the floors each column below has to beat.

**The two evaluations, and why both.** *CV* holds out 5 km regions **inside departments the
model has already seen**. *LODO* (leave-one-department-out) holds out a whole department, fits
on the other 13, and scores the one held out — trainval + pilot only, locked test excluded from
both sides. LOYO/LODYO do not apply here: every parcel is interpreted at a single endpoint
epoch (79 % of chips are 2024–25), so there is no year axis to hold out.

**Cross-validation, macro-F1 (mean ± SD over the 5 folds):**

| target | train pool | LightGBM | `rules` | **LTAE** |
|---|---|---|---|---|
| `t5` | trainval | 0.637 ± 0.064 | 0.314 ± 0.030 | **0.672 ± 0.039** |
| `t5` | + pilot | 0.660 ± 0.059 | 0.308 ± 0.030 | **0.672 ± 0.057** |
| `t4` | trainval | 0.651 ± 0.059 | 0.396 ± 0.040 | **0.709 ± 0.061** |
| `t4` | + pilot | 0.672 ± 0.064 | 0.392 ± 0.018 | **0.708 ± 0.060** |
| `t3` | trainval | 0.694 ± 0.044 | 0.550 ± 0.069 | **0.762 ± 0.072** |
| `t3` | + pilot | 0.721 ± 0.053 | 0.561 ± 0.053 | **0.760 ± 0.044** |
| `t3w` | trainval | 0.739 ± 0.059 | 0.675 ± 0.065 | **0.803 ± 0.060** |
| `t3w` | + pilot | 0.747 ± 0.040 | 0.675 ± 0.040 | **0.804 ± 0.040** |

**LTAE wins CV in all 8 arms**, by +0.013 to +0.068. Paired per-fold (both models see identical
folds), LTAE − LightGBM clears p < 0.05 in **5 of 8**: `t4` +0.058 (p = 0.027), `t4`+pilot
+0.037 (p = 0.034), `t3` +0.068 (p = 0.046), `t3w` +0.063 (p = 0.042), `t3w`+pilot +0.057
(p = 0.013). The `t5` arms run +0.013 to +0.035 at p = 0.13–0.25.

⭐ **Leave-one-department-out, macro-F1 (mean ± SD over departments), and it goes the other way:**

| target | departments held out | LightGBM | `rules` | LTAE |
|---|---|---|---|---|
| `t5` | 14 | **0.543 ± 0.129** | 0.275 ± 0.101 | 0.496 ± 0.102 |
| `t4` | 14 | **0.539 ± 0.123** | 0.360 ± 0.122 | 0.515 ± 0.077 |
| `t3` | 10 | **0.623 ± 0.103** | 0.523 ± 0.150 | 0.620 ± 0.067 |
| `t3w` | 14 | **0.697 ± 0.121** | 0.595 ± 0.137 | 0.633 ± 0.115 |

(`t3` has 10 departments, not 14, because dropping `WOODY_NON_CROP` and `NON_AGRICULTURE`
takes four departments under the 35-label floor.)

**LightGBM wins LODO in all four targets.** Paired per **department** — both models scored on
the same held-out departments — LTAE − LightGBM is **−0.047** (`t5`, p = 0.060), **−0.023**
(`t4`, p = 0.297), **−0.003** (`t3`, p = 0.931) and **−0.065** (`t3w`, p = 0.078). Consistent
in sign across four independent targets, none individually significant at p < 0.05 — the same
evidentiary shape as the CV result, with the sign flipped.

⭐ **The mechanism is visible in the size of the drop.** Same model, same labels, seen versus
unseen department:

| target | model | CV | LODO | drop |
|---|---|---|---|---|
| `t5` | LightGBM | 0.660 | 0.543 | −0.117 |
| `t5` | **LTAE** | 0.672 | 0.496 | **−0.176** |
| `t4` | LightGBM | 0.672 | 0.539 | −0.133 |
| `t4` | **LTAE** | 0.708 | 0.515 | **−0.193** |
| `t3` | LightGBM | 0.721 | 0.623 | −0.098 |
| `t3` | **LTAE** | 0.760 | 0.620 | **−0.140** |
| `t3w` | LightGBM | 0.747 | 0.697 | −0.050 |
| `t3w` | **LTAE** | 0.804 | 0.633 | **−0.171** |

(`rules` drops least of all — 0.03–0.08 — because it has the least to lose: three NDVI
thresholds cannot memorise a department.)

**LTAE's extra CV skill is skill that does not leave the training departments.** It loses **1.4–3.4× more than LightGBM** when the department
changes, in every one of the four targets. This is the fourth independent
instance of this project's central finding, and the first in which it caught an *architecture*
rather than a feature: `frac_l7` manufactured change, statics manufactured stability,
`centroid_lat` manufactured accuracy that stayed inside the training departments — and here an
attention encoder over 47 dates manufactures the same thing. **Never select on CV alone.**

⚠️ **This does not re-close "try another architecture", and it does not restore §4.3.** LTAE
is not worse than LightGBM here; it is *equal to slightly worse out of department and clearly
better within one*. What changed against the earlier reading is the evidence, not the store:
**4 departments became 14** and 354 labels became 865. A 4-department LODO mean is 4 numbers
with an SD of 0.15.

**Per-class, pooled out-of-fold on `t4` (every trainval parcel predicted exactly once):**

| class | LightGBM P / R / F1 | LTAE P / R / F1 | n parcels |
|---|---|---|---|
| `ANNUAL` | 0.683 / 0.724 / 0.703 | **0.743 / 0.827 / 0.783** | 98 |
| `OTHER` | 0.793 / 0.788 / 0.790 | **0.856 / 0.766 / 0.808** | 325 |
| `PERENNIAL` | 0.483 / 0.406 / 0.441 | **0.526 / 0.594 / 0.558** | 69 |
| `WOODY_NON_CROP` | 0.683 / 0.724 / 0.703 | **0.677 / 0.759 / 0.715** | 116 |

⚠️ **`PERENNIAL` — the class the research question is about — is the worst class in both
models**, at F1 0.44 (LightGBM) and 0.56 (LTAE) on 69 parcels. Under `t3w`, where
`WOODY_NON_CROP` counts as `PERENNIAL`, it rises to 0.744 / 0.791 on 185 parcels. **That gap
is not a modelling gain; it is the codebook decision.**

**Where the errors are — `t4`, LTAE, counts of parcels:**

| human label ↓ / model → | ANNUAL | OTHER | PERENNIAL | WOODY_NON_CROP |
|---|---|---|---|---|
| **ANNUAL** | **81** | 14 | 2 | 1 |
| **OTHER** | 23 | **249** | 24 | 29 |
| **PERENNIAL** | 3 | 13 | **41** | 12 |
| **WOODY_NON_CROP** | 2 | 15 | 11 | **88** |

The dominant residual confusion is `OTHER` ↔ `WOODY_NON_CROP` (29 + 15 = 44 parcels) and
`PERENNIAL` ↔ `WOODY_NON_CROP` (12 + 11 = 23) — i.e. exactly the codebook boundary §8.1 flags
as the hardest call. `ANNUAL` is nearly clean (81 of 98 correct, and only 5 parcels anywhere
in the matrix confuse `ANNUAL` with `PERENNIAL` in either direction).

⭐ **The learning curve has flattened with respect to the pilot.** At 354 labels, folding in the
120-parcel pilot (+48 % data) moved LightGBM **+0.050** on `t4`. At 865 labels the same pilot
(+14 % data) moves LightGBM **+0.021** and LTAE **−0.001**. The remaining 100 labellings (the
overlap shard) should be spent on **G1**, not on more training data.

**`rules` is the honest floor, not a straw man.** A depth-2 NDVI phenology rule reaches 0.675
on `t3w` CV — within 0.07 of LightGBM — and beats it out of department in 5 of 14 departments
(Lambayeque, Tumbes, Moquegua, Piura, Ayacucho). It collapses on `t4`/`t5` (0.39/0.31) for a structural
reason: it has no way to express `WOODY_NON_CROP` or `NON_AGRICULTURE` at all.

**Full tables:** `labels_s2/model_comparison.csv` (every arm, CV + LODO),
`labels_s2/ws_<target>_pilot/lodo_<model>.csv` (per department).

### 8.2b ⭐ The transition matrix — the campaign's own deliverable, no classifier in it

The weighted **declared (1996–2006) → observed (2019+)** matrix, read straight off the 865
photo-interpreted labels. Rows are what the PETT programme recorded the farmer growing; columns
are what a human sees on ≤1.2 m imagery today. Cells are **weighted row shares** (the national
sample doubles the perennial share by design, so the weights are not optional), ±95 % CI from
the effective sample size.

| declared 1996–2006 ↓ | ANNUAL | NON_AGRICULTURE | OTHER | PERENNIAL | WOODY_NON_CROP |
|---|---|---|---|---|---|
| **ANNUAL** | 0.311 ±0.084 | 0.037 ±0.034 | **0.579 ±0.089** | 0.029 ±0.030 | 0.044 ±0.037 |
| **PASTURE_FALLOW** | 0.116 ±0.059 | 0.065 ±0.045 | **0.737 ±0.081** | 0.036 ±0.034 | 0.047 ±0.039 |
| **PERENNIAL** | 0.083 ±0.036 | 0.053 ±0.029 | 0.324 ±0.061 | 0.211 ±0.053 | **0.329 ±0.061** |

⭐ **This is the descriptive conversion estimate the project has never been able to produce, and
it does not show the shift the research question was built around.** Of parcels declared
`ANNUAL` in 1996–2006, **2.9 % [0, 5.9] read as perennial today** — indistinguishable from
zero. What they overwhelmingly read as is `OTHER`: **57.9 %** are farmable ground not currently
cropped. Declared `PASTURE_FALLOW` behaves the same way (73.7 % `OTHER`).

⚠️ **And the declared-`PERENNIAL` row is a warning about every accuracy figure above.** Only
21.1 % of parcels declared perennial still read `PERENNIAL`; **32.9 % read `WOODY_NON_CROP`**
and 32.4 % read `OTHER`. Whether that third is "orchard gone wild" (a perennial that stopped
being farmed) or "riparian scrub that was never the crop" is the single hardest call in the
codebook, it carries a third of the perennial sample, and **G1 — the measurement that would
tell us how reliably it is being made — does not exist.**

⚠️⚠️ **And it is a national average that does not describe any particular department.**
Restricted to **Piura**, the same table reads completely differently: **86.7 %** of
declared-`ANNUAL` parcels still read `ANNUAL` (against 31.1 % nationally) and only **6.7 %**
read `OTHER` (against 57.9 %). The national "farmable but not currently cropped" result is
carried by other departments, not by Piura's irrigated coastal valleys. n = 65 for Piura, so
that contrast is indicative, not precise — but it is large enough that the national row must
not be read as a statement about anywhere in particular. §8.5.

⚠️ **This is a transition in *observed land state*, not evidence about titling.** It is
unweighted by area, it rests on one annotator, and `OTHER` at 2019+ includes ordinary
between-season fallow, which a single endpoint observation cannot separate from abandonment.
Source: `labels_s2/declared_to_observed_transitions.csv`.

### 8.2c ⭐ Two classes — the macro-F1 goes up and the model gets worse

Asked directly: collapse the label space to the only distinction the research question turns
on, **perennial vs not**, and see what the F1 does. Two readings, differing only in which side
`WOODY_NON_CROP` lands on, exactly as `t3`/`t3w` do:

| target | positive class | negative class | perennial prevalence in trainval |
|---|---|---|---|
| `t2` | `PERENNIAL` | `ANNUAL` + `OTHER` + `NON_AGRICULTURE` + `WOODY_NON_CROP` | **12.2 %** |
| `t2w` | `PERENNIAL` + `WOODY_NON_CROP` | `ANNUAL` + `OTHER` + `NON_AGRICULTURE` | **31.7 %** |

⚠️ `rules` is **not run** on either. Its rule maps three semantic groups onto label ids and in
a two-class space its fallback resolves `PASTURE_FALLOW` to id 1 — which is `PERENNIAL`. It
would have run and returned a number. The CLI now refuses it (`tests/test_s2_train.py`).

**Raw macro-F1 says the two-class problem is much easier:**

| target | LightGBM CV | LTAE CV | LightGBM LODO | LTAE LODO |
|---|---|---|---|---|
| `t4` (4 classes) | 0.672 | **0.708** | **0.539** | 0.515 |
| `t2` (2 classes) | 0.715 | **0.769** | **0.648** | 0.563 |
| `t3w` (3 classes) | 0.747 | **0.804** | **0.697** | 0.633 |
| `t2w` (2 classes) | 0.821 | **0.858** | **0.747** | 0.684 |

⛔ **That reading is wrong, and the reason is the baseline.** Macro-F1 averages the per-class
F1 with equal weight. With two classes, a model that never predicts the minority still collects
a full score on the majority and divides by 2, so **the floor triples as classes are removed**:

| target | classes | largest class | **macro-F1 of always guessing it** |
|---|---|---|---|
| `t5` | 5 | 45.2 % | **0.124** |
| `t4` | 4 | 51.8 % | **0.171** |
| `t3w` | 3 | 51.8 % | **0.228** |
| `t3` | 3 | 61.2 % | **0.253** |
| `t2w` | 2 | 68.3 % | **0.406** |
| `t2` | 2 | **87.8 %** | **0.467** |

⭐ **Normalised against its own floor — `skill = (macro-F1 − floor) / (1 − floor)`, so 0 is
"no better than guessing the largest class" and 1 is perfect — the two-class collapse is the
*worst* thing in the study:**

| target | classes | model | CV macro-F1 | **CV skill** | LODO macro-F1 | **LODO skill** | **`PERENNIAL` F1** |
|---|---|---|---|---|---|---|---|
| `t5` | 5 | LightGBM | 0.660 | 0.612 | 0.543 | 0.478 | 0.523 |
| `t5` | 5 | LTAE | 0.672 | 0.626 | 0.496 | 0.425 | 0.562 |
| `t4` | 4 | LightGBM | 0.672 | 0.604 | 0.539 | 0.444 | 0.471 |
| `t4` | 4 | LTAE | 0.708 | 0.648 | 0.515 | 0.415 | 0.592 |
| `t3` | 3 | LightGBM | 0.721 | 0.627 | 0.623 | 0.495 | 0.593 |
| `t3` | 3 | LTAE | 0.760 | 0.679 | 0.620 | 0.491 | 0.664 |
| `t3w` | 3 | LightGBM | 0.747 | 0.672 | **0.697** | **0.608** | 0.768 |
| `t3w` | 3 | LTAE | 0.804 | 0.746 | 0.633 | 0.525 | 0.806 |
| **`t2`** | 2 | LightGBM | 0.715 | 0.465 | 0.648 | **0.340** | 0.509 |
| **`t2`** | 2 | LTAE | 0.769 | **0.567** | 0.563 | ⛔ **0.180** | 0.620 |
| `t2w` | 2 | LightGBM | 0.821 | 0.699 | 0.747 | 0.574 | 0.759 |
| `t2w` | 2 | LTAE | 0.858 | **0.761** | 0.684 | 0.468 | 0.810 |

**`t2` is the lowest-skill arm in the entire study on both axes** — CV skill 0.465–0.567 against
`t4`'s 0.604–0.648, and **LODO skill 0.180** for LTAE, barely above "guess the largest class"
on an unseen department. Its raw 0.769 CV macro-F1 is the highest number the 5-, 4- and 3-class
targets could not reach, and it is almost entirely the 87.8 % majority.

**And the decisive check — `PERENNIAL` F1, the one number that *is* comparable across targets**
because it is the same class scored the same way regardless of how many others exist (LTAE,
pooled out-of-fold):

| | `t4` (4 cls) | `t2` (2 cls) | | `t3w` (3 cls) | `t2w` (2 cls) |
|---|---|---|---|---|---|
| `PERENNIAL` F1 | 0.592 | 0.620 | | 0.806 | **0.810** |

**Collapsing four classes into two buys +0.028 on the class of interest; collapsing three into
two buys +0.004 — nothing.** The classifier was never losing perennial parcels *to* the
annual/fallow/non-agriculture distinctions, so deleting those distinctions recovers almost
nothing and throws away four interpretable outputs.

⭐ **What the two-class run does settle is where the difficulty actually lives.** `t2` and `t2w`
differ *only* in which side `WOODY_NON_CROP` sits on, and that single choice moves `PERENNIAL`
F1 from **0.620 to 0.810** — **6× the gain from collapsing the label space at all**. The
binding constraint on this classifier is not the number of classes. It is that
`WOODY_NON_CROP` and `PERENNIAL` look alike from orbit, which is exactly the codebook boundary
§8.1 flags and exactly the boundary **G1 has never measured**.

**LightGBM now beats LTAE on LODO in 6 of 6 targets** (`t2` +0.085, `t2w` +0.063), which is the
strongest form §8.2's finding has taken.

**Practical reading:** ⛔ **do not adopt `t2`.** If a binary output is wanted, take it from the
`t3w` or `t4` model by summing probabilities — the multi-class model is at least as good at
`PERENNIAL` and still says what the other parcels are. Full table:
`labels_s2/skill_by_target.csv`.

### 8.3 The transferred Landsat baseline — ⚠️ SUPERSEDED SAMPLE, and it cannot adjudicate G4

⚠️ **Every number in this section was measured on the 354-label return, not the 865.** It is
deliberately not recomputed, because it cannot decide anything either way (below) — but that
makes it the one section here whose figures are **not comparable** with §8.1–8.2c's. Do not
read a row of it beside a row of theirs.

G4 asks the S2 model to beat "the existing Landsat model scored on the same held-out labelled
parcels". **Those predictions do not in fact exist for these parcels.** The national panel
covers **22** of the 1,112 drawn (18 after exclusions) — the campaign drew from the 614,876-parcel
eligible universe, not from the 4,565-parcel panel sample.

What can be done is to load the saved primary booster and score it on the labelled parcels'
**S2** features; its 124 pinned features are all present in the S2 store bar `n_pixels_est`,
which the sample carries.

| | n | macro-F1 | accuracy |
|---|---|---|---|
| Landsat primary → S2 features, `t3` | 245 | 0.352 | 0.400 |
| Landsat primary → S2 features, `t3w` | 354 | 0.445 | 0.441 |

⚠️ **These two rows were measured on the 354-label return and have not been recomputed on the
865.** They are not re-run because they cannot decide anything either way (see below) — but do
not compare them line-for-line against §8.2's numbers, which are measured on 2.4× the data.

⚠️ **Read this as a lower bound on the Landsat model, not a measurement of it.** The booster was
fitted on Landsat 5/7 reflectance and is being fed Sentinel-2 reflectance under the same column
names, and §6.4 established *on this project's own data* that the sensor difference is
cover-type dependent and that no global linear map removes it. The distortion is visible in the
predictions: `ANNUAL` recall **0.932** at precision **0.25–0.31** — the model calls almost
everything `ANNUAL`, which is a decision-boundary shift, not a land-use read.

The 18-parcel check confirms the direction rather than the size: the transfer agrees with the
genuine Landsat panel on **44 %** of them, and every disagreement moves toward `ANNUAL`.

**So G4 is not adjudicated, and cannot be until either (a) the Landsat model is scored on genuine
Landsat features for these parcels, or (b) the comparison is dropped in favour of the S2 model's
own held-out number.** Recording that here, before the labelling finishes, rather than after.

One thing the 18 parcels *do* settle: the genuine Landsat panel calls **5 of 5** `WOODY_NON_CROP`
parcels `PERENNIAL`. That was the assumption behind the `t3w` mapping, and it is now checked
rather than asserted.

### 8.4 What this says about finishing the campaign

* The pipeline runs end to end on the real returned CSVs — ingest, eight workspaces, three
  models × four targets × two training pools, LODO over every eligible department.
* **G2 passes at 0.139** and **G3-per-department now passes at 48**, for the first time: all 14
  departments clear the 35-label floor, so LODO is a 14-department estimate instead of a
  4-department one. That change alone is what reversed §8.2.
* ⭐ **The learning curve has flattened with respect to the pilot.** At 354 labels the pilot
  (+48 % data) moved LightGBM +0.050 on `t4`; at 865 labels the same pilot (+14 %) moves it
  +0.021 and moves LTAE −0.001. **The remaining 100 labellings should buy G1, not more training
  data.**
* **The model to carry forward is LightGBM, not LTAE** — on the evaluation that decides
  (LODO), in all four targets. That is a reversal of the previous entry here and it rests on
  10× more held-out departments.
* **G1 needs the overlap shard.** With one annotator it becomes intra-rater: re-label 100
  parcels after a gap. Until then the campaign has no measure of its own label noise, and every
  number above inherits that.

### 8.5 ⭐ CENAGRO 2012 as the "before" — the first two-declaration comparison

Every before/after above uses the **PETT declaration** as its baseline: one observation per
parcel, made when the parcel was titled. The **2012 agricultural census** is an independently
collected second declaration of what is growing on some of the same parcels, and it turns a
one-observation design into a genuine paired one, with **no satellite and no classifier in it
anywhere**.

`allperu cenagro`, module `allperu/cenagro.py`, tables in `data/processed/cenagro/`.

⚠️ **Three limits, all binding, stated before any number.**
1. **This section is Piura only**, because `IV_CENAGRO_Piura.dta` was the only census file
   available when it was run. Nothing *in §8.5* is national — and Piura is the department every
   earlier strand was built on, so it is not a neutral sample. ⭐ **The census itself is no
   longer Piura-only**: all 25 departments were extracted on 2026-08-29 (`DATA.md` §1.5) and
   **§8.6 repeats this comparison nationally**. §8.5 is kept as the first, independently built
   version of it — Piura's +12.5 pp here against §8.6's +11.6 pp there is the one external
   check either number has.
2. **The census carries no parcel key.** No `COD_PREDIO`, no `CodigoSSET`. The only link is the
   farmer's name, so the link is **farmer-level, not parcel-level**: which of a producer's
   polygons a census row refers to is uncertain (`DATA.md` Chain B). Every figure is therefore
   also reported by `link_confidence`.
3. ⚠️ **The matched subset over-represents perennials by 1.6×.** It is **23.3 %** `PERENNIAL`
   against Piura's true **14.4 %**. Name-matched producers are not a random sample of the
   department, so the *levels* below describe the matched subset, not Piura. The *change* is
   paired within parcel and is the part worth reading.

**A lexicon correction that moved the headline by 10 pp, recorded because it nearly did not
happen.** Both sides must be classified by the same rules or part of any "change" is a change
of definition. The census uses fuller crop names than the registry ("LIMON ACIDO", not
"LIMON"), and an audit of 59,855 census token-instances found **2,448 (4.09 %) falling through
to `crop_fallback: ANNUAL`** — of which **1,969 were the single token `VERGEL FRUTICOLA`,
literally "fruit orchard".** The largest unmapped token in the census was being counted as an
annual crop, biasing the 2012 perennial share *down*. `config/perennial_cenagro.yaml` (additive
only, over `perennial_allperu.yaml`) fixes it and takes the fallback to **0.03 %**. The
uncorrected run reported the headline below as **+2.4 pp**; it is **+12.5 pp**. A 4 %
unmapped-token rate is not small when 80 % of it is one token pointing one way.

#### The comparison, and the instrument difference that has to be removed first

Crossing the two raw makes `PASTURE_FALLOW` collapse **17.9 % → 1.1 %**. That is not land
change. PETT recorded a land *state* and has an explicit "EN DESCANSO" token; **CENAGRO
question 024 asks which crop is grown, so a parcel lying fallow contributes no row and leaves
the frame entirely.** The two instruments do not share a class space, and the only defensible
comparison is conditional on a crop being recorded on both sides.

⭐ **PETT declaration (median 1999) → CENAGRO 2012, same parcel, crop recorded on both sides,
n = 8,669:**

| | PETT ~1999 | CENAGRO 2012 | change | 95 % CI | ratio |
|---|---|---|---|---|---|
| **perennial**, % of parcels | 28.2 % | **40.7 %** | **+12.5 pp** | ±0.9 | ×1.44 |
| **annual**, % of parcels | 71.8 % | 59.3 % | −12.5 pp | ±0.9 | ×0.83 |
| **perennial**, % of *cadastral area* | 51.4 % | **66.0 %** | **+14.6 pp** | — | ×1.28 |
| **annual**, % of cadastral area | 48.6 % | 34.0 % | −14.6 pp | — | ×0.70 |

(Percentage *points*, and the CI is paired — McNemar discordant pairs — because a parcel that
is perennial on both sides carries no information about the change. Area is the **cadastral**
polygon area; the census self-reported area is uncorrelated with it and must not be used as a
weight, `DATA.md` Chain B.)

**The gross flows, which the net share hides:**

| PETT ~1999 ↓ / CENAGRO 2012 → | annual | perennial | | median parcel size |
|---|---|---|---|---|
| **annual** | 4,967 (79.8 %) | **1,255 (20.2 %)** | | 0.38 ha → **0.85 ha** |
| **perennial** | 174 (7.1 %) | 2,273 (92.9 %) | | 0.51 ha → 1.16 ha |

⭐ **The flow is strongly asymmetric: 20.2 % of parcels declaring an annual crop in ~1999
declared a perennial in 2012, against 7.1 % moving the other way.** Perennial declarations are
also *sticky* — 92.9 % persist — which is what a tree crop should do and an annual rotation
should not.

⭐ **And conversion happened on the larger parcels.** Parcels that switched annual→perennial had
a median cadastral area of **0.85 ha** against **0.38 ha** for those that stayed annual — which
is why the shift is **+14.6 pp by area against +12.5 pp by parcel count**. Any area-based
estimate that assumed conversion was size-neutral would be biased low.

**Does the census link drive it?** Broken out by link quality, the perennial change is
**+14.8 pp** (high, n=3,070), **+15.3 pp** (medium, n=6,037), **+17.4 pp** (low, n=1,532). The
gradient is mild and the sign and rough size are stable across it, so the finding is **not**
mainly an artefact of the name match — which was the live risk and is the reason the breakdown
exists.

#### ⚠️ How this sits beside §8.2b, which appeared to say the opposite

§8.2b, on the **photo-interpreted national** sample, found only **2.9 %** of declared-annual
parcels reading as perennial in 2019+. Here it is 20.2 % in 2012. These are not in conflict;
they are four different things at once, and the difference is the point:

* **Place.** §8.2b is national; this is Piura. **Restricted to Piura, the photo-interpreted
  picture is completely different from the national one** — 86.7 % of declared-annual Piura
  parcels still read `ANNUAL` in 2019+, against 31.1 % nationally, and only 6.7 % read `OTHER`
  against 57.9 %. ⚠️ **The national "farmable but not currently cropped" finding is not a Piura
  finding.** Piura's irrigated coastal valleys are still being cropped; the national figure is
  carried by other departments.
* **Instrument.** The census records what a farmer *declares they grow*; the S2 campaign records
  what a canopy *looks like from orbit*. A newly planted or sparse orchard is declared as mango
  and reads as bare ground.
* **Class space.** The census has no `WOODY_NON_CROP`; §8.2b puts **32.9 %** of declared-perennial
  parcels there. An orchard gone wild is still an orchard to the census.
* **Sample.** 8,669 name-matched Piura parcels here, 15 declared-annual Piura parcels in the S2
  set — the Piura-only S2 figure has an interval of roughly ±13 pp and settles nothing on its
  own.

⚠️ **What this section is not.** It is a change in **declared** land use on a non-random Piura
subsample between two administrative instruments with different class spaces. It is **not**
evidence about titling, it carries no counterfactual, and the 2012 endpoint is seven years
before the imagery every other endpoint number in §8 is measured on.

**Only 19 parcels carry both a CENAGRO observation and a 2019+ photo-interpreted label** — the
S2 campaign drew from the 614,876-parcel national eligible universe, not from Piura's census
crosswalk. Of the 15 that CENAGRO recorded as perennial, 7 read `PERENNIAL`, 6 read
`WOODY_NON_CROP` and 2 read `OTHER`. Directionally the same story as §8.2b's perennial row, at
a sample size that can only be called an anecdote.
---

### 8.6 ⭐⭐ The national PETT → CENAGRO 2012 comparison, split by tenure

§8.5 did this for Piura on 8,669 parcels. The 25-department census extract (`DATA.md` §1.5)
plus a national name link (`allperu/cenagro_link.py`) extends it to **95,941 parcels across
all 14 linkable departments** — the largest change measurement in the project, and one with
**no satellite and no classifier anywhere in it**: two *declared* observations of the same
land, ~1999 and 2012.

```
uv run python -m crop_classifier.cli allperu cenagro-link     # the crosswalk, 14 depts
uv run python -m crop_classifier.cli allperu cenagro-shift    # the comparison
```

#### The headline

⚠️ **Read the like-for-like row, not the first one.** The census cannot record fallow —
question 024 asks which crop is grown, so a fallow parcel contributes no row and leaves the
frame. `PASTURE_FALLOW` 30.8 % → 6.3 % is that instrument difference, not land change. The
defensible comparison is **conditional on a crop being recorded on both sides**.

| estimator | n | PERENNIAL ~1999 | PERENNIAL 2012 | change |
|---|---:|---:|---:|---|
| all parcels (⚠️ fallow artefact) | 95,941 | 16.6 % | 26.9 % | +10.3 pp |
| ⭐ **like-for-like, parcels** | 63,766 | 24.6 % | 33.0 % | **+8.4 pp ± 0.3** |
| ⭐ like-for-like, **cadastral area** | 63,766 | 30.8 % | 41.7 % | **+10.8 pp** |
| ⭐ **post-stratified to the national population, parcels** | 63,766 | 16.5 % | 26.4 % | **+9.9 pp ± 0.3** |
| ⭐ post-stratified, **cadastral area** | 63,766 | 21.0 % | 33.5 % | **+12.5 pp** |

Gross parcel flows: **7,158 annual → perennial against 1,832 the other way**, a 3.9:1 ratio.
The shift is real, large, and it is not a net figure hiding offsetting churn.

⚠️ **Post-stratification is not optional here.** The name link is not a random sample: it needs
a name on both sides and a district agreement, and the parcels that satisfy that are the
larger, valley-floor, better-documented ones. The linked panel is **16.6 % PERENNIAL against
the population's 9.9 %**, and its median parcel is 0.53 ha against 0.43 ha. Reweighting on
department × declared class restores the national cell counts; the change **rises** from
+8.4 to +9.9 pp, so composition was damping it, not manufacturing it. What reweighting cannot
fix is selection *within* a cell.

#### ⭐ The tenure split — the answer is no, and slightly the other way

| tenure at declaration | n | PERENNIAL ~1999 | 2012 | change |
|---|---:|---:|---:|---|
| INSCRITO (titled) | 40,161 | 15.1 % | 24.2 % | +9.2 pp |
| NO INSCRITO | 23,605 | 19.2 % | 30.5 % | +11.3 pp |
| **difference (INSCRITO − NO INSCRITO)** | 63,766 | −4.1 pp | −6.3 pp | **−2.1 pp ± 0.6** |

*(post-stratified; the raw linked panel gives −1.6 ± 0.6, the same sign and size.)*

**Titled parcels shifted to perennials slightly LESS than untitled ones**, by ~2 pp on a base
of ~10 pp. And **by cadastral area the gap disappears entirely**: INSCRITO +10.7 pp vs
NO INSCRITO +11.0 pp, a difference of +0.3 pp — opposite in sign to the parcel-count gap and
indistinguishable from zero. A tenure effect that flips sign between counting parcels and
counting hectares is not an effect; it is a composition difference in parcel size.

⚠️ **Descriptive, not causal.** Title is not randomly assigned. Note the **baseline** gap:
titled parcels started 4.1 pp *lower* on perennials, which is the same cross-sectional tenure
gap §7 documents — and a difference in change computed off different baselines is exactly the
quantity a pre-trend correction exists to discipline. **This does not overturn §7's bounded
null; it agrees with it** from a completely independent instrument (two declarations, no
imagery, no classifier).

⚠️ **The sign flips by department**, exactly as `RESULTS.md` §7's cross-sectional companion
found. 9 of 14 departments give a negative tenure difference, 5 positive — and all five
positives (Tacna +18.0, Moquegua +14.3, Huancavelica +10.2, Pasco +4.6, Ayacucho +3.9) are
small-n southern or sierra departments. **An OOD estimate over 14 units is 14 numbers**; the
national −2.1 pp is a mean over a distribution that straddles zero.

#### Per department (like-for-like)

| dept | n | ~1999 | 2012 | change | tenure diff |
|---|---:|---:|---:|---:|---:|
| ANCASH | 8,552 | 10.5 | 18.7 | +8.2 | −1.0 |
| AREQUIPA | 4,109 | 11.4 | 22.0 | +10.6 | −3.8 |
| AYACUCHO | 7,537 | 38.0 | 42.8 | +4.8 | +3.9 |
| CAJAMARCA | 13,566 | 15.9 | 25.2 | +9.3 | −1.0 |
| HUANCAVELICA | 581 | 6.9 | 6.4 | −0.5 | +10.2 |
| ICA | 4,067 | 38.7 | 47.8 | +9.0 | −0.8 |
| LAMBAYEQUE | 3,100 | 5.5 | 7.7 | +2.2 | −1.9 |
| LA_LIBERTAD | 5,507 | 10.9 | 14.1 | +3.3 | −0.1 |
| LIMA | 5,372 | 45.8 | 56.9 | +11.1 | −1.1 |
| MOQUEGUA | 613 | 62.2 | 80.1 | +17.9 | +14.3 |
| PASCO | 997 | 79.5 | 93.8 | +14.2 | +4.6 |
| PIURA | 8,182 | 27.9 | 39.5 | +11.6 | −3.0 |
| TACNA | 503 | 44.5 | 70.4 | +25.8 | +18.0 |
| TUMBES | 1,080 | 72.1 | 74.5 | +2.4 | −0.5 |

Piura's +11.6 pp reproduces §8.5's Piura-only +12.5 pp on a differently-built link — the one
external check available on this number.

#### The link is farmer-level, and it shows

| link confidence | n | change |
|---|---:|---:|
| high | 24,780 | +8.0 pp |
| medium | 36,274 | +8.0 pp |
| low | 2,712 | +15.6 pp |

**High and medium agree to the decimal** — the answer is not an artefact of link quality over
92 % of the sample. `low` (4 %) doubles it, which is what an unreliable link should do.
Validation against notebook 02's independently-built Piura crosswalk: the **high-confidence
tier reproduces the same `COD_PREDIO` on 95.5 %** of shared producers (medium 48.7 %, low
34.8 %) — the grade means what it says.

#### ⚠️ The lexicon trap fired again, and bigger

The Piura audit (§8.5) found `VERGEL FRUTICOLA` — 80 % of a 4.09 % unmapped tail, worth 10 pp
on the headline. The **national** vocabulary reran the same failure: **`MELOCOTONERO` (peach
*tree*) was 45.8 % of the national unmapped tail at 18,371 instances**, followed by
`MEMBRILLERO` (quince tree, 2,896) and `DACTYLIS` (a sown forage grass, 1,186).

`MELOCOTON` and `DURAZNO` — the *fruit* names for the same peach — were **already** PERENNIAL
in the config. The census simply uses the *-ero* tree form, so this was a vocabulary gap, not
a policy question. **The budget check passed either way** (1.35 %, budget 2 %) — the tail is
small and the error inside it was concentrated, which is precisely the failure mode the budget
cannot see. Fixing it cut the tail to **0.54 %** and moved the like-for-like headline from
+7.8 to +8.4 pp and the area figure from +10.6 to +10.8 pp.

**The rule is now twice-confirmed: print the unmapped tail sorted by frequency and read the top
ten. Never accept the budget check alone.**

### 8.7 The 2019+ photo-interpreted endpoint — consistent, and uninformative on tenure

Same 865 usable labels as §8.2, tenure known for **all 865** (489 INSCRITO / 376 NO INSCRITO).

⚠️ **The design weight is doing most of the work.** The campaign over-sampled PERENNIAL so the
rare class would be learnable; unweighted, the sample is 13.3 % PERENNIAL, and design-weighted
it is **4.7 %**. The weights validate cleanly against the population they were drawn from:

| declared class | S2 design-weighted | national PETT population |
|---|---:|---:|
| ANNUAL | 50.2 % | 50.0 % |
| PASTURE_FALLOW | 41.5 % | 40.1 % |
| PERENNIAL | 8.3 % | 9.9 % |

| reading of `WOODY_NON_CROP` | n | declared | observed 2019+ | change |
|---|---:|---:|---:|---|
| excluded | 214 | 12.5 % | 16.3 % | **+3.9 pp ± 8.0** |
| read as PERENNIAL | 364 | 21.1 % | 33.2 % | **+12.1 pp ± 8.8** |

Both are consistent with the census's +9.9 pp, and **neither can distinguish itself from it or
from zero.** The `WOODY_NON_CROP` decision moves the answer by 8 pp — more than the effect
being measured — which is §8.2c's finding again: *the binding constraint is the woody-non-crop
boundary, not the sample size.*

**On tenure the S2 arm is uninformative and should not be quoted.** WOODY excluded gives
−1.0 pp ± 16.5; WOODY as perennial gives +13.2 pp ± 18.0. Two readings of one ambiguous class,
opposite signs, CIs an order of magnitude wider than the census estimate's. The census answer
(−2.1 ± 0.6) is the one with power.

**Parcels with all three observations: 157.** Not a fixable sample size — the campaign drew
from the PETT population, not from the name-linked census subset, so the overlap is incidental.
One thing in it is worth recording: of 98 parcels the census called PERENNIAL, **47 were
photo-interpreted `WOODY_NON_CROP` or `NON_AGRICULTURE`** — nearly half of declared perennial
reads as woody non-crop from the air. That is the §8.2c boundary problem quantified against a
declared source rather than against another label.

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
| `perennial/gapfill.py` | one-time re-extraction of the parcels the 3-class map newly admits | superseded — the 3-class store was built through `allperu labels`; kept as the record of how the 12→3-class parcel gain was to be closed |

⛔ **All seven are reachable from `allperu closed …`** (`oli`, `oli-refit`, `windows`,
`window-sample`, `estimate`, `external`, `tenure-did` v2), which is a sub-group precisely so
that `allperu --help` lists live work. Moving them there changed nothing about how they run.

⚠️ **The national locked test is UNSPENT** and should stay that way until an estimand passes
its gate.
