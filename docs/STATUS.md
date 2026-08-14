# Status

**Last updated 2026-08-14.** Read this before starting work. Numbers and their evidence live in
[`RESULTS.md`](RESULTS.md) — this page says only what is done, what is closed, and what to do.

---

## The question

Has farmland in Peru shifted from domestic annual crops (rice, maize, beans) to export perennial
crops (mango, lime, coffee, banana)? And does giving a farmer secure legal title make that shift
more likely?

The data allows it because Peru's land-titling programme recorded, for ~1 M parcels, the crop
growing there **and** the parcel boundary — but only **once**, mostly 1997–2006. There is no
second visit. Satellite imagery exists every year from 1996.

---

## What works

* **A single-year 3-class land-state classifier** (`PERENNIAL` / `ANNUAL` / `PASTURE_FALLOW`).
  Piura locked test **0.681 macro-F1**; national CV **0.628**, and 3× steadier across folds.
* **The national label build** — 14 departments, 946,872 linked polygons, label years spread
  over 1997–2006.
* **The tenure difference-in-differences**, which returned the project's only actual estimate:
  a **bounded null**, headline **−0.0011 [−0.0126, +0.0104]**.
* **The evaluation protocol** — CV / LODO / LOYO / LODYO. This is what caught the rest.

---

## ⛔ What is closed, and must not be reopened

| route | why closed | where |
|---|---|---|
| 12-class "which crop?" | annual crops are not separable at 30 m — maize 0.23 F1, beans 0.11 | `RESULTS.md` §1 |
| Per-parcel annual trajectories (Piura **and** national) | flicker 0.43–0.98 against a 0.15 criterion; **no thin years nationally**, so not coverage. A ~0.55–0.59 classifier cannot support a 25-year per-parcel series | §3, §4.4 |
| "Try another architecture" | LTAE fails *harder* on every axis (flicker 0.98, LODO 0.442, LOYO 0.478) | §3.1, §4.3 |
| The 5-year window pivot | the control pool drifts 7× further than the signal, the other way, on **every** arm | §5 |
| Recalibration / quantile alignment / density matching | measured: moved the target metric by 0.0001, −0.1023, and ~⅓ of the artefact respectively | §6.3 |
| **Admitting OLI** (twice) | the sensor difference is **cover-type dependent** (0.025 NDVI between classes) — no global linear map can remove it. **Do not propose refitting the coefficients; it is done.** Reopening needs a cleaner paired sample, not better fitting | §6.4 |
| **A bigger tenure DiD** | the population is exhausted — **6,559 treated parcels is all of Peru**. The binding limit is 7× amplification of a 2.5-year placebo. Reopening needs a *dated registration event*, not two snapshots. Do not relax R4, widen the band, or drop the at-risk restriction | §7 |

---

## ⭐ What to do next

**One route remains: the S2 endpoint-labelling campaign.** Everything except the labelling is
built — 1,112 parcels drawn, chips rendered, S2 traces extracted, 9 HTML shards emitted, split
frozen, ingest round-tripped on the real files.

Full plan and gates: **[`s2_labelling/plan.md`](s2_labelling/plan.md)**.

1. **▶ THE PILOT — the immediate next action.** Both labellers do `pilot_A.html` /
   `pilot_B.html` (120 parcels, ~4 h each) → `allperu s2-labels ingest` → **G1: κ_called ≥
   0.75**. This is the only thing that can still kill the campaign. If it fails, revise the
   **codebook**, not the sample.
2. Label the four main shards + the overlap shard (~1,092 labellings, ~37 h human) → `ingest` →
   **G2** (`UNSURE` < 25 %) and **G3** (≥35/department, ≥150/class — expect `NON_AGRICULTURE`
   to fall short; pool at training time, do not re-draw).
3. Assemble → train → CV / LODO / locked test → **G4**: the S2 model must beat the existing
   Landsat model on the same held-out parcels. ⚠️ G4 has **no recovery path** — if it loses, the
   labels become a validation set. That is on the record now, before the number arrives.
4. `allperu s2-labels transitions` → the weighted declared→observed matrix with CIs and no
   classifier in it. A deliverable in its own right.

---

## Standing constraints on any new work

* **Never select a model on CV alone.** Report **CV / LODO / LOYO / LODYO**. An OOD evaluation
  is blind along the axis it holds fixed — `centroid_lat` scores +0.047 on CV *and* +0.047 on
  LOYO, and −0.060 on LODO.
* **`--drop-features meta,location` is not optional** for anything that will be applied across
  years. Time-invariant features manufacture stability; acquisition metadata manufactures change.
* **The national locked test is UNSPENT.** Keep it that way until an estimand passes its gate.
* ⚠️ **The Piura locked test is SPENT TWICE** (2026-08-05). It is no longer a clean held-out
  estimate for any selection.
* **Run a feasibility check before funding any GEE extraction** — `allperu tenure-ceiling` style:
  required n from measured variance against available n from the archive. It takes minutes.
* **Verify a finished extraction by counting its rows** (`perennial panel verify`), never by
  "the process exited" or "the file exists".
* **Any area or share figure must use `sample_weight`** — the national sample doubles the
  perennial share by design.
* **Never import torch and lightgbm in one process on macOS** (libomp segfault).
* Every accuracy number in this project is measured in ~1997–2006. **None is measured in
  2019–2023.** Do not quote one as if it were.

---

## Where things live

| | |
|---|---|
| [`RESULTS.md`](RESULTS.md) | every strand, its verdict, the numbers |
| [`LESSONS.md`](LESSONS.md) | findings that generalise beyond this project |
| [`DATA.md`](DATA.md) | raw datasets, linkage chains, the four silent traps |
| [`PIPELINE.md`](PIPELINE.md) | module reference, CLI table, cookbook |
| [`REPORT.md`](REPORT.md) | the readable narrative |
| [`s2_labelling/`](s2_labelling/) | the live campaign + frozen codebook |
