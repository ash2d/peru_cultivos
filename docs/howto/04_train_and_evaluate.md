# 4. Train a model and evaluate it honestly

The second half of this page is the important half.

---

## Train

**On the PETT declared-crop labels** (Landsat, 14 departments, ~50k parcels):

```bash
uv run cc -w national labels build
uv run cc -w national splits assign --config src/crop_classifier/config/split_allperu.yaml
uv run cc -w national train --model lightgbm --drop-features meta,location \
    --run-name lightgbm_nometa_nolat
```

**On the Sentinel-2 photo-interpreted labels** (865 parcels, all 14 departments, verified on
2019+ imagery):

```bash
uv run cc -w national_s2 labelling train prep --target t3w --climate temp
uv run cc -w national_s2 labelling train fit  --target t3w --climate temp --model lightgbm
```

`prep` builds the workspace one arm needs. **Pass it the same `--target` and `--climate` you
are about to fit with** — bare `prep` builds every plain label set and no climate arm, so
`fit --climate temp` after a bare `prep` stops and says so. To label *more* parcels and train
on them: [`06_label_more_parcels.md`](06_label_more_parcels.md).

Models: `lightgbm` (the one to carry forward), `ltae`, `psetae`, `rules` (a hand-written
baseline — a floor, not a candidate).

### ⚠️ `--drop-features meta,location` is not optional

`meta` is acquisition metadata — which satellite was overhead, how many looks, `frac_l7`. It
describes the archive, not the land, and it moves for reasons that have nothing to do with crops.
A model that uses it **manufactures a trend**.

`location` is `centroid_lat`. It is worth **+0.047 macro-F1 on cross-validation** and
**−0.060 when the department changes**. It is a lookup table.

### ⚠️ One model per process

LightGBM and torch each bundle their own libomp; co-loading them on macOS segfaults with exit
139 and no traceback. To sweep models, **loop in the shell**:

```bash
for m in lightgbm ltae; do uv run cc -w national_s2 labelling train fit --model $m; done
```

---

## Evaluate

```bash
uv run cc -w national evaluate runs/all_peru/lightgbm_nometa_nolat_aug_yleak10 \
    --tag nolat_aug_yleak10
```

⚠️ **On a fresh clone the CV row is missing from that table and the other three are not.**
`runs/` is not committed — a CV score lives in the run directory, so it comes back when you
retrain — while the LODO/LOYO/LODYO record *is* committed, because recomputing it is a model
per department. The three rows that decide are the three you get for free.

```
split     mean±sd over units   pooled   floor   skill              units         n
----------------------------------------------------------------------------------
CV           0.5811 ± 0.0024   0.5818   0.198   0.478         5 CV folds    43,419
LODO         0.4789 ± 0.0843   0.5382   0.201   0.348     14 departments    54,438
LOYO         0.5261 ± 0.0574   0.5465   0.199   0.408     14 label years    47,597
LODYO        0.5173 ± 0.0595   0.5382   0.201   0.396 14 dept x year cells         -
```

The command **reads** — it never starts a LODO refit, which is hours. Anything missing is named
along with the command that makes it:

```bash
uv run cc -w national advanced lodo --tag nolat --drop-features meta,location   # LODYO is free
uv run cc -w national advanced loyo --drop-features meta,location
```

### ⭐ Read the LODO row. Not the CV row.

| split | what it holds out | what it can see |
|---|---|---|
| **CV** | 5 km blocks *inside departments the model has already trained on* | cannot separate signal from spatial memorisation |
| **LODO** | a whole department, unseen | ⭐ the one that decides |
| **LOYO** | a whole label-year cohort, region roughly fixed | inherits CV's blind spot: a time-invariant lookup is as available here as in CV |
| **LODYO** | department **and** year at once | free — re-scored from the LODO fits |

This is not a stylistic preference. The same mistake was made three times, each time caught only
by LODO:

* **`centroid_lat`** gains +0.047 on CV *and* +0.047 on LOYO, and loses 0.060 on LODO.
* **`frac_l7`** manufactured change; static features manufactured stability.
* **LTAE**, an entire architecture, wins Sentinel-2 CV in **8 arms of 8** and loses LODO in
  **4 of 4**. The extra skill is skill that does not leave the training departments.

### ⚠️ An out-of-distribution estimate over 14 units is 14 numbers

The table leads with **mean ± sd over held-out units**, not the pooled score. They are different
numbers meaning different things: pooled weights the answer toward whichever department was
largest; the mean is what to expect from *a new* department, which is the question.

An earlier reading had LTAE *winning* LODO. What overturned it was going from **4** held-out
departments to **14** — not new data, not new code.

### ⚠️ Always read macro-F1 beside its floor

macro-F1 is not comparable across label spaces. `skill` — `(macro_f1 − floor) / (1 − floor)` —
is, and it is the only column in that table that is.

### ⭐ When a feature is a smooth function of something the model must not memorise

Run it against a **same-shaped control**, in the same folds, labelled as a control.

Mean temperature added to the S2 classifier raised CV *and* LODO — the first feature in the
project to do both. What made it readable was a fourth arm: **centroid lat/lon in place of the
two climate columns** — same count, same smoothness over space, no agro-climatic content.
LightGBM got several times more out-of-department gain from climate than from coordinates
(adopt); LTAE got the same from both (do not).

The comparison that decides is not *feature vs nothing*. It is *feature vs a same-shaped
surrogate carrying only the thing you are worried about*. And verify that the "nothing" arm
reproduces its recorded numbers before you read any delta.

---

## The locked test set

There is one, it is small (161 usable parcels over 14 departments, roughly ±7 pp), and it is
**spent** — the S2 model was scored on it on 2026-09-01 (0.774 macro-F1). Scoring anything
else on it destroys its ability to confirm anything independently.

`--eval-test` is one-way and gated behind an explicit flag for that reason. Pass it only for a
configuration already selected on CV/LODO, and write the number into `docs/RESULTS.md`.

Next: [`05_perennial_change_by_tenure.md`](05_perennial_change_by_tenure.md).
