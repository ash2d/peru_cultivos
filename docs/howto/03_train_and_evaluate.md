# 3. Train a model and read the score correctly

The second half of this page matters more than the first.

---

## Train

**The best model — Sentinel-2, 865 photo-interpreted parcels, 3 classes:**

```bash
uv run cc -w national_s2 labelling train prep --target t3w --pilot --climate temp
uv run cc -w national_s2 labelling train fit  --target t3w --pilot --climate temp --model lightgbm
```

**The national Landsat model — declared crops, 14 departments, ~50k parcels:**

```bash
uv run cc -w national labels build
uv run cc -w national splits assign --config src/crop_classifier/config/split_allperu.yaml
uv run cc -w national train --model lightgbm --drop-features meta,location \
    --run-name lightgbm_nometa_nolat
```

Notes on the options:

- `prep` builds the data one arm needs. Give it the **same** `--target` and `--climate` you
  are about to fit with, or `fit` stops and says so.
- `--target` picks the label set: `t3w` (3 classes, the one carried forward), `t4`, `t5`,
  `t3`, `t2`. Adding your own: [`07_new_label_set.md`](07_new_label_set.md).
- `--climate temp` adds each parcel's mean temperature. It is the only extra feature in this
  project that improved both cross-validation and transfer to an unseen department.
- Models: `lightgbm` (use this), `ltae`, `psetae`, `rules` (a hand-written floor, not a
  candidate).
- `--drop-features meta,location` is **not optional** on the Landsat arm. `meta` describes the
  satellite archive, not the land, and a model that uses it invents a trend. `location` is the
  parcel's latitude, which is a lookup table (see below).
- One model per command. LightGBM and torch crash if loaded together on macOS, so run each
  model as its own command, or loop in the shell rather than in Python.

Training on new labels of your own: [`06_label_more_parcels.md`](06_label_more_parcels.md).

## Evaluate

```bash
uv run cc -w national evaluate runs/all_peru/lightgbm_nometa_nolat_aug_yleak10 \
    --tag nolat_aug_yleak10
```

```
split     mean±sd over units   pooled   floor   skill              units         n
----------------------------------------------------------------------------------
CV           0.5811 ± 0.0024   0.5818   0.198   0.478         5 CV folds    43,419
LODO         0.4789 ± 0.0843   0.5382   0.201   0.348     14 departments    54,438
LOYO         0.5261 ± 0.0574   0.5465   0.199   0.408     14 label years    47,597
LODYO        0.5173 ± 0.0595   0.5382   0.201   0.396 14 dept x year cells         -
```

The command only reads what is on disk; it never starts a long refit. Anything missing is
named along with the command that produces it. On a fresh clone the CV row is missing and the
other three are there, because `runs/` is not committed while the held-out records are.

```bash
uv run cc -w national advanced lodo --tag nolat --drop-features meta,location
uv run cc -w national advanced loyo --drop-features meta,location
```

## Read the LODO row, not the CV row

| split | what it holds out | what it tells you |
|---|---|---|
| CV | 5 km blocks inside departments the model has already trained on | cannot tell real signal from memorised location |
| **LODO** | a whole department the model has never seen | the one that decides |
| LOYO | a whole label year, region roughly fixed | has the same blind spot as CV |
| LODYO | department and year at once | free — re-scored from the LODO fits |

This is not a style preference. The same mistake was caught three times, each time only by
LODO:

- **`centroid_lat`** (a parcel's latitude) gains +0.047 on CV *and* +0.047 on LOYO, and loses
  0.060 on LODO.
- **`frac_l7`** (which satellite was overhead) invented change; time-invariant features
  invented stability.
- **LTAE**, a whole architecture, wins Sentinel-2 CV in 8 arms of 8 and loses LODO in 4 of 4.

Two more rules that come with the table:

- **A held-out estimate over 14 departments is 14 numbers.** Read `mean ± sd`, not the pooled
  score: pooled leans toward whichever department was biggest. An earlier reading had LTAE
  winning LODO; what overturned it was going from 4 held-out departments to 14.
- **Always read macro-F1 beside its floor.** macro-F1 is not comparable across label sets;
  `skill = (macro_f1 − floor) / (1 − floor)` is.

## Testing a feature that tracks location

If a new feature is a smooth function of where the parcel is, compare it against a
**same-shaped control**, in the same folds — not against having no feature at all.

Mean temperature raised CV *and* LODO. What made that readable was a fourth arm: centroid
latitude and longitude in place of the two climate columns, same count, same smoothness, no
climate content. LightGBM gained several times more out-of-department from climate than from
coordinates (so it was adopted); LTAE gained the same from both (so it was not).

## The locked test set

There is one, it is small (161 parcels, roughly ±7 pp), and it is **spent** — the Sentinel-2
model was scored on it once, on 2026-09-01, at 0.774 macro-F1. Scoring anything else on it
would destroy its ability to confirm anything independently. `--eval-test` exists for that one
use and should not be used again.

Next: [`04_predict_new_parcels.md`](04_predict_new_parcels.md).
