# 6. Reference

Everything the four task pages do not need: checking the published numbers, training and testing
a model, downloading images, changing the list of crop classes, and the research questions that
are finished or abandoned.

---

## Check the published numbers

Two commands, about 90 seconds. You do not need the restricted files or an Earth Engine account:
the tables each check needs are included here.

```bash
uv sync
uv run cc reproduce
```

```
check           quantity                                          published   measured
demo            demo quickstart, CV macro-F1                        +0.5381    +0.5381  ok
national-lodo   national Landsat, LODO macro-F1 (14 departments)    +0.4789    +0.4789  ok
s2-model        Sentinel-2 3-class (t3w, --climate temp), CV        +0.7620    +0.7585  ok
perennial-shift perennial share of parcels, 1999 -> 2012 (pp)         9.900      9.900  ok
tenure-did      DiD headline effect on perennial probability        -0.0011    -0.0011  ok
```

`uv run cc reproduce --list` names the checks and says where each number was published.
`cc reproduce <check> -v` prints its full tables. Two of the checks retrain a model, so they
land within a few thousandths rather than matching exactly: LightGBM does not give identical
results on different computers. A row marked `missing` names the file it wanted;
`uv run cc data verify` says what your copy has.

## Train a model

**Sentinel-2, 865 human-labelled parcels. This is the good one:**

```bash
uv run cc -w national_s2 labelling train prep --target t4 --pilot --climate temp
uv run cc -w national_s2 labelling train fit  --target t4 --pilot --climate temp \
    --model lightgbm
```

**The Landsat model over declared crops, 14 departments, about 50,000 parcels:**

```bash
uv run cc -w national labels build
uv run cc -w national splits assign --config src/crop_classifier/config/split_allperu.yaml
uv run cc -w national train --model lightgbm --drop-features meta,location \
    --run-name lightgbm_nometa_nolat
```

- `prep` prepares the data one setup needs. Give it the **same** `--target` and `--climate` you
  are about to train with, or `fit` will stop and say so.
- Models: `lightgbm` (use this), `ltae`, `psetae`, and `rules`, a hand-written baseline that is
  there to be beaten. Run one model per command: LightGBM and the neural models crash if loaded
  together on a Mac.
- `--drop-features meta,location` is **required** for the Landsat model. `meta` describes the
  satellite archive rather than the land, and a model that uses it invents a trend. `location`
  is the parcel's latitude, which the model uses as a lookup table.

To train on labels of your own, see [`04_label_and_train.md`](04_label_and_train.md).

## Read the LODO score, not the cross-validation score

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

| test | what it hides from the model | what it tells you |
|---|---|---|
| CV (cross-validation) | 5 km blocks inside departments it has already trained on | cannot tell real skill from memorised location |
| **LODO** | a whole department it has never seen | this is the number to read |
| LOYO | a whole year of labels, with the areas roughly unchanged | has the same blind spot as CV |
| LODYO | a department and a year at once | comes free with the LODO run |

`floor` is the score from always guessing the most common class. `skill` is the score rescaled
so that 0 is the floor and 1 is perfect, which is the only fair way to compare across different
sets of classes.

The same mistake was caught three times, each time only by LODO:

- **Latitude** adds 0.047 to CV and to LOYO, and takes 0.060 off LODO.
- **`frac_l7`**, which satellite happened to be overhead, made the model invent change over
  time. Inputs that never change over time made it invent stability.
- **LTAE**, a whole model type, beat LightGBM in all 8 Sentinel-2 cross-validation runs and lost
  all 4 tests on unseen departments.

Two rules that go with the table:

- **A score over 14 departments is 14 numbers.** Read `mean ± sd`, not the pooled score, which
  leans towards whichever department was largest. An earlier reading had LTAE winning LODO; what
  overturned it was going from 4 held-out departments to 14.
- **Always read a score next to its floor.** Scores are not comparable across different sets of
  classes, but `skill` is.

**Testing an input that tracks location.** If a new input varies smoothly across the map,
compare it against a stand-in of the same shape, in the same folds, rather than against having
no input at all. Average temperature raised both CV and LODO. What made that readable was a
fourth run using the parcel's latitude and longitude in place of the temperature columns: the
same number of columns, equally smooth across the map, with no climate information in them.

**The final test set** is small (161 parcels, roughly plus or minus 7 points) and it has been
used. The Sentinel-2 model was scored on it once, on 2026-09-01, at 0.774 macro-F1. `--eval-test`
existed for that one use.

```bash
uv run cc -w national advanced lodo --tag nolat --drop-features meta,location
uv run cc -w national advanced loyo --drop-features meta,location
```

## Satellite images

[`03_score_parcels.md`](03_score_parcels.md) runs the whole chain in one command. The separate
steps, for the Landsat images:

```bash
uv run cc -w national satellite extract --stage all
uv run cc -w national satellite assemble
```

`extract` runs in two stages on purpose. First it counts, cheaply, how many cloud-free dates
each parcel has, and drops the parcels that have too few before anyone pays for their pixels.
Then it downloads the pixels for the rest, once, so that trying out new inputs never means
downloading again. `assemble` turns those pixels into per-parcel summaries. It does not fill in
gaps.

Useful options: `--stage coverage|pixels`, `--years '1999-2003'`, `--max-chunks N`,
`--chunk-size`. You can stop and restart the download safely: each batch is named after the
parcels in it, so finished work is not repeated.

**Earth Engine fails quietly in three ways.** None of them raises an error:

1. **It hangs and returns nothing.** Retrying never happens, because nothing failed. The code
   gives each request 900 seconds and then gives up. That fired 25 times during the national
   download and lost no work.
2. **Throttling arrives both as an error message and as a hang.** Messages are retried; the
   900-second limit catches the silent half.
3. **The cache is keyed on which parcels are in a batch, not on what is computed from them.**
   If you add a new measurement, every existing batch still looks complete, quietly without it.

There is a fourth problem, about cost: **a batch costs more the further apart its parcels are.**
Batches are built from nearby parcels for that reason. If you write a new download step, copy
that grouping, not just the structure.

**Check the result by counting it**, and compare the count against other years. There is a real
floor of about 0.6 % of parcels too small to return a pixel, so "98 % complete" on its own
proves nothing. A year that looks wrong next to its neighbours is a real gap.
`uv run cc -w national advanced density-audit`.

Costs and coverage: Landsat over Piura starts in 1996, and the cost rises about elevenfold by
2023. Landsat gives 13 to 24 cloud-free dates per parcel per year; Sentinel-2 gives 47. That
gap is large enough that a comparison decided on one has to be redone on the other.

## Changing the list of crop classes

A class set says which classes a model is asked to predict. Adding one means copying a YAML file
and editing it. No Python. `uv run cc labels list` shows what already exists.

A person labelling a picture records one of five classes, plus `UNSURE`, which is a refusal to
answer and never a class:

```
PERENNIAL   ANNUAL   OTHER   WOODY_NON_CROP   NON_AGRICULTURE
```

Copy the closest file in `src/crop_classifier/config/labels/`:

```yaml
# src/crop_classifier/config/labels/orchard2.yaml
name: orchard2                     # must match the filename
about: >-
  Tree and vine crops against everything else that is farmed, dropping non-farmland.
collapse:
  ANNUAL: FARMED_NOT_PERENNIAL     # merge into a new class
  OTHER: FARMED_NOT_PERENNIAL
  WOODY_NON_CROP: PERENNIAL
  NON_AGRICULTURE: null            # null removes the class from training
rules_compatible: true
```

Anything not listed under `collapse` keeps its own name. Then run `prep`, `fit` and `evaluate`
with `--target orchard2`. A name in `collapse` that is not one of the five recorded classes
raises an error listing the valid ones, rather than being ignored.

**Declared crop names** are handled separately, in `config/perennial*.yaml`. They map thousands
of messy Spanish crop names onto classes. Variants use `extends:`, so a new version is a list of
differences rather than a copy, and `add:` only appends. That matters: if a new version changed
which class an existing crop belongs to, a measured "change in the land" would partly be a
change in the definitions.

Three things that decide whether a class-set result means anything:

- **Read the list of unmatched crop names every time.** Both builders print the names that fell
  through to the catch-all class, most common first. Read the top ten. A catch-all is never
  evenly spread, and a percentage check cannot see that. When 4.09 % of 2012 census names fell
  into a blanket `ANNUAL`, 80 % of that was one name: `VERGEL FRUTICOLA`, "fruit orchard", which
  is a tree crop. The headline moved from +2.4 to +12.5 points when it was fixed, and the
  percentage check had passed either way.
- **Test your result under both treatments of non-crop trees.** `t4` and `t3w` differ in one
  decision, where `WOODY_NON_CROP` goes, and it moves the `PERENNIAL` score by 0.190. That is
  larger than most effects measured here. Adding both climate columns looked good on 12 of 14
  departments under one treatment and on only 7 of 14 under the other.
- **Scores are not comparable between class sets.** Collapsing four classes to two raised
  macro-F1 from 0.672 to 0.715, and raised the always-guess-the-most-common-class floor from
  0.171 to 0.467. Adjusted for that, the two-class version was the weakest in the study, and it
  gained 0.004 on the one class anyone cared about.

## The land-title study, and the questions that were closed

**The estimate compares parcels that gained a title with parcels that did not, before and
after.** Two dated records of registration already exist in the data: the status when the crop
was declared (1997–2006) and the status in the land survey of about 2011. 1,780,580 parcels have
both, and 8.6 % of them moved from unregistered to registered. This work is **finished**. The
6,559 treated parcels are every one that exists once the conditions are applied. The order is
enforced by the code: a placebo test runs first, and the written-down plan cannot be overwritten.

```bash
uv run cc reproduce tenure-did                      # check it without writing anything
uv run cc -w tenure_did analysis did-feasibility    # always run the feasibility check first
uv run cc -w tenure_did analysis did \
    --preds data/processed/all_peru_did/panel_predictions_nolat_aug_yleak10.parquet \
    --tag mycheck
```

`did-feasibility` compares how many parcels the question needs against how many the archive can
supply. It takes minutes and it is the most useful command on this page. Work out the size you
need for the test that will actually decide the question, and test whether the effect is small
rather than whether it is non-zero: a wide enough range passes the usual significance test
without telling you anything.

| question | why it was closed |
|---|---|
| A year-by-year history of each parcel over 25 years | predictions flip between classes 43 % to 98 % of the time, against a 15 % limit, and it is not caused by missing images |
| Comparing five-year windows | the comparison group drifts seven times further than the effect, in the opposite direction, in every version |
| Recalibrating, aligning distributions, matching image density | measured: they moved the answer by 0.0001, by −0.1023, and by about a third of the problem |
| Adding Landsat 8 and 9, tried twice | the difference between the sensors depends on what is growing (0.025 in NDVI), so no single correction removes it, and fitting one locally is worse than none |
| Collapsing to two classes | the score rises but the floor rises further |

They can still be run, under `cc archive`, and `src/crop_classifier/archive/README.md` records
which check ruled out each one. They are kept because knowing why something fails is a result.

**The three records are not one continuous series.** The declaration and the census are one kind
of record; the 2025 points come from a few hundred parcels read from pictures, drawn under
different rules. Each image-based reading starts from its own baseline rather than continuing
the census line. Joining them would invent a trend that neither record measured.

---

Every number with its caveats: [`../RESULTS.md`](../RESULTS.md). What carries over to other
projects: [`../LESSONS.md`](../LESSONS.md). Every command and every output file:
[`../PIPELINE.md`](../PIPELINE.md).
