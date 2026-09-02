# Peru crop classifier

**Has Peruvian farmland shifted from domestic annual crops to export perennials — and does
secure legal title cause it?**

Peru's land-titling programme recorded, for about a million parcels, the crop growing there and
the parcel boundary, mostly between 1997 and 2006. It never went back. Satellite imagery exists
for every year since 1996. This repository joins those two records into a land-use classifier
and reads change out of it.

![perennial share over time, by tenure](docs/figures/perennial_over_time_by_tenure.png)

---

## What do you want to do?

| | |
|---|---|
| Install it | [`docs/howto/01_setup.md`](docs/howto/01_setup.md) |
| **Reproduce the published results** — 2 commands, 90 seconds | [`docs/howto/02_reproduce_results.md`](docs/howto/02_reproduce_results.md) |
| **Train a model** and read its score correctly | [`docs/howto/03_train_and_evaluate.md`](docs/howto/03_train_and_evaluate.md) |
| **Predict on parcels**, including your own polygons | [`docs/howto/04_predict_new_parcels.md`](docs/howto/04_predict_new_parcels.md) |
| Pull new satellite imagery | [`docs/howto/05_get_satellite_data.md`](docs/howto/05_get_satellite_data.md) |
| **Label more parcels and retrain** — the loop that improves the classifier, written for someone who does not code | [`docs/howto/06_label_more_parcels.md`](docs/howto/06_label_more_parcels.md) |
| Change which classes the model predicts | [`docs/howto/07_new_label_set.md`](docs/howto/07_new_label_set.md) |
| Measure perennial change and its link to land title | [`docs/howto/08_perennial_change_by_tenure.md`](docs/howto/08_perennial_change_by_tenure.md) |

Quick start:

```bash
uv sync                 # Python 3.11, exact versions
uv run cc reproduce     # re-derive every published number, ~90 s
uv run cc data verify   # what this clone can do, and what is missing
uv run cc --help
```

The data needed to reproduce the published results is committed — 642 MB of parcels, polygons,
labels and satellite features. No Earth Engine account and no licensed archive are needed for
that. [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md) says what is here and what is not.

---

## The answers

**The shift is real and large.** Between about 1999 and 2012, on 63,766 parcels linked across
all 14 departments, perennial crops go from 16.5 % to 26.4 % of parcels (+9.9 pp) and from
21.0 % to 33.5 % of cadastral area (+12.5 pp). Flows run 3.9 : 1 toward perennials, so it is
not a net figure hiding churn in both directions. No satellite imagery and no classifier enter
that measurement: it compares two official declarations of the same land.

**Title does not appear to cause it.** Titled parcels shifted slightly less (−2.1 pp ± 0.6),
and measured in hectares the gap disappears (+0.3 pp). A two-period difference-in-differences
over 14,625 parcels returns a bounded null: −0.0011 [−0.0126, +0.0104].

**The classifier works for one year, not for a time series.** Four attempts to turn it into a
per-parcel trend all failed their pre-registered checks, for measured reasons.

Every number, with intervals and caveats: [`docs/RESULTS.md`](docs/RESULTS.md).

---

## The current best classifier — Sentinel-2, 3 classes

LightGBM over per-parcel Sentinel-2 summaries for 2019 onward, plus the parcel's mean annual
temperature. Labels come from photo-interpretation: 865 usable parcels across all 14
departments, on a split frozen before any labelling began — 704 for training and
cross-validation, 161 held back as a locked test.

Classes (`t3w`): `ANNUAL`, `PERENNIAL` (including woody non-crop) and `OTHER`.

| evaluation | what it holds out | macro-F1 | accuracy | `PERENNIAL` F1 | floor |
|---|---|---:|---:|---:|---:|
| Cross-validation (704 parcels) | 5 km blocks inside seen departments | 0.762 ± 0.038 | 0.783 | 0.789 | 0.228 |
| Leave-one-department-out | a whole department, 14 of them | 0.724 ± 0.111 | 0.789 | 0.777 | 0.228 |
| Locked test (161 parcels) | regions frozen before labelling | **0.774** [0.697, 0.845] | 0.789 | 0.780 | 0.219 |

Per class on the locked test (κ = 0.647):

| class | precision | recall | F1 | n |
|---|---:|---:|---:|---:|
| `ANNUAL` | 0.938 | 0.600 | 0.732 | 25 |
| `OTHER` | 0.810 | 0.810 | 0.810 | 79 |
| `PERENNIAL` | 0.727 | 0.842 | 0.780 | 57 |

How to read it:

1. **The cross-validated estimate was honest** — the test scored 0.774 against a CV estimate
   of 0.762, and `PERENNIAL` F1 reproduced to within 0.01.
2. **0.774 is the in-department number.** Every department appears on both sides of the test
   split. For a department the model has never seen, expect 0.724 ± 0.112.
3. **The test is spent** (2026-09-01) and must not be scored again.
4. **The label-noise floor is unmeasured** — one annotator, so this is agreement with those
   labels, not with ground truth.

For comparison, the national Landsat model over the 1997–2006 label years scores 0.628 CV and
0.479 leave-one-department-out, on a different 3-class question and much sparser imagery
(13–24 clear dates per parcel-year against 47).

```bash
uv run cc reproduce s2-model     # refit it and compare against the published number
```

---

## The one thing to know before modelling anything

A parcel's latitude (`centroid_lat`) is worth +0.047 macro-F1 on cross-validation and −0.060
when the department changes.

Cross-validation holds out 5 km blocks inside departments the model has already seen, so it
cannot separate real signal from memorised location. Never select a feature or a model on
cross-validation alone. `cc evaluate` prints CV, LODO, LOYO and LODYO together, each beside the
score you would get by always guessing the largest class.

The same lesson caught three features and then a whole architecture: LTAE wins Sentinel-2
cross-validation in 8 arms of 8 and loses leave-one-department-out in 4 of 4. More:
[`docs/LESSONS.md`](docs/LESSONS.md).

---

## The data, in one paragraph

Three files chain into the training set, and the middle one is mandatory:

```
BD SSET (crop, year) ──CodigoSSET──► Grafica_Tabular ──COD_PREDIO──► QGIS polygons
```

`Grafica_Tabular` is the only file carrying both keys; fifteen exist and Callao yields nothing,
so 14 departments are linkable and that limit is structural. There are no sierra or selva
labels and there never will be. A fourth dataset, the 2012 agricultural census, links only by
the farmer's name.

Four traps in the raw data each return a plausible empty result instead of an error:
zero-padded keys, 19 shapefiles shipping their attribute table under the wrong name, one 3D
cadastre Earth Engine rejects, and departments whose rows are not in "their" workbook. Read raw
data through `allperu.sources`, which handles all four. [`docs/DATA.md`](docs/DATA.md).

---

## Reference

| | |
|---|---|
| [`docs/STATUS.md`](docs/STATUS.md) | what is done, what is closed, what to do next |
| [`docs/RESULTS.md`](docs/RESULTS.md) | every strand, its verdict, the numbers of record |
| [`docs/LESSONS.md`](docs/LESSONS.md) | what generalises beyond this project |
| [`docs/DATA.md`](docs/DATA.md) | datasets, linkage chains, the four traps |
| [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md) | how to obtain the data, and what runs without it |
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | module and command reference |
| [`docs/s2_labelling/`](docs/s2_labelling/plan.md) | the photo-interpretation campaign and its codebook |
| `reports/peru_report.tex` | the written-up narrative (PDF beside it) |
| [`docs/repo_layout.md`](docs/repo_layout.md) | why the repository is shaped like this |
| [`CLAUDE.md`](CLAUDE.md) | orientation for a coding agent |

Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md). The [`LICENSE`](LICENSE) covers the code
only — the data is not redistributed and is not covered by it.
