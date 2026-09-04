# Peru crop classifier

**Has Peruvian farmland moved from food crops grown for the home market to tree and vine crops
grown for export? And does a legal land title make that move more likely?**

Between 1997 and 2006, Peru's land-titling programme wrote down the crop growing on about a
million farm parcels, along with the shape of each parcel. Nobody went back to check what
happened since. Satellite images cover every year from 1996 on. This repository joins those two
records, trains a model to read crops from the images, and measures the change.

![perennial share over time, by tenure](docs/figures/perennial_over_time_by_tenure.png)

---

## What do you want to do?

| I want to | page |
|---|---|
| Install it. 15 minutes, once | [`docs/howto/01_setup.md`](docs/howto/01_setup.md) |
| **Get one table of every parcel and its crop type**, from all four records: what the farmer declared, the 2012 census, what a person saw in 2025 images, and what the model predicts for 2025 | [`docs/howto/02_parcel_table.md`](docs/howto/02_parcel_table.md) |
| **Run the model on parcels nobody has labelled.** A sample from all of Peru, or your own field boundaries | [`docs/howto/03_score_parcels.md`](docs/howto/03_score_parcels.md) |
| **Label more images and train a better model.** Written for someone who does not write code | [`docs/howto/04_label_and_train.md`](docs/howto/04_label_and_train.md) |
| **Get summary numbers:** share of land in tree crops, share of parcels with a title, and how both changed. For Peru, or one department at a time | [`docs/howto/05_summary_stats.md`](docs/howto/05_summary_stats.md) |
| Everything else: check the published numbers, train a model, download images, change the list of crop classes | [`docs/howto/06_reference.md`](docs/howto/06_reference.md) |

Quick start:

```bash
uv sync                 # install, Python 3.11
uv run cc reproduce     # recompute every published number, about 90 seconds
uv run cc data verify   # what this copy can do, and what is missing
uv run cc -w national analysis summary          # the main numbers, in one table
uv run cc -w national analysis parcel-table     # one row per parcel, all four records
uv run cc --help
```

The data needed to check the published results is included here: 642 MB of parcels, boundaries,
labels and satellite measurements. You do not need a Google Earth Engine account or the
restricted government files for that. [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md) lists what is
included and what is not.

---

## The answers

**The shift is real and large.** Between about 1999 and 2012, across 63,766 parcels in all 14
departments, tree and vine crops rose from 16.5 % to 26.4 % of parcels (+9.9 points) and from
21.0 % to 33.5 % of the land area (+12.5 points). Parcels moving into tree crops outnumber those
moving out by about 4 to 1, so this is not a small net figure hiding large movement both ways.
No satellite images and no model are used in this measurement. It compares two official records
of the same land.

**A land title does not appear to cause the shift.** Parcels with a title shifted slightly less
(−2.1 points, give or take 0.6), and measured in hectares the difference disappears
(+0.3 points). The formal estimate is close to zero, and its range is narrow enough to say so:
−0.0011, between −0.0126 and +0.0104.

**The model is reliable for one year at a time, not for a trend.** Four attempts to turn it into
a year-by-year history of each parcel all failed the checks written down in advance.

Every number, with its uncertainty and its caveats: [`docs/RESULTS.md`](docs/RESULTS.md).

---

## The current best model: Sentinel-2, 3 classes

The model is LightGBM. Its inputs are summaries of Sentinel-2 satellite images from 2019 on,
one set per parcel, plus the parcel's average yearly temperature. Its training labels come from
people looking at aerial pictures: 865 usable parcels across all 14 departments. Which parcels
would be used for training and which would be held back was decided before any labelling
started: 704 for training, 161 kept aside as a final test.

Classes (`t3w`): `ANNUAL`, `PERENNIAL` (tree and vine crops, including non-crop tree cover) and
`OTHER`.

Scores are macro-F1, an average accuracy that counts each class equally, so a rare class matters
as much as a common one. "Floor" is the score you would get by always guessing the most common
class. A score is only meaningful next to its floor.

| test | what was hidden from the model | macro-F1 | accuracy | `PERENNIAL` F1 | floor |
|---|---|---:|---:|---:|---:|
| Cross-validation (704 parcels) | 5 km blocks, inside departments it had seen | 0.762 ± 0.038 | 0.783 | 0.789 | 0.228 |
| Leave one department out | a whole department, each of the 14 in turn | 0.724 ± 0.111 | 0.789 | 0.777 | 0.228 |
| Final test (161 parcels) | areas set aside before labelling began | **0.774** [0.697, 0.845] | 0.789 | 0.780 | 0.219 |

Per class on the final test:

| class | precision | recall | F1 | n |
|---|---:|---:|---:|---:|
| `ANNUAL` | 0.938 | 0.600 | 0.732 | 25 |
| `OTHER` | 0.810 | 0.810 | 0.810 | 79 |
| `PERENNIAL` | 0.727 | 0.842 | 0.780 | 57 |

How to read this:

1. **The cross-validation estimate was honest.** The final test scored 0.774 against an estimate
   of 0.762, and the `PERENNIAL` score matched to within 0.01.
2. **0.774 applies to departments the model has already seen.** For a new department, expect
   0.724, give or take about 0.11.
3. **The final test has been used** (2026-09-01). Do not score anything on it again, or it stops
   being an independent check.
4. **We do not know how often the human labels themselves are wrong.** Most parcels were labelled
   by one person, so these scores measure agreement with those labels, not with the truth on the
   ground.

For comparison, the older Landsat model, trained on the 1997–2006 declarations, scores 0.628 in
cross-validation and 0.479 on a new department. It answers a slightly different question and has
far fewer clear images per parcel (13–24 a year against 47).

```bash
uv run cc reproduce s2-model     # refit it and compare against the published number
```

---

## The one thing to know before training anything

Giving the model the parcel's latitude adds 0.047 to its cross-validation score and takes 0.060
off its score on a department it has not seen.

Cross-validation hides small blocks of land, but the model has already seen the rest of that
department. So a high cross-validation score can mean the model has memorised where things grow,
not that it has learned to read the images. Never choose a feature or a model on
cross-validation alone. `uv run cc evaluate` prints four tests side by side, each next to the
score you would get by always guessing the most common class.

The same mistake was caught three times with single features, and once with a whole model type:
LTAE beat LightGBM in all 8 cross-validation runs and lost all 4 tests on unseen departments.
More: [`docs/LESSONS.md`](docs/LESSONS.md).

---

## The data, in short

Three files link together to make the training set. The middle one is required:

```
BD SSET (crop, year) ──CodigoSSET──► Grafica_Tabular ──COD_PREDIO──► QGIS polygons
```

`Grafica_Tabular` is the only file that carries both keys. Fifteen of these files exist and one
(Callao) has no usable rows, so only 14 departments can be linked. That limit cannot be lifted
with more work: the files do not exist. There are no labels for the highlands or the Amazon, and
there never will be. A fourth dataset, the 2012 agricultural census, can only be matched by the
farmer's name.

Four problems in the raw files each return an empty but believable result instead of an error:
keys padded with zeros on one side only, 19 map files that store their table under the wrong
name, one department's boundaries stored in 3D which Earth Engine refuses, and departments whose
rows sit in another department's spreadsheet. Read raw data through `allperu.sources`, which
handles all four. [`docs/DATA.md`](docs/DATA.md).

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
| [`docs/s2_labelling/`](docs/s2_labelling/plan.md) | the labelling work: how parcels were picked, and the rules labellers followed |
| `reports/peru_report.tex` | the written-up narrative (PDF beside it) |
| [`docs/repo_layout.md`](docs/repo_layout.md) | why the repository is shaped like this |
| [`CLAUDE.md`](CLAUDE.md) | orientation for a coding agent |

Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md). The [`LICENSE`](LICENSE) covers the code
only — the data is not redistributed and is not covered by it.
