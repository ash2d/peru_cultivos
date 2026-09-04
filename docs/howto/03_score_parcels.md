# 3. Run the model on parcels nobody has labelled

One command. It picks the parcels, downloads the Sentinel-2 images, turns them into numbers,
and predicts a crop type for each parcel.

```bash
uv run cc predict-s2 --n 500                        # 500 random parcels from across Peru
uv run cc predict-s2 --parcels my_farms.shp --n 0   # your own boundaries, all of them
```

You need a Google Earth Engine account ([`01_setup.md`](01_setup.md)). Downloading the images
is the slow part: a few minutes for a few hundred parcels, hours for tens of thousands. The
result is written to `data/predict_s2/predictions.parquet`.

---

## What you get

```
COD_PREDIO        year  quality_ok  prob_ANNUAL  prob_OTHER  prob_PERENNIAL  prob_WOODY_NON_CROP  pred_label  pred_proba  abstained
7_5109460_066795  2025  True              0.988       0.008           0.003                0.001  ANNUAL           0.988  False
```

`pred_label` is the model's answer and `pred_proba` is how sure it is, from 0 to 1. `abstained`
is `True` when the model gave no answer at all; the table at the bottom of this page says why
that happens. The command also prints how many parcels fell into each class.

To open the file in Excel:

```bash
uv run python -c "import pandas as pd; pd.read_parquet('data/predict_s2/predictions.parquet').to_csv('preds.csv', index=False)"
```

To put these predictions next to what was declared on the same parcels in the 1990s and
recorded in the 2012 census:

```bash
uv run cc -w national analysis parcel-table --preds data/predict_s2/predictions.parquet
```

## The options

| option | default | what it does |
|---|---|---|
| `--n` | 500 | how many parcels to pick. `0` means all of them |
| `--parcels` | none | your own file of boundaries: shapefile, GeoPackage, GeoJSON or parquet. Leave it out to sample Peru's own parcels |
| `--date` | `2025-03-01` | the date to look at. The model reads the farming year around it, August to July |
| `--tau` | 0 | if the model is less sure than this, it says nothing instead of guessing |
| `--model` | the four-class model | the folder of a model, for example one you trained on [page 4](04_label_and_train.md) |
| `--work` | `data/predict_s2` | working folder for the parcels, images and numbers |
| `--resume` | off | go straight to predicting, because the images in `--work` are already downloaded |

Without `--parcels`, the command picks at random from the 726,808 parcels in the titling
records, leaving out any parcel that already has a human label. Your own boundaries must have a
coordinate system set. Use `--id-col` to say which column names each parcel; without it the
rows are just numbered.

## What a prediction can and cannot tell you

- **One year, not a trend.** The model is reliable for a single year. Four attempts to turn it
  into a year-by-year history of a parcel all failed the checks written down in advance. Do not
  build a trend by predicting several years and comparing them.
- **Expect the lower score outside familiar ground.** In departments the model was trained on
  it scores 0.774 (macro-F1, where 1.0 is perfect). In a department it has never seen, expect
  about 0.724, give or take 0.11. Outside the Peruvian coast, expect less than either.
- **Shares need care.** A random sample of the national records represents the country. Any
  other set of parcels does not, and the project's own national training sample deliberately
  contains twice as many tree-crop parcels as the country does.
- **Very small and very large parcels are guesses.** The model was trained on parcels between
  0.09 and 50 hectares. It still predicts outside that range, and the command tells you how
  many such parcels there were.

## When something goes wrong

| what you see | what it means |
|---|---|
| `abstained` with `low_confidence` | the model was less sure than your `--tau`. Lower it, or accept the blank |
| `abstained` with `no_features` | the parcel got no usable images. Check how many rows the download produced |
| `feature store is missing 1 model features` | the model needs a temperature column. Keep the default `--climate temp`, or use a model trained without it |
| `none of these parcels is in parcel_climate_normals` | your boundaries are outside the project's temperature table. Use `--climate none` together with `--model runs/s2_labels/ws_t4_pilot/lightgbm` |
| Earth Engine seemed to finish but nothing appeared | it did not finish. Count the rows it produced. See [`06_reference.md`](06_reference.md) |

If you need to run the five underlying steps separately, they are in
[`06_reference.md`](06_reference.md).
