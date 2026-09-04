# 2. The parcel table

One row per parcel, with what each record says was growing on it.

```bash
uv run cc -w national analysis parcel-table                    # about 10 seconds
uv run cc -w national analysis parcel-table --out parcels.csv  # a CSV you can open in Excel
```

This writes `data/processed/cenagro/parcel_table.parquet`. It contains no analysis, just the
four records side by side.

---

## What is in it

| column | what it is |
|---|---|
| `COD_PREDIO`, `dept`, `area_ha` | the parcel: its id, its department, its size in hectares |
| `tenure`, `frac_inscrito`, `reg_year` | whether the parcel was registered when the crop was declared, and in which year |
| `pett_year`, `pett_class` | **1. the crop the farmer declared**, and the year they declared it (1997–2006) |
| `cen_class`, `cen_sown_ha`, `cen_any_export`, `n_producers`, `link_confidence` | **2. the 2012 agricultural census** |
| `s2_label`, `s2_class`, `imagery_date`, `weight` | **3. what a person saw** looking at 2019 or later images |
| `s2_pred_label`, `s2_pred_class`, `s2_pred_proba`, `s2_pred_source` | **4. what the model predicts** from the same images |
| `n_observations` | how many of the four records this parcel actually has |

All four crop columns use the same four words, so you can compare any two of them directly, or
count how many parcels moved from one class to another:

```
PERENNIAL        a tree or vine crop: orchard, plantation, vineyard
ANNUAL           a crop replanted each year: rice, maize, cotton
WOODY_NON_CROP   trees that are not a crop
OTHER            farmable ground with nothing growing on it, and anything not farmland
```

`PERENNIAL` here does **not** include `WOODY_NON_CROP`. Only the two image-based records can
tell those apart. The declared crop and the census both record a crop name, so their parcels
are never `WOODY_NON_CROP`.

## Which parcels are included

```bash
uv run cc -w national analysis parcel-table --universe all
```

| `--universe` | parcels | census columns | needs |
|---|---|---|---|
| `linked` (the default) | **95,941**: those that have both a declaration and a 2012 census record | filled in for all of them | nothing |
| `all` | **726,808**: every parcel in the titling programme, 14 departments | filled in for 95,941 | the restricted government files |

The list of parcels is fixed by `--universe`, and the other records are matched onto it. A
column is left empty where a record has nothing for that parcel. `--universe all` gives you the
whole country. It needs
`data/processed/all_peru_full/modeling_parcels.parquet`, a 437 MB file that is not included
here ([`../DATA_ACCESS.md`](../DATA_ACCESS.md)).

## Four things to check before you calculate anything

- **The image columns are mostly empty, and that is expected.** 865 parcels have a human label,
  and only 157 of them are also in the census-linked set. The labelling was drawn from the much
  larger set of declared parcels, so the overlap is a coincidence. Filter on `n_observations`
  rather than assuming a column is filled.
- **Use the `weight` column for any share based on the image columns.** The labelling
  deliberately picked about three times as many tree-crop parcels as the country has, so an
  unweighted share is about three times too high. `weight` is empty for the 120 parcels of the
  first trial batch, which were picked differently.
- **`s2_pred_source` tells you whether a prediction is a fair test.** The model was trained on
  these parcels, so the column is only filled from predictions made on parcels that were held
  back during training (`out_of_fold`, `locked_test`). A prediction you supply yourself with
  `--preds` is marked `applied`. A model scoring parcels it was trained on always looks much
  better than it is.
- **Crop names are not in this table, only the four classes.** The declared crop names need a
  417 MB file and the census crop names need the restricted files. Neither is included.

## Adding 2025 predictions for your own parcels

Score them first ([`03_score_parcels.md`](03_score_parcels.md)), then pass the result in:

```bash
uv run cc -w national analysis parcel-table --preds data/predict_s2/predictions.parquet
```

## The other set of classes

`--classes 3` uses the three classes of the declared data instead: `PERENNIAL`, `ANNUAL` and
`PASTURE_FALLOW`. There is no separate class for non-crop trees, so those parcels are reported
twice: once left out (`s2_class`) and once counted as tree crop (`s2_class_woody_perennial`).
That choice moves the 2025 result by 26 points, against a census change of about 10 points, so
if it matters to your question, report both.

```bash
uv run cc -w national analysis parcel-table --classes 3
```

With `--classes 3` the prediction column comes from a model trained on three classes, which
counts non-crop trees as `PERENNIAL`. So under `--classes 3` the model's `PERENNIAL` column
matches `s2_class_woody_perennial`, not `s2_class`.

Next: [`05_summary_stats.md`](05_summary_stats.md) turns these columns into shares and changes.
