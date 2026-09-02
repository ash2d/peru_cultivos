# 4. Predict on parcels

Apply a trained model to parcels and get one row per parcel: the predicted class, the
probability behind it, and whether the model refused to answer.

You need a trained model first. `runs/` is not committed, so train one from
[`03_train_and_evaluate.md`](03_train_and_evaluate.md), or run
`uv run cc reproduce s2-model`, which fits the Sentinel-2 model of record in about a minute.

---

## A. Parcels the project already has features for

This is the quick case: the parcels are in a workspace and their imagery has already been
turned into features.

```bash
uv run cc -w demo predict runs/demo/demo --tau 0.5 --out preds.parquet
```

To score a subset, pass a file listing the parcel ids you want, in a column called
`COD_PREDIO`:

```bash
uv run cc -w demo predict runs/demo/demo --polygons my_parcels.parquet --out preds.parquet
```

## B. Parcels of your own

Say you have a shapefile of farm boundaries and want the model's reading of them for 2020.
Four steps.

**1. Turn the polygons into the table the pipeline reads.**

```bash
uv run python tools/prepare_parcels.py my_farms.shp --year 2020 --out data/mine
```

Shapefile, GeoPackage, GeoJSON or parquet all work. The file needs a coordinate system set.
Options: `--id-col` names the column identifying each parcel (otherwise rows are numbered),
and `--year-col` gives a different year per parcel instead of one `--year` for all of them.

The tool reports any parcel outside 0.09–50 ha. Those are kept, but the models were fitted
inside that range, so a prediction outside it is an extrapolation.

**2. Add a workspace for them** — one block at the end of `workspaces.yaml`:

```yaml
  mine:
    proc: mine
    feat: mine/features
    runs: mine
    about: my own parcels
```

Check it: `uv run cc workspaces`.

**3. Get imagery for them.** This needs an Earth Engine account
([`01_setup.md`](01_setup.md)) and is the slow step — minutes for a few hundred parcels,
hours for tens of thousands.

```bash
uv run cc -w mine satellite extract --stage all
uv run cc -w mine satellite assemble
```

`extract` first counts how many clear satellite dates each parcel has, drops the ones that
cannot support a measurement, then downloads pixels for the rest. It is resumable: if it
stops, run it again. More on it, and on the ways Earth Engine fails quietly, in
[`05_get_satellite_data.md`](05_get_satellite_data.md).

For the Sentinel-2 model, use section C below instead.

**4. Predict.**

```bash
uv run cc -w mine predict runs/all_peru/lightgbm_nometa_nolat_aug_yleak10 \
    --tau 0.5 --out preds.parquet
```

The model and the features must match: a model trained on Landsat features needs a Landsat
extraction, which is what `cc satellite extract` produces.

## C. Parcels read with the Sentinel-2 model

The Sentinel-2 model is the better one, and it reads a **season** (Aug–Jul) around a date you
choose, rather than a calendar year. Its features come from the labelling campaign's
extractor, so the steps differ slightly.

**1. Prepare the parcels, with the date you want read.** To score a sample of the project's
own national parcels for the 2024/25 season:

```bash
uv run python tools/prepare_parcels.py data/processed/all_peru/modeling_parcels.parquet \
    --for s2 --imagery-date 2025-03-01 --sample 500 --out data/mine2025
```

Your own polygons work the same way — pass the shapefile instead. `--for s2` also writes
`labels_s2/label_sample.parquet`, which is what the extractor reads.

**2. Add the workspace** to `workspaces.yaml`:

```yaml
  mine2025:
    proc: mine2025
    feat: mine2025/features
    runs: mine2025
    about: parcels to score with the S2 model
```

**3. Get the imagery and build the features** (Earth Engine; a few minutes for a few hundred
parcels):

```bash
uv run cc -w mine2025 labelling campaign extract
uv run cc -w mine2025 labelling campaign assemble
```

**4. Add the climate column, if your model was trained with `--climate`.** The model of record
was, so this step is needed for it — without it `cc predict` stops with "feature store is
missing 1 model features".

```bash
uv run python tools/prepare_parcels.py --attach-climate temp --out data/mine2025
```

The values come from the committed per-parcel normals, so this covers parcels that are in the
project's national table. For polygons from elsewhere, either build the normals over them
(`cc allperu climate normals`, which needs the 10 GB WorldClim rasters) or use a model trained
without `--climate`, such as `runs/s2_labels/ws_t3w_pilot/lightgbm`.

**5. Predict.**

```bash
uv run cc -w mine2025 predict runs/s2_labels/ws_t3w_pilot__clim_temp/lightgbm \
    --tau 0.5 --out preds2025.parquet
```

That path is the model of record. A model you trained yourself on a labelling round is under
`runs/labels_s2_<round>/ws_<target>[__clim_<arm>]/<model>` — for example
`runs/labels_s2_round2_all/ws_t3w__clim_temp/lightgbm`
([`06_label_more_parcels.md`](06_label_more_parcels.md)).

## Reading the output

```
COD_PREDIO        year  quality_ok  prob_ANNUAL  prob_PASTURE_FALLOW  prob_PERENNIAL  pred_label  pred_proba  abstained  abstain_reason
7_5109460_066795  2005  True              0.988                0.011           0.001  ANNUAL           0.988  False
```

To open it in Excel:

```bash
uv run python -c "import pandas as pd; pd.read_parquet('preds.parquet').to_csv('preds.csv', index=False)"
```

`--tau` is the confidence floor: below it the model abstains instead of guessing. `--tau 0`
(the default) always answers. Every abstention says why:

| `abstain_reason` | what happened | what to do |
|---|---|---|
| `low_confidence` | the best class scored below `--tau` | lower `--tau`, or accept the abstention |
| `quality_gate` | too few clear satellite dates for this parcel-year | try a different year |
| `coverage_unmeasured` | the coverage step has not run for this parcel | run `cc satellite extract --stage coverage` |
| `no_features` | the parcel has no features in the store | run the extraction for it |

## What a prediction is and is not

- **One year, not a trend.** The classifier is reliable for a single year. Four attempts to
  turn it into a per-parcel time series all failed their pre-registered checks, so do not
  build a trend by predicting each year and comparing.
- **Expect the out-of-department number.** On parcels in departments the model trained on,
  the Sentinel-2 model scores 0.774 macro-F1. On a department it has never seen, expect
  0.724 ± 0.112. Anywhere outside Peru's coast, expect less than either.
- **Areas and shares need weights.** Predictions on a sample are not population shares. The
  national sample doubles the perennial share by design, so any share or area figure must use
  `sample_weight`.

Next: [`05_get_satellite_data.md`](05_get_satellite_data.md).
