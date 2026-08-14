# Peru crop classifier — agent orientation

## Read these, in this order

1. **[`docs/STATUS.md`](docs/STATUS.md)** — what is done, what is closed, what to do next.
2. **[`docs/RESULTS.md`](docs/RESULTS.md)** — every strand, its verdict, the numbers.
3. **[`docs/PIPELINE.md`](docs/PIPELINE.md)** — module reference, CLI table, cookbook.
4. **[`docs/DATA.md`](docs/DATA.md)** — datasets, linkage chains, the four silent traps.

Also: [`docs/LESSONS.md`](docs/LESSONS.md) (method findings worth reusing),
[`docs/REPORT.md`](docs/REPORT.md) (the readable narrative),
[`docs/s2_labelling/plan.md`](docs/s2_labelling/plan.md) (the only live work).

**Do not put status updates or results in this file.** They go in `STATUS.md` / `RESULTS.md`.
This file is orientation only and is auto-loaded into every context.

---

## ⭐ The one result to know before modelling

`centroid_lat` is worth **+0.047 macro-F1 on spatial CV, +0.047 on leave-one-year-out, and
−0.060 on leave-one-department-out.**

Spatial CV holds out 5 km cells *inside departments the model has already seen*, and LOYO holds
region approximately fixed — so neither can separate spatial memorisation from real signal.
**Never select a feature or a model on CV alone.** Report **CV / LODO / LOYO / LODYO**; LODYO is
free (`allperu lodyo` re-scores existing LODO predictions).

The same lesson has three independent instances: `frac_l7` manufactured *change*, statics
manufacture *stability*, and `centroid_lat` manufactures *accuracy that does not leave the
training departments*.

---

## What this project is

Build a **crop classifier from satellite imagery for Peru**, to answer: *has land shifted from
domestic annual crops to export perennial crops, and does secure land title cause it?*

The training signal, per parcel: **(polygon boundary) + (declared crop) + (a year) → matched
satellite imagery → features → label.** The raw material is legacy Peruvian land-titling and
agricultural-census data, and a large part of the work was **forensic** — establishing what each
file is and how they link.

Three strands share one pipeline: a 12-class crop classifier (closed), a 3-class
perennial/annual/pasture classifier (works), and a national extension to 14 departments and
946,872 linked polygons. A Sentinel-2 endpoint-labelling campaign is the live work.

---

## Environment

* **Package manager: `uv`** — never bare `pip`/`python`. Project `crop-classifier`, Python
  **3.11**, `src/` layout, installed editable. Run everything with `uv run …`. Add deps with
  `uv add <pkg>`; never hand-edit `pyproject.toml` for deps.
* **GEE auth is required** for any extraction: `uv run earthengine authenticate`, plus a GCP
  project id (default `peru-crop-classifier`, hard-coded in `landsat_gee.py` — another user
  needs their own).
* Key libs: `geopandas`, `rasterio`, `rioxarray`, `xarray`, `shapely`, `pyproj`,
  `scikit-learn`, `lightgbm`, `torch`, `earthengine-api`, `geemap`, `contextily`, `pyreadstat`.
  Dev: `ruff`, `pytest`, `ipykernel`, `jupyterlab`.
* **Workspace env vars decide which dataset you touch** — `CC_PROC`, `CC_FEAT`, `CC_RUNS`. Get
  them wrong and you silently read or write the wrong store. Table in `PIPELINE.md` §2.
* `data/` (~20 GB), `runs/` (~780 MB) and `logs/` are gitignored and local-only.

---

## The data, in brief

Full detail in [`docs/DATA.md`](docs/DATA.md).

* **`data/raw/BD_SSET/`** — the crop registry, 8 workbooks, ~5.6 M rows. One row per parcel-crop
  declaration from the PETT land-titling programme, **not a census**. Key `Codigo SSET`; free
  text `CULTIVO`; `AREA` in **m²** and dirty; `FECHA EMPADRONAMIENTO` **unreliable**;
  `ESTADO en RRPP` = tenure at declaration.
* **`data/raw/Grafica_Tabular/<Dept>.dta`** — the cadastral bridge, 15 files. **Mandatory**: it
  is the only file carrying *both* `CodigoSSET` and `COD_PREDIO`. Also carries the cadastre's
  own `estado` + `fech_tran` (≈2011–12) — the second dated tenure observation.
* **`data/raw/QGIS/<DEPT>/`** — 24 parcel shapefiles, ~2.9 M polygons, key `COD_PREDIO`.
  Provenance is **PETT, not CENAGRO**, despite the filename.
* **`IV_CENAGRO_Piura.dta`** — the 2012 census, 947,884 × 409, 938 MB. Read with
  `encoding="latin1"` and **always an explicit `usecols`**. It has **no PETT code** — the only
  link is the farmer's name, in **unlabelled** columns `P009_01/02/03`.

**The linkage chain (Chain A) is the training set.** Both joins use real keys:

```
BD SSET (crop, year) ──CodigoSSET──► Grafica_Tabular ──COD_PREDIO──► QGIS polygons
```

**Only 14 departments are linkable** and that is structural — the bridge is mandatory and only
15 files exist, of which Callao yields nothing. There are **no sierra/selva labels and never
will be**.

---

## ⚠️ The four silent data traps

Every one returns a plausible empty or column-less result instead of an error. Each is pinned by
a test in `tests/test_allperu.py`. Read raw data through `allperu.sources`, which handles them.

1. **Bridge keys are zero-padded to 9 chars** (`030406693`) while BD SSET stores them unpadded.
   A string join returns **zero rows** and reads as "no data for this department". Ancash: 0 →
   369,089 keys. Fixed by `build_labels.canon_key`.
2. **19 of 24 shapefiles ship their attribute table under the wrong basename**
   (`QGIS/ANCASH/ANCASH.dbf`). GDAL opens them and returns **zero columns** — `COD_PREDIO`
   vanishes silently. Fixed by `sources.shapefile_view` (symlinks; raw data untouched).
3. **La Libertad's cadastre is 3D** and Earth Engine rejects 3D GeoJSON outright — 9.6 % of the
   sample, and it kills a multi-hour extraction partway through. Fixed by
   `build_labels.clean_geometry`.
4. **A department's rows are not confined to "its" workbook** (Lima's are in three). Fixed by
   `build_dept_sset_caches`, which scans every workbook once.

---

## Code layout

```
src/crop_classifier/
  paths.py                 workspace resolution (CC_PROC / CC_FEAT / CC_RUNS)
  crop_normalization.py    one dirty CULTIVO cell → list of (crop, category)
  build_training_data.py   Chain A end-to-end for Piura
  labels.py  splits.py     label policy; spatially blocked splits + autocorrelation audit
  data.py                  datasets, fold selection, feature exclusion
  train.py evaluate.py infer.py
  features/                landsat_gee, s2_gee, indices, assemble, s2_assemble
  models/                  lazy registry: trees (LightGBM), ltae, psetae, torch_common
  perennial/               3-class strand: labels3, panel, diagnostics (the gate), rules, …
  allperu/                 national: sources, build_labels, sample, lodo, loyo, tenure_did, …
  labelling/               chips, build_html, ingest  (the S2 campaign)
  config/                  data.yaml, split*.yaml, perennial*.yaml
```

One CLI for everything: `uv run python -m crop_classifier.cli --help`. Command table in
`PIPELINE.md` §3.

---

## Notebooks

Exploratory and historical; the pipeline in `src/` supersedes them for anything reproducible.
Paths inside them are relative to `notebooks/` (use `../data/raw/…`).

| notebook | what it is |
|---|---|
| `01_explore_raw_datasets.ipynb` | the original forensic exploration (81 cells): characterises every file, establishes the polygon↔crop join, validates it against year-matched Landsat / Sentinel-2 / Esri imagery, and settles the **PETT provenance** of the polygons |
| `02_merge_cenagro_sset_polygons.ipynb` | census↔PETT by farmer name (Chain B) |
| `03_pett_crop_polygon.ipynb` | the Chain-A training build, since scripted into `src/` |
| `04_inspect_parcel_basemaps.ipynb` | per-parcel inspector — enter a `COD_PREDIO`, get it and its neighbours on three basemaps, captioned by crop and year |
| `05_crop_label_cleaning.ipynb` | label-normalisation development |
| `06_perennial_trends.ipynb` | 3-class exploration |

---

## ⚠️ Gotchas

* **Never import torch and lightgbm in one process on macOS** — each bundles its own libomp and
  co-loading segfaults (exit 139). The lazy model registry exists for this. The test suite runs
  LightGBM training in a subprocess.
* **Always `uv run …`.** The notebooks need the project venv (Python 3.11).
* **Read every large `.dta` with `pyreadstat.read_dta(path, encoding="latin1")`**, and for
  `IV_CENAGRO` always pass `usecols` (938 MB / 409 cols). The census name lives in *unlabelled*
  columns — check values, not labels.
* **Shapefiles are EPSG:32717** (Piura); reproject to 4326 for GEE and 3857 for contextily. The
  country spans UTM 17S–19S, so the national split grid uses 32718.
* **`quality_ok` is a nullable boolean** (NA = unmeasured). Compare with `== True` / `.isna()`,
  never truthiness.
* **GEE fails silently in three ways**: hangs with no exception (fixed by a 900 s wall-clock
  deadline on a **daemon** thread), throttles that arrive *both* as error strings and as hangs,
  and content-addressed caches that skip new columns. **Verify a finished job by counting its
  output**, never by "the process ended".
* **Editing notebook cells programmatically**: cells contain triple-double-quoted docstrings, so
  build source strings with `'''…'''`. Never start a second `nbconvert --inplace` on a notebook
  while one is running — they share the Jupyter runtime and deadlock.
* **Any area or share figure must use `sample_weight`** — the national sample doubles the
  perennial share by design.

---

## Where results live

No numbers in this file except the one at the top. Everything else:

* **[`docs/STATUS.md`](docs/STATUS.md)** — current state, closed routes, next actions.
* **[`docs/RESULTS.md`](docs/RESULTS.md)** — the numbers of record, per strand, with verdicts.
* **[`docs/LESSONS.md`](docs/LESSONS.md)** — what generalises.
* `docs/figures/` — generated by `uv run python -m crop_classifier.perennial.report_figures`.
* `runs/all_peru/selected_model.json` — the current primary model of record.
