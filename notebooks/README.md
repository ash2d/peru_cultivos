# Notebooks

`04_inspect_parcel_basemaps.ipynb` is a **live tool**: enter a `COD_PREDIO`, get that parcel and
its neighbours on three basemaps, captioned by crop and year. It is the quickest way to see what
the polygons actually look like. Its large-parcel contact sheet is checked in as
[`../docs/figures/big_parcels_grid.png`](../docs/figures/big_parcels_grid.png).

Everything in [`exploratory/`](exploratory/) is **historical**. It is the forensic record of how
the datasets were characterised and how the joins were established, and it is worth reading for
that — but the pipeline in `src/` supersedes it for anything reproducible. Where a notebook and
`src/` disagree, `src/` is right.

| notebook | what it is |
|---|---|
| `exploratory/01_explore_raw_datasets.ipynb` | the original forensic exploration, 81 cells: characterises every raw file, establishes the polygon↔crop join, validates it against year-matched Landsat / Sentinel-2 / Esri imagery, and settles the **PETT provenance** of the polygons (the filenames say CENAGRO; they are wrong) |
| `exploratory/02_merge_cenagro_sset_polygons.ipynb` | census↔PETT linkage by farmer name — "Chain B" |
| `exploratory/03_pett_crop_polygon.ipynb` | the Chain-A training build, since scripted into `src/crop_classifier/build_training_data.py` |
| `04_inspect_parcel_basemaps.ipynb` | the per-parcel inspector (live) |
| `exploratory/05_crop_label_cleaning.ipynb` | label-normalisation development |
| `exploratory/06_perennial_trends.ipynb` | 3-class exploration |

## Running them

```bash
uv run jupyter lab
```

They need the project venv (Python 3.11), so `uv run`, not a system Jupyter. Paths inside them
are relative to this directory — `../data/raw/…` from `04`, `../../data/raw/…` from
`exploratory/`.

**Editing cells programmatically**: cells contain triple-double-quoted docstrings, so build
source strings with `'''…'''`. And never start a second `nbconvert --inplace` on a notebook while
one is running — they share the Jupyter runtime and deadlock.
