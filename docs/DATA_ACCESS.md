# Getting the data

`docs/DATA.md` describes what each dataset **is**. This page is the other half: what you must
obtain, from whom, and what you can still do while you wait.

`data/` is gitignored and local-only (~20 GB raw, ~33 GB with everything derived). Cloning this
repository gives you the code and none of the inputs.

---

## 1. What does a clone already have?

**Most of it.** The tables needed to reproduce this project's published results are committed —
642 MB, 317 files. You do **not** need the raw archive and you do **not** need to run an Earth
Engine extraction to rerun the analyses. The direct proof of that:

```bash
uv run cc reproduce          # ⭐ re-derive every published headline number, ~90 s
uv run cc data verify        # what this clone can actually do, file by file
```

Every capability comes back as one of three things:

| | |
|---|---|
| `ok` | everything it needs is here |
| `derivable` | the gap is rebuildable by a command, which the output names |
| `needs-data` | the gap is the licensed archive or a fresh extraction — this page |

On a bare clone that is: the demo, the national Landsat model, the Sentinel-2 label set, the
PETT→CENAGRO comparison, the tenure difference-in-differences, and the Piura strand — all `ok`.
Only rebuilding from the raw archive, and extracting *new* imagery beyond the 56,419 parcels
already linked, need anything more.

Run it first, and again after obtaining anything. A missing input here usually does **not**
crash: four of the raw datasets return a plausible empty or column-less result instead of an
error (§4 of [`DATA.md`](DATA.md)), so "start the job and see" was never a check.

### What is committed

| | |
|---|---|
| national PETT parcels — **with the real polygons** — and Landsat features | 160 MB |
| CENAGRO 2012, the 14 linkable departments, plus the name crosswalk | 126 MB |
| Piura strand: 12-class and 3-class tables and features | 171 MB |
| tenure: per-parcel status, two-period, the DiD sample and its predictions | 107 MB |
| per-parcel climate covariates | 24 MB |
| the Sentinel-2 endpoint labels (865 parcels) and S2 features | 23 MB |
| LODO / LOYO / LODYO metrics, summaries and held-out predictions | 8 MB |

The allowlist is at the bottom of [`.gitignore`](../.gitignore), with a block saying what is
excluded and why.

### What is not, and whether it matters

| excluded | size | does it block anything? |
|---|---|---|
| the raw PETT archive (`BD_SSET`, `Grafica_Tabular`, `QGIS`) | 5.0 GB | only rebuilding the label tables from scratch — the derived tables are committed |
| WorldClim rasters | 10 GB | no — the per-parcel covariates extracted from them are committed |
| raw pixel stores, and the multi-year panels | ~6 GB | no for the published results; yes if you want to re-derive features from pixels |
| labelling-UI imagery (tiles, chips, HTML) | 4.0 GB | no — the labels themselves are committed |
| the other 11 CENAGRO departments | 93 MB | no — they have no PETT counterpart and cannot be linked |
| the full national parcel table (`all_peru_full/`, 726,808 parcels) | 437 MB | no — the two things built from it that the results use, the CENAGRO panel and the department × class population counts, are committed. A **new** labelling draw falls back to the 56,419-parcel table, which is a smaller frame, not a broken one |
| the INEI question-024 crop workbook (`.xlsx`) | 0.1 MB | no — the 3,351-code list it contains is committed as `Cenagro_IV/crop_code_table.csv`, which is what the code reads when the workbook is absent |

⭐ **`modeling_parcels.parquet` carries the real polygons** (56,419, EPSG:4326), so a fresh Earth
Engine extraction can be run from this repository alone. You do not need the 3 GB shapefile
archive to pull new imagery.

---

## 1b. The one real gap

The **multi-year panel extractions** are not committed and cannot be, at ~6 GB. That matters in
exactly one place: the difference-in-differences.

Its classifier *predictions* are committed, so the estimate reproduces from this clone. Rebuilding
those predictions from pixels is a ~13.5-hour Earth Engine job. `cc data verify` marks that
input `optional` and shows it as `○` rather than counting it as a gap.

---

## 2. The four raw inputs

| what | where it lives | size | needed for |
|---|---|---|---|
| **`BD_SSET/`** — crop registry, 8 workbooks, ~5.6 M rows | Peruvian land-titling programme (PETT/COFOPRI) archive, via the project's UDEP contacts | ~1 GB | any label build |
| **`Grafica_Tabular/`** — cadastral bridge, 15 `.dta` | same archive | ~0.5 GB | ⭐ **mandatory** — the only file carrying both keys |
| **`QGIS/<DEPT>/`** — 24 parcel shapefiles, ~2.9 M polygons | same archive | ~5 GB | any satellite extraction |
| **`Cenagro_IV/`** — 2012 agricultural census, 25 departments | UDEP OneDrive share `MARAVI MENESES CRISTIAN ADDERLY - Departamentos_IV_CENAGRO (sin posesionario)` | 17.5 GB raw → 0.21 GB extracted | the before/after perennial comparison only |

**These are not open data.** They were obtained under a research agreement with the Universidad
de Piura. If you are joining this project, ask the repository owner for access; do not
redistribute the raw files, and do not commit any of them (`.gitignore` already blocks `data/`).

Expected layout once you have them:

```
data/raw/
  BD_SSET/*.xlsx
  Grafica_Tabular/<Dept>.dta
  QGIS/<DEPT>/*.shp
  Cenagro_IV/*.parquet          # produced by `cc data cenagro-extract`, see §4
```

Then `uv run cc workspaces` will show `[ok]` against the directories you have.

---

## 3. What each dataset unlocks, and what dies without it

You do **not** need all four to do useful work.

| you have | you can run |
|---|---|
| nothing | the `demo` workspace; read the code and `docs/RESULTS.md` |
| bridge + registry | label builds, the declared-crop tables, the tenure cross-section |
| \+ shapefiles | ⭐ everything satellite: extraction, training, evaluation, LODO/LOYO |
| \+ CENAGRO | the PETT → 2012 paired perennial comparison and its tenure split |

⚠️ **The bridge is mandatory and it is why only 14 departments exist.** `Grafica_Tabular/` is the
only file carrying *both* `CodigoSSET` (crop side) and `COD_PREDIO` (polygon side). Fifteen files
exist, Callao yields nothing, so 14 departments are linkable and no amount of extra imagery
changes that. **There are no sierra or selva labels and there never will be** — that is a
property of the archive, not of the code.

---

## 4. Regenerating the CENAGRO extract

`data/raw/Cenagro_IV/*.parquet` is derived, not given. To rebuild it:

1. Mount the OneDrive share and set `cenagro_source_dir:` in `workspaces.yaml` to the folder
   holding the 25 `.dta` files.
2. **Warm the files first.** OneDrive Files-On-Demand ships them as *dataless placeholders*;
   every read streams at ~2.7 MB/s with `%CPU` sitting at **0.0**, which is indistinguishable
   from a hang. Force them local (`dd if=<file> of=/dev/null`, four in parallel) — a warmed
   department extracts in 5–15 s instead of 200–500 s.
3. Run it:

```bash
uv run cc data cenagro-extract                # all 25, ~1 h (mostly download)
uv run cc data cenagro-extract --verify       # the audit; writes _extract_audit.csv
```

17,750,195 rows, 409 columns → 76, 85× smaller, zero rows lost. The audit is the record of
that claim — read it, don't assume it.

---

## 5. Earth Engine

Any satellite extraction needs your **own** Google Cloud project — Earth Engine bills a project
and the one in this repository's history is not yours.

```bash
uv run earthengine authenticate
# then set `gee_project:` in workspaces.yaml to your project id
```

Sign-up and free research access: <https://earthengine.google.com/>. What extraction costs, and
the three ways Earth Engine fails silently, are in [`PIPELINE.md`](PIPELINE.md) §6.

---

## 6. Licence

The **code** in this repository is released under the terms in [`LICENSE`](../LICENSE). The
**data** is not covered by it and is not redistributed here. Any use of the PETT/COFOPRI or
CENAGRO files is governed by the agreement under which you obtained them.
