# Getting the data

[`DATA.md`](DATA.md) describes what each dataset **is**. This page says what you must obtain,
from whom, and what you can do while you wait.

`data/` is gitignored and local-only (~20 GB raw, ~33 GB with everything derived). Cloning
gives you the code and none of the raw inputs.

---

## 1. What a clone already has

**Most of what matters.** The tables needed to reproduce the published results are committed —
642 MB, 317 files. You do not need the raw archive and you do not need Earth Engine to rerun
the analyses.

```bash
uv run cc reproduce     # re-derive every published number, ~90 s
uv run cc data verify   # what this clone can do, file by file
```

`data verify` marks every capability `ok` (everything is here), `derivable` (a command rebuilds
it, and the output names the command) or `needs-data` (the licensed archive or a new
extraction). On a bare clone the demo, the national Landsat model, the Sentinel-2 labels, the
census comparison, the tenure estimate and the Piura strand are all `ok`.

Run it first, and again after obtaining anything. A missing input usually does **not** crash:
four of the raw datasets return a plausible empty or column-less result instead of an error
([`DATA.md`](DATA.md) §4), so "start the job and see" was never a check.

**Committed:**

| | |
|---|---|
| national PETT parcels with the real polygons, and Landsat features | 160 MB |
| CENAGRO 2012 for the 14 linkable departments, plus the name crosswalk | 126 MB |
| Piura strand: 12-class and 3-class tables and features | 171 MB |
| tenure: per-parcel status, two-period table, the DiD sample and its predictions | 107 MB |
| per-parcel climate covariates | 24 MB |
| the Sentinel-2 labels (865 parcels) and S2 features | 23 MB |
| LODO / LOYO / LODYO metrics and held-out predictions | 8 MB |

The allowlist is at the bottom of [`.gitignore`](../.gitignore).

**Not committed, and whether it blocks anything:**

| excluded | size | blocks |
|---|---|---|
| the raw PETT archive (`BD_SSET`, `Grafica_Tabular`, `QGIS`) | 5.0 GB | only rebuilding the label tables from scratch |
| WorldClim rasters | 10 GB | nothing — the per-parcel covariates are committed |
| raw pixel stores and the multi-year panels | ~6 GB | nothing published; only re-deriving features from pixels |
| labelling-UI imagery (tiles, chips, HTML) | 4.0 GB | nothing — the labels are committed |
| the other 11 CENAGRO departments | 93 MB | nothing — they have no PETT counterpart |
| the full national parcel table (726,808 parcels) | 437 MB | nothing — a new labelling draw falls back to the 56,419-parcel table, a smaller frame rather than a broken one |
| the INEI question-024 crop workbook | 0.1 MB | nothing — its 3,351-code list is committed as a CSV |

`modeling_parcels.parquet` carries the real polygons (56,419, EPSG:4326), so a fresh Earth
Engine extraction can run from this repository alone.

**The one real gap** is the multi-year panel extraction, ~6 GB, which cannot be committed. It
matters in one place: rebuilding the difference-in-differences predictions from pixels is a
~13.5-hour Earth Engine job. The predictions themselves are committed, so the estimate
reproduces here; `cc data verify` marks that input optional rather than missing.

---

## 2. The four raw inputs

| what | where it comes from | size | needed for |
|---|---|---|---|
| **`BD_SSET/`** — crop registry, 8 workbooks, ~5.6 M rows | the Peruvian land-titling (PETT/COFOPRI) archive, via the project's UDEP contacts | ~1 GB | any label build |
| **`Grafica_Tabular/`** — cadastral bridge, 15 `.dta` | same archive | ~0.5 GB | mandatory: the only file carrying both keys |
| **`QGIS/<DEPT>/`** — 24 shapefiles, ~2.9 M polygons | same archive | ~5 GB | any satellite extraction |
| **`Cenagro_IV/`** — 2012 census, 25 departments | UDEP OneDrive share | 17.5 GB raw → 0.21 GB extracted | the before/after perennial comparison |

**These are not open data.** They were obtained under a research agreement with the Universidad
de Piura. Ask the repository owner for access; do not redistribute or commit them.

Expected layout:

```
data/raw/
  BD_SSET/*.xlsx
  Grafica_Tabular/<Dept>.dta
  QGIS/<DEPT>/*.shp
  Cenagro_IV/*.parquet          # produced by `cc data cenagro-extract`, see §4
```

`uv run cc workspaces` then shows `[ok]` against the directories you have.

## 3. What each input unlocks

| you have | you can run |
|---|---|
| nothing | the `demo` workspace, `cc reproduce`, and the docs |
| bridge + registry | label builds, declared-crop tables, the tenure cross-section |
| \+ shapefiles | everything satellite: extraction, training, evaluation, LODO/LOYO |
| \+ CENAGRO | the PETT → 2012 perennial comparison and its tenure split |

**The bridge is why only 14 departments exist.** `Grafica_Tabular/` is the only file carrying
both `CodigoSSET` (crop side) and `COD_PREDIO` (polygon side). Fifteen files exist, Callao
yields nothing, so 14 departments are linkable and no amount of extra imagery changes that.

## 4. Regenerating the CENAGRO extract

1. Mount the OneDrive share and set `cenagro_source_dir:` in `workspaces.yaml`.
2. **Warm the files first.** OneDrive ships them as dataless placeholders; every read streams
   at ~2.7 MB/s with CPU at 0.0 %, which looks exactly like a hang. Force them local
   (`dd if=<file> of=/dev/null`, four at a time) — a warmed department extracts in 5–15 s
   instead of 200–500 s.
3. Run it:

```bash
uv run cc data cenagro-extract              # all 25, ~1 h, mostly download
uv run cc data cenagro-extract --verify     # the audit; writes _extract_audit.csv
```

17,750,195 rows, 409 columns down to 76, 85× smaller, zero rows lost. The audit is the record
of that claim — read it rather than assuming it.

## 5. Earth Engine

Any extraction needs your own Google Cloud project; Earth Engine bills a project and the id in
this repository's history is not usable by anyone else.

```bash
uv run earthengine authenticate
# then set `gee_project:` in workspaces.yaml
```

Free research access: <https://earthengine.google.com/>. What extraction costs and how it fails
quietly: [`howto/06_reference.md`](howto/06_reference.md).

## 6. Licence

The **code** is released under [`LICENSE`](../LICENSE). The **data** is not covered by it and is
not redistributed here; use of the PETT/COFOPRI or CENAGRO files is governed by the agreement
under which you obtained them.
