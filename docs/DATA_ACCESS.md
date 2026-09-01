# Getting the data

`docs/DATA.md` describes what each dataset **is**. This page is the other half: what you must
obtain, from whom, and what you can still do while you wait.

`data/` is gitignored and local-only (~20 GB raw, ~33 GB with everything derived). Cloning this
repository gives you the code and none of the inputs.

---

## 1. Can I run anything with no data at all?

Yes — the `demo` workspace:

```bash
uv run cc -w demo labels build
uv run cc -w demo train
uv run cc -w demo evaluate
```

It ships a small parcel sample with its satellite features already extracted, so it needs no raw
archive and no Earth Engine account. It exists to prove your install works and to let you read
the pipeline end to end in a few minutes.

⚠️ **Numbers from the demo workspace are not results.** It is far too small, and it is a single
convenience sample. Never quote a metric from it.

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
