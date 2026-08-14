# Peru crop classifier — project context

> Handoff/status doc for a coding agent. Read this first to understand the goal, the data, and
> exactly where the work stands.
>
> **⚠️ 2026-08-08 — this file is titled "Piura" for history; the project is now NATIONAL.**
> Three strands share one pipeline: the 12-class Piura crop classifier, the 3-class
> perennial/annual/pasture classifier + its 28-year Piura panel (**gate failed**), and
> **all of Peru** — 14 departments, 946,872 linked polygons. Sections 1–7 below describe the
> Piura data forensics and remain accurate for Piura. The national strand is summarised in
> §8 and documented in **[`docs/all_peru/`](docs/all_peru/)**.
>
> **The one result to know before doing any modelling here:** `centroid_lat` is worth
> **+0.047 macro-F1 on spatial CV and −0.060 on leave-one-department-out**. Spatial CV holds
> out 5 km cells *inside departments the model has already seen*, so it cannot separate
> spatial memorisation from real signal. Do not select features or models on CV alone.
>
> **The modelling pipeline (labels → spatial splits → GEE Landsat
> extraction → features → train/eval/infer for LightGBM + LTAE + PSE-LTAE) — its code guide and
> current state live in
> [`docs/PIPELINE.md`](docs/PIPELINE.md); read that alongside this file.** This doc covers the
> upstream data forensics + label build that the pipeline starts from. Last updated after adding the
> modelling pipeline; earlier milestones still current: notebook 01's §11 provenance re-check settled
> that the `qgis_stefany` polygons are a **PETT cadastre** (not CENAGRO census geometry) and re-ran
> the crop↔polygon linkage with the wider bridge; merge notebooks 02 (census↔PETT by farmer name) and
> 03 (PETT crop→polygon, the recommended training set) are also current; and notebook 03 was scripted
> into the reusable `src/crop_classifier/` label pipeline (`build_training_data.py` +
> `crop_normalization.py`) that cleans the free-text crop labels into a normalised **list of crops per
> polygon** (see §5b).

## 1. What this project is

Build a **crop classifier from satellite imagery for Piura, Peru**, with **Google Earth Engine
(GEE)** integration. The intended training signal is, per farm parcel:

> **(polygon boundary) + (declared crop) + (a date/year) → a matching satellite image → features → label**

The raw material is a set of **legacy Peruvian land-titling / agricultural-census datasets**. A
large part of the work so far has been *forensic*: figuring out what each legacy file actually is,
how they link, and whether they can be assembled into `(polygon, crop, year)` training examples.

There are now **four notebooks**:
- **`notebooks/01_explore_raw_datasets.ipynb`** (81 cells) — the original forensic exploration:
  characterises the data, establishes the polygon↔crop join, validates it against year-matched
  Landsat / Sentinel-2 / sub-metre Esri imagery (§§7–8, 10), analyses crop-per-parcel and neighbour
  adjacency (§9), and — §11 — re-verifies the polygon **provenance** (PETT, not CENAGRO) and re-runs
  the linked-crop-year analysis with the wider `grafica_tabular_Piura.dta` bridge.
- **`notebooks/02_merge_cenagro_sset_polygons.ipynb`** — merges the **2012 CENAGRO census** to the
  PETT side **on the farmer's name** (the census has no shared code), producing an enriched
  parcel table + a comparison against the prior merge `Base_Cenagro_PETT_Piura.dta`.
- **`notebooks/03_pett_crop_polygon.ipynb`** — the **recommended training-data build**: the clean
  PETT-only `(geometry, crop, year)` extraction using **`grafica_tabular_Piura.dta`** as the bridge.
- **`notebooks/04_inspect_parcel_basemaps.ipynb`** — an interactive **per-parcel inspector**: enter a
  `COD_PREDIO` and it reproduces notebook 01 §8 for that parcel — the parcel + its neighbours drawn on
  three basemaps (year-matched Landsat, earliest Sentinel-2, recent Esri), each outlined and captioned
  by its normalised crop(s) and reported year. Reads `training_crop_polygon.parquet` (see §5b).

There is now also a **scripted pipeline** in `src/crop_classifier/`: the reproducible label-cleaned
build (§5b, the `training_crop_polygon.parquet` version of notebook 03) **plus a full modelling
pipeline on top of it** — label policy, spatial splits, GEE Landsat extraction, feature assembly,
and train/eval/infer for three models. **[`docs/PIPELINE.md`](docs/PIPELINE.md) is the authoritative
guide to that modelling code and its current state**; §5b here covers only the label build it starts
from.

**(Stale line removed 2026-08-08: models HAVE been trained on real data, at both Piura and
national scale — see §8 and `docs/all_peru/RESULTS.md`.)** For a Piura classifier, start from
`data/processed/training_crop_polygon.parquet` (one row per polygon, geometry + a normalised
**list of crops** + year); for a national one, `data/processed/all_peru/modeling_parcels.parquet`.
See `docs/DATASETS.md` for a plain-language summary of every dataset, each merge and its yield.

## 2. Environment & how to run

- **Package manager: `uv`** (do not use bare `pip`/`python`). Project is `crop-classifier`,
  Python **3.11**, `src/` layout (`src/crop_classifier/`, installed editable).
- Run anything with `uv run …` (e.g. `uv run python …`, `uv run jupyter …`). Deps are in
  `pyproject.toml` + locked in `uv.lock`. Add deps with `uv add <pkg>` (never edit pyproject by
  hand for deps).
- Key libs: `geopandas`, `rasterio`, `rioxarray`, `xarray`, `shapely`, `pyproj`, `scikit-learn`,
  `earthengine-api`, `geemap`, `contextily`, `matplotlib`, `seaborn`, `openpyxl`, `pyreadstat`.
  Dev group: `ruff`, `pytest`, `ipykernel`, `jupyterlab`.
- **GEE auth is required for Sections 8c/8d (Landsat/Sentinel-2).** Run once:
  `uv run earthengine authenticate`. The notebook initialises with
  `ee.Initialize(project="peru-crop-classifier")` (hard-coded in cell 8a — a different user needs
  their own authenticated GCP project id there).
- **Running notebook 01 takes ~5–6 min** and needs internet: it fetches small-AOI basemap tiles
  (§§5,7,8e,10) and several GEE composites (§8). Tiles are **not** cached between runs. Run headless:
  `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_explore_raw_datasets.ipynb`
  (there is no partial-execute; nbconvert always runs top-to-bottom). The imports cell sets a global
  `socket.setdefaulttimeout(90)` so no tile/GEE fetch can hang the run.
- **Do NOT launch a second `nbconvert` on the same notebook while one is running.** Two concurrent
  `--inplace` runs share the Jupyter runtime and deadlock (this cost ~30 min once). Before launching,
  `pkill -f 'jupyter-nbconvert.*01_explore'` and confirm
  `ps aux | grep -c '[.]venv/bin/python3.*nbconvert'` is 0. A per-cell timeout
  (`--ExecutePreprocessor.timeout=300`) is a good safety net, but note a *timeout* kills the kernel
  (unlike `--allow-errors`, which only tolerates cell *exceptions*).

## 3. Data inventory — `data/raw/`

**None of these are in git-friendly form; treat `data/` as local-only** (it is gitignored).

> **2026-08-07 — the three national folders supersede the Piura paths below.**
> `data/raw/BD_SSET/` (8 workbooks), `data/raw/Grafica_Tabular/` (15 bridge `.dta`),
> `data/raw/QGIS/<DEPT>/` (24 shapefiles). **The Piura members are byte-identical (sha256)**
> to the files described below, so nothing here is invalidated — but new work should read
> the national folders through `crop_classifier.allperu.sources`, which handles four traps
> that all fail *silently* (zero-padded keys, `.dbf` under the wrong basename, 3D geometry,
> departments split across workbooks). See [`docs/all_peru/DATA_AUDIT.md`](docs/all_peru/DATA_AUDIT.md).

### `BD SSET(MOQUEGUA-PASCO-PIURA).xlsx`  — the crop registry
- Single sheet `DATOS`, **509,684 rows**, covers Moquegua/Pasco/**Piura (~348k)**.
- This is a **SSET/PETT land-titling crop registry**, one row per parcel-crop declaration — *not*
  a population census.
- Key columns: `Codigo SSET` (parcel key, **the join key to the crop side**), `CULTIVO` (free-text
  crop label; ~42% null in Piura), `AREA` (in **m²**, has negatives/outliers — not hectares),
  `FECHA EMPADRONAMIENTO` (registration date, **unreliable** — see caveats), `ESTADO en RRPP`.
- It is a **panel**: a parcel reappears across years (mean 1.8 records/parcel, up to 27).

### `grafica_tabular_Piura.dta`  — the cadastral **bridge** ⭐ USE THIS ONE
- Stata file, **158,427 rows × 48 cols**, Piura only. Read with `pyreadstat.read_dta(path,
  encoding="latin1")`.
- Carries **both keys** — `COD_PREDIO` (e.g. `7_6359485_001871`) **and** `CodigoSSET` — so it is the
  bridge that connects polygons to crops. **This is the best bridge**: it covers **68.8%** of SSET
  crop keys and **strictly supersedes** `grafica_tabular_catastro_Piura.dta` (catastro's pairs are a
  subset — it adds 0 pairs). Using it instead of catastro recovers **+12,842 crop records /
  +8,392 polygons** (notebook 03).
- (Earlier docs said "ignore this file" — that was wrong; it is now the primary bridge.)

### `grafica_tabular_catastro_Piura.dta`  — smaller, older bridge (superseded)
- Stata file, **109,796 rows × 65 cols**, Piura only. Same two keys (`COD_PREDIO`, `CodigoSSET`) but
  covers only **59.8%** of SSET crop keys. Notebook 01 used this one; notebooks 02/03 replaced it
  with `grafica_tabular_Piura.dta`. Has centroids (`centroid_e/n`, PSAD56/UTM 17S) but no polygon
  geometry and no crop (`cod_uso` 100% null). Keep only for reference.

### `IV_CENAGRO_Piura.dta`  — the 2012 agricultural census (the big one)
- Stata file, **947,884 rows × 409 cols**, Piura. Read with `encoding="latin1"` + explicit `usecols`
  (938 MB — never pull all 409 columns). **Long format**: one row per *parcel × crop*.
- Uses census question codes `P0xx`. Key columns: `P001`=dept, `P002`=prov, `P003`=dist,
  `P007X`=SEA, `P008`=UA, `NPRIN`=cédula (productor id), `NPARC`=parcel; crop code `P024_03` (→
  name via the dictionary below), sown area `P025`; `LONG_DECI`/`LAT_DECI` = **SEA-centroid** coords
  (coarse — shared by all UAs in a SEA, *not* a parcel location).
- **The farmer name is here but the columns are unlabelled**: `P009_01`=apellido paterno,
  `P009_02`=apellido materno, `P009_03`=nombres (given names, often two), `P009_04`=razón social
  (companies/heirs), `P009_05`=RUC. **No `COD_PREDIO`, no `CodigoSSET`, no DNI number** → the *only*
  link to PETT is the **name** (see §4).

### `IV CENAGRO - Tabla_Cultivos_Totales.xlsx`  — the census crop-code dictionary
- Sheet `Permanente`, question **Preg. 024**, columns `CODIGO`/`TITULO`/`Exportable` (skip the
  header rows). Maps `P024_03` codes → crop names (3,351 codes, covers 100% of census crop codes).

### `Base_Cenagro_PETT_Piura.dta`  — a PRIOR census↔PETT merge (reference / comparison target)
- Stata file, **142,348 rows × 469 cols**, `encoding="latin1"`. One row per census productor
  (`NPRIN`), carrying census `P0xx` codes **plus** appended PETT fields (`COD_PREDIO`, `CodigoSSET`,
  `NOMBRE`, `DNI`, `CULTIVO`, `FECHAEMPADRONAMIENTO`, `_merge`, `PETT`). `PETT==1` = 82,914 matched
  productors → 26,028 distinct predios; `PETT==0` = 59,434 unmatched. Its `NOMBRE` equals the census
  `P009_01+02+03` (~93% exact) → **the prior merge was itself name-based**. Used in notebook 02 only
  as a comparison/union target; it carries **no geometry**.

### `qgis_stefany/`  — the parcel polygons (the missing geometry, added mid-project)
- **`CATASTRO_CENAGRO_PIURA_WGS84_Z17S_FINAL.shp`** (+ `.shx/.dbf/.prj/.cpg/.qix/.shp.xml`):
  **190,098 parcel polygons**, CRS **EPSG:32717 (WGS84 / UTM 17S)**. Read with `gpd.read_file`.
  Key `COD_PREDIO` (179,608 non-null; **10,476 have a null code → unlinkable, drop them**).
- **Provenance = PETT, not CENAGRO** (settled in notebook 01 §11 despite the "CENAGRO" in the
  filename): the `.shp.xml` lineage shows the layer was `CopyFeatures`-d from a pre-existing
  `PIURA.shp` into a `CENAGRO.gdb` container, feature class **`CAT_PIURA`** ("Catastro" = cadastre).
  The **census fields `COD_INEI` and `N_SEA` are 100% empty** while the **PETT fields
  (`COD_PREDIO` 94.5%, `NUM_PREDIO` 99.3%, `COND_JUR` 100%) are populated**, and 27,593 `NOMBRE='SN'`
  match the lineage's placeholder step. So the geometry is a **PETT/COFOPRI land-titling cadastre**
  packaged into the 2012 CENAGRO geodatabase as a base layer — the `2012-04-10` date is the
  *packaging* date, not a survey date (PETT titling predates it). `COD_PREDIO` keys ~99% into the
  PETT bridge tables. (The genuinely CENAGRO census data is the separate `IV_CENAGRO_Piura.dta`.)
- **`qgis_stefany/PIURA.dta`**: the CENAGRO attribute table, **158,638 rows**, must be read with
  **`encoding="latin1"`**. Companion (attribute table) of the shapefile — its `COD_PREDIO` just
  duplicates the shapefile's. **Has no `CodigoSSET` → useless for bridging crops to polygons** (it
  cannot recover any records catastro/`grafica_tabular_Piura` miss). Confirmed dead-end.

## 4. The linkage chains (two of them)

**Chain A — PETT crop → polygon (reliable, real keys). This is the training set (notebook 03).**
```
BD SSET (crop, FECHA→year)  ──CodigoSSET──►  grafica_tabular_Piura.dta  ──COD_PREDIO──►  qgis polygons
                                             (bridge: CodigoSSET↔COD_PREDIO)             (geometry)
```
- **116,599 crop records reach a polygon, over 66,363 distinct polygons** (68,717 distinct
  `polygon × crop × year`). All joins use real keys — no name-matching.
- BD SSET has no `COD_PREDIO` and the shapefile has no `CodigoSSET`, so the **bridge is mandatory**;
  use `grafica_tabular_Piura.dta` (68.8% coverage), not catastro (59.8%).
- **Crop years are ~1998–2007** (bulk 1998–99), a handful later. Re-verified in §11 with the wider
  bridge: it recovers **+12,842 records / +8,392 polygons** but **does not extend the years** (2015+
  only 36→40 records), so Sentinel-2 stays infeasible and Landsat 5/7 is the only year-matched
  sensor. And — see §6 — **almost no polygon has crops in more than one year** (1 of 66,363). Treat
  as *one ~1998–99 crop label per polygon*.

**Chain B — CENAGRO census → PETT, on the farmer NAME only (notebook 02).**
```
IV_CENAGRO name (P009_01/02/03)  ──normalised name──►  SSET / catastro / polygon NOMBRE(S)
(2012 census crop P024_03)                             (→ CodigoSSET / COD_PREDIO / geometry)
```
- The census shares **no code** with PETT, so the name is the only bridge. Names are normalised into
  components (`name1..4`, Peru has ≥2 surnames + ≥2 given names) and matched on several routes
  (exact / token-set / core). Recovers **45,942 productors** to a polygon on its own; **union with
  `Base` = 91,636** (> Base's 82,914).
- **Caveat:** there is no parcel-level key, so pinning the *exact* polygon is uncertain (~43% exact
  `COD_PREDIO` agreement with `Base`, even at high confidence). The name link is good for the
  *person/district/crop*, weak for the *exact parcel*. Adds a 2012 crop declaration but at lower
  reliability than Chain A.

## 5. Notebook walkthrough

### `notebooks/03_pett_crop_polygon.ipynb` ⭐ (the recommended training-data build)
- Chain A above. Compares the two bridges (proves `grafica_tabular_Piura` ⊃ catastro), proves
  `qgis/PIURA.dta` is useless, builds `(COD_PREDIO, crop, year)`, checks the multi-year reality.
- Outputs (in `data/processed/`): `pett_polygons.parquet` (66,363 geometries, EPSG:4326),
  `pett_crop_year.parquet` (116,599 records), `pett_crop_year_distinct.parquet` (68,717 deduped).
  Join crop tables to polygons on `COD_PREDIO`.

## 5b. The scripted pipeline — `src/crop_classifier/` ⭐ (start here for modelling)

The reproducible, label-cleaned version of notebook 03's Chain A. Run it with:
`uv run python -m crop_classifier.build_training_data` (`--no-save` to dry-run). Two modules:

- **`crop_normalization.py`** — turns one dirty free-text `CULTIVO` cell into a **list of
  `(crop, category)` pairs**. `normalize_label(raw)` is the public entry point. It splits multi-crop
  cells on every separator seen (`, / + & -`, whole-word `Y`/`CON`, and **percentage weightings** —
  `CAFE 50%-PLATANO 50%`, `PLATANO (30%) CAFE (30%)…` all split), fixes typos/accents/plurals and
  documented regional synonyms via a curated `CANON`/`_SYNONYMS` map (e.g. `AROZ→ARROZ`,
  `ALGODONERO→ALGODON`, `MERKERON→MELQUERON`), and tags each token `crop` / `pasture` / `fallow` /
  `land_prep` / `unspecified`. **Nothing is dropped by crop type** — non-crops are only *flagged*.
  Genuinely distinct crops are never merged (bean varieties `FRIJOL DE PALO`/`FRIJOL CAUPI` stay
  separate). Compresses 9,394 raw Piura labels → ~1,647 canonical tokens. Tests:
  `tests/test_crop_normalization.py` pin the tricky cases.
- **`build_training_data.py`** — runs the SSET→bridge→polygon join (same real keys as notebook 03,
  `grafica_tabular_Piura.dta` bridge), applies the normaliser, and aggregates to **one row per
  polygon**. Cleans errors: implausible `year` (outside 1990–2020) nulled, non-positive `area_m2`
  nulled, keys stripped, duplicate/null-code polygons dissolved.

Outputs (in `data/processed/`):
- **`training_crop_polygon.parquet`** ⭐ — 66,352 polygons, EPSG:4326. Columns: `COD_PREDIO`,
  `geometry`, `crops` (**list** of normalised names), `crop_categories` (aligned list), `n_labels`,
  `is_vegetated`, `codigo_sset` (list), `year` (modal) + `years` (list), `n_records`, `area_ha`.
- **`training_crop_records.parquet`** — 80,618 rows, the exploded long form, one row per
  `(COD_PREDIO, CodigoSSET, crop, category, year)`, keeps the raw label. **Deduped**: repeated
  same-crop re-declarations collapse (136,012→80,618, i.e. 55,394 dup rows removed); `n_records` on
  the polygon table preserves the raw count. Same crop in a *different* year would be kept (year is in
  the dedup key), but in this data that never occurs (0 of 80,618) — see §6 "~1 crop-year per polygon".
- **`crop_normalization_map.csv`** — every distinct raw label → its normalised crop list + categories
  + row count. The full audit trail of cleaning decisions.

**Not yet applied here (deliberately):** no crop is dropped and intercrops are kept as a list — the
fallow/land-prep filter and any intercrop policy are downstream modelling choices (filter on
`crop_categories` / `is_vegetated`). Latest linked crop-year is **2019** (4 records), but ~80% is
1998–99; the year is a titling date, not a verified season.

### `notebooks/04_inspect_parcel_basemaps.ipynb` (interactive per-parcel viewer)
- Set `RECORD_ID` to a `COD_PREDIO` (+ optional `HALF_WIDTH_DEG`), *Run All* → a 1×3 figure of that
  parcel **and its neighbours** on year-matched **Landsat-5/7**, earliest **Sentinel-2**, and recent
  **Esri** imagery, each parcel outlined+captioned by its normalised **crop list** and **reported
  year**, with a colour legend and a per-parcel summary table. `inspect_parcel("<COD_PREDIO>")` re-runs
  for any id. Reuses the §8 basemap fetchers (`landsat_rgb`, `sentinel2_rgb`); **needs GEE auth** for
  the two satellite panels (degrades gracefully to Esri-only without it). Reads the §5b polygon table.

### `notebooks/02_merge_cenagro_sset_polygons.ipynb` (census↔PETT by name)
- Chain B above. Normalises the three name formats, matches census→PETT, resolves a best predio per
  productor with a `link_confidence`, compares to / unions with `Base`. Outputs
  `data/processed/merged_parcels.parquet` (45,942 predio-linked productors, with geometry) and a
  `crop_records.parquet`. Use the `high`-confidence rows only if you need the 2012 census crop.

### `notebooks/01_explore_raw_datasets.ipynb` (original forensic exploration)

| § | What it does | Key result |
|---|---|---|
| 1 | Load & characterise the SSET xlsx | It's a PETT crop registry; dates unreliable; AREA in m²; panel structure |
| 2 | Load & characterise the catastro `.dta` | ~2011 cadastral base table; centroids only, **no polygons, no crops** |
| 3 | Join xlsx ↔ dta on `Codigo SSET` | ~99.8% overlap — reliable key |
| 4 | Gap analysis toward training data | Missing polygons, unreliable dates, crop cleanup, area outliers, panel |
| 5 | Load the parcel shapefile (`qgis_stefany/`) | 190k polygons, EPSG:32717; sidecar `PIURA.dta` (latin1) |
| 6 | Confirm polygon→crop linkage | ~58k polygons reach a crop label — geometry gap closed |
| 7 | Build labelled layer + 1-AOI visual check on Esri | Polygons track real field edges; join is spatially sound |
| 8b | Build `parcels_clean` (clean whitelist crops, 1 representative crop/parcel) | ~35k clean labelled polygons |
| 8c | 2×2 grid, 4 AOIs, **year-matched Landsat 5/7** (1999–2006, 30 m) | Year-correct but coarse; small parcels are blocky |
| 8d | Same AOIs on **earliest Sentinel-2** (2015-16, 10 m) | Higher-res but 10–17 yr after → **location check only** |
| 8e | Same AOIs on **recent sub-metre Esri** | Sharpest; present-day → location check only |
| 9 | Multi-crop-per-parcel + **neighbour adjacency** analysis | Crops form single-crop **blocks** (see findings) |
| 10 | `plot_point_parcels(lat, lon)` helper | Spot-check parcels around any coordinate on Esri imagery |
| 11 | **Provenance re-check + wider bridge** | Polygons are **PETT** (census fields empty, `CAT_PIURA` lineage); `grafica_tabular_Piura` bridge adds +12,842 records/+8,392 polygons but **not more years** (~1998–2007 holds) |

## 6. Key findings & caveats (carry these forward)

- **Dates are unreliable.** `FECHA EMPADRONAMIENTO` clusters on stub/batch values (e.g.
  `2001-01-01` alone ≈ 35% of rows). Some AOIs have genuine campaign dates (tight day/week
  clusters), others are pure stubs. **Do not trust an exact date to pick a single scene**; prefer a
  season/phenology window, and per panel we required a single dominant year.
- **Sentinel-2 year-matching is impossible with this data** (linked crops are pre-2015). Landsat 5/7
  is the only sensor that covers the linked years — at 30 m (coarse for ~0.5 ha parcels).
- **Crops are strongly spatially autocorrelated.** ~86% of adjacent parcels share the same crop
  (vs ~46% under random placement); 88.7% among same-year neighbours. Piura grows in **single-crop
  blocks**. ⇒ **train/test split MUST be blocked by area**, not random parcels, or neighbours leak.
- **Intercropping is real (~14% of declarations)** — labels like `CAFE Y PLATANO`, `ARROZ Y MAIZ`.
  Needs an explicit policy (drop / multi-label / "mixed" class). The §5b pipeline now **splits and
  keeps them as a list per polygon** (16% of linked polygons carry >1 crop) — the modelling policy is
  still open, but the data is no longer lossy. Notebook 01 §8+ instead excluded them via a clean-crop
  whitelist (`KEEP`).
- **Multi-record ≠ multi-crop.** ~60% of parcels have ≥2 SSET records, but only ~7% ever record a
  different *raw* crop label (and only ~1% a different *clean* whitelist crop) — mostly
  re-declaration of the same crop. (~56% of farmers, by DNI, hold >1 parcel.)
- **~10k polygons have a null `COD_PREDIO`** (unlinkable) and ~half of all parcels never reach a
  crop record — filter accordingly.
- **AREA units differ:** xlsx `AREA` ≈ m² (dirty); dta/shapefile `area_ha`/`AREA_S_HA` ≈ hectares.
- **Bridge choice matters:** use `grafica_tabular_Piura.dta` (68.8% of SSET crop keys), not
  `grafica_tabular_catastro_Piura.dta` (59.8%, a subset). `qgis/PIURA.dta` cannot bridge (no
  `CodigoSSET`).
- **~1 crop-year per polygon.** Only 1 of 66,363 linked polygons has crops in >1 distinct year — the
  data is a single ~1998–99 titling snapshot, not a time series. A second time point (2012) exists
  only in the census, reachable via the weaker name link.
- **The census has farmer names but no PETT code.** `IV_CENAGRO` links to PETT only by name
  (normalise the 3 formats). Reliable for person/crop, unreliable for the exact parcel — see §4
  Chain B.

## 7. Reusable objects/functions

### In `src/crop_classifier/` (importable — the durable API)
- `crop_normalization.normalize_label(raw) -> list[(crop, category)]` — clean one `CULTIVO` cell.
  Helpers: `clean_text`, `canonicalize_token`; edit `_SYNONYMS`/`_FALLOW`/`_LAND_PREP` to extend the map.
- `build_training_data.build(save=True)` — run the whole Chain-A pipeline; returns
  `(polygons_gdf, records_df, sset_raw_df)`. Loaders `load_sset()`, `load_bridge()`, `load_polygons()`
  are reusable on their own.

### In notebook 01 (in scope after a full run)
- `sset` / `piura` — the full SSET table / Piura subset (crop records, `CULTIVO`, `FECHA…`).
- `piura_dta` — the catastro bridge (`COD_PREDIO`, `CodigoSSET`, …).
- `parcels_geom` — all CENAGRO polygons (native EPSG:32717), keyed by `COD_PREDIO`.
- `parcels_labelled` — polygons ∪ crop label (broad; EPSG:4326).
- `parcels_clean` — one clean representative crop per parcel (whitelist `KEEP`, has `YEAR`);
  basis for the Section 8 panels and the Section 9 adjacency analysis.
- `LOCATIONS` — the 4 pre-scouted AOIs (centre lon/lat, half-width, dominant crop year).
- `landsat_rgb(bbox, year)` (8a), `sentinel2_rgb(bbox)` (8d), `plot_point_parcels(lat, lon, radius_m, zoom)` (10).

## 8. Open questions / next steps

Items 1–3 below are now **built** in the `src/crop_classifier/` modelling pipeline (label policy,
two-stage GEE Landsat extraction, spatial splits, LightGBM/LTAE/PSE-LTAE) and verified end-to-end on
a small extraction — **see [`docs/PIPELINE.md`](docs/PIPELINE.md) for the code, run cookbook
(smoke → medium → full), and the live open-decisions list** (accepted 1.5 km buffer leakage, rare-class
evaluability, unknown 1998 L5-only attrition, minimal intercrop `merge` map).

**➡ NEW STRAND (2026-08-07): ALL OF PERU, not just Piura.** The raw data grew from one
department to the whole country — three new folders `data/raw/BD_SSET/`,
`data/raw/Grafica_Tabular/`, `data/raw/QGIS/`. **The Piura members of all three are
byte-identical (sha256) to the files this pipeline already used**, and rebuilding Piura
through the new code path reproduces its documented figures exactly (66,352 polygons /
80,618 records), so no existing result is invalidated. New code: `src/crop_classifier/allperu/`
(`sources.py` registry, `build_labels.py` Chain-A per department, `sample.py`, `lodo.py`).
Docs: **[`docs/all_peru/`](docs/all_peru/)** — `plan.md` (decisions), `DATA_AUDIT.md`
(provenance + traps), `RESULTS.md` (numbers), and **[`window_plan.md`](docs/all_peru/window_plan.md)
— the route proposed after both panels failed their gate: aggregate predictions to 5-year
windows, take the baseline from the *observed* PETT label instead of predicting it, and
identify the tenure contrast within-year.**

**⛔ 2026-08-10 — THE WINDOW PIVOT'S THREE PRE-GATES ALSO FAILED (RESULTS.md §8,
window_plan.md §9). No extraction was funded; the locked test is still unspent.**

* **M1 works.** Aggregating window-mean *probability* (not modal class) over 5-year windows
  collapses annual 3-class flicker 0.75 → **8.8 % of parcels with ≥2 window-state changes**;
  82 % never change state. The Phase-7 instability is genuinely fixed by aggregation.
* **T3 fails on W2, every arm.** The PETT-`PERENNIAL` control pool — which must be flat for
  the at-risk series to mean anything — drifts **−0.044 / −0.059 / −0.103 per decade**
  (`nometa` / `nolat` / `ltae`) while the at-risk pool rises only **+0.018**. The yardstick
  moves 7× further than the signal, the other way. Not composition: a balanced panel
  (parcels in all 5 windows) gives the same series. ⚠️ Piura's §1 evidence for M2 came from
  `lightgbm_nometa` — the arm LODO disqualified; on the *selected* `nolat` arm Piura already
  drifts −0.017/decade.
* **A measured partial mechanism: probability compression.** Landsat observation density
  falls ~24 → ~13 clear obs/parcel-year (L5 retired, L7 SLC-off). Within parcel, window
  probability tracks it: **+0.052 per log-observation on true perennials, −0.022 on true
  annuals** (p < 1e-6). Both classes revert to the base rate as evidence thins, which is
  indistinguishable from conversion at the level of a share. Explains 0–29 % of the drift, and
  is **monotone in how few statics the arm carries** — the third appearance of that pattern.
* **T2 (leave-one-YEAR-out, new — `allperu/loyo.py`) fails.** Worst non-1998 cohort **0.406**
  vs CV 0.581 (tolerance 0.10); `PERENNIAL` recall spans **0.370–0.882**. 1998 is mid-pack
  nationally (0.523). A 2019–23 prediction is not licensed.
* **T1 fails on the leg that matters.** Accuracy is non-differential (tenure coef +0.013,
  p 0.107) but the **at-risk false-positive rate is 0.111 INSCRITO vs 0.155 NO INSCRITO**
  (−1.75 pp conditional on region+size, p 0.016) — the size of the target effect, pointing the
  opposite way to the cross-sectional association.
* **⭐ A SECOND DATED TENURE OBSERVATION EXISTS AND WAS ALREADY ON DISK.** The bridge `.dta`
  carries the cadastre's own titling status (`estado`) and its cut date (`fech_tran` ≈ 2011-12)
  alongside BD SSET's declaration-time `ESTADO en RRPP` (~1997–2006): **1,780,580 parcels with
  two dated observations, 8.6 % moving NO INSCRITO → REGISTERED** (Piura 25.5 %). That makes a
  two-period difference-in-differences available with no new data — and it differences out
  exactly the drift that broke M2. `allperu.tenure.tenure_two_period`.
* Also done: `ESTADO en RRPP` is now on `COD_PREDIO` for 1.78 M parcels
  (`allperu/tenure.py`, 100 % of held-out CV parcels resolve); the §4.4 year×label-balanced
  test draw (`splits.pick_test_units`, `split_window*.yaml`); `buffer_m=3000` costs **nothing**
  (CV 0.580 ± 0.024 vs 0.5805 ± 0.002); the stratified window sample draws 13,002 parcels at a
  measured `deff` **2.36**, not the assumed 1.4; PERENNIAL is **57 % export / 20 % mixed /
  23 % domestic**; declared woody non-crop is **2.79 %** of the pool — bigger than the effect.

**⛔ 2026-08-11 — THE TEMPORAL-OOD PLAN RAN IN FULL. Steps 1–3 are DONE; step 1's
acceptance FAILED, step 3 FAILED, and no 2019–23 estimate is licensed.** Full account:
[`docs/all_peru/RESULTS.md`](docs/all_peru/RESULTS.md) **§9** and
[`temporal_ood_plan.md`](docs/all_peru/temporal_ood_plan.md) **§6**. New code:
`allperu/{density,yearleak,oli_overlap}.py`, `loyo.lodo_by_cohort`; tests
`tests/test_{density,temporal_ood}.py`.

* **⭐ THE FINDING THAT MATTERS MOST: LOYO inherits spatial CV's blind spot.** `loyo.py`
  holds region approximately fixed so year varies — so a *time-invariant* feature is as
  exploitable there as in CV. `centroid_lat` is worth **+0.0476 on CV, +0.0474 on LOYO
  (the same number), −0.0596 on LODO**. The free fix is **`allperu lodyo`**
  (`loyo.lodo_by_cohort`): re-score the *existing* LODO predictions per label-year cohort,
  so department **and** year are out of distribution at once — no new fits. There
  `centroid_lat`'s advantage becomes **−0.005**. **An OOD evaluation is only blind-sided
  along the axis it holds fixed. Report CV / LODO / LOYO / LODYO for every candidate.**
* **NEW PRIMARY MODEL: `lightgbm_nometa_nolat_aug_yleak10`** (`selected_model.json`; the
  previous record is preserved at `selected_model_20260809.json`). Degradation augmentation
  + the top-10 year-identifying features withheld. Beats the incumbent on **all five**
  criteria — CV +0.0006, LODO +0.0024, LOYO +0.0025, LODYO +0.0040, **W2 control slope
  −0.0592 → −0.0382 (35 % flatter)**. ⚠️ **No labelled difference is individually
  significant** (paired p 0.14–0.54): adopted for consistency in sign at zero cost, not for
  a demonstrated effect. The two changes are **not additive** — `yleak10` alone makes W2
  *worse* (−0.0679).
* **Step 1 acceptance ⛔ FAILED.** W2 target |0.01| (best 0.038); LOYO worst-cohort gap
  target 0.10 (best 0.151). Observation density is a real mechanism worth ~a third of the
  artefact, **not** the artefact.
* **Two routes are now CLOSED, measured rather than assumed.** (1) **Recalibration cannot
  undo probability compression**: fitted temperature *falls* with density
  (T = 1.448 − 0.160·log n; ECE 0.076 at n≈5 vs 0.008 at n≈19), so at endpoint density the
  model is **over**-confident and the correct fix pushes probabilities further toward the
  base rate — applying it moved W2 by **0.0001**. (2) **Per-year quantile alignment is worse
  than doing nothing**: W2 **−0.1023**, nearly 2× the baseline.
* **A structural feature result:** order statistics (`_min`/`_max`/`_amp`) move **0.347**
  within-SD per e-fold of observation count against **0.080** for the harmonic/slope fits —
  a 4.3× gap, land held fixed (`allperu density-audit`). ⚠️ **Do not act on that ranking
  alone**: dropping all 33 of them (`--drop-features order`) is the arm LOYO *rejects*
  (mean −0.0026, worst `PERENNIAL` recall 0.344). Also measured: a *parcel*'s L7 SLC-off
  loss is **11 pp, not the nominal scene-level 22 %**.
* **The features identify the label year at 0.508 accuracy vs a 0.155 baseline**, held out
  by region (`allperu year-leak`). `frac_l7` was one symptom of a systemic property.
* **⛔ STEP 3 (OLI) FAILED, and the Roy harmonisation is HARMFUL.** OLI alone would give
  **21.8 clear observations per parcel-year against L7's 13.1** — the largest lever on the
  density mechanism anyone has found. But it shifts the PETT-`PERENNIAL` control pool by
  **−0.042** (split-half self-noise: −0.0035), and `perennial/harmonization.py` — implemented
  months ago, documented as the safe path, **never once run** — makes that **−0.107** and
  lowers agreement too. Worse than no correction on every axis. **An unused correction is an
  untested correction.** The registered 0.95 agreement criterion was also **never
  achievable**: the model's split-half self-agreement ceiling is **0.655**
  (`allperu oli split-half`).
* **LTAE is now the worst arm on the temporal axis too** (LOYO 0.4784, LODYO 0.4965,
  W2 −0.1025), extending the §6.2c negative result to a second dimension.

* **⛔ AND REFITTING THAT CORRECTION ON OUR OWN DATA ALSO FAILS — the OLI route is CLOSED**
  (RESULTS.md §9.5, `allperu/oli_refit.py`, `allperu oli-refit`). 27,573 **same-day** L7/OLI
  parcel pairs exist in WRS sidelaps, but they **cannot identify a slope**: the SD of the two
  sensors' difference (0.064–0.102) **exceeds the SD of either sensor's own values**
  (0.065–0.088) and blue correlates at 0.12, so OLS returns a physically impossible 0.116
  (de-diluted by binning it still only reaches 0.26). Only a per-band **offset** is
  estimable. That offset is the **best arm on every axis** — agreement 0.6452 (98.5 % of the
  split-half ceiling), at-risk pool bias **−0.0003** — and **still fails**: the control pool
  overshoots −0.0423 → **+0.0360**. **The reason is structural: the sensor difference is
  cover-type dependent.** Raw OLI reads **+0.070 NDVI greener on perennial parcels and +0.045
  on pasture**, and that **0.025 between-class spread survives every arm** (0.015–0.025). A
  global band-level linear map moves all classes *together* and cannot remove a difference
  *between* them; the only map that would work is conditioned on cover type, which is what the
  model predicts. **Do not propose refitting the coefficients — it is done.** Reopening step 3
  needs a *cleaner paired sample* (pixel-level co-registration or larger parcels, not parcel
  medians in scene sidelaps).
* **LODYO now runs automatically at the end of every `allperu lodo`** — an evaluation nobody
  remembers to run is an evaluation that does not exist, and this is the one that overturned
  the selection.

**⭐ A FREE PILOT OF THE TWO-PERIOD TENURE DiD WAS RUN AND THE DRIFT DOES CANCEL**
(⚠️ **superseded 2026-08-11 — the pilot's −0.038 reverses to +0.034 under a valid pre-period;
read the block below this one before using any number here**)
(RESULTS.md §9.6). The two dated tenure observations supply the **treatment**; the classifier
still supplies the **outcome** at both dates, drift and all — the point is that the drift is
now *common to both groups*. On the at-risk pool (PETT-`ANNUAL`) split by whether the parcel
became registered, the two arms drift at **+0.0127 and +0.0115 per decade and their difference
is flat (−0.0013)**. Contrast M2's control pool, which drifted −0.0163 *the other way* because
it was a different kind of parcel and the compression is class-specific. **Power is much
better than feared** — parcel FE removes most of the variance, so **~1,300 treated parcels
detect 2 pp at 80 %** and **44,957 at-risk parcels nationally became registered**; the
extraction is panel-sized, not the 12,500/group the window plan assumed.
⚠️ **But the placebo does NOT clear**: W99→W04 is entirely pre-treatment and should be zero,
and it is **−0.0169 (se 0.0132), 44 % of the headline −0.0383 and pointing the same way**.
With 334 treated parcels it cannot be resolved. **Do not quote −0.038 as a result** — the
pilot is exactly powered to mislead.

**⛔ 2026-08-11 — THE TENURE DiD RAN AND IS NOT FUNDABLE (v2). Stopped at N3, before any GEE
spend. No estimate exists** (RESULTS.md **§10**, tenure_did_plan.md **§8**).
**⚠️ SUPERSEDED 2026-08-12 — the study was reopened as v3 with a corrected gate and RAN TO
COMPLETION; see the ⚖️ block further down and RESULTS.md §11. The two findings below still
stand (the pilot's sign reversal, and 6,559 treated being all that exist); what did not stand
is the *gate*, which turned out to be unpassable at any sample size.** New code:
`allperu/tenure_did.py`, CLI `allperu tenure-did` / `allperu tenure-ceiling`; tests
`tests/test_tenure_did.py` (17).

* **⭐ THE PILOT'S HEADLINE REVERSES SIGN.** §9.6's **−0.0383 (p 0.007)** becomes
  **+0.0338 (p 0.173)** once the pre-period is required to precede *each parcel's own
  declaration*. The placebo moves −0.0169 → +0.0049 at the same time. The pilot was measuring
  a pre-existing divergence. **Do not quote −0.038; it is reproduced by a test only so §9.6
  stays on the record (N-D7).**
* **The two tenure observations are SNAPSHOTS OF A ROLLING PROGRAMME, not two shared dates** —
  measured, not assumed: declarations spread 1996–2009, `fech_tran` is a per-record
  transaction date (3–112 distinct dates *per department*, La Libertad 1999–2015),
  **24.6 % of parcels carry a status with no date**, and the registration rate is
  **non-monotone** in the gap between observations (campaign waves, not duration). So a
  calendar pre-period is contaminated by early treatment; the pre-window must end before the
  parcel's own declaration (**R4**).
* **⛔ G3a fails against the POPULATION, not the budget.** The placebo needs **25,202 parcels
  per arm** from measured variance; after R1–R4 **all of Peru holds 6,559 treated** (effective
  n 5,231) — **short by 2.41×**. The trade is structural: cohort ≥2004 gives 6,559 treated but
  only *one* clean pre-window; cohort ≥2009 gives two pre-windows but **380 treated**. Either
  the pre-period is clean or it is powered, never both.
* **G2 (integrity) fails 3 of 4, and two checks had never been run here**: training-set
  membership differs **6–8 pp** between arms (the model was trained on these parcels *labelled*
  `ANNUAL`, which anchors their predicted probability), and only **0.37** of parcels sit in a
  region holding both arms, so region-clustered SEs were not doing what they appeared to.
  Differential attrition — the new check the plan added — **passes cleanly everywhere**.
* **The v1 gate would have PASSED this and funded the extraction.** `|coef| < 0.005` **and**
  "CI contains 0" is satisfied by +0.0049 with a wide CI. The v2 equivalence gate returns
  **INCONCLUSIVE**. *A gate that rewards imprecision is not a gate.*
* **`ltae` FAILS the placebo outright** (−0.0878, CI excluding 0) where LightGBM is flat: even
  the *pre-treatment* coefficient is architecture-dependent.
* **Method to carry forward:** `population_ceiling` + `precision` + `feasibility` answers "can
  this study exist?" in minutes, from variance and the archive. **Run it before funding any
  extraction, ever again.**

**⚖️ 2026-08-12 — THE TENURE DiD WAS REOPENED AS v3, RAN IN FULL, AND RETURNED AN ANSWER: A
BOUNDED NULL.** The first estimand in this project to produce a number rather than a failure.
Full account: [`docs/all_peru/RESULTS.md`](docs/all_peru/RESULTS.md) **§11**,
[`tenure_did_plan.md`](docs/all_peru/tenure_did_plan.md) **§9**. New code:
`allperu/did_sample.py`, the correction in `allperu/tenure_did.py`,
`perennial/panel.py`'s `rebuild_year_stores` / `verify_years`; tests
`tests/test_tenure_did.py` (31) + `tests/test_coverage_years.py` (+2). **278 tests pass.**

* **What changed was the GATE, not the design.** R1–R5 and N-D1…N-D9 all still bind — **R4
  unchanged**. v2 demanded *proof* that the pre-trend was negligible (equivalence, 25,202
  parcels/arm against 6,559 that exist). v3 **measures the pre-trend and subtracts it**,
  carrying its uncertainty: `corrected = headline − M·placebo`,
  `se = sqrt(se_h² + M²se_p²)`, with **M = 7 derived from window midpoints in code**, never
  hard-coded (17.5 y headline / 2.5 y placebo).
* **⭐ THE RESULT.** 14,625 parcels (all 6,559 qualifying treated + 8,066 controls), 15 years
  extracted (211,567 parcel-years, 63.6 M pixel-obs, ~13.5 h), 219,375 parcel-years inferred.
  **Placebo −0.0029 (se 0.0037) — no anticipation. Headline −0.0011 (se 0.0059), CI
  [−0.0126, +0.0104]. Corrected at M = 7: +0.0195 (se 0.0268), CI [−0.0330, +0.0720] ⇒
  NOT-SEPARABLE.** The headline
  is a **tight null**: the correction widens ±1.3 pp to ±5.3 pp, it does not move a large
  effect off zero. Registered before extraction in `did2_registration.json`.
* **⭐ THE v2 GATE WAS UNPASSABLE AT ANY SAMPLE SIZE — measurable only now.** The placebo's
  point estimate (−0.00295) sits **outside** the registered ±0.0025 band, so the SE needed for
  its CI to fit inside is **negative**; more precision returns FAIL, never PASS. And it would
  have failed on −0.0118/decade, which cannot change a null. §10 recorded that v1 *rewarded
  imprecision*; §11 records that v2 **punished a point estimate for missing an arbitrary band
  on a question whose answer does not depend on it.** A pre-trend gate must ask *"could this
  pre-trend overturn my conclusion?"* — which the sensitivity curve answers directly.
* **The pilot's placebo was noise, now demonstrated**: −0.0169 (se 0.0132) → −0.0029
  (se 0.0037). And the correction's **sign flips between outcomes** (−0.0029 on probability,
  +0.0029 on thresholded share), which is what a noise term does. **Never read a direction
  from the corrected point estimate.**
* **G2 went from 1 of 4 passing to 3 of 4, by sample design not by tuning**: training-set
  membership **−0.062 → +0.0026** (draw from the full national table, not the model's own
  sample) and shared regions **0.366 → 0.849** (draw controls region-first). Common support
  still fails (0.409, department) and is **structural** — 75 % of treated parcels are La
  Libertad + Cajamarca, so the estimate applies to where titling actually happened.
* **⚠️ Magnitude is architecture-dependent even though sign is not.** `ltae` gives −0.0154
  (W14 −0.0252, p 0.006) against LightGBM's −0.0011. `ltae` **no longer fails the placebo**
  (−0.0026 at n = 6,520): §10.2's −0.0878 was small-sample noise at n = 60.
* **⚠️ THE CAVEAT THAT BINDS EVERY NUMBER**: tenure is last observed ~2011 but the outcome
  runs to 2023, and Peru's titling programme kept running — so part of the control arm was
  certainly treated and unobserved. That attenuates **toward zero**, so this null is **not**
  evidence that titling has no effect. Now listed in the plan's §6 risks.
* **⚠️ THE CROSS-SECTIONAL CONTRAST WAS COMPUTED AS A DESCRIPTIVE COMPANION AND IS NOT AN
  ESTIMATE** (`allperu tenure-xsec`, RESULTS.md §11.13). Among at-risk parcels the
  INSCRITO − NO INSCRITO gap runs **−0.0429 at W99 → −0.0216 at W19**. Three findings:
  (a) the W99 gap **is** the classifier's false-positive-rate gap, reproducing T1's
  −0.044 (§8.5) from a completely different direction — the baseline "association" is
  the instrument, not the land; (b) the **sign is department-specific** (INSCRITO higher
  in 5 of 14 departments at W99, 4 of 14 at W19), so the pooled number describes a
  quantity that does not exist; (c) ⚠️ **`../perennial/RESULTS.md` §7.5's Piura
  "24.9 % vs 10.9 %, INSCRITO higher" does NOT replicate** — Piura's gap here is
  **−0.081**, INSCRITO *lower*. Do not carry that figure forward as a national fact.
* **Two infrastructure lessons.** (1) `verify_years` cried wolf on a *complete* extraction
  because its tolerance was guessed (0.995) and there is a **structural sub-pixel floor**: the
  same ~85 parcels are missing every year (Jaccard 0.95, median 0.11 ha) because a parcel
  smaller than a Landsat pixel passes the coverage gate and returns no rows. Fixed by measuring
  the floor and adding a floor-independent `deficit_ratio_to_median`. (2) **GEE throttling is
  real and mostly invisible**: 5 workers hit *"Exceeded Earth Engine concurrency limit …
  Restricted Mode"*, and it arrived as **8 silent 900 s hangs**, not errors — the `_retry`
  deadline absorbed all eight.

**➡ START HERE FOR NEW WORK.** Everything cheaper has been tried and measured (density
matching, recalibration, feature pruning, quantile alignment, OLI twice, and the tenure DiD
twice). **One plan remains, and it is the only one that verifies rather than mitigates:**

1. ~~**[`docs/all_peru/tenure_did_plan.md`](docs/all_peru/tenure_did_plan.md)**~~ — **RUN TO
   COMPLETION 2026-08-12 as v3; result is a bounded null** (above). Do **not** reopen it for a
   bigger sample: the population is exhausted (6,559 treated is all of Peru) and the binding
   limit is the 7× amplification of a 2.5-year placebo onto a 17.5-year headline. Reopening
   needs a **dated registration event** rather than two snapshots. The v2 closure reasons in
   RESULTS.md §10.6 still stand — do not relax R4, widen the band, or drop the at-risk
   restriction. Briefing kept at
   [`docs/all_peru/AGENT_PROMPT_tenure_did.md`](docs/all_peru/AGENT_PROMPT_tenure_did.md).
2. ~~**[`docs/all_peru/endpoint_labels_plan.md`](docs/all_peru/endpoint_labels_plan.md)**~~ —
   superseded by **[`docs/all_peru/s2_labelling_plan.md`](docs/all_peru/s2_labelling_plan.md)**,
   which is **BUILT AND RUN up to the point where humans take over** (see the ⭐ block
   below). Everything except the labelling itself is done.

**⭐ 2026-08-12 — THE S2 ENDPOINT-LABELLING CAMPAIGN IS BUILT AND THE LABELLING HTML
EXISTS.** Plan §11 steps 1–2 and 4–7 are complete; only the human steps (the pilot, and
the labelling) stand between here and a 2020+ Sentinel-2 classifier. Full record:
**[`docs/all_peru/s2_labelling_RESULTS.md`](docs/all_peru/s2_labelling_RESULTS.md)**;
frozen codebook: [`s2_labelling_codebook.md`](docs/all_peru/s2_labelling_codebook.md).
New code: `allperu/{label_sample,s2_campaign}.py`, `features/{s2_gee,s2_assemble}.py`,
`labelling/{chips,build_html,ingest}.py`, `config/split_s2labels.yaml`; **67 new tests,
suite 348, all pass**. One entry point: `allperu s2-labels <step>`.

* **Eligible universe 614,876 parcels** (14 depts, `area_ha ≥ 0.15`); **4,519 Esri
  centroids probed**; **G0 PASSES at 87.6 %** ≤1.2 m and ≥2019. Delivered:
  **2,224 chips (0 failures), 117,768 S2 parcel-dates, 9 HTML shards covering all 1,112
  parcels**, largest 13.5 MB.
* **⭐ Sentinel-2 gives a median 19–113 clear dates per parcel-agricultural-year by
  department, against the Landsat store's 13–24.** Even Pasco, the cloudiest, sits at the
  top of the Landsat range. **Observation density is the one mechanism this project has
  *measured* driving the panel failures** (probability compression, RESULTS.md §8), and S2
  roughly triples it. That does not mean the artefact is gone — measure it, do not assume.
* **The harmonisation was verified, and verifying it took three attempts.** The obvious
  before/after-the-cut comparison reports a +0.055 NDVI "step" that is **entirely the
  growing season**; a within-month-of-year pairing looks principled and still returns
  ±0.018 on a pure sine. What works: fit the seasonal cycle out per parcel, subtract a
  **placebo cut** one year earlier as the noise floor, and read the verdict off **raw
  bands, never an index**. Result: SWIR1/R/NIR steps **+0.009 / +0.009 / +0.002** against
  **+0.10** for an unremoved +1000 DN offset. NDVI reads −0.034 and that is Peruvian
  weather, not Sentinel-2.
* **A third silent failure, fixed in shared code:** GEE throttles
  (`Too many concurrent aggregations`) were not in `landsat_gee._TRANSIENT_MSGS`, so
  `_retry` re-raised instead of backing off and killed a running extraction at 144 of 215
  chunks. Now classified transient — **and note §11 recorded the *same* throttle arriving
  as eight silent 900 s hangs instead. It comes both ways; handle both.** 4 workers is over
  this project's quota; 2 is stable.
* ⚠️ **Piura is one of the three worst-covered departments for recent high-res imagery**
  (0.61 eligible, with Cajamarca 0.60 and Pasco 0.39; eight departments are 1.00). Every
  earlier strand was built on Piura. A campaign scoped to Piura would have lost a third of
  its draw and it would have looked like a sampling bug.
* **Draw: 992 main + 120 pilot, split frozen before labelling** (`split_s2labels.yaml`,
  buffer 3,000 m costs 9.9 % of trainval; test = 201 parcels in 150 regions, every
  department on both sides). The 8-parcel shortfall is **places, not parcels**:
  Huancavelica has 44 populated 5 km regions in the entire universe against a 2/region cap.
* ⚠️ **Two silent failures, both caught only by looking at the output.** (1) **Esri returns
  a valid flat-grey image above the zoom it serves** — `zoom="auto"` on a 120 m extent
  rendered blank chips with no error; zoom now comes from the probed resolution
  (30cm→z19 / 60cm→z18 / **1.2m→z17**) with a placeholder detector. (2) **Chunking by
  extraction window alone spans the whole country**, because parcels sharing an imagery
  month sit in all 14 departments — `filterBounds` then reduces every granule in that
  rectangle and the first run stalled at 3 of 89 chunks at 0 % CPU. Fixed with a 1 deg²
  bbox cap. **`landsat_gee._chunk_todo` already carried that exact lesson in a comment; it
  was learned twice because the new module copied the old one's structure, not its
  chunker.**
* **The labeller sees six values** (PERENNIAL / ANNUAL / OTHER / WOODY_NON_CROP / **UNSURE**
  / NON_AGRICULTURE) and a boundary-mismatch flag, on two Esri chips plus a **24-month S2
  NDVI trace** — the trace is what makes ANNUAL-vs-OTHER callable at all.
  **Blindness is asserted on the raw HTML string**: no declared class, split or fold appears
  anywhere in the file (`tests/test_build_html.py`).
* **⭐ 2026-08-13 — three changes after opening the emitted HTML** (RESULTS §11). **(1)** The
  right panel became a **zoom**, not a 600 m context view — both panels shared a centre, so
  at 600 m neither resolved canopy on a large parcel, and **texture is the actual
  PERENNIAL-vs-WOODY_NON_CROP discriminator** the codebook asks for. Scale bars are per
  panel. **(2)** `UNSURE` is a **label, not a low confidence** — the old design had
  no abstain, and a guess at confidence-1 is *indistinguishable from a real label
  downstream*. G1 is now `kappa_called` (κ over parcels both labellers actually called, so
  "we disagree" and "one abstained" stay separable), G2 is the `UNSURE` share, G3 counts
  only the real classes. **(3)** The labeller **types their own name**; it goes into
  the CSV and the filename, progress is keyed to it, and `_A`/`_B` in the filename is now
  only a routing suggestion. Item order is md5-seeded per (shard, slot) so the two overlap
  shards hold the same parcels in different orders.
* **⭐ 2026-08-13 (second review) — five more changes; §11's panel widths, chip sizes and
  five-value vocabulary are SUPERSEDED by RESULTS §12.** Suite **407 tests**, ruff clean.
  **(1)** `ZOOM_M` 100 → **200 m**. **(2)** The left panel is a real **context** view
  (`<item_id>_context.jpg`): `max(span + 2·min(0.5·span, 300 m), 400 m)` — proportional
  padding, capped, floored — because *the same crop reads differently in different
  geographies*. ⚠️ These two are **one change**: at `ZOOM_M = 200` the old 120 m-floored
  left panel would have been **narrower than the zoom for half the draw**. And the neighbour
  box now derives from the panel extent, or a 2.7 km panel leaves outer fields unoutlined —
  which reads as "no neighbours", not as a bug. **(3)** **`NON_AGRICULTURE` on key 6**, with
  `OTHER` **narrowed** to "farmable land not currently a crop" — the old `OTHER` explicitly
  listed water/built-up/road/riverbed, so adding the class without narrowing it would have
  put two labellers on opposite sides of the same parcel. Separating test, in the codebook
  and asserted on the emitted HTML: ***could this ground be sown next season exactly as it
  stands?*** Plus a **priority ladder** (`WOODY_NON_CROP` before `NON_AGRICULTURE`, so
  riparian trees are woody and the gravel beside them is not). ⚠️ G3's ≥150 floor now spans
  **five** classes and `NON_AGRICULTURE` has no stratum — expect it to fail, and pool at
  training time rather than re-draw. **(4)** **Confidence removed entirely.** §11 added
  `UNSURE` *because* a control you must actively set goes unused, then kept confidence
  anyway — two ways to record doubt, one of which is indistinguishable from a real label
  downstream. G2 is now a single number. **(5)** **NDVI p25–p75 ribbon** — see below; it is
  the only one of the five that cost a re-extraction.
* **⚠️ THE RIBBON NEEDED DATA THAT DID NOT EXIST — check the store before designing a
  visual.** `s2_perdate.parquet` held band **medians** and a pixel count, no within-parcel
  quantiles, and *no arithmetic on medians recovers a quantile*. Fixed by adding
  `ee.Reducer.percentile([25,75])` to the `perdate_chunk` combine (`sharedInputs=True`, so
  it is one pass over the pixels, not a second `reduceRegions`) — **and by forming NDVI per
  pixel server-side** (`S2_NDVI_BAND`), because **a quantile of a ratio is not the ratio of
  the quantiles**. That in turn forced the *line* to move to `NDVI_px_p50`: it differs from
  band-median NDVI by 0.0025, enough to put the plotted median outside its own band on some
  dates. Model features are untouched and the columns are named `NDVI_px_*` so
  `add_indices`'s `NDVI` cannot collide. The 215 cached chunks were **content-addressed on
  parcel-set + window**, so the new columns would have been silently skipped — the cache had
  to be invalidated by hand. Re-extract: 215 chunks / 2 workers / ~2 h, **no errors**, with
  GEE in *"restricted mode — exceeded the noncommercial compute quota"* throughout.
  **100 % of the 117,755 old parcel-dates reproduced**, 0.079 % of bands moving >1 DN,
  0 quartile-ordering violations, median IQR **0.0835 NDVI**, line inside its own band on
  100 % of dates; coverage and the harmonisation check both reproduce §4b/§3b.
* ⚠️ **A pre-existing blindness gap surfaced while re-verifying**: `FORBIDDEN_FIELDS`
  contains `overlap` and `overlap_A/B.html` carry it once — as `SHARD_ID="overlap"`, the
  file's own name — because **the blindness test's fixture sets `overlap=False`, so it had
  never built an overlap shard**. Now tested, with that one occurrence documented and
  pinned. Not renamed (the filenames carry the routing), but note a labeller can tell which
  shard is double-labelled, and **κ is measured on exactly that shard**.
* ⚠️ **What 1,000 buys**: a national model and one national test number. **Not**
  per-(dept × class) accuracy (42 cells at ~24 each). **No sierra/selva labels ever** — the
  8 polygon-only departments have no bridge, so no declared class, so no stratified draw.
  That is a permanent model-card limit, not a to-do. And the training prior is a **~5x
  `PERENNIAL` over-sample**: prior-correct the model, not just the estimates.

**Only 15 of 24 departments are linkable and this is structural, not a choice**: the bridge
(`Grafica_Tabular/<Dept>.dta`) is mandatory — BD SSET has no `COD_PREDIO`, shapefiles have no
`CodigoSSET` — and only 15 bridge files exist. Callao then yields 0 linked records, leaving
**14 departments, 946,872 linked polygons (14.3x Piura), 726,808 3-class-eligible parcels**.
**The label years are spread over 1997–2006** instead of Piura's ~80 % in 1998–99, which is
the single most valuable thing the national data adds for temporal generalisability.

**Four traps in the new folders, all of which fail SILENTLY (see DATA_AUDIT §4):**
1. **Bridge keys are zero-padded to 9 chars** (`030406693`) while BD SSET stores them
   unpadded — a string join returns **zero** rows and reads as "no data for this
   department". Ancash: 0 → 369,089 keys matched. Fixed by `build_labels.canon_key`
   (no-op for Piura).
2. **19 of 24 shapefiles ship their attribute table under the wrong basename**
   (`QGIS/ANCASH/ANCASH.dbf`). GDAL opens them and returns **zero columns** — `COD_PREDIO`
   vanishes with no error. Fixed by `sources.shapefile_view` (symlinks; raw data untouched).
3. **La Libertad's cadastre is 3D** and Earth Engine rejects 3D GeoJSON outright
   (`EEException: Invalid GeoJSON geometry`) — 9.6 % of the sample, and it kills a
   multi-hour extraction partway through. Fixed by `build_labels.clean_geometry`.
4. **A department's rows are not confined to "its" workbook** (Lima's are in three).
   Fixed by `build_dept_sset_caches`, which scans every workbook once.

**Working set = a Piura-scale sample (56,419 parcels), so cost is unchanged**: whole 5 km
regions, departments allocated sqrt-proportionally with a floor and a per-region cap
(`allperu sample`). ⚠️ **That allocation doubles the raw `PERENNIAL` share (9.9 % population
→ 20.2 % sample)** because small departments are the coastal perennial ones — good for
training, but **any area/share figure must use `sample_weight`**, which reconstructs the
population exactly. Workspace switches: `CC_PROC=data/processed/all_peru`,
**`CC_FEAT=data/processed/all_peru/features`** (new env var — the Piura store is no longer
implicitly shared), `CC_RUNS=runs/all_peru`.

**⭐ THE HEADLINE NATIONAL RESULT: `centroid_lat` is spatial memorisation, not
agro-climatic signal — and only leave-one-department-out can see it.** With 14 departments
the pipeline can finally hold out a *place* rather than a neighbouring 5 km cell:

| | CV (unseen cell) | LODO mean | LODO pooled | LODO std |
|---|---|---|---|---|
| `lightgbm_nometa` (has `centroid_lat`) | **0.628 ± 0.013** | 0.417 | 0.539 | 0.098 |
| **`lightgbm_nometa_nolat`** ⭐ selected | 0.581 ± 0.002 | **0.477** | 0.537 | **0.082** |
| Δ | **−0.047** | **+0.060** | −0.002 | −0.016 |

**Latitude's contribution reverses sign when the test is a new place**; **12 of 14
departments improve without it** (Piura most, +0.178). In Peru latitude is nearly a
climate-zone label, so it looks like real information — but it transfers like a lookup
table. This is the third route to the same lesson as `frac_l7` (manufactured *change*) and
`centroid_lat` in the Piura panel (manufactured *stability*): here it manufactures
*accuracy that does not leave the training departments*. **Selection therefore reverses
Piura's** (`runs/all_peru/selected_model.json`) — same rule, different evidence.

Two more results that matter: **the spatial-generalisation gap is ~0.09 macro-F1**
(0.629 unseen-cell → 0.539 unseen-department), invisible to every evaluation Piura could
run; and — ⚠️ **corrected 2026-08-09** — the old claim that *"Piura is one of the hardest
departments to predict"* (0.301) **was a property of `centroid_lat`, not of Piura**. That
figure comes from `lightgbm_nometa`, the variant that was rejected. Under the **selected**
`nolat` model Piura is **0.479, 6th of 14**; under **LTAE, which has no location feature at
all, 0.512, 2nd of 14**. Piura is the most distinctive latitude band, so it is where a
latitude lookup table fails hardest — not an atypical department.

**➡ LTAE was added nationally (2026-08-09) and it does NOT win — a negative result.**
Trained on the identical sample/folds/store, calibrated the same way: CV **0.602 ± 0.010**
(between the two LightGBM variants, *reversing* Piura where tuned LTAE beat LightGBM on all
5 folds), and far the worst calibrated before scaling (ECE 0.120, **T = 2.08**). On the
deciding criterion it loses too — **LODO mean 0.442** vs `nolat`'s 0.477, and **worst of the
three on pooled LODO** (0.508). It wins only across-department *spread* (**0.056** vs 0.082
/ 0.098) and is the only arm that lifts the two hardest departments. **Selection is
unchanged: `lightgbm_nometa_nolat` stays primary.** The lesson: LTAE carries **no statics at
all**, so the memorisation mechanism cannot apply to it — yet it still transfers worse.
*"Fewer statics" is not a monotone recipe for generalisation; the specific feature was the
problem, not staticness.* RESULTS.md §6.2c.

**Other national findings:** CV 0.628 vs Piura's 0.648, but **fold variance falls ~3x**
(±0.036 → ±0.013) and calibration is far better out of the box (ECE 0.026, T = 1.09 vs
0.090, T = 1.41). Per-class the profile is **much more balanced** — `PASTURE_FALLOW`
0.485 → 0.606 (the sierra supplies grazing land Piura lacked), `ANNUAL` 0.781 → 0.627
(Piura's rice/cotton monoculture made it nearly trivial), `PERENNIAL` 0.685 → 0.655.
**The El Niño exclusion is Piura-specific**: `--train-years 1999-2023` cost −0.0003 in Piura
but −0.0048 nationally, because 1996–98 also holds sierra/southern parcels that are
perfectly readable. The primary national model keeps all years.

**⛔ 2026-08-09 — THE NATIONAL PANEL FAILED ITS GATE TOO, ON ALL THREE ARMS.** The panel
was assembled (25 years, every year verified by parcel count against its pixel store),
inferred for three arms (**114,125 parcel-years each** = 4,565 × 25 exactly) and gated:

| arm | statics | S4 | S5 `PERENNIAL` flicker (crit. 0.15) | gate |
|---|---|---|---|---|
| `lightgbm_nometa` | 3 | **PASS** (dev 0.029) | **0.517** | ⛔ FAIL |
| `lightgbm_nometa_nolat` ⭐ | 2 | **PASS** (dev 0.013) | **0.744** | ⛔ FAIL |
| `ltae` | **0** | FAIL (dev 0.107) | **0.980** | ⛔ FAIL |

**The pre-registered prediction is confirmed exactly** (registered in `docs/all_peru/plan.md`
§7b, and in RESULTS.md §7.4 of the pre-gate revision, before any of this ran; the outcome is
recorded against it in plan.md §7b.1): flicker is **monotone in how many time-invariant
features the model carries**, so
statics manufacture *stability* nationally as they did in Piura — and every rung is worse
here (Piura: 0.428 / 0.547 / 0.780). Meanwhile **k = 0 accuracy barely moves** (0.586 /
0.544 / 0.581): statics contribute almost nothing to being right and almost everything to
not changing your mind. `ltae` is the only static-free arm and therefore the honest end.

**Three things make this a cleaner failure than Piura's, which is what makes it
informative.** (1) **S4 PASSES on both LightGBM arms** — the panel starts in 1999 and label
years spread over 1997–2006, so no k-bin is dominated by one cohort. (2) **There are no thin
years**: Piura's flagged coverage failures (2009 49.2 %, 2011 43.3 %, 2012 83.6 %) pass at
95.4 / 96.2 / 97.6 % nationally, minimum 93.4 % over all 25 years — **so 0.98 flicker cannot
be a coverage artefact**. (3) The El Niño arm shows **no class-specific collapse** — but ⚠️
**do not read that as a clearance**: the national 1998 arm is only **5.5 % Piura** (6.2 %
north coast), so it never tested the flood.

**What that isolates:** not the 1996–98 baseline, not coverage, not the sensor boundary, not
the architecture. **A single-year 3-class classifier at ~0.55–0.59 accuracy is simply not
accurate enough for a per-parcel annual trajectory** — per-year errors are near-independent,
so a 25-year series is dominated by classification noise. Most promising remaining option
(unrun, and not implied by anything above): **drop per-parcel trajectories as the estimand**
and report a weighted aggregate share per year with CIs. RESULTS.md §7.4–§7.10.

**Status of the national strand (2026-08-09):** labels, sample, splits, GEE extraction
(14.75 M pixel-obs over 54,438 parcels), assemble, **four** trained + calibrated models
(3 LightGBM + LTAE), leave-one-department-out **for all three candidates**, panel extraction
(25 years, 2,949 chunks, **37.6 M pixel-obs**), panel assemble, panel inference and the
**Phase-7 gate — all complete**. **The locked test is STILL UNSPENT** and should stay that
way: §6.2b argued the moment to spend it is after the gate returns, and it returned FAIL, so
the estimand has to change first. **No national trajectories, transitions or area estimates
exist, and none should be produced** — that code is built and unit-tested and stays unrun.

**Two shared-code fixes made here that matter to the Piura pipeline too:**
* **The silent-GEE-hang durable fix is now IMPLEMENTED** (previously listed as outstanding):
  `_retry` runs each call under a 900 s wall-clock deadline on a worker thread, so a *hang*
  — which is not an error and so was never caught — raises `ChunkTimeout` and is retried.
  **It fired 25 times during the national panel extraction** (5 workers, plus 96 ordinary
  transient network errors) with zero workers lost and zero tracebacks. Each of those 25
  would previously have been a permanent silent stall — the failure that once cost 13.4 h.
  ⚠️ **The first version of this fix had a defect of its own**: it used a
  `ThreadPoolExecutor`, whose threads are **non-daemon** and are *joined* by an `atexit` hook,
  so every timeout orphaned an unjoinable thread and all five workers then sat at 0 % CPU for
  up to **3 h after their work was complete**, never exiting. (`cancel_futures=True` does not
  help — it only drops futures still *queued*.) Now `_call_with_deadline` uses an explicit
  `threading.Thread(daemon=True)`. Tests pin it. The lesson: **the fix for a silent failure
  introduced a second silent failure one layer down** — always verify a finished job by
  counting its output, never by "the process ended" or "the file exists".
* **Chunk extent is now packed for locality** (`_chunk_todo`). Grouping by year alone gave
  chunks spanning up to **110 deg²** nationally; greedy longitude-ordered packing with a
  4 deg² cap gives 240 chunks at a 0.53 deg² median.

**➡ NEW STRAND (2026-08-04): the 3-class perennial/annual/pasture classifier.** The 12-class
"which crop?" question has been joined by the one the research actually needs — *"has land
shifted from non-export (annual) to export (perennial) crops?"* — as a **3-class land-state
classifier** run as annual inference over a multi-year panel. It reuses this whole pipeline
via a `CC_PROC` workspace switch (`src/crop_classifier/paths.py`); new code is in
`src/crop_classifier/perennial/`. **Single-year locked-test macro-F1 0.681** (vs 0.427 for
the 12-class problem). Plan: [`docs/perennial/plan.md`](docs/perennial/plan.md); results +
current state: [`docs/perennial/RESULTS.md`](docs/perennial/RESULTS.md).

**📄 Read [`docs/SUMMARY_FULL.md`](docs/SUMMARY_FULL.md) first** — the complete narrative of
all modelling work (12-class → 3-class → panel → gates), with figures: per-class seasonal
profiles showing *why* the 12-class problem is unsolvable and the 3-class one is, the El Niño
signature in the imagery, and the flicker-vs-statics result. Figures are generated by
`uv run python -m crop_classifier.perennial.report_figures` into `docs/figures/`.

**⛔ 2026-08-06 — THE PANEL FAILED ITS VALIDATION GATE. The classifier works; the trend
does not.** Phases 1–6 are complete, the panel is extracted + assembled + inferred
(**215,320 parcel-years**, `panel_predictions.parquet`), and Phase 7's gate
(`uv run python -m crop_classifier.cli perennial diagnostics`) was run for the first time:

* **S4 temporal transfer: FAIL** — accuracy 0.746 at k=0 but **0.600 at k=−3** (0.146 >
  0.10 tolerance). The bin is 66 % panel-year 1996 and contains 1997 at accuracy 0.326.
  Forward transfer (k=+1…+3) is fine, within 0.010.
* **S5 flicker: FAIL** — **0.428** of `PERENNIAL`-labelled parcels flicker vs a 0.15
  criterion. **Not** a coverage artefact (0.421 with the four thin years dropped); smoothing
  only halves it (0.221), which is the case plan §9.2 warned would *hide* the problem.
* **El Niño confound test: FAIL (the §8 "smoking gun")** — train 1999+2000 → test 1998 with
  the required control arm: `PASTURE_FALLOW` recall 0.589→0.100, `PERENNIAL` 0.361→**0.000**,
  `ANNUAL` *up* 0.049. Robust across seeds, region definitions, and with/without the
  metadata ablation — a **separate** failure from `frac_l7`. 1996–98 are the panel baseline.
* **Sensor drift: PASSES** — on a balanced 3,434-parcel panel, **0 of 30 mission-boundary
  steps exceed 2× the within-era year-to-year movement** (raw bands included). But
  cross-parcel *spread* grows 1.37× into the L7-only era and spikes in 2021–23.

**It is the DATA, not the model (RESULTS.md §7.0.3).** The panel was re-inferred with
**LTAE** (different architecture, no statics at all) and the gate **fails harder**: flicker
**0.780** vs LightGBM's 0.428, S4 still fails. A third run with `lightgbm_nometa_nolat` gives
a clean monotone ordering — flicker 0.428 (has `centroid_lat`) → 0.547 (no lat) → 0.780 (no
statics at all) — while k=0 accuracy barely moves (0.746/0.752/0.712). **Time-invariant
features manufacture *stability* exactly as `frac_l7` manufactured *change*;** S5 is therefore
gameable, LightGBM's 0.428 is the optimistic end and LTAE's 0.780 the honest one. The El Niño
arm now runs through the model registry (`--elnino-model`) and **both architectures put
`PERENNIAL` recall on 1998 at ~0.02–0.03** (controls 0.54/0.66). "Try the other model" is a
closed route: any fix must change the data or the estimand, not the classifier.

**No trajectories, transitions or area estimates exist, and none should be produced.** That
code is built and unit-tested; it stays unrun. RESULTS.md §7.0.2 lists the options (the most
promising: drop per-parcel annual trajectories, report an aggregate share with CIs).

**2026-08-07 — option 2 ("restrict the baseline") was tested and is NOT sufficient
(RESULTS.md §8.3).** New flag `train --train-years 1999-2023` restricts the **train** side
only (both CV folds and the final refit); val/test membership is untouched so CV stays
comparable. Run `runs/perennial/lightgbm_nometa_from1999`, panel
`panel_predictions_from1999.parquet` (192,320 parcel-years, 1999–2023). **S4 flips to PASS
(0.146 → 0.063); S5 still FAILS (0.428 → 0.356 vs a 0.15 criterion).** CV cost is *zero* on
the years it still covers — pooled CV on `year ≥ 1999` validation parcels is 0.6412 vs the
baseline's 0.6415 — the −0.024 headline drop is entirely the 1998 val parcels it no longer
trains for. **A control gate is what makes this readable**: the *baseline* model's own
predictions truncated to ≥1999 (`--tag trunc1999`) also passes S4 at 0.076, so most of the S4
gain is the shorter panel (the failing k = −3 bin was 66 % panel-year 1996 and no longer
exists), not the retrain. Dropping 1998 costs only 40 of 5,647 `PERENNIAL` train parcels and
3 of 221 regions — **no spatial coverage gap** (panel→nearest-train-parcel median 1.21 →
1.52 km). The ≥1999 model does shift 2017/2023 (the other Piura flood years) toward
`PERENNIAL` by +0.016/+0.018 share — right direction, too small to matter. **Gate still
fails; no trend produced.**

**2026-08-07 — the El Niño *mechanism*, and a BSI audit (RESULTS.md §8.2).** §8.1 shows
*that* 1998 breaks the model; §8.2 shows *why*. The perennial signature is a **contrast**
with annual neighbours, and the flood collapsed it from both sides: the `NDVI_p25` class gap
falls **+1.47 SD → +0.46 SD** and `NDVI_amp` is annihilated (−0.60 → −0.10 SD), because
perennials lost canopy (floor 0.518→0.448) while flooded annual ground stayed green
(0.363→0.404). **`BSI_max` is the one feature that holds** (−0.76 → −0.90 SD; the *raw* gap
widens too, −0.049 → −0.072, so it is not a normalisation artefact). Numbers persisted to
`docs/figures/elnino_signature_collapse.csv`. ⚠️ **Do not read this as a missing-feature
problem** — LightGBM *has* `BSI_max` (rank 9/135) and still lost `PERENNIAL` on 1998; the
`meta`/`location` ablations never touched any `BSI_*` column. The deficit is **weighting**
(all `BSI_*` = 6.51 % of gain vs a dominant greenness family), and reweighting is a
*testable hypothesis, not a demonstrated fix*. LTAE/PSE-LTAE get per-date `BSI` as channel
11 of 11 but **no** precomputed `BSI_max`.

**2026-08-07 — the `rules` control was under-specified (RESULTS.md §4.1).** It reads only
its 3 configured columns, so `BSI_max` sat unused in the same store. Swapping
`NDVI_max`→`BSI_max` wins 5/5 folds (CV 0.388→0.437, pooled 0.395→0.451, acc 0.367→0.519)
and repairs a pathological class mix — the registered rule predicts `PASTURE_FALLOW`, its
*fall-through* branch, for 69 % of parcels against a true 22 %, which is why its accuracy is
*below* the majority baseline. **Not adopted**: `PERENNIAL` F1 *falls* 0.493→0.379, balanced
accuracy falls 0.468→0.455, and it forfeits the control's one clean win — threshold
stability (level spread 0.426–0.590 vs 0.548–0.568). Run persisted as
`runs/perennial/rules_3c_bsi/`; both are kept. Read S2's "+0.257" as ~+0.20.

**✅ RESOLVED — `frac_l7` (RESULTS.md §4.6).** `data.py` fed LightGBM every column except
`COD_PREDIO`/`label_id`, so acquisition metadata were model inputs; `frac_l7` was the
2nd-highest-gain feature and ramps 0.000 across the 1996–98 baseline → 1.000 from 2002.
Fixed: `train --drop-features meta|location|<cols>` (see `data.META_FEATURES`), and
**`infer()` now pins the column list to the model's saved `feature_names`** so an ablated
model cannot silently regain a feature the panel store still contains. Ablation cost
**+0.0013** macro-F1 (i.e. nothing — gain ≠ contribution, as the 12-class audit also found).
**Panel model is now `lightgbm_nometa`** (CV 0.648, T=1.409),
`runs/perennial/selected_model_panel.json`; it supersedes `selected_model.json` *for panel
inference only*. LTAE (CV 0.658) is co-admissible — it receives no statics at all — and lost
the ~0.02 tie-break on cost; it is the recommended sensitivity arm if the gate is ever passed.

**2026-08-05 additions** (all detailed in RESULTS.md):
* **LTAE was swept** (§4.5) to close the plan's gap. It does *not* overturn the LightGBM
  selection: 30 trials mean 0.639 ± 0.006, and LightGBM's 0.642 on the same folds sits at
  the ~70th percentile. Best-of-30 (0.650) is exactly what selection noise predicts.
* **⚠️ The locked test was used a SECOND time** — tuned LTAE, at the user's explicit request,
  for interest only. Nothing was selected on it. It is no longer a clean held-out estimate
  for any future selection. Tuned LTAE **beats LightGBM on all 5 CV folds (0.658 vs 0.647)
  but loses the locked test (0.661 vs 0.681)** — CV rank and test rank disagree, and spatial
  CV did not catch it. `test − CV` is +0.034 for LightGBM vs +0.003 for LTAE.
* **⚠️ Registration year is badly confounded with label AND with the split** — 1998 is 0.4 %
  perennial vs 2000's 33.8 %, and the purely spatial split made the test set 48.3 % 1998 vs
  trainval's 32.9 %. Coupled with the 1997–98 El Niño this is the main unquantified threat to
  the panel baseline. A controlled train-on-1999/2000, test-on-1998 diagnostic is designed but
  not run.
* **Tenure (`ESTADO en RRPP`) is NOT in any processed table** — `load_sset()` omits it from
  `usecols`; read it from the raw xlsx. Panel coverage is 100 %; a within-region
  fixed-effects tenure contrast is feasible on 93 % of the at-risk pool without re-sampling.
* **⚠️ Silent GEE hangs are RECURRING** — three in one run (13.4 h unsupervised, then two
  ~30 min ones), at unrelated points in unrelated years. Process alive, no output, no
  exception, no retry: `socket.setdefaulttimeout(120)` and the 5-try backoff both fail to
  fire because **`_retry` only catches errors that *raise*, and a hang is not an error.**
  Budget ~1 hang per 30–40 min of sustained GEE work. **The durable fix (a wall-clock
  deadline per chunk in `_retry`) is still not implemented** and this pipeline has the same
  exposure. Interim: a stall-watchdog, restarted with *only* the outstanding years (a full
  `--years` range replays every cached year on each recovery). Note `--years` needs `lo-hi`,
  so one year is `2023-2023`.

Three things from that work that matter to *this* pipeline:
* **`run_coverage` had a cross-year contamination bug** (globbed all chunk files, deduped on
  `COD_PREDIO` alone) — fixed; it never affected the single-year training store.
* **`assemble` merges years silently** if a parcel ever has pixels in two years — now
  guarded by `assert_one_year_per_parcel` + a `years`/`out_dir` filter.
* **Landsat over Piura effectively starts in 1996** (1992 has *zero* clear acquisitions;
  1995 gate pass 4.7 % vs 1996's 100 %), and 1997 is half-lost to the El Niño.
* **The panel is TM/ETM+ only (L5+L7), 1996–2023** — training is 52.8 % L5 / 47.2 % L7 and
  0.0 % OLI, so admitting L8/L9 would infer 2013+ on unseen radiometry. L7 acquisitions stop
  after 2023 (2024 gate pass 0.0 %), which is what bounds the panel.
  See [`docs/perennial/l7_coverage.md`](docs/perennial/l7_coverage.md).

**➡ Status (2026-07-30): medium run, full run, and 12-class retrain+sweeps all DONE.** The medium
de-risking run passed (1998 gate ~95%), the full extraction ran (4.53M pixel-obs, 47,851 parcels,
~2.5 h, milestone 3 passed), and all three models were retrained + Optuna-swept on the revised
12-class label space. **Current state of the art: tuned LTAE, pooled spatial-CV macro-F1 0.427**
(> PSE-LTAE 0.421 > LightGBM 0.383; majority baseline acc 0.425). A one-page summary is in
[`docs/SUMMARY.md`](docs/SUMMARY.md); full detail + the live open-decisions list is in
[`docs/PIPELINE.md`](docs/PIPELINE.md) §1. **The remaining next step is the single locked-test
evaluation** (never yet touched) of the chosen model — plus, optionally, the `max_gap` ablation and
temperature scaling before relying on confidence-based abstention.

1. ~~Assemble the training table~~ **DONE** — `training_crop_polygon.parquet`; label policy now applied
   in the pipeline (`labels.py`): intercrops resolved via a config `merge` map, fallow/land-prep
   handled as land-cover/drop. ~1 crop-year per polygon, so it is **one row per parcel**.
2. ~~GEE feature extraction~~ **BUILT** — `features/landsat_gee.py` (coverage gate + raw dated pixels)
   + `features/assemble.py` (whole-year summaries/harmonics for LightGBM; per-date & pixel-set tensors
   for the attention models). Not yet run at full scale.
3. ~~Baseline classifier~~ **BUILT** — LightGBM/LTAE/PSE-LTAE with a spatially blocked, buffered
   train/test split (`splits.py`). Not yet trained on real data.
4. **Resolve with the domain side** (still open): per-crop planting/harvest windows given unreliable
   dates; how to treat intercropped parcels; whether a finer-grained per-parcel date source exists.

## 9. Gotchas for an agent

- Always `uv run …`; the notebook needs the project venv (Python 3.11).
- GEE cells need `earthengine authenticate` + project `peru-crop-classifier`; without it, 8a onward
  fails. `ee.Initialize` also needs a GCP project (not just auth).
- Read every large `.dta` (`IV_CENAGRO`, `Base_...`, both `grafica_*`, `qgis/PIURA.dta`) with
  `pyreadstat.read_dta(path, encoding="latin1")`; for `IV_CENAGRO` always pass `usecols` (938 MB /
  409 cols). The census name lives in **unlabelled** cols `P009_01/02/03` — check values, not labels.
- Shapefile is EPSG:32717; reproject to 4326 for GEE/`cx` lon-lat slicing and to 3857 for contextily.
- Editing notebook cells programmatically: cells contain triple-double-quoted docstrings, so build
  source strings with `'''…'''` to avoid delimiter clashes (bit us twice).
- Full re-runs of notebook 01 are ~5–6 min and hit the network; never start a second concurrent
  `nbconvert` on it (see §2). The §2 centroid plot is a plain scatter by default (`fetch_basemap=False`)
  because a department-wide tile fetch there repeatedly stalled the run; the satellite-overlaid views
  live in §§5,7,8,10.
- Notebook-01 paths are relative to `notebooks/` (CWD when nbconvert runs it): use `RAW =
  Path("../data/raw")`, i.e. `../data/raw/…`, not `data/raw/…` (a root-relative path fails).
