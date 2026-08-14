# Piura datasets & merges — plain-language summary

Goal: build `(polygon, crop, year)` training examples for a Piura crop classifier. This doc explains
**what each raw dataset is**, **how they were merged**, **the yield of each merge**, and **which
dataset to use for the classifier**. All figures are Piura-only.

> ## ⚠️ 2026-08-07 — the data is now NATIONAL, and these Piura paths are superseded
>
> Three new folders cover the whole country and **contain the Piura files byte-identically**
> (sha256 verified — see [`all_peru/DATA_AUDIT.md`](all_peru/DATA_AUDIT.md) §1):
>
> | new folder | contents | replaces |
> |---|---|---|
> | `data/raw/BD_SSET/` | 8 multi-department crop-registry workbooks, ~5.6 M rows | #1 below |
> | `data/raw/Grafica_Tabular/` | 15 bridge `.dta`, one per department, ~1.87 M rows | #2 below |
> | `data/raw/QGIS/<DEPT>/` | 24 parcel shapefiles, ~2.9 M polygons | #4, #5 below |
>
> **Only 15 departments are linkable and 14 yield data** — the bridge is mandatory and
> `Grafica_Tabular/` ships only 15 files; Callao then yields nothing. Result: **946,872
> linked polygons (14.3x Piura), label years spread over 1997–2006** instead of ~80 % in
> 1998–99.
>
> **Four traps in the new folders all fail silently** — zero-padded bridge keys, attribute
> tables under the wrong basename, 3D geometry that Earth Engine rejects, and departments
> whose rows sit in several workbooks. Read `all_peru/DATA_AUDIT.md` §4 **before** touching
> them; each is fixed in `crop_classifier/allperu/` and pinned by `tests/test_allperu.py`.
>
> Everything below remains accurate for Piura and for the merge logic, which is unchanged.

---

## 1. The raw datasets (`data/raw/`)

| # | file | what it is | rows | key fields |
|---|------|-----------|------|-----------|
| 1 | `BD SSET(...).xlsx` | **PETT land-titling crop registry** — one row per parcel-crop declaration (a panel) | 348,076 | `Codigo SSET`, `CULTIVO`, `FECHA EMPADRONAMIENTO`, `NOMBRES`, `DNI`, `AREA` |
| 2 | `grafica_tabular_Piura.dta` ⭐ | **cadastral bridge** carrying both keys — the best one | 158,427 | `CodigoSSET` ↔ `COD_PREDIO` |
| 3 | `grafica_tabular_catastro_Piura.dta` | older/smaller bridge, **superseded** by #2 | 109,796 | `CodigoSSET` ↔ `COD_PREDIO` |
| 4 | `qgis_stefany/...FINAL.shp` | **CENAGRO 2012 parcel polygons** — the geometry | 190,098 | `COD_PREDIO`, `NOMBRE`, `UBIGEO` |
| 5 | `qgis_stefany/PIURA.dta` | shapefile's attribute table — **no `CodigoSSET`** | 158,638 | `COD_PREDIO` only |
| 6 | `IV_CENAGRO_Piura.dta` | **2012 agricultural census** — one row per parcel×crop | 947,884 | census codes `P0xx`, crop `P024_03`, name `P009_01/02/03` |
| 7 | `IV CENAGRO - Tabla_Cultivos_Totales.xlsx` | census crop-code → crop-name dictionary (Preg. 024) | 3,351 codes | `CODIGO` → `TITULO` |
| 8 | `Base_Cenagro_PETT_Piura.dta` | a **prior** census↔PETT merge — used only for comparison | 142,348 | census codes + `COD_PREDIO`, `CodigoSSET`, `CULTIVO` |

Two independent "crop" sources exist: the **PETT/SSET registry** (#1, ~1998–2007) and the **2012
census** (#6). They never share a parcel code — see §3.

---

## 2. Profiling the two crop sources

### BD SSET (PETT registry) — #1
- **177,695 of 348,076 rows have a crop label (51%)**; the rest are blank `CULTIVO`.
- **209,485 distinct parcels** (`Codigo SSET`).
- **Panel:** 125,879 parcels (60%) appear in **more than one** SSET record…
- …but only **7,002 parcels (7.1%) ever record a *different* crop** over time (and only ~1% a
  different *clean/whitelist* crop) — multi-record is mostly re-declaration of the same crop.
- **Farmers own multiple parcels:** of 94,262 distinct farmers (by `DNI`), **52,846 (56%) have >1
  parcel** (max 242).

### IV CENAGRO (2012 census) — #6
- **289,824 crop rows** (`P024_03` not null) across **145,890 producers** (`NPRIN`).
- **106,795 producers have ≥1 crop; 47,500 (44%) declare >1 distinct crop** (census captures
  intercropping/rotation better than SSET).
- **61,791 producers (42%) work >1 parcel** (max 33).
- Carries the **farmer name** (`P009_01`=paternal surname, `P009_02`=maternal, `P009_03`=given
  names) but **no `COD_PREDIO`/`CodigoSSET`/DNI-number** → can only reach PETT by name.

---

## 3. The merges and their yields

### Merge A — PETT crop → polygon  ⭐ **(notebook 03, recommended for the classifier)**

Joins on **real keys only** (no name-matching):
`BD SSET (crop, year)` → `grafica_tabular_Piura.dta` (bridge) → `qgis` polygons.

**Bridge choice was the decisive lever:**

| bridge | SSET crop-keys covered | crop records with a polygon | distinct polygons |
|---|---|---|---|
| `grafica_tabular_catastro` | 59.8% | 103,757 | 57,971 |
| **`grafica_tabular_Piura`** ⭐ | **68.8%** | **116,599** | **66,363** |

Catastro's pairs are a **subset** of `grafica_tabular_Piura` (adds 0), and `qgis/PIURA.dta` has no
`CodigoSSET` so it recovers nothing. Using the better bridge gains **+12,842 records / +8,392
polygons**.

**Result:** **116,599 crop records over 66,363 polygons** → **68,717 distinct `(polygon, crop,
year)`** (the rest are co-owner duplicates of the same parcel-crop-year).
**Fields:** `COD_PREDIO, geometry, crop, year, CodigoSSET, area_m2, AREA_S_HA, fecha, owner`.

⚠️ **Almost no time series:** only **1 of 66,363 polygons** has crops in more than one distinct year.
SSET is a single **~1998–99 titling snapshot** per parcel, and the "year" is the *registration*
date (batch/stub values, e.g. `2001-01-01`), not a growing season.

### Merge B — CENAGRO census → PETT, by farmer NAME (notebook 02)

The census shares no code with PETT, so it is matched on the **normalised farmer name** (3 different
formats parsed into components `name1..4`; routes: exact / token-set / core). Adds the census's own
2012 crop declaration + demographics.

- **45,942 producers linked to a polygon** on names alone (31.5% of 145,890).
- **Union with the prior `Base` merge = 91,636 producers** (name-only 8,722 · both 37,220 · Base-only
  45,694) — the most complete census↔PETT link available.
- Confidence flags: **high 15,317 · medium 23,817 · low 6,808**.
- ⚠️ **Exact-parcel uncertainty:** no parcel-level key exists, so even *high*-confidence links agree
  with `Base` on the exact `COD_PREDIO` only ~43% of the time (multi-parcel owners / shared names).
  The name link is trustworthy for **person + district + crop**, weak for the **exact polygon**.

### Comparison to the previous merge (`Base_Cenagro_PETT_Piura.dta`)

| | prior merge (`Base`) | Merge A (notebook 03) | Merge B (notebook 02) |
|---|---|---|---|
| producers / predios linked | 82,914 / 26,028 | — / **66,363** | 45,942 (union **91,636**) |
| crop records | 82,914 (1 crop each) | **116,599** (full panel) | 129,950 (census+SSET) |
| carries polygon geometry? | ❌ no | ✅ **yes (99%)** | ✅ yes |
| join basis | name (fuzzy, opaque) | **real keys** | name (transparent, flagged) |

`Base` linked more *producers* (its fuzzy name match), but carried **no geometry**, only **one** crop
per producer, and reached **fewer distinct predios** than Merge A.

---

## 4. Recommended dataset for the classifier

**Use Merge A — notebook 03's output.** It is the only set with reliable `(geometry, crop, year)` via
real keys.

Files (join on `COD_PREDIO`):
- `data/processed/pett_polygons.parquet` — 66,363 polygon geometries (EPSG:4326)
- `data/processed/pett_crop_year_distinct.parquet` — 68,717 `(COD_PREDIO, crop, year, …)` rows

```python
import geopandas as gpd, pandas as pd
poly = gpd.read_parquet("data/processed/pett_polygons.parquet")
cy   = pd.read_parquet("data/processed/pett_crop_year_distinct.parquet")
gdf  = poly.merge(cy, on="COD_PREDIO")     # ready (geometry, crop, year)
```

**Before modelling, still to do:**
- **Crop-label cleanup:** there are 4,842 raw free-text labels. Drop **fallow/non-crop** (14.4%:
  `EN DESCANSO`, `TERRENO EN DESCANSO`, etc.) and decide an **intercrop** policy (4.2%: `X Y Z`). A
  clean single-crop whitelist leaves **~55,900 rows** (dominated by `ARROZ` 25.5k, then `MAIZ`,
  `ALGODON`, `TRIGO`, `MANGO`…).
- **Modelling grain:** effectively **one row per parcel** (no multi-year), so no per-parcel-year
  panel — a second date only exists via the weaker census link (Merge B).
- **Spatially blocked train/test split** — crops grow in single-crop blocks (~86% of neighbours
  share a crop), so a random split leaks. Split by area/block, never by random rows.
- **Sensor:** crop years are pre-2015, so only **Landsat 5/7** (30 m) covers them; Sentinel-2 is
  location-check only.

**Do NOT use** `grafica_tabular_catastro_Piura.dta` (fewer matches), `qgis/PIURA.dta` (can't bridge),
or the raw `Base_...` merge (no geometry) as the crop→polygon source.
