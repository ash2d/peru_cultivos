# Data

Every raw dataset, how they link, what each merge yields, and the traps. `data/` is
**gitignored and local-only** (~20 GB).

Modelling results are in [`RESULTS.md`](RESULTS.md); the code that reads all of this is in
[`PIPELINE.md`](PIPELINE.md).

---

## 1. Raw inventory — `data/raw/`

The data is **national**. Three folders cover the whole country; read them through
`crop_classifier.allperu.sources`, which handles the four silent traps in §4.

| folder | contents |
|---|---|
| `BD_SSET/` | 8 crop-registry workbooks, ~5.6 M rows |
| `Grafica_Tabular/` | 15 bridge `.dta` files, ~1.87 M rows |
| `QGIS/<DEPT>/` | 24 parcel shapefiles, ~2.9 M polygons |

The older Piura-only files (`BD SSET(MOQUEGUA-PASCO-PIURA).xlsx`, `grafica_tabular_Piura.dta`,
`qgis_stefany/`) are **byte-identical (sha256 verified)** to their members of the national
folders, so no Piura result is invalidated. Rebuilding Piura through the national code path
reproduces its figures exactly (66,352 polygons / 80,618 records). Prefer the national paths.

### 1.1 BD SSET — the crop registry

One row per parcel-crop declaration, from the SSET/PETT land-titling programme. **Not a
population census.**

* Key columns: **`Codigo SSET`** (parcel key — the join key to the crop side), `CULTIVO`
  (free-text crop label, ~42 % null in Piura), `AREA` (in **m²**, dirty — negatives and
  outliers), `FECHA EMPADRONAMIENTO` (registration date, **unreliable** — §5),
  `ESTADO en RRPP` (tenure at declaration).
* It is a **panel**: a parcel reappears across years (mean 1.8 records/parcel, up to 27).
* ⚠️ `ESTADO en RRPP` is **not** in any processed table — `load_sset()` omits it from
  `usecols`. Read it from the raw workbook, or use `allperu/tenure.py`, which has resolved it
  onto `COD_PREDIO` for 1.78 M parcels.

### 1.2 `Grafica_Tabular/<Dept>.dta` — the cadastral bridge ⭐ mandatory

Stata, read with `pyreadstat.read_dta(path, encoding="latin1")`. Carries **both keys** —
`COD_PREDIO` (e.g. `7_6359485_001871`) **and** `CodigoSSET` — so it is the only thing
connecting polygons to crops. BD SSET has no `COD_PREDIO`; the shapefiles have no
`CodigoSSET`.

It also carries the cadastre's own titling status (`estado`) and its cut date
(`fech_tran` ≈ 2011–12) — the **second dated tenure observation** that made the DiD possible
(`RESULTS.md` §7).

⚠️ `grafica_tabular_catastro_Piura.dta` is a smaller, older bridge covering 59.8 % of SSET crop
keys against `grafica_tabular_Piura.dta`'s **68.8 %**, and its pairs are a strict subset. Use
the latter. Keep catastro for reference only.

### 1.3 `QGIS/<DEPT>/` — the parcel polygons

Piura: 190,098 polygons, native **EPSG:32717**; nationally the country spans UTM 17S/18S/19S.
Key `COD_PREDIO`; ~10.5 k Piura polygons have a **null code and are unlinkable — drop them**.

**Provenance is PETT, not CENAGRO**, despite `CENAGRO` in the filename. The `.shp.xml` lineage
shows a `CopyFeatures` from a pre-existing `PIURA.shp` into a `CENAGRO.gdb` container, feature
class `CAT_PIURA` ("Catastro" = cadastre). The census fields `COD_INEI` and `N_SEA` are **100 %
empty** while the PETT fields (`COD_PREDIO` 94.5 %, `NUM_PREDIO` 99.3 %, `COND_JUR` 100 %) are
populated. The 2012-04-10 date is the *packaging* date, not a survey date.

⚠️ `qgis_stefany/PIURA.dta` (158,638 rows, latin1) is the shapefile's attribute table. It has
**no `CodigoSSET`, so it cannot bridge anything.** Confirmed dead end.

### 1.4 The 2012 agricultural census — `IV_CENAGRO_Piura.dta`

947,884 rows × 409 cols, 938 MB. Read with `encoding="latin1"` **and always an explicit
`usecols`**. Long format: one row per parcel × crop.

* `P001`=dept, `P002`=prov, `P003`=dist, `P007X`=SEA, `P008`=UA, `NPRIN`=productor id,
  `NPARC`=parcel; crop code `P024_03`, sown area `P025`.
* `LONG_DECI`/`LAT_DECI` are **SEA centroids** — shared by all UAs in a SEA, **not** a parcel
  location.
* ⚠️ **The farmer name is here but the columns are unlabelled**: `P009_01` apellido paterno,
  `P009_02` apellido materno, `P009_03` nombres, `P009_04` razón social, `P009_05` RUC. Check
  values, not labels.
* **No `COD_PREDIO`, no `CodigoSSET`, no DNI** → the only link to PETT is the **name**.

Companions: `IV CENAGRO - Tabla_Cultivos_Totales.xlsx` (sheet `Permanente`, Preg. 024) maps
`P024_03` → crop names, covering 100 % of census codes. `Base_Cenagro_PETT_Piura.dta` is a
**prior name-based merge** (142,348 rows; `PETT==1` = 82,914 matched productors → 26,028
predios) kept only as a comparison target — it carries **no geometry**.

---

## 2. The two linkage chains

### Chain A — PETT crop → polygon ⭐ this is the training set

```
BD SSET (crop, FECHA→year) ──CodigoSSET──► Grafica_Tabular/<Dept>.dta ──COD_PREDIO──► QGIS polygons
```

All real keys, no name matching. Nationally: **946,872 linked polygons across 14 departments**,
726,808 of them resolving to one of the three land states. Piura alone: 116,599 crop records
over 66,363 polygons.

### Chain B — CENAGRO census → PETT, on the farmer name only

```
IV_CENAGRO name (P009_01/02/03) ──normalised name──► SSET / bridge / polygon NOMBRE
```

Recovers 45,942 productors to a polygon on its own; union with the prior `Base` merge = 91,636.
⚠️ **There is no parcel-level key**, so pinning the *exact* polygon is uncertain (~43 %
`COD_PREDIO` agreement with `Base`, even at high confidence). Good for the person, district and
crop; weak for the parcel. Not used in any current model.

---

## 3. Coverage — only 14 departments, and that is structural

`Grafica_Tabular/` ships **15 files**, and the bridge is mandatory, so only 15 departments can
be linked. Callao then yields nothing (395 polygons, 366 declarations, no surviving overlap).

* **14 with data:** Ancash, Arequipa, Ayacucho, Cajamarca, Huancavelica, Ica, La Libertad,
  Lambayeque, Lima, Moquegua, Pasco, Piura, Tacna, Tumbes.
* **Polygons but no bridge (8):** Amazonas, Apurímac, Cusco, Huánuco, Junín, Madre de Dios,
  Puno, Ucayali — ~653 k polygons unreachable from any crop label.
* **Crops but no polygons (2):** Loreto, San Martín.

⚠️ **Consequence:** there are **no sierra/selva labels and there never will be** from this
source. That is a permanent model-card limit, not a to-do.

---

## 4. ⚠️ Four traps, all of which fail SILENTLY

Each produces a plausible empty or column-less result rather than an error. Each is pinned by a
test in `tests/test_allperu.py`.

### 4.1 Zero-padded bridge keys — the big one

Most departments' bridge stores `CodigoSSET` **zero-padded to 9 characters** (`030406693`);
BD SSET stores it unpadded (`30406693`). A string join returns **zero** rows, which reads as
"this department has no linkable data".

> Ancash: **0 of 422,769 keys matched before the fix, 369,089 after.**

Fixed in `build_labels.canon_key`. **A no-op for Piura** (both sides already 9 digits).

### 4.2 Attribute tables under the wrong basename

19 of 24 departments ship `QGIS/ANCASH/ANCASH.dbf` beside
`CATASTRO_CENAGRO_ANCASH_…_FINAL.shp`. **GDAL opens such a shapefile happily and returns zero
attribute columns** — `COD_PREDIO` vanishes with no error. `sources.shapefile_view` builds a
symlink directory with consistent basenames; raw data is never modified.

### 4.3 A department's rows are not confined to "its" workbook

Selecting by the workbook whose *filename* mentions the department loses data — Lima's rows are
in three workbooks (165,923 vs 187,849 total), Cajamarca's in three, Piura's in three.
`build_dept_sset_caches` scans **every** workbook once and concatenates.

### 4.4 3D geometry, sheet and naming variants

* **La Libertad's cadastre is 3D** and Earth Engine rejects 3D GeoJSON outright
  (`EEException: Invalid GeoJSON geometry`) — 9.6 % of the sample, and it kills a multi-hour
  extraction partway through. Fixed by `build_labels.clean_geometry`.
* The Arequipa/Ayacucho/Cajamarca workbook holds 1.6 M rows across **`DATOS1` + `DATOS2`**
  (Excel's row limit). A `startswith("DATOS")` fix that took only the first sheet would have
  silently lost a department.
* Callao is filed under `PROV.CONST.DEL CALLAO`; Lima's rows also appear under
  `LIMA METROPOLITANA`. Both aliased in `sources._DEPT_ALIASES`.
* **Arequipa ships the same 135,780 parcels twice**, in UTM 18S and 19S. The 18S copy is used.
* Departments span UTM 17S/18S/19S, so the national split config projects the whole country to
  **UTM 18S** for one continuous block grid (scale error < 0.7 %, i.e. < 11 m on a 1.5 km
  buffer).

---

## 5. ⚠️ Caveats that bind every use of this data

* **Dates are unreliable.** `FECHA EMPADRONAMIENTO` clusters on stub/batch values —
  `2001-01-01` alone is ~35 % of Piura rows. Some areas have genuine campaign dates (tight
  day/week clusters), others are pure stubs. **Never trust an exact date to pick a scene**; use
  a season window. The year is a **titling date, not a verified growing season**.
* **Crops are strongly spatially autocorrelated.** ~86 % of adjacent Piura parcels share a crop
  against ~46 % under random placement — Piura grows in single-crop **blocks**. ⇒ **train/test
  splits MUST be blocked by area.** Nationally this is milder (0.72 at 0–100 m) but still
  binding: label agreement is **0.563 at 4–5 km** against a 0.361 baseline, so it never
  decorrelates inside the audit window and `buffer_m = 1500` controls leakage only partially.
  `splits.assign()` prints a warning saying so. **This is the quantitative reason model
  selection was made on leave-one-department-out.**
* **Intercropping is real (~14 % of declarations)** — `CAFE Y PLATANO`, `ARROZ Y MAIZ`. The
  label pipeline splits and keeps them as a **list per polygon** (16 % of linked polygons carry
  >1 crop); the modelling policy is a config `merge` map.
* **~1 crop-year per polygon.** Only 1 of 66,363 linked Piura polygons has crops in >1 distinct
  year. This is a single titling snapshot, **not a time series**. A second time point (2012)
  exists only in the census, via the weaker name link.
* **Multi-record ≠ multi-crop.** ~60 % of parcels have ≥2 SSET records but only ~7 % ever record
  a different *raw* crop label — mostly re-declaration.
* **AREA units differ:** the workbook's `AREA` is m² (dirty); the `.dta`/shapefile
  `area_ha`/`AREA_S_HA` are hectares.
* **Year confound (Piura).** Registration year is badly confounded with label *and* with the
  split — 1998 is 0.4 % perennial against 2000's 33.8 %, and the purely spatial split made the
  Piura test set 48.3 % 1998 against trainval's 32.9 %. Nationally this is much milder (test
  46.2 % `ANNUAL` vs trainval 42.0 %) but not zero.

---

## 6. Processed tables — `data/processed/`

Built by `uv run python -m crop_classifier.build_training_data` (Piura) or
`allperu labels` (national).

| file | rows | contents |
|---|---|---|
| **`training_crop_polygon.parquet`** ⭐ | 66,352 | Piura. One row per polygon, EPSG:4326: `COD_PREDIO`, `geometry`, `crops` (**list**), `crop_categories`, `n_labels`, `is_vegetated`, `codigo_sset`, `year` (modal) + `years`, `n_records`, `area_ha` |
| `training_crop_records.parquet` | 80,618 | Piura, exploded long form, one row per `(COD_PREDIO, CodigoSSET, crop, category, year)`, keeps the raw label. Deduped from 136,012 |
| `crop_normalization_map.csv` | — | every distinct raw label → normalised crops + categories + row count. The audit trail |
| `all_peru_full/modeling_parcels.parquet` | **726,808** | the national **population** |
| **`all_peru/modeling_parcels.parquet`** ⭐ | **56,419** | the national **modelling sample** |

**These two are not interchangeable.** 946,872 polygons link to a crop record; 726,808 resolve
to one of the three land states; the sample is sized to Piura's 56,419 so modelling cost is
unchanged.

### 6.1 Label cleaning

`crop_normalization.normalize_label(raw)` turns one dirty `CULTIVO` cell into a list of
`(crop, category)` pairs. It splits on every separator seen (`, / + & -`, whole-word `Y`/`CON`,
and **percentage weightings** — `CAFE 50%-PLATANO 50%`), fixes typos/accents/plurals and
regional synonyms via a curated map (`AROZ→ARROZ`, `ALGODONERO→ALGODON`), and tags each token
`crop` / `pasture` / `fallow` / `land_prep` / `unspecified`. **Nothing is dropped by crop
type** — non-crops are only *flagged*, and genuinely distinct crops are never merged
(`FRIJOL DE PALO` and `FRIJOL CAUPI` stay separate). 9,394 raw Piura labels → ~1,647 canonical
tokens.

### 6.2 ⚠️ The national sample's class mix is not the population's

| | ANNUAL | PASTURE_FALLOW | **PERENNIAL** |
|---|---|---|---|
| population share | 0.5002 | 0.4009 | **0.0989** |
| raw sample share | 0.4284 | 0.3700 | **0.2016** |
| **weighted sample share** | 0.5002 | 0.4009 | **0.0989** |

Square-root department allocation upweights small departments, and the small departments are
the coastal perennial ones (Tumbes sample 74.8 % perennial, Pasco 60.2 %, Huancavelica 5.2 %).
It is *good for training* (11,375 perennial parcels instead of ~5,600) and **fatal for any
area or share statistic computed from raw counts**. Use `sample_weight` / `population_weight`
— it reconstructs the population exactly (total weight = 726,808).

### 6.3 Label years — the main thing the national data adds

| year | 1996 | 1997 | 1998 | 1999 | 2000 | 2001 | 2002 | 2003 | 2004 | 2005 | 2006 | 2007 | 2008 | 2009+ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| population % | 0.8 | 12.2 | 13.9 | 16.4 | 4.9 | 3.7 | 4.0 | 13.8 | 6.9 | 4.7 | 8.8 | 4.1 | 4.4 | 1.3 |

Piura was ~80 % concentrated in 1998–99; nationally the mass spreads over 1997–2008 (≈94 %),
which is what makes temporal generalisation testable at all. 2010 onward is effectively empty.

### 6.4 National splits

`split_allperu.yaml`, metric CRS 32718: 5,693 blocks of 1 km, **677 regions of 5 km** as the
assignment unit. **Locked test 125 regions / 11,441 parcels (20.3 %) — UNSPENT.** Trainval 552
regions / 44,978 parcels, 5 `StratifiedGroupKFold` folds grouped on `region_id`. Coverage gate
`n_valid_obs ≥ 4`: 55,028 pass, 1,391 fail.

---

## 7. Satellite imagery

### 7.1 Landsat: L5 + L7 only, and the mission mixing is sound

**Mission policy (user decision): no L8/L9.** The training store is 52.8 % L5 (TM) / 47.2 % L7
(ETM+) / **0.0 % OLI** (530 observations). Admitting L8 would infer everything from 2013 on
using radiometry absent from training, with the sensor step landing exactly where an export-crop
expansion would also appear.

Two independent audits of the multi-mission handling both came back clean:

* **Radiometry is correct** — Collection 2 Tier 1 Level-2 surface reflectance, equivalent
  reflective bands mapped to a common six-band schema, correct C2 scale/offset, and masking of
  cloud, cloud-shadow, dilated-cloud, cirrus and radiometric-saturation flags. The measured
  L5↔L7 step in *this* dataset is **0.005–0.012 reflectance**.
* **In the panel, 0 of 30 mission-boundary steps exceed 2×** the within-era year-to-year
  movement, raw bands included.

⛔ The OLI harmonisation route is **closed** — see `RESULTS.md` §6.4. Do not reopen it.

**Archive limits (measured, not assumed):**

| | |
|---|---|
| **Landsat over Piura starts in 1996** | 1992 has **zero** clear acquisitions; 1995 gate pass 4.7 % vs 1996's 100 % |
| 1997 | half-lost to the El Niño |
| L7 usable | 2000–2023; launched April 1999 (half-year), **stops dead at 2024** (0.07 obs/parcel) |
| Thin years under *any* TM/ETM+ policy | **2009 (0.470)** and **2011 (0.373)** — L5 contributed nothing in 2008–11, its degraded final years |
| Cost | ~0.30 s/parcel-year in 1996–2010, ~0.50 in 2011–18, ~0.77 in 2019–24 |

Raw numbers: `figures/l7_coverage.csv`, `figures/panel_budget.csv`.

⚠️ A parcel's L7 SLC-off loss is **11 pp, not the nominal scene-level 22 %**.

### 7.2 Sentinel-2 — for the 2019+ endpoint campaign only

**Year-matching Sentinel-2 to the PETT labels is impossible** (linked crops are pre-2015, and
the wider bridge does not extend the years — 2015+ is 40 records). S2 is used only for the
endpoint labelling campaign, where it gives a **median 19–113 clear dates per
parcel-agricultural-year by department, against the Landsat store's 13–24**. See
[`s2_labelling/plan.md`](s2_labelling/plan.md).

### 7.3 Esri high-resolution basemap

Probed for date and resolution before use (`allperu esri-dates`): **87.6 % of the eligible
universe is ≤1.2 m and ≥2019**.

⚠️ **Piura is one of the three worst-covered departments** (0.61 eligible, with Cajamarca 0.60
and Pasco 0.39; eight departments are 1.00). Every earlier strand was built on Piura — a
campaign scoped to Piura would have lost a third of its draw and it would have looked like a
sampling bug.

⚠️ **Esri returns a valid flat-grey image above the zoom it serves**, not an error. Derive zoom
from the probed resolution (30 cm→z19, 60 cm→z18, 1.2 m→z17) and keep a placeholder detector.
