# All-Peru data audit

> What arrived on 2026-08-07, what it links to, and every trap found while wiring it up.
> **§5 is the summary of the data itself** — population vs modelling sample, class mix, crop
> and year coverage, per-department allocation, and the splits.
> Design decisions are in [`plan.md`](plan.md); modelling results in [`RESULTS.md`](RESULTS.md).

## 1. The new Piura files are byte-identical to the old ones ✔

The first question was whether the new folders are consistent with the files the Piura
pipeline has been using. They are — not merely equivalent, **identical**:

```
$ shasum -a 256 <old> <new>
2eddf7f1648196b692fd26c307658d4ed1e6dc7298cb5ceb8822da58117b05e7  qgis_stefany/CATASTRO_CENAGRO_PIURA_WGS84_Z17S_FINAL.shp
2eddf7f1648196b692fd26c307658d4ed1e6dc7298cb5ceb8822da58117b05e7  QGIS/PIURA/CATASTRO_CENAGRO_PIURA_WGS84_Z17S_FINAL.shp
9e76f4353278932a8ba26f8e621e00b6297a5ae63f7026cee077d025f237a697  qgis_stefany/…FINAL.dbf
9e76f4353278932a8ba26f8e621e00b6297a5ae63f7026cee077d025f237a697  QGIS/PIURA/…FINAL.dbf
df4fd5028e0a22bffbcf22473b4dfcbf7f92a77e44ab09815904feafc61e7f90  qgis_stefany/PIURA.dta
df4fd5028e0a22bffbcf22473b4dfcbf7f92a77e44ab09815904feafc61e7f90  QGIS/PIURA/PIURA.dta
96e90c6dde7caa1f74031da528ef614af55c6580a61a8095109592982d58f0aa  BD SSET(MOQUEGUA-PASCO-PIURA).xlsx
96e90c6dde7caa1f74031da528ef614af55c6580a61a8095109592982d58f0aa  BD_SSET/BD SSET(MOQUEGUA-PASCO-PIURA).xlsx
9493e0effc64486d4457132e68baa3145e45250d2690fd2b0de98de4abb509a6  grafica_tabular_Piura.dta
9493e0effc64486d4457132e68baa3145e45250d2690fd2b0de98de4abb509a6  Grafica_Tabular/Piura.dta
```

So **no Piura result is invalidated by the new drop**, and the old paths can be retired in
favour of the new folders.

**Second, stronger check — an end-to-end rebuild.** Piura was rebuilt through the *new*
all-Peru code path and reproduces the documented figures exactly:

| quantity | CLAUDE.md §5b | rebuild via `allperu.build_labels --only PIURA` |
|---|---|---|
| polygons | 66,352 | **66,352** ✔ |
| exploded records | 80,618 | **80,618** ✔ |
| polygons with >1 crop label | 16 % | **16.0 %** ✔ |
| polygons with >1 crop-year | 1 | **1** ✔ |

*(One caveat, stated because it is a real if tiny difference: once the loader scans **all**
workbooks rather than only the one named after the department — see §4 — Piura picks up 341
extra crop declarations filed in other workbooks, 348,076 → 348,417. Those are genuine Piura
rows that the single-workbook read was silently dropping. It does not change the polygon or
record counts above, which are post-join.)*

## 2. What the new data actually adds

| source | old (Piura only) | new (all Peru) |
|---|---|---|
| crop registry | 1 workbook, 509,684 rows | **8 workbooks, ~5.6 M rows** |
| bridge | 1 `.dta`, 158,427 rows | **15 `.dta`, ~1.87 M rows** |
| polygons | 1 shapefile, 190,098 | **24 shapefiles, ~2.9 M** |

## 3. Only 15 of 24 departments are linkable, and only 14 yield data

The link chain is fixed by what carries which key:

```
BD SSET (crop, year) ──CodigoSSET──► Grafica_Tabular/<Dept>.dta ──COD_PREDIO──► QGIS polygons
```

**The bridge is mandatory**: BD SSET has no `COD_PREDIO`, and the shapefiles have no
`CodigoSSET`. `Grafica_Tabular/` ships only 15 files, so only 15 departments can be linked.

* **Linkable (15):** Ancash, Arequipa, Ayacucho, Cajamarca, Callao, Huancavelica, Ica,
  La Libertad, Lambayeque, Lima, Moquegua, Pasco, Piura, Tacna, Tumbes.
* **...of which Callao yields nothing** — 395 polygons and 366 declarations, with no
  surviving overlap after the join. **14 departments carry the data.**
* **Polygons but no bridge (8):** Amazonas, Apurímac, Cusco, Huánuco, Junín, Madre de Dios,
  Puno, Ucayali — ~653 k polygons unreachable from any crop label.
* **Crops but no polygons (2):** Loreto, San Martín.

See `plan.md` §3 (decision D1) for why the CENAGRO name-match is *not* used to rescue the
eight bridge-less departments.

## 4. Four traps, all of which fail silently

Every one of these produces a plausible-looking empty or column-less result rather than an
error, so each is pinned by a test in `tests/test_allperu.py`.

### 4.1 Zero-padded bridge keys (the big one)

Most departments' bridge stores `CodigoSSET` **zero-padded to 9 characters**
(`030406693`); BD SSET stores the same key unpadded (`30406693`). A string join returns
**zero** rows, which reads as "this department has no linkable data".

> Ancash: **0 of 422,769 keys matched before the fix, 369,089 after.**

Fixed in `build_labels.canon_key` (strip whitespace + leading zeros, both sides). **It is a
no-op for Piura** — both sides are already 9 digits — verified by rebuilding Piura
bit-identically.

### 4.2 Attribute tables under the wrong basename

19 of 24 departments ship `QGIS/ANCASH/ANCASH.dbf` beside
`CATASTRO_CENAGRO_ANCASH_…_FINAL.shp`. **GDAL opens such a shapefile happily and returns
zero attribute columns** — `COD_PREDIO` vanishes with no error at all. Record counts were
checked to match for every department, confirming these really are the shapefiles' own
attribute tables. `sources.shapefile_view` builds a symlink directory with consistent
basenames; raw data is never modified.

### 4.3 A department's rows are not confined to "its" workbook

Selecting rows from the workbook whose *filename* mentions the department loses data:

| department | declarations in its "own" workbook | total across all workbooks |
|---|---|---|
| Cajamarca | 1,083,707 | **1,086,075** (3 workbooks) |
| La Libertad | 629,218 | **629,220** (2) |
| Lima | 165,923 | **187,849** (3) |
| Piura | 348,076 | **348,417** (3) |
| Lambayeque | 135,806 | **135,808** (2) |
| Ayacucho | 294,662 | **294,732** (3) |
| Ancash | 694,230 | **694,499** (2) |

`build_dept_sset_caches` therefore scans **every** workbook once and concatenates.

### 4.4 Sheet and naming variants

* The Arequipa/Ayacucho/Cajamarca workbook holds 1.6 M rows across **`DATOS1` + `DATOS2`**
  (Excel's 1,048,576-row sheet limit). Assuming a single `DATOS` sheet throws
  `Worksheet named 'DATOS' not found` — and, worse, a `startswith("DATOS")` fix that took
  only the first sheet would have silently lost a department.
* Callao is filed under **`PROV.CONST.DEL CALLAO`**, not `CALLAO`; Lima's rows also appear
  under `LIMA METROPOLITANA`. Both are aliased in `sources._DEPT_ALIASES`.
* **Arequipa ships the same 135,780 parcels twice**, projected to UTM 18S and 19S. The 18S
  copy is used (it is the one with a correctly-named `.dbf`; everything is reprojected to
  EPSG:4326 downstream regardless).
* Departments span **UTM 17S / 18S / 19S**. Block and region ids must come from one
  continuous grid, so the all-Peru split config projects the whole country to UTM 18S
  (scale error < 0.7 %, i.e. < 11 m on the 1.5 km buffer).

## 5. The population and the modelling sample

Two tables exist and they are **not** interchangeable:

| | file | rows |
|---|---|---|
| **Population** — every 3-class-eligible linked parcel | `data/processed/all_peru_full/modeling_parcels.parquet` | **726,808** |
| **Sample** — what is actually modelled | `data/processed/all_peru/modeling_parcels.parquet` | **56,419** |

(946,872 polygons link to a crop record; 726,808 of those resolve to one of the three land
states. The sample is sized to Piura's 56,419 so total modelling cost is unchanged.)

### 5.1 Composition

| | population | sample |
|---|---|---|
| ANNUAL | 50.0 % | 42.8 % |
| PASTURE_FALLOW | 40.1 % | 37.0 % |
| **PERENNIAL** | **9.9 %** | **20.2 %** |
| distinct `crop_set` values | 56,743 | 7,975 |
| single-crop / mixed-priority labels | 83.0 % / 17.0 % | 81.7 % / 18.3 % |
| median parcel area | 0.43 ha | 0.57 ha |
| total area | 942,278 ha | 100,256 ha |

Top crops in the sample: `DESCANSO` (fallow) 16.1 %, `MAIZ` 8.6 %, `ALFALFA` 7.7 %,
`ARROZ` 3.0 %, `TRIGO` 2.7 %, `ALGODON` 2.4 %, `BARBECHO` 2.0 %, `PAPA` 1.6 %.

> ⚠️ **The sample doubles the perennial share (9.9 % → 20.2 %) and enlarges the median
> parcel.** This is a direct consequence of sqrt-proportional allocation: the small
> departments are the coastal perennial ones. It is good for training and **fatal for any
> area or share statistic computed from raw counts**. Use `population_weight`, which
> reconstructs the population exactly.

### 5.2 Year coverage — the main thing the national data adds

Piura was ~80 % concentrated in 1998–99. Nationally the label years spread across 1997–2008
(≈94 % of parcels), which is what makes temporal generalisation testable at all.

| year | 1996 | 1997 | 1998 | 1999 | 2000 | 2001 | 2002 | 2003 | 2004 | 2005 | 2006 | 2007 | 2008 | 2009+ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| population % | 0.8 | 12.2 | 13.9 | 16.4 | 4.9 | 3.7 | 4.0 | 13.8 | 6.9 | 4.7 | 8.8 | 4.1 | 4.4 | 1.3 |
| sample % | 0.9 | 11.6 | 13.3 | 15.1 | 6.6 | 6.6 | 4.0 | 11.7 | 5.9 | 6.6 | 8.1 | 3.0 | 4.9 | 2.0 |

Range is 1996–2019, but 2010 onward is effectively empty (< 0.6 % combined). The year is a
**titling date, not a verified growing season** — see the root `CLAUDE.md` §6.

### 5.3 Per-department allocation

Sampling rate varies 10x by design (`allperu.sample`: whole 5 km regions, sqrt-proportional
quota, floor 800/dept, cap 220 parcels/region, seed 42):

| dept | population | sample | pop % | samp % | rate | sample PERENNIAL % |
|---|---:|---:|---:|---:|---:|---:|
| CAJAMARCA | 159,021 | 7,210 | 21.9 | 12.8 | 4.5 % | 12.2 |
| ANCASH | 154,263 | 7,113 | 21.2 | 12.6 | 4.6 % | 11.6 |
| LA_LIBERTAD | 82,311 | 5,411 | 11.3 | 9.6 | 6.6 % | 6.9 |
| AYACUCHO | 80,723 | 5,367 | 11.1 | 9.5 | 6.6 % | 21.1 |
| AREQUIPA | 60,354 | 4,749 | 8.3 | 8.4 | 7.9 % | 10.6 |
| PIURA | 56,422 | 4,618 | 7.8 | 8.2 | 8.2 % | 28.3 |
| LIMA | 42,150 | 4,100 | 5.8 | 7.3 | 9.7 % | 31.8 |
| ICA | 31,274 | 3,642 | 4.3 | 6.5 | 11.6 % | 27.4 |
| LAMBAYEQUE | 15,491 | 2,801 | 2.1 | 5.0 | 18.1 % | 6.1 |
| MOQUEGUA | 14,128 | 2,710 | 1.9 | 4.8 | 19.2 % | 25.1 |
| HUANCAVELICA | 13,079 | 2,638 | 1.8 | 4.7 | 20.2 % | 5.2 |
| TACNA | 7,861 | 2,225 | 1.1 | 3.9 | 28.3 % | 22.0 |
| PASCO | 5,663 | 2,010 | 0.8 | 3.6 | 35.5 % | 60.2 |
| TUMBES | 4,068 | 1,825 | 0.6 | 3.2 | 44.9 % | 74.8 |

The rightmost column is the mechanism behind the perennial inflation: the most heavily
sampled departments (Tumbes 74.8 %, Pasco 60.2 %) are the most perennial.

### 5.4 Splits

Spatially blocked, from `splits_meta.json` (config `split_allperu.yaml`, metric CRS 32718):

| | |
|---|---|
| 1 km blocks (reporting) / 5 km regions (assignment unit) | 5,693 / 677 |
| **Locked test** | 125 contiguous regions, **11,441 parcels (20.3 %)** — **unspent** |
| **Trainval** | 552 regions, **44,978 parcels** |
| **CV folds** | 5, `StratifiedGroupKFold` grouped on `region_id`, 8,994–8,999 each |
| 1.5 km buffer dead-zone excluded | 2,208 (test), 861–1,889 (per fold) |
| Coverage gate `n_valid_obs ≥ 4` | 55,028 pass / 1,391 fail |

Test and trainval are mildly unbalanced — test is 46.2 % ANNUAL vs trainval 42.0 %, and
1999/2001 are over-represented in test (17.5 %/10.4 % vs 14.4 %/5.6 %). Much milder than
Piura's 48.3 %-vs-32.9 % 1998 confound, but not zero.

> ⚠️ **The buffer is smaller than the measured decorrelation range.** The autocorrelation
> audit finds label agreement still **0.563 at 4–5 km** against a **0.361** baseline — it
> never decorrelates inside the 5 km audit window, so `buffer_m = 1500` controls neighbour
> leakage only partially, and `splits.assign()` prints a warning saying so. This is the
> quantitative reason spatial CV cannot be trusted alone here, and why model selection was
> made on **leave-one-department-out** instead (`RESULTS.md` §6.2).

## 6. Per-department yield

See `data/processed/all_peru_full/build_report.csv` for the machine-readable version
(declarations, bridge pairs, polygons, linked records, key coverage, seconds per stage).
