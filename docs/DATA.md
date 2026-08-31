# Data

Every raw dataset, how they link, what each merge yields, and the traps. `data/` is
**gitignored and local-only** (~20 GB).

Modelling results are in [`RESULTS.md`](RESULTS.md); the code that reads all of this is in
[`PIPELINE.md`](PIPELINE.md).

---

## 1. Raw inventory — `data/raw/`

The data is **national**. Four folders cover the whole country; read the first three through
`crop_classifier.allperu.sources`, which handles the four silent traps in §4.

| folder | contents |
|---|---|
| `BD_SSET/` | 8 crop-registry workbooks, ~5.6 M rows |
| `Grafica_Tabular/` | 15 bridge `.dta` files, ~1.87 M rows |
| `QGIS/<DEPT>/` | 24 parcel shapefiles, ~2.9 M polygons |
| `Cenagro_IV/` | the 2012 census, **all 25 departments**, slimmed to 76 columns (§1.5) |

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

### 1.4 The 2012 agricultural census, Piura — `IV_CENAGRO_Piura.dta`

⚠️ **This is one department of 25.** The other 24 are in `Cenagro_IV/` — see §1.5.

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

### 1.5 ⭐ The 2012 census, all 25 files — `data/raw/Cenagro_IV/`

`IV_CENAGRO_Piura.dta` (§1.4) is **one department of twenty-five**. The full set lives on the
UDEP OneDrive share `MARAVI MENESES CRISTIAN ADDERLY - Departamentos_IV_CENAGRO (sin
posesionario)` as 25 `.dta` files, **409 columns each, 17.5 GB**. `data/` is gitignored, so
`PIPELINE.md` §7 records how to regenerate; the extractor is
`allperu/cenagro_extract.py`.

```
uv run python -m crop_classifier.cli allperu cenagro-extract            # all 25
uv run python -m crop_classifier.cli allperu cenagro-extract --verify   # the audit below
```

**17,750,195 rows, 17.5 GB → 0.21 GB (85× smaller), 409 columns → 76, zero rows lost.**
Output is **long** — one row per producer × parcel × crop-order — because each consumer
aggregates differently. `_extract_audit.csv` beside the Parquet files carries the table below. (A stray
`Cenagro_IV/Tacna.dta`, a 148 MB hand-copy from an earlier session, is superseded by
`Tacna.parquet` and can be deleted.)

⚠️ **The OneDrive files are `dataless` placeholders.** Every read streams over the network at
~2.7 MB/s, so the first extraction of a 2 GB department takes ~9 minutes and `%CPU` sits at
**0.0** the whole time — which looks exactly like a hang. Warm them first
(`dd if=<file> of=/dev/null`, four in parallel); a warmed department then extracts in
**5–15 s** instead of 200–500 s. Total wall clock ≈ 1 h either way, but it is download time,
not compute.

Per-department row counts, name coverage, crop-row share and posesionario rate:
**[`cenagro_columns.md`](cenagro_columns.md)**, regenerable with
`allperu cenagro-extract --verify` (the copy of record is `_extract_audit.csv` beside the
Parquet files).

**Linkable: 14 of 25** — Ancash, Arequipa, Ayacucho, Cajamarca, Huancavelica, Ica, La Libertad,
Lambayeque, Lima, Moquegua, Pasco, Piura, Tacna, Tumbes (§3). The other 11 are extracted anyway
— valid census data for a national descriptive, they just cannot reach a polygon or a PETT crop,
because the cadastral bridge does not exist for them. Structural, not a to-do.

⚠️ **The Piura file is not new data.** `Piura.dta` here extracts to **947,884 rows**, the exact
row count of `data/raw/IV_CENAGRO_Piura.dta` (§1.4). It was extracted under the same scheme so
that all 25 are consistent; `IV_CENAGRO_Piura.dta` is untouched and remains what §1.4 and
notebook 02 read.

#### The kept columns, the name columns, and what "sin posesionario" did not filter

76 columns. A producer-level value is repeated on every one of that producer's rows, and a
parcel-level value on every one of that parcel's crop rows — the file is **long**, so
de-duplicate on `NPRIN` (or `NPRIN`+`NPARC`) before averaging anything.

⭐ **Full column dictionary, with units: [`cenagro_columns.md`](cenagro_columns.md).** Four
things in it bind any use of this data and are repeated here because getting them wrong is
silent:

* ⚠️ **The farmer name columns `P009_01/02/03` carry an empty variable label.** They are the
  **only** link from the census to the rest of the project — no `COD_PREDIO`, no `CodigoSSET`,
  no DNI. A `usecols` built from labels finds nothing; a column picked by position finds a
  plausible string column that is not the name. **Identify them by value.** Coverage is
  97.2–99.7 % of rows, 92.0–99.0 % of producers (worst: Madre de Dios).
* ⚠️ **Derive tenure security from `P037_01_03` yourself.** Six derived registration variables
  (`registrado`, `registrado_1/2`, `p_registrado`, `GP`, `GP_1`, `X_t`) ship with the source,
  are absent from the INEI questionnaire, and match their apparent definitions on only 96–98 %
  of producers. `GP` also runs the *opposite* direction to the raw code. Cross-check only.
* ⚠️ **`P037_SS` is not the polygon area.** The census self-reported parcel surface is
  **uncorrelated** with the cadastral polygon area (Pearson ~0.01). Weight areas by the
  cadastral `area_ha`, never by `P037_SS`.
* ⚠️ **"sin posesionario" is a folder name, not a row filter — nothing was removed.**
  `P037_04_01 == 1` on 2.1–31.9 % of producers in **25 of 25** departments, 94,063 producers
  hold land *only* as posesionario, and incomplete/refused enumerations survive too. So the 25
  files stay **mutually comparable** — which is what makes any cross-department tenure
  comparison valid. The geographic gradient is real (Callao 31.9 %, Tacna 25.4 %, Madre de Dios
  17.8 %): a variable to model, not a nuisance to drop.

**`P024_03` resolves 100 %** against `IV CENAGRO - Tabla_Cultivos_Totales.xlsx` in all 25
departments. Codes are 1–4 characters and are **not** consistently zero-padded on either side;
`crop_code_table()` canonicalises both to an integer.

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
crop; weak for the parcel. Not used in any classifier.

⭐ **It is used for one thing: a second *declared* observation.** `allperu cenagro`
(`RESULTS.md` §8.5) crosses the PETT declaration with the 2012 census on the same parcel —
**10,639 parcels with both, 8,669 with a crop recorded on both sides** — which is the project's
only paired before/after with no satellite and no classifier in it. Three limits bind it:

* **⭐ No longer Piura only.** With the 25-department extract (§1.5) and
  `allperu/cenagro_link.py`, Chain B now runs on **all 14 linkable departments**:
  **307,807 census producers linked, 95,941 parcels with both observations, 63,766 with a
  crop on both sides** (`RESULTS.md` §8.6). `IV_CENAGRO_Piura.dta` was the only census
  file until 2026-08-29.
* ⚠️ **The matched subset is not a random sample.** Piura's is 23.3 % `PERENNIAL` against
  its true 14.4 %; nationally the linked panel is **16.6 % against the population's
  9.9 %**, with a larger median parcel (0.53 vs 0.43 ha). Matching needs a name on both
  sides *and* a district agreement, and the parcels that satisfy that are the larger,
  valley-floor, better-documented ones. Read the paired *change*, not the levels — and
  for a national claim use `cenagro_shift.poststratify()`, which reweights on department
  × declared class.
* ⚠️ **The two instruments do not share a class space.** PETT records a land *state* and has an
  explicit "EN DESCANSO"; census question 024 asks which **crop is grown**, so a fallow parcel
  contributes no row and leaves the frame. Crossing them raw makes `PASTURE_FALLOW` appear to
  collapse 17.9 % → 1.1 %, which is the instrument, not the land. Restrict to parcels with a
  crop on both sides.
* ⚠️ **The census vocabulary needs its own lexicon.** It uses fuller crop names than the
  registry ("LIMON ACIDO", not "LIMON"). Audited, **4.09 % of census tokens fell through to
  `crop_fallback: ANNUAL` — and 1,969 of those 2,448 were `VERGEL FRUTICOLA`, "fruit
  orchard".** `config/perennial_cenagro.yaml` (additive over `perennial_allperu.yaml`) takes
  that to 0.03 %. **The uncorrected headline was +2.4 pp; the corrected one is +12.5 pp.** Run
  `cenagro.token_audit()` before trusting any figure built this way.
  ⚠️ **The same trap fired again on the national vocabulary**, and the budget check passed
  both times. `MELOCOTONERO` — the peach *tree*, where `MELOCOTON` and `DURAZNO` were
  already PERENNIAL — was **45.8 % of a 1.35 % national tail at 18,371 instances**, with
  `MEMBRILLERO` (quince tree) and `DACTYLIS` (a sown forage grass) behind it. Fixed in the
  same config; tail now 0.54 %. **Print the tail sorted by frequency and read the top ten;
  a budget check cannot see a concentrated error inside a small tail.**

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

### 7.4 Climate covariates — per-parcel temperature and rainfall

Built by `allperu climate normals|rainfall`, module `allperu/climate.py`. Two tables,
keyed on `COD_PREDIO`, covering **all 726,808 parcels** in
`all_peru_full/modeling_parcels.parquet` — which is a superset of every other parcel table
in the project (the 56,419-parcel national sample, the 614,876-parcel S2 eligible universe
and the 1,112 labelled parcels are all contained in it, verified, 0 outside).

| file | source | resolution | time | rows × cols |
|---|---|---|---|---|
| `processed/climate/parcel_climate_normals.parquet` | WorldClim 2.1 | 30 arc-sec (~914 m, **85 ha**) | **static**, the 1970–2000 normal | 726,808 × 36 |
| `processed/climate/parcel_rainfall_annual.parquet` | CHIRPS 2.0 | 0.05° (~5.5 km, ~3,000 ha) | **one value per calendar year**, 1996–2024 | 726,808 × 33 |

**Sanity, from the built table** — mean annual temperature / rainfall by department, which is
the fastest way to see the file is right: **Ica 19.5 °C / 10 mm/yr** (hyper-arid coastal
desert), **Tumbes 25.4 °C / 266 mm** (equatorial coast), **Huancavelica 11.9 °C / 772 mm**
(high sierra), **Pasco 21.2 °C / 1,669 mm** (Amazon-facing selva). Dry months (<50 mm) run from
**12 of 12 in Ica to 1.3 in Pasco**.

**Columns and their units.** Normals: `tmean_c` (annual mean temperature, °C),
`precip_mm_yr` (annual rainfall total, mm/year), `tmean_c_warmest_month` /
`tmean_c_coolest_month` / `t_range_c` (°C), `precip_mm_wettest_month` /
`precip_mm_driest_month` (mm), `precip_seasonality_cv` (SD of monthly rainfall as a % of
the monthly mean — WorldClim BIO15; high = one short wet season), `aridity_index_dm`
(De Martonne, P/(T+10); <10 arid, >40 humid), `n_dry_months` (months under 50 mm), plus the
twelve monthly `tmean_c_mMM` and `precip_mm_mMM`. Annual rainfall: `precip_mm_<year>` for
1996–2024 (mm), plus `precip_mm_mean` and `precip_mm_cv` across those years.

**Why a centroid sample and not a zonal mean over the polygon.** The largest parcel in the
national table is **50 ha**; one WorldClim cell is **~86 ha** and one CHIRPS cell
**~3,000 ha**. *No parcel is larger than one climate pixel*, so a polygon mean and a
centroid sample return the same number, at a few hundred times the cost.
`climate.build_normals` prints this check rather than assuming it — measured, **0 of
726,808 parcels exceed a cell; the largest is 0.59 of one** — and `tests/test_climate.py` fails
if it ever stops being true.

⚠️ **`precip_seasonality_cv` is NaN for 1,275 Ica parcels, and that is correct.** They receive
**exactly 0 mm/year** in the WorldClim normal, so the seasonality of their rainfall is 0/0 —
genuinely undefined. It is left NaN rather than imputed, because filling it with 0 would assert
"rain is evenly spread through the year", which is a claim about rain that does not fall. It is
the **only** NaN in either table (`tmean_c`, `precip_mm_yr` and all 29 CHIRPS years are
complete). LightGBM handles it natively; a torch model needs an explicit fill, and *what you
fill it with is a modelling decision, not a cleanup step*.

⚠️ **A masked cell returns NaN, not an error** — the fifth instance of this project's
recurring failure shape. Climate rasters mask the ocean and Peru's cadastre runs to the
shoreline, so a coastal centroid can land one cell seaward; the column would come back NaN
and that parcel would silently vanish from any model using it. `_sample_points` falls back
to the nearest valid cell within 3 cells and reports how many needed it. Measured: **0 of 726,808 unresolved**, for CHIRPS in every one of the 29 years and for
WorldClim in all 24 monthly rasters.

⚠️ **The normals are time-invariant and this project has measured what that costs.** A
column with the same value in 1998 and 2023 cannot express change, so a model given one
reads stability into a panel whether or not the land was stable (§4.4, §5 of `RESULTS.md`).
Use the **normals for a single-year classifier** and **`parcel_rainfall_annual` for anything
applied across years**. Either way evaluate on **LODO as well as CV**: a 1 km climate
surface is a smooth function of location, so it is a proxy for `centroid_lat`, whose
CV/LODO story is the one `CLAUDE.md` opens with.

⚠️ **Year-resolved temperature does not exist in these tables.** CHIRPS is rainfall only,
and no equally cheap public temperature product was available (GEE was in restricted mode;
TerraClimate needs `netcdf4`, which is not a project dependency). Temperature is normals-only
and therefore static.

⭐ **The rainfall series validates itself against the project's own El Niño finding.**
CHIRPS was extracted with no reference to any project result, yet 1998 against 1997 gives
Tumbes **×6.24**, Piura **×3.36**, Lambayeque **×2.54** — while the southern sierra went
*drier* (Tacna ×0.66, Moquegua ×0.72), the textbook ENSO dipole. The 2017 coastal El Niño
repeats it (Tumbes ×2.45, Piura ×1.73 against 2016), and **2023 is the wettest year of the
29 nationally** (806 mm against a 610 mm mean). This is independent corroboration of the
El Niño mechanism `RESULTS.md` established from BSI, from a completely different instrument.

⚠️⚠️ **The two rainfall sources disagree by 25× on the hyper-arid coast, and the 1 km one is
right.** Mean annual rainfall per parcel, by department:

| dept | WorldClim (1 km) | CHIRPS (5.5 km) | ratio |
|---|---|---|---|
| **ICA** | **10 mm/yr** | **252 mm/yr** | **×25.2** |
| MOQUEGUA | 124 | 357 | ×2.9 |
| AREQUIPA | 237 | 431 | ×1.8 |
| TACNA | 135 | 242 | ×1.8 |
| TUMBES | 266 | 451 | ×1.7 |
| … | … | … | … |
| CAJAMARCA | 899 | 862 | ×0.96 |
| LAMBAYEQUE | 154 | 92 | ×0.60 |

Parcel-level Pearson r = **0.88** (Spearman 0.87) — they agree on the broad pattern and
disagree systematically where it matters. A CHIRPS cell is **~3,000 ha** and on Peru's coast a
single cell spans both the desert floor and the western Andean slope, so mountain rainfall
bleeds onto desert parcels. Ica genuinely receives a few mm a year; 252 mm is a resolution
artefact, not a period difference.

**So use them for different things:**
* **the spatial *level* of rainfall → `parcel_climate_normals.precip_mm_yr`** (1 km);
* **the year-to-year *anomaly* → CHIRPS as a ratio to its own mean**,
  `precip_mm_<year> / precip_mm_mean`, which is scale-free and therefore immune to the bias
  above. One line to compute; deliberately not stored as 29 more columns.
* **Do not feed CHIRPS's absolute level to a model on coastal parcels.**

**Provenance.** WorldClim 2.1 (Fick & Hijmans 2017), downloaded once to
`data/raw/worldclim/` (`wc2.1_30s_tavg.zip` 4.3 GB, `wc2.1_30s_prec.zip` 1.0 GB); gitignored
and not redistributed. CHIRPS 2.0 (Funk et al. 2015) is **never downloaded in bulk** — the
annual GeoTIFFs are uncompressed and row-striped, so a `/vsicurl` windowed read over Peru's
bounding box transfers ~320 of 2,000 rows per year. All 29 years take ~4 minutes and leave
nothing on disk.
