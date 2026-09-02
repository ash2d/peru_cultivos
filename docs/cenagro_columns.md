# CENAGRO 2012 — the 25-department extract, column by column

Reference for `data/raw/Cenagro_IV/<Dept>.parquet`, built by `allperu cenagro-extract`
(`allperu/cenagro_extract.py`). What the extract *is* and why it exists: [`DATA.md`](DATA.md)
§1.5. What was measured with it: [`RESULTS.md`](RESULTS.md) §8.6.

**17,750,195 rows, 17.5 GB → 0.21 GB (85× smaller), 409 columns → 76, zero rows lost.**

---

## Per-department audit

Regenerate with `allperu cenagro-extract --verify`; the numbers are written to
`_extract_audit.csv` beside the Parquet files, which is the copy of record.

| dept | rows | producers | MB in → out | P009_01 non-blank (rows / producers) | crop rows | posesionario | linkable |
|---|---:|---:|---|---|---:|---:|:--:|
| Amazonas | 503,474 | 69,811 | 492 → 6.2 | 99.5 / 98.9 % | 36.0 % | 2.1 % | — |
| Ancash | 1,428,751 | 175,069 | 1,419 → 16.9 | 99.1 / 96.4 % | 27.9 % | 4.4 % | ✅ |
| Apurimac | 751,234 | 84,275 | 740 → 8.2 | 99.5 / 98.1 % | 32.8 % | 4.5 % | — |
| Arequipa | 437,790 | 58,392 | 434 → 5.8 | 99.3 / 98.8 % | 35.7 % | 3.5 % | ✅ |
| Ayacucho | 960,893 | 119,762 | 946 → 10.6 | 98.9 / 94.3 % | 32.6 % | 5.6 % | ✅ |
| Cajamarca | 2,297,014 | 346,779 | 2,263 → 27.5 | 99.6 / 97.9 % | 27.5 % | 2.1 % | ✅ |
| Callao | 18,582 | 3,096 | 18 → 0.2 | 99.5 / 97.1 % | 1.1 % | **31.9 %** | — |
| Cusco | 1,536,606 | 185,129 | 1,517 → 16.8 | 99.3 / 97.6 % | 33.4 % | 3.8 % | — |
| Huancavelica | 691,448 | 79,050 | 679 → 7.0 | 98.8 / 93.8 % | 35.0 % | 4.8 % | ✅ |
| Huanuco | 874,291 | 107,633 | 860 → 9.6 | 99.7 / 99.0 % | 33.0 % | 6.5 % | — |
| Ica | 164,600 | 32,573 | 163 → 2.9 | 97.2 / 97.7 % | 35.7 % | 6.5 % | ✅ |
| Junin | 952,104 | 137,406 | 947 → 13.1 | 99.2 / 98.2 % | 38.6 % | 2.5 % | — |
| La_Libertad | 981,821 | 130,181 | 972 → 11.6 | 99.3 / 97.3 % | 22.7 % | 4.1 % | ✅ |
| Lambayeque | 390,528 | 61,229 | 387 → 4.8 | 99.1 / 96.2 % | 30.5 % | 4.9 % | ✅ |
| Lima | 523,461 | 79,190 | 522 → 7.7 | 98.0 / 97.3 % | 35.9 % | 8.3 % | ✅ |
| Loreto | 412,529 | 71,233 | 402 → 5.0 | 98.7 / 94.0 % | 47.1 % | **15.9 %** | — |
| Madre_de_Dios | 45,903 | 7,163 | 45 → 0.7 | 98.3 / **92.0 %** | 39.2 % | **17.8 %** | — |
| Moquegua | 129,750 | 15,039 | 128 → 1.7 | 98.9 / 93.6 % | 40.5 % | 6.9 % | ✅ |
| Pasco | 266,980 | 33,012 | 262 → 3.0 | 99.1 / 97.6 % | 30.8 % | 9.4 % | ✅ |
| Piura | 947,884 | 145,890 | 939 → 11.4 | 99.4 / 97.6 % | 30.6 % | 6.0 % | ✅ |
| Puno | 2,560,596 | 222,753 | 2,520 → 24.2 | 99.4 / 95.8 % | 25.2 % | 2.5 % | — |
| San_Martin | 528,087 | 93,846 | 522 → 7.4 | 99.3 / 97.0 % | 44.1 % | 4.5 % | — |
| Tacna | 149,736 | 22,820 | 148 → 2.1 | 98.7 / 95.8 % | 32.4 % | **25.4 %** | ✅ |
| Tumbes | 43,239 | 8,299 | 42 → 0.7 | 99.4 / 98.0 % | 41.4 % | 6.4 % | ✅ |
| Ucayali | 152,894 | 25,978 | 150 → 2.0 | 99.2 / 97.4 % | 44.8 % | **12.3 %** | — |
| **total** | **17,750,195** | **2,315,608** | **17,514 → 207** | | | | **14 ✅** |

`crop rows` is the share of rows carrying a `P024_03` code — the rest are parcel/tenure rows
with no crop. Callao's 1.1 % is why it yields nothing.

**Linkable: 14 of 25** — Ancash, Arequipa, Ayacucho, Cajamarca, Huancavelica, Ica, La Libertad,
Lambayeque, Lima, Moquegua, Pasco, Piura, Tacna, Tumbes (§3). The other 11 are extracted anyway
— they are valid census data for a national descriptive, they just cannot reach a polygon or a

`crop rows` is the share of rows carrying a `P024_03` code — the rest are parcel/tenure rows
with no crop. Callao's 1.1 % is why it yields nothing.

**Linkable: 14 of 25** — Ancash, Arequipa, Ayacucho, Cajamarca, Huancavelica, Ica, La Libertad,
Lambayeque, Lima, Moquegua, Pasco, Piura, Tacna, Tumbes (`DATA.md` §3). The other 11 are
extracted anyway — they are valid census data for a national descriptive, they just cannot reach
a polygon or a PETT crop, because the cadastral bridge does not exist for them. That is
structural, not a to-do.

---

#### The kept columns

76 columns. A producer-level value is repeated on every one of that producer's rows, and a
parcel-level value on every one of that parcel's crop rows — the file is **long**, so
de-duplicate on `NPRIN` (or `NPRIN`+`NPARC`) before averaging anything.

| column | what it is | unit / type |
|---|---|---|
| `dept` | department, from the source filename | str, added here |
| `TIPO_REC` | record type — `"01"` throughout | str |
| `P001` `P002` `P003` | departamento / provincia / distrito | str, zero-padded 2 |
| `UBIGEO` | `P001+P002+P003` | str(6), added here |
| `P007X` | Sector de Enumeración Agropecuario (SEA) | str(5) |
| `P008` | Unidad Agropecuaria (UA) within the SEA | str(4) |
| `NPRIN` | **producer id** (número de cédula principal) | str(7) |
| `NPARC` | parcel number within the producer | float |
| `RESULTADO` | 1 Completa · 2 Incompleta · 3 Rechazo · 4 Ausente | code |
| `P009_01` `P009_02` `P009_03` | **apellido paterno / materno / nombres — UNLABELLED** | str |
| `P009_04` `P009_05` | razón social · RUC (11 digits, kept as text) | str |
| `LONG_DECI` `LAT_DECI` | **SEA centroid**, shared by every UA in the SEA | decimal degrees |
| `WALTITUD` | altitude | m.a.s.l. |
| `WREGION` | 1 Costa · 2 Sierra · 3 Selva | code |
| `WPISO` | piso altitudinal, 1 Chala … 9 Yunga Marítima | code |
| `P016` | condición jurídica, 1 Persona natural … 9 Otra | code |
| `WP111` `WP112` | producer sex (1 H / 2 M) · age | code · years |
| `P019` `P019_01` | parcels worked in this district · UA without land | count · flag |
| `P020_01` | total surface of all parcels in this district | **ha** |
| `P021` `P022` `P022_01` | lives on a parcel · works parcels elsewhere · how many | code |
| `WSUP03` `WSUP04` | agricultural · non-agricultural surface | **ha** |
| `WSUP07` `WSUP10` | transitory-crop · **permanent-crop** land | **ha** |
| `WSUP11` `WSUP12` `WSUP13` | cultivated pasture · forestry · associated crops | **ha** |
| `WSUP18` | total cultivated surface | **ha** |
| `P023_01` `P023_04` `P023_07` | out-of-district parcel: order · surface (ha) · ubigeo | |
| `P037_0k_01` (k=1…5) | **tenure regime** = 1 when it applies: 1 propietario · 2 comunero · 3 arrendatario · **4 posesionario** · 5 otro | flag |
| `P037_0k_02` (k=1…5) | surface conducted under regime *k* | **ha** |
| `P037_01_03` | owner's title status: **1 título inscrito en RRPP** · 2 título no inscrito · 3 sin título, en trámite · 4 sin título ni trámite | code |
| `P037_SS` | **self-reported total parcel surface** | **ha** |
| `P038_01…04` | acquired by: herencia · compra-venta · adjudicación · otro | flag |
| `P039_01` `P040` | community type (campesina/nativa) · scattered trees | code |
| `registrado`, `registrado_1/2`, `p_registrado`, `GP`, `GP_1`, `X_t` | **undocumented derived registration variables** — see the warning below | float |
| `P024_01` `P024_03` | crop order within the parcel · **crop code** | code |
| `P025` | **sown area of that crop** | **ha** |
| `P026` `P027` | riego/secano · irrigation method | code |
| `P028` `P029_01/02/03` | destination of most output · sold to national market / **export market** / agroindustry | code · flags |
| `P036` | reason the land is uncultivated | code |

**`P024_03` resolves 100 %** against `IV CENAGRO - Tabla_Cultivos_Totales.xlsx` (sheet
`Permanente` — despite the name it is the whole 3,351-code vocabulary, with an `Exportable`
flag) in **all 25 departments, 0.00 % unresolved**. Codes are 1–4 characters and are **not**
consistently zero-padded on either side; `crop_code_table()` canonicalises both to an integer.

**Six derived registration variables ship with the source and none of them is documented.**
`registrado`, `registrado_1`, `registrado_2`, `p_registrado`, `GP`, `GP_1`, `X_t` are unlabelled
and are **not** in the INEI questionnaire — someone added them upstream. Reverse-engineered on
Piura they are *nearly* but not exactly what they look like:

* `registrado` is **constant within `NPRIN`** (100 %) and equals "this producer has ≥1 parcel
  with an inscribed title" on **98.2 %** of producers;
* `registrado_1` = "all of the producer's parcels are inscribed" on 98.5 %;
* `registrado_2` = the count of inscribed parcels, and `p_registrado` = that count over the
  parcel count, each on **96.0 %**;
* `GP` is an exact reversed recode of `P037_01_03` wherever `P037_01_03` is present
  (4 ↔ inscrito, 3 ↔ no inscrito, 2 ↔ en trámite, 1 ↔ sin título) — i.e. **higher = more
  secure**, the opposite direction to the raw code.

A 96–98 % match to an unstated definition is not a definition. **Derive tenure security from
`P037_01_03` yourself** and use these only as a cross-check.

**`P037_SS` is not the polygon area.** The census self-reported parcel surface is
**uncorrelated** with the cadastral polygon area (Pearson ~0.01, Chain B below). Weight areas
by the cadastral `area_ha`, never by `P037_SS`.

#### The name columns are the whole link, and they are unlabelled

`P009_01/02/03` carry an **empty variable label** in the `.dta`. A `usecols` built from labels
finds nothing; a column picked by position finds a plausible string column that is not the
name. **Identify them by value.** They are the *only* link from the census to the rest of the
project — there is no `COD_PREDIO`, no `CodigoSSET`, no DNI (§1.4, Chain B).

Coverage is high and even: **`P009_01` non-blank on 97.2–99.7 % of rows** and 92.0–99.0 % of
producers. The row/producer gap is real — producers with many rows are more likely to be named.
Worst producer-level coverage: **Madre de Dios 92.0 %**, Moquegua 93.6 %, Huancavelica 93.8 %,
Loreto 94.0 %. Adding `P009_04` (razón social) as a fallback recovers almost nothing —
**8.0 % of Madre de Dios producers have no personal name and no razón social**, against 1.0–3.0 %
in most departments. None of this is a broken extraction; it is the census's own coverage.

#### "sin posesionario" is a folder name, not a row filter

The source folder is named *(sin posesionario)*, which reads as a promise that
posesionario — occupant-without-title — records were stripped. **They were not.** Checked three
ways:

* `P037_04_01 == 1` on **2.1 %–31.9 % of producers, in 25 of 25 departments**;
* **94,063 producers (4.1 %) hold land *only* as posesionario** and are still in the files —
  precisely the rows a posesionario filter would have to delete;
* incomplete/refused/absent enumerations (`RESULTADO != 1`, 87,470 producers) survive too, so
  no completeness filter was applied either.

So nothing detectable has been removed, and the 25 files stay **mutually comparable**. This
mattered: an undocumented row filter applied to some departments and not others would have made
every cross-department tenure comparison invalid, and the folder name is the only thing that
ever suggested one.

What the numbers do show is a **real geographic gradient**, not an artefact: posesionario is
2–6 % across most of the sierra and coast but **31.9 % in Callao, 25.4 % in Tacna, 17.8 % in
Madre de Dios, 15.9 % in Loreto, 12.3 % in Ucayali** — urban-fringe and Amazon-frontier
tenure. Treat the 2012 posesionario share as a variable to model, not a nuisance to drop.

---

---
