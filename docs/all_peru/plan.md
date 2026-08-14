# All-Peru extension — plan and design decisions

> Sister document to [`../perennial/plan.md`](../perennial/plan.md), which this reuses
> wholesale. **Everything here is a delta on the Piura pipeline, not a replacement.**
> Results live in [`RESULTS.md`](RESULTS.md); the data audit is in
> [`DATA_AUDIT.md`](DATA_AUDIT.md).
>
> No question was put to the user for any of this. Where a decision was needed it was made
> here and recorded with its reasoning, per the brief ("do not ask me any questions, decide
> yourself and note it for later").

## 1. What changed and why

On 2026-08-07 the raw data grew from **one department to the whole country**: three new
folders (`data/raw/BD_SSET/`, `data/raw/Grafica_Tabular/`, `data/raw/QGIS/`) carrying the
same three source types the Piura pipeline already used.

**Step zero was verification, and it passed cleanly.** The Piura members of all three new
folders are **byte-identical** (sha256) to the files the existing pipeline reads:

| file | old path | new path | sha256 |
|---|---|---|---|
| crop registry | `BD SSET(MOQUEGUA-PASCO-PIURA).xlsx` | `BD_SSET/` same name | `96e90c6d…` ✔ |
| bridge | `grafica_tabular_Piura.dta` | `Grafica_Tabular/Piura.dta` | `9493e0ef…` ✔ |
| polygons (.shp) | `qgis_stefany/CATASTRO_…PIURA….shp` | `QGIS/PIURA/` same name | `2eddf7f1…` ✔ |
| polygons (.dbf) | `qgis_stefany/…PIURA….dbf` | `QGIS/PIURA/` same name | `9e76f435…` ✔ |
| sidecar | `qgis_stefany/PIURA.dta` | `QGIS/PIURA/PIURA.dta` | `df4fd502…` ✔ |

So the new folders are a strict superset and no Piura result is invalidated. Confirmed a
second way: rebuilding Piura through the *new* all-Peru code path reproduces the documented
figures exactly — **66,352 polygons / 80,618 records / 16.0 % multi-crop / 1 polygon with
>1 crop-year**.

## 2. The governing objective

From the brief: *"the main overall aim is to classify these land parcels through time to see
if they change from perennial to annual, so the classifier needs to be spatially
generalisable and temporally generalisable."*

That objective, plus the Piura findings, sets every design choice below. In particular the
Piura panel **failed its validation gate** (`../perennial/RESULTS.md` §7): the classifier
works single-year but per-parcel annual trajectories flicker far too much, and the 1997–98 El
Niño makes the baseline years unreadable. **Going national is a direct test of whether that
failure was a Piura-specific data limitation or a general one** — which is the most valuable
thing this extension can establish, whichever way it comes out.

## 3. What is usable, and what is not (D1)

Only **15 of Peru's 24 departments** can be linked — and only **14** yield any data once
Callao comes back empty. The reason is structural, not a choice. The link chain is

```
BD SSET (crop, year) ──CodigoSSET──► Grafica_Tabular/<Dept>.dta ──COD_PREDIO──► QGIS polygons
```

and **the bridge is mandatory** — BD SSET has no `COD_PREDIO`, the shapefiles have no
`CodigoSSET`. `Grafica_Tabular/` ships only 15 files.

* **Linkable (15):** Ancash, Arequipa, Ayacucho, Cajamarca, Callao, Huancavelica, Ica,
  La Libertad, Lambayeque, Lima, Moquegua, Pasco, **Piura**, Tacna, Tumbes. **Callao then
  yields 0 linked records** (395 polygons, 366 declarations, no overlap), leaving **14**.
* **Polygons but no bridge (8, excluded):** Amazonas, Apurímac, Cusco, Huánuco, Junín,
  Madre de Dios, Puno, Ucayali. ~653 k polygons that cannot be reached from a crop label.
* **Crops but no polygons (3, excluded):** Loreto, San Martín, Lima Metropolitana.

**D1 — do not attempt to recover the 8 bridge-less departments.** The only alternative link
is the CENAGRO farmer-name match (CLAUDE.md Chain B), which the Piura work already measured
as *"good for the person/district/crop, weak for the exact parcel"* (~43 % exact
`COD_PREDIO` agreement). Feeding parcel-level satellite features a label that is attached to
the wrong parcel ~half the time would poison a training set that is otherwise built purely
from real keys. Recorded as a limitation, not a gap to fill.

## 4. Two data traps found in the new folders (D2, D3)

Both fail **silently** — they produce plausible-looking empty or column-less results rather
than errors — so both are pinned by tests in `tests/test_allperu.py`.

**D2 — zero-padded bridge keys.** Most departments' bridge stores `CodigoSSET` zero-padded
to 9 characters (`030406693`) while BD SSET stores it unpadded (`30406693`). A string join
returns **zero** matches, which reads as "this department has no linkable data". Ancash:
**0 keys matched before the fix, 369,089 of 422,769 after.** Fixed by
`build_labels.canon_key`, which strips whitespace and leading zeros on both sides. **It is a
no-op for Piura** (both sides already 9 digits) — verified by rebuilding Piura bit-identically.

**D3 — shapefile attribute tables under the wrong basename.** 19 of 24 departments ship
`QGIS/ANCASH/ANCASH.dbf` next to `CATASTRO_CENAGRO_ANCASH_…_FINAL.shp`. GDAL opens such a
shapefile happily and returns **zero attribute columns**, so `COD_PREDIO` vanishes without an
error. `sources.shapefile_view` builds a symlink directory with consistent basenames; raw
data is never modified, and record counts were verified to match for every department.

Minor: the Arequipa/Ayacucho/Cajamarca workbook holds 1.6 M rows split across sheets
`DATOS1`/`DATOS2` (Excel's 1,048,576-row limit); assuming a single `DATOS` sheet silently
drops a department. Arequipa ships the same 135,780 parcels twice, in UTM 18S and 19S — 18S
is taken (it is the copy with a correctly-named `.dbf`; everything is reprojected to 4326
anyway).

## 5. Sampling back to Piura scale (D4, D5, D6)

The brief: *"aim to have in total as much data as was previously used … for just Piura as
for the whole of the new data. (so if there is 10x more data, sample it back down 10x)"*.

**D4 — the sample size is the Piura modelling table: 56,419 parcels.** Every downstream cost
(GEE hours, feature-store size, training time, panel extraction) scales with parcel count,
so sampling once, here, holds the whole budget fixed.

**D5 — sample whole 5 km regions, never scattered parcels.** Crops in Peru grow in
single-crop blocks (~86 % of adjacent parcels share a crop). A scattered 1-in-10 sample would
leave every region too sparse for the spatially-blocked split's 1.5 km buffer dead-zone to
mean anything, and would make the task look artificially hard by deleting each parcel's
neighbours. Sampling whole regions keeps local parcel density realistic.

**D6 — allocate departments by square-root-proportional share, with a floor, and cap parcels
per region.** Cajamarca and Ancash together hold over half the linked parcels; proportional
allocation would produce an "all-Peru" model that is really two departments — defeating the
spatial-generalisation objective. Square-root allocation is the standard compromise between
proportional (efficient for a national total) and equal (efficient for between-department
contrasts). A per-department floor keeps Callao and Moquegua present at all; a per-region cap
converts "more parcels" into "more places", which is the binding constraint for spatial
generalisation.

**Class balance is deliberately NOT forced.** The perennial/annual/pasture prior is a real
property of Peruvian agriculture; re-weighting it here would corrupt any area share.
Sampling weights (stratum population / stratum sample, stratum = department × label) are
written alongside so population quantities remain recoverable.

## 6. Workspace layout (D7)

**D7 — the all-Peru work gets its own feature store.** The Piura 12-class and 3-class
workspaces deliberately *share* one pixel store because they are the same parcels with a
different label column. All-Peru is different parcels, so `paths.py` gained a `CC_FEAT`
override (defaulting to the existing shared store, so nothing about Piura changes).

```bash
export CC_PROC=data/processed/all_peru          # sampled workspace: labels, splits, panel
export CC_FEAT=data/processed/all_peru/features # its own pixel store
export CC_RUNS=runs/all_peru
```

| directory | holds |
|---|---|
| `data/processed/all_peru_full/` | the **full** 15-department Chain-A build + read caches |
| `data/processed/all_peru/` | the Piura-scale **sample**: labels, splits, panel, predictions |
| `data/processed/all_peru/features/` | its pixel/feature store (`panel/` beneath it) |
| `runs/all_peru/` | trained models, CV metrics, gate artifacts |

Three small generalisations were needed in shared code, all defaulting to previous
behaviour:

* `paths.feat()` + `CC_FEAT` (above);
* `splits.assign` reads `metric_crs` from its config — Piura fits in UTM 17S, Peru spans
  17S–19S and needs **one** continuous grid, so the all-Peru config uses UTM 18S (scale
  error < 0.7 %, i.e. < 11 m on the 1.5 km buffer);
* `labels3.source_tables()` prefers the workspace's own `training_crop_polygon.parquet`
  before the shared Piura one.

## 7. Modelling deltas

Otherwise the Piura recipe is reused **unchanged and on purpose** — same 3-class label
policy, same spatially-blocked split with a 1.5 km buffered dead-zone, same LightGBM /
LTAE / PSE-LTAE, same Phase-7 gate — because the point is a like-for-like comparison against
Piura, not a new modelling study. Three deliberate changes:

1. **`test_frac` 0.15 → 0.20.** The locked test is what measures *spatial* generalisation,
   which is the whole point of going national, and the sample spreads over many more regions
   than Piura's, so a larger held-out share costs proportionally less per region.
2. **D10 — the panel is sized for the GATE, not for the area estimate: 4,565 parcels, not
   ~8,000.** (`--n 3500`; the floor of 3,000 forced locked-test parcels, which S4 requires,
   takes the total to 4,565 — of which 969 are `PERENNIAL`, against the 1,086 Piura's S5
   flicker rests on.) Piura's panel used 7,690 parcels because it was sized for the headline
   deliverable (a per-year area share) as well as for validation. That ordering is wrong
   here, and Piura proved it: **the gate failed, so the area share was never produced and
   the parcels bought for it were wasted.** The Phase-7 gate is a validation instrument, and
   its two criteria are per-parcel statistics (S5 flicker) and per-k accuracy curves (S4)
   that converge long before 8,000 parcels — Piura's own S5 rests on 1,086 `PERENNIAL`
   parcels and its S4 on 3,553 locked-test parcel-years. Sizing for the gate first cuts
   extraction from ~200,000 parcel-years to ~87,500 and makes the answer reachable; **if the
   gate passes, a larger panel is extracted then**, for the estimate that would actually use
   it. Measured national rate is ~0.18–0.5 s/parcel-year, so this is the difference between
   roughly 6–16 h (114,125 parcel-years) and 20–28 h (~200,000).

   Two consequences to carry forward. **(a)** 3,000 of the 4,565 are forced locked-test
   parcels (66 %, against 39 % in Piura's larger panel), so this panel is deliberately
   skewed toward the test regions — right for S4, wrong for a national area share, which is
   another reason it must not be reused for one. **(b)** Panel weights now **compose two
   sampling stages**: `population_weight` (modelling sample → all 14 departments, from
   `allperu sample`) times the panel's own `sample_weight` (panel → modelling sample).
   `build_panel` multiplies them, giving weights that expand to 708,756 parcels — the
   population minus those that failed the coverage gate. The two columns were originally
   both called `sample_weight` and collided; that is fixed and regression-tested.
3. **Panel years 1999–2023, not 1996–2023.** Task-1's ≥1999 experiment
   (`../perennial/RESULTS.md` §8.3) showed the 1996–98 El Niño years are the ones that break
   temporal transfer, and excluding them flipped S4 from FAIL to PASS at zero CV cost. There
   is no reason to re-import a known-bad baseline into a new workspace.

**A deliberate extra evaluation that Piura could not support: leave-one-department-out.**
With 14 departments there is, for the first time, a real test of transfer to *unseen
regions* rather than unseen 5 km cells. This is the single most informative thing the
national data adds and it is reported in `RESULTS.md`.

**D9 — `centroid_lat` must be re-examined nationally, not inherited.** Piura kept it: the
ablation cost 0.006 CV and, being time-invariant, it cannot manufacture a *trend*
(`../perennial/RESULTS.md` §4.6). Two things change at national scale. First, latitude now
spans ~14° instead of ~1°, so it is a far stronger memorisation handle. Second, §7.0.3
showed time-invariant features **manufacture stability** — they damp a panel series
mechanically and make S5 flicker look better than the spectral data supports. Leave-one-
department-out is exactly the test that exposes this: a model leaning on latitude cannot
transfer to a department it never saw. So `--drop-features meta` and `meta,location` are
both trained and both put through LODO, and the *gap between them* is reported rather than
one being assumed.

## 7b. D11 — model selection is made on LODO, not CV (decided 2026-08-08, on evidence)

D9 asked whether `centroid_lat` earns its place nationally. It was answered, and the answer
inverted the choice: latitude is worth **+0.047 macro-F1 on spatial CV and −0.060 on
leave-one-department-out**, with **12 of 14 departments improving without it**
(`RESULTS.md` §6.2).

**So the selection criterion itself had to be decided, not just the feature.** Spatial CV
holds out 5 km cells *inside departments the model has already seen*; the deliverable is a
classifier applied to places and years it was not trained on. LODO is therefore the
estimator that matches the intended use and CV is the optimistic one. **Selected:
`lightgbm_nometa_nolat`** (`runs/all_peru/selected_model.json`).

This **reverses Piura's choice** (`../perennial/RESULTS.md` §4.6 kept `centroid_lat` because
dropping it cost 0.006 and bought nothing *there*). Both follow the same rule — prefer the
model whose failure mode is benign for the intended use — applied to different evidence. It
is not a contradiction, and the Piura decision was not wrong on the evidence Piura had; the
point is that **Piura could not run the experiment that settles it.**

Consequence for the panel: the selected model holds *less* time-invariant information than
the alternative, and Piura's §7.0.3 found time-invariant features manufacture *stability*.
So the gate is run on **both** models — if that finding generalises, the selected model must
flicker **more**, not less. Predicted before running; reported in §7 either way.

### 7b.1 Outcome (2026-08-09) — prediction CONFIRMED, decision UNCHANGED

**The pre-registered prediction held exactly.** A third arm was added first — `ltae`, which
carries **no** time-invariant features at all — and the gate was run on all three. Flicker is
**monotone in static count**: `nometa` 0.517 < `nolat` 0.744 < `ltae` 0.980, every rung above
Piura's (0.428 / 0.547 / 0.780). The selected model does flicker **more**, as predicted, and
k = 0 accuracy barely moves across the three (0.586 / 0.544 / 0.581). `RESULTS.md` §7.5.

**Fixing the gate arms before LODO was known was load-bearing.** LTAE turned out to *lose*
selection (LODO mean 0.442 vs 0.477; `RESULTS.md` §6.2c), so a rule of "gate the winner" would
have dropped the only static-free arm — the one that supplies the honest end of the flicker
estimate. Selection names the *primary* model; it does not choose the gate arms.

**D11 itself is re-confirmed**, now against three candidates rather than two:
`lightgbm_nometa_nolat` still leads the LODO mean and is best in 8 of 14 departments. But
note a limit the LTAE arm exposes: **"fewer statics" is not a monotone recipe for
transfer.** Dropping `centroid_lat` from LightGBM buys +0.060; dropping *every* static by
switching architecture costs 0.035 against that. D9's specific feature was the problem, not
staticness as such — which is the opposite sign to what staticness does to *flicker*.

⚠️ **The gate FAILED on all three arms**, so no trajectories, transitions or area estimates
were produced. Steps 9+ of §8 below stay unrun. `RESULTS.md` §7.9/§7.10.

## 8. Order of operations

```bash
export CC_PROC=data/processed/all_peru CC_FEAT=data/processed/all_peru/features \
       CC_RUNS=runs/all_peru

# 1. Chain A over every linkable department -> data/processed/all_peru_full/
uv run python -m crop_classifier.cli allperu labels

# 2. 3-class labels on the FULL set, with the NATIONAL lexicon (§9). Note CC_PROC points at
#    the full workspace here: labels3 reads training_crop_polygon.parquet from CC_PROC.
CC_PROC=data/processed/all_peru_full \
  uv run python -m crop_classifier.cli perennial labels \
      --config src/crop_classifier/config/perennial_allperu.yaml

# 3. Sample back to Piura scale. Writes into CC_PROC (the sampled workspace).
uv run python -m crop_classifier.cli allperu sample --source data/processed/all_peru_full

# 4. Spatial splits, national config (UTM 18S grid, test_frac 0.20)
uv run python -m crop_classifier.cli splits assign \
      --config src/crop_classifier/config/split_allperu.yaml

# 5. GEE — resumable; `all` = stage-1 coverage + gate, then stage-2 pixels for survivors
uv run python -m crop_classifier.cli features extract --stage all
uv run python -m crop_classifier.cli features assemble

# 6. Models. Two variants, because the 1997-98 El Nino was a NORTHERN COASTAL event
#    (Piura/Tumbes/Lambayeque/La Libertad) — the sierra and the south were not hit the same
#    way, so whether ../perennial/RESULTS.md §8.3's exclusion is needed nationally is an
#    empirical question, and `--train-years` now makes it a one-flag experiment.
uv run python -m crop_classifier.cli train --model lightgbm \
      --run-name lightgbm_nometa --drop-features meta
uv run python -m crop_classifier.cli train --model lightgbm \
      --run-name lightgbm_nometa_from1999 --drop-features meta --train-years 1999-2023
for r in lightgbm_nometa lightgbm_nometa_from1999; do
  uv run python -m crop_classifier.cli perennial pool-cv   runs/all_peru/$r
  uv run python -m crop_classifier.cli perennial calibrate runs/all_peru/$r
done

# 7. The evaluation Piura could not run
uv run python -m crop_classifier.cli allperu lodo --drop-features meta

# 8. Panel + gate. 1999-2023, per §7.
uv run python -m crop_classifier.cli perennial panel build --n 7500
uv run python -m crop_classifier.cli perennial panel extract  --years 1999-2023
uv run python -m crop_classifier.cli perennial panel assemble --years 1999-2023
uv run python -m crop_classifier.cli perennial panel infer \
      --run runs/all_peru/<selected> --years 1999-2023
uv run python -m crop_classifier.cli perennial diagnostics     # THE GATE
```

**The gate binds here exactly as it does in Piura.** If S4 or S5 fails, no trajectories, no
transitions and no area estimates are produced. A trend from a panel that failed validation
is not a weaker finding, it is a wrong one.

## 9. The lexicon had to be extended (D8), and how

**D8 — a national lexicon, `config/perennial_allperu.yaml`, derived from the Piura one by
ADDING tokens only.** No Piura assignment is changed and every policy flag is identical, so
the two label spaces stay comparable.

The Piura 3-class lexicon resolves 79,164 of 80,618 Piura records but only **80.8 %** of the
national ones — 19.23 % fell through to the blanket `crop_fallback: ANNUAL` guess against a
2 % budget the build enforces by raising `AssertionError`. Piura's registry is a
coastal-valley crop list; Peru adds the sierra and the selva. The largest single missing
token is not a crop at all: **`KIKUYO`, the Andean grazing grass, at 36,098 records** — the
sierra registrars recorded grazing land *by grass species*.

Three mechanisms, because a longer list alone could not have worked
(19.23 % → 6.92 % → 2.54 % → **1.92 %**, under budget):

1. **Lexicon additions** — sierra forage species, sierra staples, Andean/selva tree crops,
   and bare/idle-ground vocabulary.
2. **A growth-stage stripper** (`stage_words` + `labels3._stage_regex`). Sierra registrars
   wrote the crop *and its phenological stage*: `MAIZ EN FLORACION`, `PAPA EN FASE DE
   CRECIMIENTO`. That is a **productive** pattern — any crop × any stage — so it cannot be
   enumerated; ~5 % of national records resolve only after stripping it. Stage is irrelevant
   to a label space describing a parcel's land *state* over a whole year.
3. **Word-level resolution** (`word_match: true`) as the last resort before guessing:
   `PLANTACION DE VID` → `VID`; `CONTIENE RASTROJO DE MAIZ` → {PASTURE_FALLOW, ANNUAL},
   combined by the same `group_priority` a multi-crop parcel uses. Plus singular/plural
   matching. Worth a further 3.8 %.

Every hit records *how* it resolved (`lexicon`, `stage_stripped:…`, `word_match`,
`crop_fallback`) in `class_lexicon_resolved.csv`, so no mechanism hides behind another.

**Judgement calls worth knowing about**, all made here rather than asked:
`ALFALFA`/`KIKUYO`/`TREBOL`/`ICHU` → `PASTURE_FALLOW` (cut or grazed forage: the land-state
is grass, not a crop stand); `COCA` → `PERENNIAL` (a woody multi-year shrub — a land-state
call, not a legal one); `PIÑA` and `ESPARRAGO` → `ANNUAL`, following MapBiomas' temporary-crop
convention already used for Piura; `EUCALIPTO`/`PINO`/`TARA`/`PECANO` → `woody_noncrop` and
therefore excluded, consistent with Piura's treatment of `ALGARROBO`.
