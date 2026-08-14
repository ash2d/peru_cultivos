# All-Peru results

> Design decisions: [`plan.md`](plan.md). Data provenance and traps:
> [`DATA_AUDIT.md`](DATA_AUDIT.md). Piura baseline this is compared against:
> [`../perennial/RESULTS.md`](../perennial/RESULTS.md).
>
> Everything below is measured. Sections still running are marked **IN PROGRESS**.
>
> Last updated 2026-08-09: LTAE trained and put through LODO (it does **not** win — §6.2c),
> selection re-confirmed as `lightgbm_nometa_nolat` (§6.3), and the panel assembled, inferred
> for three arms and **gated — the gate FAILED on all three** (§7.4). **No trajectories,
> transitions or area estimates exist and none should be produced.** The locked test is still
> unspent.

## 0. Headline

### ⛔ The panel gate failed on every arm — the classifier works, the annual trend does not

| arm | statics | S4 temporal transfer | S5 `PERENNIAL` flicker (criterion 0.15) | gate |
|---|---|---|---|---|
| `lightgbm_nometa` | 3 | PASS (dev 0.029) | **0.517** | ⛔ FAIL |
| **`lightgbm_nometa_nolat`** ⭐ | 2 | PASS (dev 0.013) | **0.744** | ⛔ FAIL |
| `ltae` | **0** | FAIL (dev 0.107) | **0.980** | ⛔ FAIL |

The **pre-registered prediction is confirmed exactly**: flicker is monotone in how many
time-invariant features a model carries, so statics manufacture *stability* nationally just
as they did in Piura — and every rung is worse here (Piura: 0.428 / 0.547 / 0.780). The
static-free arm is the binding one and it changes its mind about a parcel in 98 % of cases
over 25 years. **Crucially, the national panel has no thin years** (minimum gate pass 93.4 %
against Piura's 43.3 %), so this cannot be blamed on coverage. §7.4–§7.5.

### Adding a static-free architecture did not fix transfer either — a negative result

LTAE carries **no** statics, so the `centroid_lat` mechanism below cannot apply to it. It
still loses selection: LODO mean 0.442 against `nolat`'s 0.477, and worst of the three on
pooled LODO. It wins only across-department *spread* (0.056 vs 0.082 / 0.098). "Fewer
statics" is not a monotone recipe for generalisation — the specific feature was the problem,
not staticness. §6.2c.

### The finding: `centroid_lat` is spatial memorisation, and only a national dataset can see it

| | CV (unseen 5 km cell) | LODO mean (unseen department) | LODO pooled |
|---|---|---|---|
| `lightgbm_nometa` (has `centroid_lat`) | **0.628 ± 0.013** | 0.417 | 0.539 |
| **`lightgbm_nometa_nolat`** ⭐ selected | 0.581 ± 0.002 | **0.477** | 0.537 |
| Δ | **−0.047** | **+0.060** | −0.002 |

**Latitude's contribution reverses sign the moment the held-out unit is a place rather than a
neighbouring cell**, and **12 of 14 departments improve without it**. Spatial CV — the best
tool Piura had — cannot distinguish memorisation from signal, because it holds out 5 km cells
*inside departments the model has already seen*. §6.2.

Two companions to it:

* **The spatial-generalisation gap is ~0.09 macro-F1** (0.629 unseen-cell → 0.539
  unseen-department). Invisible to every evaluation the single-department project could run.
* **"Piura is one of the hardest departments to predict" is a property of `centroid_lat`, not
  of Piura.** It scores 0.301 (13th of 14) under `lightgbm_nometa` — but **0.479 (6th of 14)
  under the selected `nolat` model and 0.512 (2nd of 14) under LTAE**, which has no location
  feature at all. The apparent atypicality is the latitude lookup table failing hardest on
  the most distinctive latitude band. Corrected 2026-08-09; earlier revisions reported the
  `nometa` figure as if it characterised the department. §6.2c.

### The data: better in exactly the two ways the research question needs

| | Piura (existing) | all-Peru | change |
|---|---|---|---|
| linked polygons | 66,352 | **946,872** | **14.3x** |
| departments | 1 | **14** | — |
| 3-class eligible parcels | 56,419 | **726,808** | 12.9x |
| label years with >5 % of mass | **2** (1998, 1999) | **8** (1997–2006) | **4x** |
| 5 km regions (in the modelling sample) | 256 | **677** | 2.6x |
| neighbour label agreement at 0–100 m | ~0.86 | **0.72** | less spatially trivial |

The last three rows are the point. Piura's fatal limitation was that it is **one place at
one moment**: ~80 % of labels came from the 1998–99 titling wave, in a single department,
during a catastrophic El Niño. The national data breaks both degeneracies — label mass is
spread over 1997–2006 and over 14 departments from the Tumbes coast to the Puno-adjacent
altiplano.

## 1. What was built

Sample of **56,419 parcels** — deliberately the same size as the Piura modelling table, so
GEE hours, feature-store size and training cost are unchanged (plan §5). Drawn as whole 5 km
regions, allocated across departments by square-root-proportional share with a floor and a
per-region cap.

| dept | sampled | of available | regions sampled | of available |
|---|---|---|---|---|
| CAJAMARCA | 7,210 | 159,021 | 74 | 757 |
| ANCASH | 7,113 | 154,263 | 69 | 384 |
| LA_LIBERTAD | 5,411 | 82,311 | 66 | 394 |
| AYACUCHO | 5,367 | 80,723 | 37 | 231 |
| AREQUIPA | 4,749 | 60,354 | 56 | 369 |
| PIURA | 4,618 | 56,422 | 49 | 262 |
| LIMA | 4,100 | 42,150 | 37 | 264 |
| ICA | 3,642 | 31,274 | 46 | 219 |
| LAMBAYEQUE | 2,801 | 15,491 | 51 | 181 |
| MOQUEGUA | 2,710 | 14,128 | 24 | 81 |
| HUANCAVELICA | 2,638 | 13,079 | 26 | 44 |
| TUMBES | 1,825 | 4,068 | 31 | 59 |
| TACNA | 2,225 | 7,861 | 49 | 93 |
| PASCO | 2,010 | 5,663 | 62 | 212 |

**A consequence worth stating plainly: the sample's class mix is not the population's.**
Square-root allocation upweights the small departments, and the small departments are the
coastal perennial ones (Tumbes is 75 % perennial, Huancavelica 5 %), so `PERENNIAL` runs at
**20.2 % of the sample against 9.9 % of the population**. This was not a class-balancing
step and it was not forced — it falls out of the spatial allocation. It is *helpful* for
training (11,375 perennial parcels rather than ~5,600) and it is **not** a problem for area
estimation, because `sample_weight` (stratum population / stratum sample, stratum =
department × label) reconstructs the population exactly:

| | ANNUAL | PASTURE_FALLOW | PERENNIAL |
|---|---|---|---|
| population share | 0.5002 | 0.4009 | 0.0989 |
| **weighted sample share** | **0.5002** | **0.4009** | **0.0989** |
| raw (unweighted) sample share | 0.4284 | 0.3700 | 0.2016 |

Total weight = 726,808 = the population count, exactly. **Any area or share figure computed
on this sample must use `sample_weight`.**

## 2. Label build

Chain A run per department, identical join logic and identical crop normalisation to the
Piura pipeline. **Piura rebuilt through the new code path reproduces its documented figures
exactly** (66,352 polygons / 80,618 records — see DATA_AUDIT §1).

Key coverage (SSET crop keys reaching a polygon) is **better than Piura's outside Piura**:
Arequipa 87.8 %, Ancash 87.3 %, Cajamarca 81.7 %, Lambayeque 81.2 %, Ica 80.3 % against
Piura's 67.4 %. The weakest are Pasco 41.8 % and Huancavelica 48.7 %.

**Callao yields nothing** (395 polygons, 366 declarations, no overlap after the join) and
drops out, leaving 14 departments.

### 2.1 The lexicon had to be extended, and the reason is interesting

The Piura 3-class lexicon covers 79,164 of 80,618 Piura records but only **80.8 %** of the
national ones — 19.23 % of records fell through to the blanket `crop_fallback: ANNUAL`
guess, nearly 10x the 2 % budget the build enforces. Piura's registry is a coastal-valley
crop list (rice, mango, lime, cotton). Peru adds the sierra and the selva, and the single
largest missing group is not a crop at all:

> **`KIKUYO` — Andean grazing grass — is 36,098 records**, the largest unassigned token by a
> factor of four. The sierra registrars recorded grazing land **by grass species**
> (`KIKUYO`, `TREBOL`, `ICHU`, `ALFALFA`, `BRACHIARIA`), which Piura had almost no occasion
> to do.

Three fixes, in order of how much they bought (19.23 % → 6.92 % → 2.54 % → **1.92 %**,
under budget):

1. **A national lexicon** (`config/perennial_allperu.yaml`) — additive only, so no Piura
   assignment changes and the two label spaces stay comparable. Sierra forage species,
   sierra staples (lupin, oca, quinoa), Andean and selva tree crops (apple, peach, quince,
   oil palm, coffee cultivars), plus bare/idle-ground vocabulary (`RASTROJO` stubble,
   `BARBECHADO`, `SURCADO`).
2. **A growth-stage stripper.** Sierra registrars wrote the crop *and its phenological
   stage*: `MAIZ EN FLORACION`, `PAPA EN FASE DE CRECIMIENTO`, `TREBOL EN MADURACION`. That
   is a **productive** pattern — any crop × any stage — so it cannot be enumerated; ~5 % of
   national records resolve only after stripping it. Stage is irrelevant to a label space
   describing a parcel's land *state* over a whole year.
3. **Word-level resolution** as a last resort before guessing: `PLANTACION DE VID` → `VID`,
   `CONTIENE RASTROJO DE MAIZ` → {PASTURE_FALLOW, ANNUAL}, combined by the same
   `group_priority` used for multi-crop parcels. Plus singular/plural matching (`TUNAS` →
   `TUNA`). Worth a further 3.8 % of records.

Final resolution provenance, all auditable in `class_lexicon_resolved.csv`:

| source | share of records |
|---|---|
| exact lexicon hit | 63.1 % |
| category default (pasture/fallow/land-prep) | 23.6 % |
| stage-phrase stripped, then lexicon | 5.0 % |
| word match | 3.8 % |
| **`crop_fallback` (a guess)** | **1.9 %** |
| woody non-crop / sugarcane policy | 2.6 % |

### 2.2 Class mix

| | ANNUAL | PASTURE_FALLOW | PERENNIAL |
|---|---|---|---|
| Piura | ~64 % | ~22 % | ~14 % |
| **all-Peru (population)** | **50.0 %** | **40.1 %** | **9.9 %** |

Pasture nearly doubles — the sierra is grazing country — and perennial thins. In absolute
terms perennial *grows* from ~7.9 k parcels to **71,906**.

## 3. Spatial splits

Run with `config/split_allperu.yaml`: identical policy to Piura except one continuous UTM
18S grid for the whole country (Peru spans zones 17S–19S) and `test_frac` 0.15 → 0.20.

* 5,693 populated 1 km blocks; **677 populated 5 km regions** (median 39 parcels/region)
* locked test: **11,441 parcels in 125 contiguous regions** (20.3 %)
* buffer (1.5 km) excludes only **2,208 parcels** (3.9 %) from final training — far cheaper
  than Piura, because the sampled regions are spread out rather than contiguous
* every class has real test support: ANNUAL 5,283 parcels / 112 regions, PASTURE_FALLOW
  3,917 / 100, **PERENNIAL 2,241 / 79**

**The autocorrelation audit reports a materially easier-to-trust picture than Piura's.**
Baseline (random-pair) agreement is 0.361 and neighbour agreement at 0–100 m is **0.718**,
decaying to 0.563 at 4–5 km. Piura's figure was ~0.86 at adjacency. Crops are still strongly
clustered — the same `buffer_m < decorrelation range` warning fires — but a national sample
is substantially less spatially degenerate than a single department.

## 4. Feature extraction

**Stage-1 coverage gate: 55,028 of 56,419 parcels pass (97.5 %), 1,391 fail, 0 unmeasured** —
better than Piura, whose 1998 cohort passed at ~95 %. The gate is `n_valid_obs >= 4` clear
Landsat acquisitions in the parcel's label year.

Failures concentrate where cloud does, which is the expected pattern rather than a data
problem: Tumbes 11.8 % (the wettest coastal corner), Cajamarca 6.8 % (north-sierra cloud),
then Lambayeque 2.6 %, Moquegua 2.3 %, Ancash 2.2 %, everything else under 2 %. The class mix
is essentially untouched by the gate (43.1 / 37.1 / 19.8 vs 42.8 / 37.0 / 20.2 before it), so
no class is being silently thinned by cloud.

Stage-2 raw pixel export runs over the 55,028 survivors in 1,466 chunks.

### 4.0 Why the national extraction costs ~4x Piura's for the same parcel count

Worth stating because it looks like a bug and is not. Piura's full extraction was ~2.5 h for
47,851 parcels; the national one runs at roughly 4x that per parcel. The cause is **pixel
volume, not parcel count**:

| | Piura | all-Peru |
|---|---|---|
| parcels | 47,851 | 55,028 |
| mean parcel area | 0.97 ha | **1.79 ha** |
| total area | 53,171 ha | **98,613 ha** (1.85x) |
| pixel-observations | 4.53 M | **14.75 M** (measured) |

Peru's parcels are simply bigger — Pasco averages 8.3 ha against Ancash's 0.87 — and better
observed. So the brief's "as much data as was previously used" is met on **training
examples** (56,419 parcels, matched exactly) while the *computational* volume is 3.3x. That
is a bonus for the model, and the reason for the longer wall time.

**Wall time: ~7 h**, against an estimated ~9 h serial. Extraction is latency-bound rather
than quota-bound, so it was split across five concurrent processes on disjoint year ranges,
which took throughput from 3.5 to a peak of 12 chunks/min. **Zero transient errors, zero
timeouts and zero torn chunk files** across the whole run — all 1,463 chunk parquets were
re-read and verified. (Concurrency is safe here only because chunk filenames are
content-addressed on the parcel set; a worker that meets an existing chunk skips it. The
year ranges must genuinely be disjoint, which mine initially were not — the original
all-years process overlapped the workers added later, wasting some calls before it was
pruned.)

**Two optimisations were tested and both rejected on measurement**, which is worth recording
so neither is retried:

* **Larger GEE chunks.** The expectation was that a bigger chunk amortises per-request
  overhead. Measured on identical parcels: **0.180 s/parcel at `chunk_size=40` vs 0.278 s at
  120** — bigger is *worse*, because cost scales super-linearly with the pixel volume in a
  request, not with the number of requests. The existing setting was already right.
* **Capping pixels per parcel-date.** Downstream only ever uses a per-date median and 8
  sampled pixels, so extracting 554 pixels for a 50 ha parcel looks wasteful. But the
  distribution is long-tailed rather than top-heavy: a 30-pixel cap saves only **44 %** of
  volume while changing the features of **16.3 %** of parcels. That forfeits like-for-like
  comparability with Piura's feature computation — the entire point of the exercise — for
  well under a 2x speedup. Not adopted.

Three engineering problems had to be solved before a national extraction was affordable or
even possible. All three are documented in DATA_AUDIT §4 and pinned by tests.

**4.1 Chunk extent.** A chunk's bounding box becomes the `ee.Geometry.Rectangle` that every
`filterBounds` runs against, so extent drives GEE cost directly. Piura is one compact valley
system, so grouping chunks by year alone was fine. Nationally, one label year draws parcels
from all 14 departments and lon/lat sorting produced chunks spanning **up to 110 deg²**
(~1,200 × 1,000 km). Splitting strictly by `(year, dept)` fixed extent but overshot — 89 of
204 groups held under 50 parcels, each still paying full per-request overhead. The
implemented answer visits departments in longitude order and packs them greedily, cutting a
chunk whenever the next department would push its bbox past 4 deg²:

| chunking | chunks | median bbox | p90 | max |
|---|---|---|---|---|
| by year (Piura's rule) | 159 | 0.98 deg² | 17.0 | **109.9** |
| by (year, dept) | 286 | 0.25 | 1.66 | 6.72 |
| **greedy lon-packed (used)** | **240** | 0.53 | 2.89 | 6.72 |

**4.2 Three-dimensional geometry.** **La Libertad's cadastre stores 3D polygons** and Earth
Engine's GeoJSON validator rejects any coordinate triple outright — `EEException: Invalid
GeoJSON geometry`. Shapely calls them perfectly valid and geopandas round-trips them
happily, so nothing complains until the extraction dies partway through, on the first
department it reaches that has them. 5,411 of 56,419 sampled parcels (9.6 %), all from one
department. Fixed by forcing 2D and repairing the 19 genuinely self-intersecting rings.

**4.3 A durable fix for the recurring silent GEE hangs — and it fired 15 times in the panel
run.** During the panel extraction the log recorded **15 separate**
`no response from GEE in 900s — treating the hang as a transient error and retrying`
events across the five workers (4/4/4/2/1), on top of **110 ordinary transient network
errors**. Every one of those 15 would have been a **permanent silent stall** under the
previous code: the process stays alive, produces no output, raises nothing, and `_retry`
only ever saw exceptions. That is the failure that cost the Piura panel 13.4 h in a single
incident. Zero workers died and no year was lost.

Original note follows. The Piura work measured ~1 hang
per 30–40 min of sustained GEE work — process alive, no output, no exception, no retry —
including one that burned 13.4 h. The documented cause is that `_retry` only ever saw
*exceptions*, and **a hang is not an error**. Rather than run a national extraction with the
same exposure, the durable fix was implemented: each GEE call now runs under a 900 s
wall-clock deadline on a worker thread, and a blown deadline raises `ChunkTimeout`, which is
transient-classified and retried like any other. Regression:
`tests/test_pipeline.py::TestGeeFaultTolerance`.

## 5. Model results

Feature store: **14.75 M pixel-obs → 54,438 parcels × 141 flat features**, plus per-date
`[54438, 64, 11]` and pixel-set `[54438, 64, 8, 11]` tensors. Five spatially-blocked folds,
identical policy to Piura. **The locked test has not been touched.**

| run | CV mean ± std | pooled CV | acc | `ANNUAL` F1 | `PASTURE_FALLOW` F1 | `PERENNIAL` F1 | T |
|---|---|---|---|---|---|---|---|
| **`lightgbm_nometa`** | **0.628 ± 0.013** | **0.629** | 0.625 | 0.627 | 0.606 | **0.655** | 1.093 |
| `lightgbm_nometa_from1999` | 0.606 ± 0.026 | 0.607 | 0.604 | 0.613 | 0.583 | 0.625 | 1.160 |
| `ltae` | 0.602 ± 0.010 | 0.603 | 0.598 | 0.587 | 0.588 | 0.634 | **2.081** |
| `lightgbm_nometa_nolat` | 0.581 ± 0.002 | 0.581 | 0.576 | 0.574 | 0.554 | 0.615 | 1.066 |
| *Piura `lightgbm_nometa`* | *0.648 ± 0.036* | *0.650* | *0.699* | *0.781* | *0.485* | *0.685* | *1.409* |

All four pooled figures are over the **identical 43,419 CV parcels**, so the columns are
directly comparable. `ltae` was added 2026-08-09 on exactly the same sample, the same
`splits_meta.json` folds and the same feature store; it receives per-date tensors and **no
statics at all**, which is why it is worth running. Its fold scores are 0.600 / 0.612 /
0.587 / 0.610 / 0.602.

**LTAE loses to LightGBM on CV nationally, reversing Piura**, where tuned LTAE beat
LightGBM on all five folds (0.658 vs 0.647). It is also by far the **worst calibrated before
scaling** — ECE 0.120 and T = 2.08, against 0.023–0.026 and T ≈ 1.07–1.09 for the LightGBM
runs — which matches its training curves: train macro-F1 climbs past 0.79 while validation
plateaus near 0.60 within three or four epochs. Best-epoch checkpointing on validation
macro-F1 is what keeps it competitive at all. Defaults were used (`d_model=128`,
`dropout=0.2`, `lr=1e-3`); it was not swept, because the deciding criterion is LODO (§6.2c)
and LTAE loses that too, by a margin a sweep is not going to close.

### 5.1 Going national costs 0.02 macro-F1 and buys a much steadier estimate

0.628 against Piura's 0.648 — but **fold-to-fold variance falls nearly 3x, 0.036 → 0.013**
(folds span 0.612–0.648). 677 regions across 14 departments give a far more stable estimate
than 256 regions inside one. The model is also **better calibrated before scaling**:
ECE 0.026 and T = 1.09, against Piura's ECE 0.090 and T = 1.41.

**The per-class picture changes more than the headline does**, and in a way that favours the
national data:

* `PASTURE_FALLOW` **0.485 → 0.606**. Piura's weakest class was pasture/fallow, which it had
  little of; the sierra is grazing country and supplies it in quantity.
* `ANNUAL` **0.781 → 0.627**. Piura's rice-and-cotton monoculture made `ANNUAL` nearly
  trivial. Nationally it spans irrigated coastal rice, sierra potato and highland barley —
  genuinely harder, and the 0.781 was flattering.
* `PERENNIAL` 0.685 → 0.655, roughly held.

So the national model is **markedly more balanced across classes** (0.61–0.66 vs Piura's
0.49–0.78). For a research question about movement *between* these classes, an even error
profile matters more than a headline point.

### 5.2 The El Niño exclusion was a Piura-specific remedy, not a general one

Task-1 established that dropping the 1997–98 cohort was free in Piura: pooled CV on ≥1999
validation parcels moved 0.6415 → 0.6412, i.e. **−0.0003**. Nationally the same restriction
costs **−0.0048** on the same kind of comparison (0.6306 → 0.6258), and −0.0225 overall.

That is the expected result and it is worth stating: the 1997–98 El Niño was a **northern
coastal** event. Nationally, 1996–98 also holds sierra and southern parcels whose imagery is
perfectly readable, so excluding those years now discards genuine training signal rather than
only noise. **`--train-years` remains the right tool; the years to exclude are regional.**
`lightgbm_nometa` (all years) is therefore the primary national model, where Piura's primary
is the ≥1999 one.

### 5.3 Latitude is doing real work nationally — but which kind?

Dropping `centroid_lat` costs **0.047** nationally against **0.006** in Piura. That is the
D9 concern made measurable: latitude spans ~14° here instead of ~1°, and in Peru latitude is
very close to a climate-zone label.

The CV number alone **cannot** distinguish legitimate agro-climatic signal from spatial
memorisation, because CV holds out 5 km cells that sit inside departments the model has
already seen. Leave-one-department-out is the discriminating test, and both variants are put
through it in §6.

The whole downstream chain — `features assemble` → `train` → `allperu lodo` — was **verified
end-to-end** partway through extraction, on the 184 pixel chunks completed at that point
(2.2 M pixel-obs, 6,867 parcels, all 14 departments). Tensors build at the expected shapes
(`[6867, 64, 11]` per-date, `[6867, 64, 8, 11]` pixel-set), 141 flat features, five spatial
folds, and LODO runs over every department with the buffer applied.

**No numbers from that smoke run are reported here.** Those 6,867 parcels are the 1996–97
cohort alone — the El Niño years, and a year-skewed slice of one corner of the sample — so
anything measured on them describes the slice, not Peru.

Two bugs it caught, both of which would have surfaced only at the end of a multi-hour run:

* **`label_map.json` was never copied into the sampled workspace.** The label build writes it
  to `all_peru_full/`; `train.py` and `infer.py` read it from the current workspace. Every
  downstream command would have died with `FileNotFoundError`. `allperu sample` now carries
  it across.
* **A `sample_weight` name collision.** `allperu.sample` wrote a column of that name
  (modelling sample → national population) and `perennial/panel.build_panel` computes its own
  (panel → modelling sample); the merge suffixed both and `build_panel` raised `KeyError`.
  They are successive stages of one design and must **multiply**, not collide. The first is
  now `population_weight` and `build_panel` composes the two, so a panel weight expands the
  whole way: 708,756 parcels — the population minus coverage-gate failures.

## 6. Leave-one-department-out — the evaluation Piura could not run

Piura's spatial CV holds out 5 km *regions*: neighbouring cells, same department, same
agro-climate, same titling campaign. With 14 departments there is finally a test of transfer
to an unseen **place**. Implemented in `allperu/lodo.py` — same 1.5 km buffer applied at
department borders, and an inner validation split carved from the *training* departments so
early stopping never sees the held-out one.

### 6.1 The spatial-generalisation gap, measured

| | macro-F1 |
|---|---|
| unseen 5 km cell (pooled CV) | **0.629** |
| **unseen department (pooled LODO)** | **0.539** |
| unseen department (unweighted mean over 14) | 0.417 |

**Transfer to a department the model has never seen costs ~0.09 macro-F1** against transfer
to an unseen cell inside a familiar one. That gap is invisible to every evaluation the Piura
project could run, and it is the single most important thing the national data buys.

The unweighted mean (0.417) sits well below the pooled figure (0.539) because macro-F1
punishes departments with skewed class mixes — Tumbes is 75 % perennial, Pasco 60 %. Pooled
is the better summary of "what one national model achieves on unseen ground"; the mean is the
better summary of "how it does on an arbitrary new department".

Piura scores 0.301 here — 13th of 14 — which earlier revisions reported as "Piura is one of
the hardest departments to predict from the rest of Peru". **⚠️ Corrected 2026-08-09: that is
a property of this particular model, not of the department.** The figures in this section are
all `lightgbm_nometa`, the variant that was subsequently *rejected*. Under the selected
`nolat` model Piura scores **0.479 (6th of 14)**, and under LTAE — which has no location
feature whatsoever — **0.512 (2nd of 14)**. Piura is the most distinctive latitude band in
the sample, so it is where a latitude lookup table fails hardest; remove the lookup table and
it becomes one of the *easier* departments. §6.2c.

### 6.2 ⭐ Latitude is spatial memorisation, not agro-climatic signal

This is the result LODO existed to produce, and it is decisive.

| | CV (unseen cell) | LODO mean | LODO pooled | LODO std |
|---|---|---|---|---|
| `lightgbm_nometa` (has `centroid_lat`) | **0.628** | 0.417 | 0.539 | 0.098 |
| `lightgbm_nometa_nolat` | 0.581 | **0.477** | 0.537 | **0.082** |
| **difference** | **−0.047** | **+0.060** | −0.002 | −0.016 |

**Latitude's contribution reverses sign the moment the test is a new place.** It is worth
+0.047 when the model has already seen the department, and −0.060 (per-department mean) when
it has not. **12 of 14 departments improve when it is removed**, and the spread across
departments narrows:

| dept | with lat | no lat | Δ |
|---|---|---|---|
| PIURA | 0.301 | **0.479** | **+0.178** |
| MOQUEGUA | 0.302 | 0.467 | +0.164 |
| ANCASH | 0.374 | 0.525 | +0.151 |
| AREQUIPA | 0.366 | 0.478 | +0.112 |
| ICA | 0.476 | 0.560 | +0.083 |
| LA_LIBERTAD | 0.316 | 0.395 | +0.079 |
| LIMA | 0.529 | 0.590 | +0.061 |
| TUMBES | 0.408 | 0.465 | +0.057 |
| HUANCAVELICA | 0.443 | 0.475 | +0.032 |
| TACNA | 0.498 | 0.522 | +0.023 |
| PASCO | 0.274 | 0.297 | +0.022 |
| AYACUCHO | 0.597 | 0.598 | +0.001 |
| CAJAMARCA | 0.460 | 0.399 | −0.061 |
| LAMBAYEQUE | 0.492 | 0.422 | −0.070 |

In Peru latitude is very nearly a climate-zone label, so the *plausible* reading was that it
carries genuine agro-climatic information. It does not transfer like information; it
transfers like a lookup table. A model given latitude learns "parcels near −5.2° are mango"
rather than "this phenology is perennial", and that knowledge is worthless — actively
harmful — on ground it has not seen. Piura gains most from its removal precisely because
Piura is the most distinctive latitude band in the sample.

**This is the same lesson as Piura's §4.6/§7.0.3 arriving by a third route.** `frac_l7`
manufactured *change*; `centroid_lat` manufactured *stability* in a panel; here it
manufactures *accuracy* that does not leave the training departments. All three are the model
answering from something other than this parcel's spectral year.

### 6.2c A structurally static-free model does NOT transfer better — a negative result

If time-invariant features are what stops a model transferring, the clean test is an
architecture that has **none of them**. LTAE consumes per-date spectral sequences plus DOY
and a padding mask; `centroid_lat`, `area_ha`, `n_pixels_est` and the acquisition metadata
never reach it. The mechanism §6.2 identifies simply cannot apply to it.

It still does not win.

| run | statics it carries | CV | LODO mean | LODO pooled | LODO std |
|---|---|---|---|---|---|
| `lightgbm_nometa` | `centroid_lat`, `area_ha`, `n_pixels_est` | **0.628** | 0.417 | **0.539** | 0.098 |
| **`lightgbm_nometa_nolat`** ⭐ | `area_ha`, `n_pixels_est` | 0.581 | **0.477** | 0.537 | 0.082 |
| `ltae` | **none** | 0.602 | 0.442 | 0.508 | **0.056** |

LTAE lands **between** the two LightGBM variants on the LODO mean and is **worst of the
three on pooled LODO** (−0.029 against both). Per department it is best in only 4 of 14
(`nolat` is best in 8), and it beats `nolat` in just 6 of 14.

| dept | `nometa` | `nolat` | `ltae` |
|---|---|---|---|
| AYACUCHO | 0.597 | **0.598** | 0.552 |
| LIMA | 0.529 | **0.590** | 0.447 |
| ICA | 0.476 | **0.560** | 0.417 |
| ANCASH | 0.374 | **0.525** | 0.415 |
| TACNA | 0.498 | **0.522** | 0.503 |
| PIURA | 0.301 | 0.479 | **0.512** |
| AREQUIPA | 0.366 | **0.478** | 0.379 |
| HUANCAVELICA | 0.443 | **0.475** | 0.464 |
| MOQUEGUA | 0.302 | **0.467** | 0.390 |
| TUMBES | 0.408 | 0.465 | **0.491** |
| LAMBAYEQUE | **0.492** | 0.422 | 0.433 |
| CAJAMARCA | **0.460** | 0.399 | 0.426 |
| LA_LIBERTAD | 0.316 | 0.395 | **0.404** |
| PASCO | 0.274 | 0.297 | **0.356** |
| **mean** | 0.417 | **0.477** | 0.442 |
| **std** | 0.098 | 0.082 | **0.056** |

**What LTAE does win is consistency**: the lowest across-department spread of the three
(0.056 vs 0.082 and 0.098), and it lifts `nolat`'s two worst departments — PASCO
0.297 → 0.356 and LA_LIBERTAD 0.395 → 0.404. Its own floor (PASCO, 0.356) sits above
`nolat`'s floor and well above `nometa`'s four weakest. So the static-free architecture does
deliver the flatter
transfer profile the mechanism predicts; it just delivers it around a **lower mean**,
because its ceiling is lower everywhere (CV 0.602 vs 0.628).

**Read this as bounding the §6.2 claim rather than contradicting it.** Removing
`centroid_lat` from LightGBM buys +0.060 of transfer. Removing *every* static, by switching
architecture, does not buy more — it costs 0.035 against `nolat`. "Fewer statics" is
therefore not a monotone recipe for generalisation: the specific feature identified in §6.2
was the problem, not staticness as such. **Note also that this is the exact opposite of what
the panel gate finds about *flicker* (§7.5), where the ordering by staticness is perfectly
monotone.** A model can be simultaneously the least stable across years and the most even
across places, and LTAE is.

Notably, **PIURA is LTAE's 6th-best department (0.512) where it is LightGBM's worst
(0.301)** — the department the whole project was built on is the one an architecture with no
location information reads best.

### 6.2b ⛔ The locked test is UNSPENT, deliberately, and should stay that way for now

11,441 parcels across 125 contiguous regions (20 %), never scored. **Do not run
`--eval-test` yet.**

Selection above used spatial CV and LODO only, so the test is still a clean, unbiased
estimate — and it is the *only* one left. Piura's equivalent was spent twice (once for
selection, once "for interest" on a tuned LTAE) and is no longer a valid basis for any
future decision (`../perennial/RESULTS.md` §4.4, §4.5). That is a mistake worth not
repeating.

**The right moment to spend it is after the Phase-7 gate returns**, because the outcome
determines whether the modelling is finished:

* **gate passes** → the pipeline is done, and the locked test is the final headline number.
* **gate fails** → models or the estimand will have to change, and an unspent test is worth
  far more than a number obtained before that iteration.

Spending it now would buy one figure and forfeit the ability to evaluate whatever comes next.

### 6.3 Model selection

**Selected for national work: `lightgbm_nometa_nolat`.** It loses 0.047 of CV and wins the
question that actually matters — +0.060 on unseen departments, lower variance across them,
and a pooled figure that is a statistical tie (−0.002). The deliverable is a classifier
applied across places and years it was not trained on, so LODO is the relevant estimator and
CV is the optimistic one.

This deliberately reverses Piura's choice (§4.6 kept `centroid_lat` because dropping it cost
0.006 and bought nothing there). Both decisions follow the same rule — *prefer the model
whose errors are benign for the intended use* — applied to different evidence.

**Re-selection 2026-08-09, after adding LTAE: the choice is unchanged.** LTAE was trained,
calibrated and put through the identical LODO protocol, and it loses the deciding criterion
— LODO mean 0.442 against `nolat`'s 0.477, and worst of the three on pooled LODO. It also
loses CV to `lightgbm_nometa`. It wins only across-department *spread* (§6.2c). **This is a
negative result and is recorded as one:** the pre-run expectation was that an architecture
immune to the §6.2 mechanism would transfer best, and it did not. `runs/all_peru/selected_model.json`
carries all three candidates.

**Selection labels the *primary* model; it does not decide which models are gated.** All
three arms went through the Phase-7 panel gate, because they differ in exactly the variable
the gate's S5 criterion is sensitive to. §7.5.

## 7. Panel — **GATE RUN. ⛔ FAILED ON ALL THREE ARMS.**

Panel years **1999–2023**, not 1996–2023: Task-1's ≥1999 experiment
(`../perennial/RESULTS.md` §8.3) showed 1996–98 are the years that break temporal transfer,
and excluding them flipped S4 from FAIL to PASS at zero CV cost. There is no reason to
re-import a known-bad baseline into a new workspace.

**The gate binds here exactly as it does in Piura.** If S4 or S5 fails, no trajectories, no
transitions, no area estimates.

### 7.1 Extraction — complete (2026-08-08)

| | |
|---|---|
| Panel parcels | 4,565 |
| Years | 25 (1999–2023) |
| Pixel chunks | 2,949 (+450 coverage chunks) |
| **Pixel-observations** | **37,620,414** |
| Parcels per year (min–max) | 4,232 (2013) – 4,529 (2000) |
| Wall clock | ~7 h 40 m across 5 concurrent workers, disjoint 5-year ranges |

Per-year stores are `data/processed/all_peru/features/panel/pixels_<year>.parquet`.

**Completeness was verified by parcel count against each year's coverage-gate survivors, not
by file existence.** This matters: `run_pixels` combines a year by globbing every chunk on
disk at the moment it runs, so a worker that finishes early rewrites the per-year file for
years another worker is still extracting. Caught in the act — `pixels_2023.parquet` sat on
disk at 2,719 of 4,355 parcels (62 %) while looking perfectly well-formed. Every per-year
store was therefore rebuilt once, from the full chunk set, after all workers exited
(`rebuild_panel_years.py`). A file-existence check would have fed a 38 %-truncated year
straight into the gate.

### 7.2 What the deadline fix bought — and the bug it hid

The 900 s wall-clock deadline in `_retry` fired **25 times** across the five workers
(8 / 8 / 1 / 2 / 6), alongside **96** ordinary transient network errors. **Zero workers were
lost and zero tracebacks were raised.** Each of those 25 would previously have been a
permanent silent stall — the failure mode that once cost 13.4 h. This supersedes the "15
hangs" figure recorded mid-run.

**But the fix had a defect of its own, found only because the run was watched to the end.**
All five workers finished their extraction and then sat at **0 % CPU for up to three hours**,
never exiting. The data was complete the whole time; only the processes were stuck. Cause:
`_retry` ran each call on a `ThreadPoolExecutor`, whose threads are **non-daemon**, and
`concurrent.futures.thread` registers an `atexit` hook that *joins* them. A thread parked in
a hung GEE socket read is never joinable, and `shutdown(wait=False, cancel_futures=True)`
does not help — `cancel_futures` only drops futures still *queued*, never one already
running. So every timeout orphaned a thread, and the process then blocked forever in
interpreter shutdown joining it.

Fixed: `_call_with_deadline` now runs the call on an explicit `threading.Thread(daemon=True)`,
which the interpreter abandons at exit instead of joining. Regression tests pin both the
daemon flag and the absence of any surviving futures worker
(`tests/test_pipeline.py::TestGeeFaultTolerance`).

⚠️ **The generalisable lesson is the same one this project keeps re-learning**: the fix for a
silent failure introduced a *second* silent failure one layer down. A hung process that has
already written all its output is indistinguishable, from the outside, from a hung process
that has written none — which is precisely why completeness here is measured in parcels, not
in files or in "the job finished".

### 7.3 What ran (2026-08-09)

Assemble, inference for **three** arms and the Phase-7 gate for **three** arms, in that
order. Nothing else. Workspace `CC_PROC=data/processed/all_peru`,
`CC_FEAT=data/processed/all_peru/features`, `CC_RUNS=runs/all_peru` throughout; the locked
test was never touched.

```bash
# A. assemble — 25 years
uv run python -m crop_classifier.cli perennial panel assemble --years 1999-2023

# B. infer — LightGBM arms, one process, no torch (macOS libomp)
uv run python -m crop_classifier.cli perennial panel infer \
    --run runs/all_peru/lightgbm_nometa_nolat --years 1999-2023 \
    --out data/processed/all_peru/panel_predictions_nolat.parquet
uv run python -m crop_classifier.cli perennial panel infer \
    --run runs/all_peru/lightgbm_nometa --years 1999-2023 \
    --out data/processed/all_peru/panel_predictions_nometa.parquet

# C. infer — LTAE arm, SEPARATE process
uv run python -m crop_classifier.cli perennial panel infer \
    --run runs/all_peru/ltae --years 1999-2023 \
    --out data/processed/all_peru/panel_predictions_ltae.parquet

# D. gate every arm (exits non-zero on failure — all three exited 1)
uv run python -m crop_classifier.cli perennial diagnostics --tag nolat \
    --preds data/processed/all_peru/panel_predictions_nolat.parquet
uv run python -m crop_classifier.cli perennial diagnostics --tag nometa \
    --preds data/processed/all_peru/panel_predictions_nometa.parquet
uv run python -m crop_classifier.cli perennial diagnostics --tag ltae --elnino-model ltae \
    --preds data/processed/all_peru/panel_predictions_ltae.parquet
```

Assemble produced all 25 per-year bundles. **Verified by parcel count, not file existence**:
for every year the LightGBM feature table, the per-date tensor and the raw pixel store agree
exactly (1999: 4,520 / 4,520 / 4,520 … 2023: 4,325 / 4,325 / 4,325). Each arm then produced
**114,125 parcel-years** — 4,565 × 25 exactly — with an identical abstention profile
(4,325 predicted, 240 abstained: 210 `quality_gate`, 30 `no_features`).

**Gate arms were fixed before LODO was known**, so the comparison could not be chosen after
seeing which model won. That mattered: LTAE turned out to lose selection (§6.2c) and is
nevertheless the most informative arm here.

### 7.4 ⛔ The gate FAILS. S5 flicker fails on every arm, by a wide margin.

| arm | statics | S4 | k=0 acc | worst k (\|k\|≤3) | deviation | S5 | `PERENNIAL` flicker | gate |
|---|---|---|---|---|---|---|---|---|
| `nometa` | `centroid_lat`, `area_ha`, `n_pixels_est` | **PASS** | 0.586 | k=+2, 0.616 | 0.029 | **FAIL** | **0.517** | ⛔ FAIL |
| `nolat` ⭐ | `area_ha`, `n_pixels_est` | **PASS** | 0.544 | k=+1, 0.557 | 0.013 | **FAIL** | **0.744** | ⛔ FAIL |
| `ltae` | **none** | **FAIL** | 0.581 | k=−3, 0.474 | **0.107** | **FAIL** | **0.980** | ⛔ FAIL |

Criteria: S4 within 0.10 of the k = 0 accuracy for |k| ≤ 3; S5 under 0.15 for
`PERENNIAL`-labelled parcels. Artifacts: `phase7_gate_{nometa,nolat,ltae}.json`,
`s4_temporal_transfer_*.csv`, `s5_flicker_report_*.csv`, `feature_drift_*_*.csv`, figures in
`runs/all_peru/diagnostics_*`.

**The same conclusion as Piura, reached on 14 departments and a spread of label years: the
classifier works and the annual trend does not.** No trajectories, transitions or area
estimates exist, and none should be produced.

### 7.5 ⭐ The pre-registered prediction is confirmed exactly, and it is worse nationally

Registered in [`plan.md`](plan.md) §7b and in §7.4 of this file's pre-gate revision, before
any of this ran: *if Piura's §7.0.3 mechanism generalises, flicker must be ordered*
`nometa` < `nolat` < `ltae`, *because time-invariant features manufacture stability and
`ltae` is the only static-free arm.* A `nolat` below `nometa` would have been evidence
**against** the mechanism. The outcome is recorded against the prediction in plan.md §7b.1.

| arm | statics carried | Piura flicker | **all-Peru flicker** |
|---|---|---|---|
| `nometa` (has `centroid_lat`) | 3 | 0.428 | **0.517** |
| `nolat` | 2 | 0.547 | **0.744** |
| `ltae` (no statics at all) | 0 | 0.780 | **0.980** |

**Monotone, in the predicted direction, at every rung, and higher than Piura at every rung.**
The national spread is also wider: 0.463 from `nometa` to `ltae`, against Piura's 0.352.

And as in Piura, **k = 0 accuracy barely moves across the three** — 0.586 / 0.544 / 0.581, a
range of 0.042 — while flicker nearly doubles. The statics are contributing almost nothing to
whether a given parcel-year is classified correctly, and almost everything to whether the
answer stays the same next year. That is the definition of manufactured stability.

**So `nometa`'s 0.517 is the optimistic end and `ltae`'s 0.980 is the honest one.** A
static-free model changes its mind about a parcel in **98 % of cases** over 25 years. Read
against the flat 4-year-window smoothing supplement (0.195 / 0.393 / 0.759) the point stands:
smoothing halves the number without supplying any evidence that the underlying series is
real, which is exactly what plan §9.2 warned it would do.

**One national result is genuinely new and rules out the most attractive excuse.** Piura
could not separate "flicker is real instability" from "flicker is thin, El Niño-damaged
coverage", because Piura had four badly degraded years (1997 at 48.3 % gate pass, 2009 at
49.2 %, 2011 at 43.3 %, 2012 at 83.6 %). **The national panel has no thin years at all** —
the same years pass at 95.4 %, 96.2 % and 97.6 %, and the *minimum* over all 25 years is
93.4 % (2013), against a mean of 4,444 of 4,565 parcels per year. Fourteen departments span
many Landsat path/rows, so a cloudy year in one place is covered elsewhere. **Flicker of
0.98 on a panel with no coverage failures cannot be blamed on coverage.** (The
`flagged_years_dropped` supplement confirms it arithmetically — 0.526 / 0.749 / 0.983,
essentially unchanged — but the coverage table is the stronger argument, since
`diagnostics.FLAG_YEARS` is a Piura-measured constant and is merely conservative here.)

### 7.6 S4 passes for LightGBM and fails for LTAE — and 1999 is why

S4 **passes comfortably on both LightGBM arms** (deviation 0.013 and 0.029, tolerance 0.10),
where Piura's baseline panel **failed** at 0.146. Two things changed and both help:

* **The panel starts in 1999.** Piura's failing bin was 66 % panel-year 1996 and contained
  1997 at accuracy 0.326; ≥1999 removes it, which is exactly what Piura's own `--train-years
  1999-2023` experiment (`../perennial/RESULTS.md` §8.3) showed flips S4 to PASS.
* **Label years are spread over 1997–2006 nationally**, so no k bin is dominated by a single
  cohort. Every |k| ≤ 3 bin here holds 1,375–3,163 test parcels.

**LTAE fails S4 at 0.107** — barely over the 0.10 tolerance, and entirely at **k = −3**
(0.474 against 0.581 at k = 0); k = −2 … +3 are all within tolerance. Backward transfer is
its weak direction, the same asymmetry Piura saw. Given it also fails S5 at 0.980, the S4
margin is not the interesting part.

### 7.7 The El Niño arm is **diluted nationally and must not be read as a clearance**

The 1999+2000 → 1998 confound test returns "generic temporal-transfer decay" on all three
arms — all classes degrade roughly equally against the control, `PERENNIAL` recall on 1998
is 0.498 (control 0.618) for LightGBM and 0.598 (control 0.814) for LTAE. That is a very
different picture from Piura's smoking gun, where `PERENNIAL` recall on 1998 collapsed to
0.000 against a control of 0.361.

**This is not evidence that the El Niño confound is resolved.** The 1997–98 El Niño was a
**northern coastal** event, and the national 1998 arm is drawn from regions that hold both
1998 and 1999/2000 parcels — which nationally means **4,358 parcels of which only 241 (5.5 %)
are Piura**, and 6.2 % the whole north coast. TACNA (876), AYACUCHO (601), ANCASH (590) and
ICA (561) dominate it. The test is therefore measuring generic 1998-versus-1999/2000 transfer
over mostly southern and sierra parcels that the flood never touched. It says the *national*
1998 cohort is not catastrophically unreadable; it says nothing new about Piura's, and it
does not lift Piura's §8.1 finding.

### 7.8 Sensor drift — only one era boundary is measurable

**2 of 15 boundary steps exceed 2× the within-era year-to-year movement**, both raw bands at
the L5+L7 → L7-only boundary (`G_median` −3.58×, `B_median` −2.52×); no index feature does.
Cross-parcel spread grows into the L7-only era (index 0.350 → 0.376, raw-band amplitude
0.208 → 0.280), the same widening Piura saw.

**Only 15 steps, not 30, and that is structural**: `era_steps` compares consecutive mission
eras, and the "L5 only" era is 1996–1998, which this panel deliberately does not contain. So
the TM→ETM+ boundary is untestable here by construction. Drift is context for the gate, not
part of its verdict, and it is not what failed.

### 7.9 What was NOT run, and why

* **Trajectories, transitions and area estimates.** The gate failed on every arm, and the
  binding arm is the static-free one (`ltae`, flicker 0.980). The code is built and
  unit-tested; it stays unrun. A trend from a panel that failed validation is not a weaker
  finding, it is a wrong one.
* **The locked test** (11,441 parcels, 125 regions). Still unspent. §6.2b argued the right
  moment to spend it is *after* the gate returns, and the gate returned FAIL — so the
  estimand or the data has to change first, and an unspent test is worth more than a number
  obtained before that iteration.
* **An LTAE sweep.** LTAE loses LODO by 0.035 and the gate by a wide margin; tuning changes
  neither conclusion. Piura's 30-trial sweep moved LTAE's CV by ~0.011, which is inside
  selection noise.

### 7.10 Where this leaves the national panel

The failure is **not** the one Piura diagnosed. Piura's panel failed S4 *and* S5, with a
class-specific El Niño collapse underneath. The national panel **passes S4 on both LightGBM
arms**, has no thin years, and shows no class-specific El Niño collapse in the arm it can
measure — and still fails S5 at 0.517 / 0.744 / 0.980.

That isolates the problem. It is not the 1996–98 baseline, not coverage, not the sensor
boundary, and not the architecture — "try the other model" was already a closed route in
Piura and is closed here too, in the opposite direction (the *best*-transferring architecture
flickers *most*). What remains is that **a single-year 3-class land-state classifier at
~0.55–0.59 accuracy is simply not accurate enough to support a per-parcel annual
trajectory**. Per-year error is largely independent across years, so a parcel's 25-year
series is dominated by classification noise rather than land-use change.

The options are unchanged from `../perennial/RESULTS.md` §7.0.2, and the most promising is
still the same one: **drop per-parcel annual trajectories as the estimand** and report an
aggregate share per year with confidence intervals, where independent per-parcel errors
partly cancel instead of compounding. Nationally that estimand would additionally have to be
weighted — `population_weight` × panel `sample_weight` — because the sampling design doubles
the raw `PERENNIAL` share (9.9 % population → 20.2 % sample). **That analysis has not been
run and is not implied by anything above.**

---

## 8. The window pivot — ⛔ **T1, T2 and T3 all FAIL** (2026-08-10)

[`window_plan.md`](window_plan.md) proposed changing the *estimand* rather than the model:
aggregate predictions to 5-year windows, take the baseline from the **observed** PETT label
instead of predicting it, and identify the tenure contrast **within** year. It registered
three cheap falsification tasks to run before spending any GEE budget. All three were
implemented and run. **All three failed.** No extraction was funded and no estimate was
produced — the plan's own stop rule.

Code: `allperu/windows.py` (T1 + T3), `allperu/loyo.py` (T2), `allperu/tenure.py`,
`allperu/window_sample.py`, `allperu/estimate.py`, `allperu/external.py`,
`allperu/export_crops.py`. Tests: `tests/test_window_pivot.py` (21).

### 8.1 The Piura §1 numbers reproduce — and they are arm-specific

`windows.py` reproduces window_plan §1 exactly on annual flicker (3-class 0.432 / 0.552 /
0.787 and perennial-vs-rest 0.350 / 0.460 / 0.762 for `nometa` / `nolat` / `ltae`). The
window-change fractions come out slightly *lower* than the plan's scratch figures (≥2 changes
0.034 / 0.047 / 0.197 vs 0.039 / 0.053 / 0.218) because parcels are required to qualify in
≥2 windows before they can be counted as changing.

The important part is what the plan's §1 table did not split out: **the Piura window
diagnostic passes on `lightgbm_nometa` and fails on `lightgbm_nometa_nolat`.** The control
pool — PETT-`PERENNIAL` parcels, which must be flat — moves +0.003/decade for `nometa` and
−0.017/decade for `nolat`. The plan's evidence for M2 came from the arm that carries the most
time-invariant features, i.e. the arm LODO disqualified (§6.2).

### 8.2 T3 nationally: the control pool drifts on **every** arm

`uv run python -m crop_classifier.cli allperu windows --preds … --tag …` (exits non-zero on
failure; all three exited 1).

| | W1 ≥2 changes <0.10 | W2 control flat | W3 symmetric noise | W4 monotone at-risk | gate |
|---|---:|---:|---:|---:|---|
| `lightgbm_nometa` | **0.076 PASS** | −0.044/dec **FAIL** | +0.004 PASS | −0.011 **FAIL** | ⛔ |
| `lightgbm_nometa_nolat` ⭐ | **0.088 PASS** | −0.059/dec **FAIL** | +0.003 PASS | −0.001 PASS | ⛔ |
| `ltae` | 0.133 **FAIL** | −0.103/dec **FAIL** | −0.002 PASS | −0.026 **FAIL** | ⛔ |

**W1 is the good news and it is real.** Window aggregation does what M1 claimed: annual
3-class flicker of 0.75 (`nolat`) collapses to an 8.8 % rate of ≥2 window-state changes, and
82 % of parcels never change window state at all. The per-parcel instability that killed
Phase 7 is genuinely fixed by aggregating probabilities.

**W2 is the failure and it is fatal to the design as written.** The at-risk pool rises
+1.8 pp W99→W19 (0.017 → 0.035) while the control pool falls 12.5 pp (0.511 → 0.386). M2's
whole argument was that generic drift would move both pools together and therefore a control
that stays flat licenses reading the at-risk rise. Here the control does not stay flat; it
moves **seven times further than the effect**, in the opposite direction. A +1.8 pp rise
measured against a −12.5 pp drifting yardstick is not interpretable.

It is not a composition artefact: restricting to the 4,409 of 4,527 parcels that qualify in
**all five** windows gives the same series (0.511 → 0.368).

### 8.3 Part of the drift is measurable: probability compression as observations thin out

Landsat observation density is not stationary — the national panel averages 24 clear
observations per parcel-year in W04 and 13 in W19 (L5 retired, L7 SLC-off). A classifier with
a weaker signal reverts toward its prior, which *simultaneously* pushes true perennials down
and true annuals up. That is indistinguishable, at the level of a share, from a real
conversion.

`windows.density_confound_test` measures it **within parcel** (parcel FE, SEs clustered by
parcel), regressing the window-mean probability on `log(n_valid_obs)`:

| arm | statics | PETT-`PERENNIAL` dp/dlog n | PETT-`ANNUAL` | implied control drift | actual |
|---|---:|---:|---:|---:|---:|
| `lightgbm_nometa` | 3 | +0.001 (p 0.95) | −0.027 | −0.000 | −0.090 |
| `lightgbm_nometa_nolat` | 2 | **+0.052** (p 4e-07) | −0.022 | −0.020 | −0.125 |
| `ltae` | **0** | **+0.085** (p 2e-13) | −0.043 | −0.033 | −0.111 |

Two things follow. First, the compression is **real and statistically overwhelming** but
explains only 0–29 % of the observed control drift, so it is a mechanism, not *the*
mechanism. Second — and this is the third independent appearance of the same pattern —
**the compression is monotone in how few time-invariant features the arm carries**, exactly as
`frac_l7` manufactured change and `centroid_lat` manufactured stability. Statics anchor a
prediction against a degrading signal; the arm that transfers best across departments is the
arm whose window series wanders most.

### 8.4 T2 (leave-one-year-out): the endpoint prediction is **not licensed**

New in `allperu/loyo.py` — window_plan §4.3, the temporal analogue of LODO and the single
most valuable evaluation the project had never run. Regions holding ≥2 label-year cohorts
(48,435 of 55,028 parcels, 463 regions), so region is held approximately fixed while year
varies; 1.5 km buffer against the held-out cohort; 14 cohorts with ≥300 parcels.

* LOYO macro-F1 **mean 0.524 ± 0.057, pooled 0.542**, against spatial-CV 0.581.
* **Worst cohort excluding 1998: 2008 at 0.406 — a gap of 0.175 against a 0.10 tolerance.**
* `PERENNIAL` recall ranges **0.370 (2009) to 0.882 (2007)**; CV is 0.674, so the worst-cohort
  gap is 0.305 against the same 0.10 tolerance.
* 1998 — the cohort the plan expected to fail — scores **0.523, mid-pack**. Nationally it is
  only 5.5 % Piura, so the El Niño cohort is not the problem here.

The cohort effect is roughly ±0.09 macro-F1 and ±0.25 `PERENNIAL` recall, which is the same
order as the spatial-generalisation gap of §6.1. **A model whose accuracy on an unseen
*year* moves that much has not earned a 2019–23 prediction**, and 2019–23 is 13 years past
the last cohort it can be tested on.

### 8.5 T1: error is non-differential in **accuracy** but differential in **false positives**

`allperu/tenure.py` puts `ESTADO en RRPP` on `COD_PREDIO` for the first time (it is in every
one of the eight workbooks, on every `DATOS*` sheet, 100 % non-null, binary):
**1,780,552 parcels**, and **100 % of the 43,419 held-out CV parcels resolve**. Parcel-level
INSCRITO share runs from Ayacucho 0.80 to Piura 0.16, confirming window_plan §M4.

| tenure | n | true perennial | predicted | sensitivity | FPR | accuracy |
|---|---:|---:|---:|---:|---:|---:|
| INSCRITO | 26,614 | 0.196 | 0.226 | 0.679 | 0.116 | 0.576 |
| NO INSCRITO | 16,805 | 0.206 | 0.258 | 0.667 | 0.152 | 0.575 |

The plan's own criterion nearly passes: the tenure coefficient in
`correct ~ tenure + true_class + log(area)` with region FE and region-clustered SEs is
**+0.013 (p 0.107)**, and the logit with department FE agrees (p 0.133). Sensitivity is
within 0.013 marginally — though it fails the stratified leg, reaching **0.103** in the
largest area quartile.

**The leg that matters was not in the plan.** The estimand is the perennial share of parcels
whose PETT label is `ANNUAL`; their true perennial share at the label year is ≈0 by
construction, so their *measured* share is essentially the classifier's false-positive rate.
That rate is **0.1114 for INSCRITO and 0.1554 for NO INSCRITO — a −4.4 pp gap, −1.75 pp after
conditioning on region and parcel size (se 0.0072, p 0.016)**.

The design is powered for a ~2 pp differential (§4.6). The classifier's own differential
false-positive rate is **the same size, and points the other way** from the cross-sectional
association (`../perennial/RESULTS.md` §7.5 found INSCRITO parcels ~1.7× *more* perennial).
A contrast of that magnitude cannot be separated from this artefact by any correction that
assumes non-differential error — which is exactly what T1 exists to detect.

### 8.6 The split-design work (§4.1, §4.4) — both done, and §4.1 is reassuring

* **§4.4 balanced test draw.** `splits.pick_test_units` now optionally draws N candidate
  whole-region test sets and keeps the one minimising total-variation distance between test
  and trainval on the joint `year × label` histogram; the achieved distance is written to
  `splits_meta.json` **always**, optimised or not. Over 200 draws of the national table the
  distance ranges 0.130–0.309 (median 0.207); the balanced pick takes the minimum, and the
  resulting test set is 16.5 % label-year 1999 against trainval's 14.7 % — against Piura's
  accidental 48.3 % vs 32.9 %. `n_candidates=1` reproduces the old draw exactly (pinned by a
  test). New configs: `split_window.yaml`, `split_window_b3000.yaml`.
* **§4.1 buffer 3000 m.** Retrained the selected model on a 3 km dead-zone
  (`runs/all_peru/lightgbm_nometa_nolat_b3000`): **CV 0.580 ± 0.024 vs 0.5805 ± 0.002** at
  1.5 km. Doubling the buffer costs **nothing** in mean CV; only fold variance rises. So the
  headline CV number is not measurably inflated by buffer width — though the audit still
  measures label agreement above baseline at 5 km, so this bounds one leakage channel, not
  all of them.
* **§4.5 stratified draw.** `allperu/window_sample.py` draws `dept × tenure × label` strata
  over whole 5 km regions. A dry run gives **13,002 parcels** — 5,001 / 4,999 in the at-risk
  cells, 750 per control cell, all 14 departments represented. **The measured design effect
  is 2.36, not the 1.4 the plan assumed**, so §4.6's ~2,100-per-group becomes ~3,550; the
  ≥5,000 target still covers a 2 pp differential, but a 1 pp differential would need ~12,500
  per group — about twice the whole budget.

### 8.7 The risks (§6) — one of them turned into an opportunity

* **§6.2 reverse causality — a SECOND dated tenure observation exists, and it was already on
  disk.** The bridge `.dta` is not only a key table: it carries the cadastre's own titling
  status (`estado`, 20–26 values) and the date the cadastre was cut (`fech_tran`, a Stata day
  number resolving to **2011–2012** in every department checked). BD SSET's `ESTADO en RRPP`
  is the status at *declaration* (~1997–2006). Together, **1,780,580 parcels carry two dated
  observations of registration status**. Consistency is good — 84.2 % of INSCRITO parcels are
  `REGISTERED` in the cadastre against 21.6 % of NO INSCRITO ones — and **8.6 % (152,742
  parcels) move NO INSCRITO → REGISTERED**, reaching 25.5 % in Piura, 31.8 % in Tumbes,
  20.7 % in Lima. That is a treatment variable with variation *inside* the panel, i.e. the
  two-period difference-in-differences window_plan §6.2 said the design would need to become
  causal. Caveats: `fech_tran` dates the snapshot, not the inscription event; `estado` is a
  pipeline state (73.6 % of NO INSCRITO parcels sit in `IN_PROCESS`), so it is one extra
  period, not a staggered event study. Built as `tenure.tenure_two_period`.
* **§6.4 perennial ≠ export, quantified.** Weighted over the national sample, PERENNIAL
  parcels are **57.3 % export basket, 19.9 % mixed (banana, lime, unspecified orchard),
  22.8 % domestic** — so a measured perennial-share change should be discounted by roughly a
  quarter to a fifth before it is read as export. It varies enormously by department: Tacna
  0.92 export (olive), Piura 0.69, Huancavelica 0.16, Lambayeque 0.13.
  (`allperu/export_crops.py`.)
* **§6.3 woody non-crop, bounded from the declarations.** 20,835 parcels were excluded from
  training as declared woody non-crop — **2.79 % of the eligible pool**, against a national
  PERENNIAL share of 9.9 %. If they all read PERENNIAL at inference, they inflate the
  perennial pool by more than the effect being measured. This is a **lower** bound: it counts
  only *declared* woody non-crop, not vegetation that invaded an abandoned parcel over the
  following 20 years, which nothing in this project observes. MapBiomas cannot close this —
  over Piura it only ever assigns `MOSAIC` and `ANNUAL` (memory: it has no perennial class).
* **§6.1 / T6 external validation — harness built, data not fetched.** `allperu/external.py`
  joins districts to parcels through the bridge (`DISTRITO`, `id_dist` — the only place a
  district lives) and rank-compares district perennial growth against a MIDAGRI/SIEA
  district-crop-year CSV. SIEA data is not redistributed here and is not downloaded
  automatically; the command takes a path. **This remains the only external check on the
  endpoint and it has not been run.**

### 8.8 Where this leaves the pivot

The pivot's first move works and its second does not.

**M1 is vindicated**: aggregating probabilities into 5-year windows removes the per-parcel
instability that failed Phase 7 (≥2 changes 8.8 % against annual flicker 0.75). **M2 is
not**: the control pool that was supposed to license reading the at-risk series drifts
several times harder than the signal, on every arm, and the part of that drift we can
attribute is Landsat observation density — a property of the *archive*, not of the model or
the estimand, and one that no re-weighting of the same predictions fixes. **M3 is untestable
until M2 is fixed**, and T1 shows that even in-window identification has a differential
false-positive rate the size of the target effect.

Three routes remain, and they are ordered by how much they cost:

1. **Make the endpoint comparable to the baseline in observation density**, not just in
   sensor. Subsample every year's observations to a fixed count (or a fixed DOY skeleton)
   before assembling features, so W19 is read with the same amount of evidence as W99. This
   attacks the measured mechanism directly, needs no new labels, and can be tested on the
   *existing* panel store by re-assembling — the cheapest experiment on the list.
2. **Get a second supervision point.** The 2012 CENAGRO census reaches parcels only through
   the weak name link (Chain B) and only in aggregate, but a district-level 2012 anchor would
   turn "the control drifts" from an uninterpretable artefact into a measurable correction.
   T6/SIEA is the external version of the same idea.
3. **Change the treatment, not the outcome.** §8.7's two-period tenure gives a within-parcel
   registration event. A difference-in-differences on *that* differences out any drift common
   to both tenure groups — which is precisely the class of artefact §8.2 and §8.3 found, and
   is a stronger design than the cross-sectional contrast the plan set out to estimate.

**Not run, deliberately:** the T4 extraction (the sample is drawn but not extracted), the T5
estimate (`allperu/estimate.py` refuses while any gate is failed, and records the override if
forced), and the national locked test, which is **still unspent**.

---

## 9. Temporal out-of-distribution skill — [`temporal_ood_plan.md`](temporal_ood_plan.md)

Three successive estimands failed on the same underlying fact: the model is trained on
1996–2009 and asked to predict 2019–2023, and nothing in the design ever made it good at
that or measured whether it was. This section attacks the fact instead of routing around it.

Governing rule **T-D1**: a change is adopted only if **out-of-distribution-*year*** skill
improves. Spatial CV is explicitly *not* the criterion and a CV regression is acceptable.
Deciding metrics (**T-D2**): LOYO mean + worst-cohort macro-F1, worst-cohort `PERENNIAL`
recall, and T3's W2 control-pool slope. Every candidate reports CV, LODO **and** LOYO
together (**T-D3**); nothing overwrites an existing run (**T-D4**); the national locked test
stays unspent (**T-D5**).

New code: `allperu/density.py` (1a audit, degradation, density-conditional temperature, 2c
quantile alignment), `allperu/yearleak.py` (2a), `allperu/oli_overlap.py` (3).
Tests: `tests/test_density.py`.

### 9.1 Step 1 — make the endpoint look like the training years ⛔ **acceptance FAILED**

#### 9.1.0 Scoreboard

All arms are LightGBM on the identical sample, folds and store. `nolat` is the incumbent
selected model; `aug` = degradation augmentation (1b); `norder` = the 33 order-statistic
features dropped (1a-derived); `dcal` = the incumbent's panel re-scaled by a
density-conditional temperature (1c), which cannot change any argmax and therefore has no
CV/LODO/LOYO of its own.

| arm | CV | LODO mean | LODO pooled | LOYO mean | LOYO pooled | LOYO worst¹ | LOYO worst (n≥1000)² | worst `PERENNIAL` recall | **W2 slope/decade** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `nolat` (incumbent) | **0.5805** | 0.4765 | **0.5374** | 0.5236 | 0.5423 | 0.4063 | 0.4063 | 0.3696 | −0.0592 |
| `nolat_aug` | 0.5795 | **0.4770** | 0.5369 | **0.5291** | **0.5499** | 0.3934 | 0.4206 | 0.3636 | **−0.0512** |
| `nolat_norder` | 0.5763 | 0.4754 | 0.5334 | 0.5210 | 0.5423 | 0.3981 | 0.4222 | 0.3442 | −0.0596 |
| `nolat_aug_norder` | 0.5776 | **0.4770** | 0.5356 | 0.5253 | 0.5464 | **0.4269** | **0.4269** | 0.3659 | −0.0698 |
| `nolat_dcal` | (=`nolat`) | — | — | — | — | — | — | — | −0.0591 |

¹ the registered statistic: minimum over cohorts with ≥300 parcels, excluding 1998.
² the same statistic over cohorts with ≥1000 parcels — see §9.1.5 for why both are reported.

**Verdict against the plan's step-1 acceptance (all three required):**

| criterion | target | best achieved | |
|---|---|---|---|
| LOYO worst-cohort gap vs CV | < 0.10 (from 0.175) | **0.151** (`aug_norder`) | ⛔ FAIL |
| T3 W2 control slope | \|slope\| < 0.01/decade | **−0.0512** (`aug`) | ⛔ FAIL |
| LODO mean must not fall > 0.01 | ≥ 0.4665 | 0.4754–0.4770 | ✅ PASS |

**Step 1 does not fix the problem.** The best arm closes 14 % of the W2 gap and 14 % of the
LOYO worst-cohort gap. Density is a real mechanism and it is not the dominant one.

#### 9.1.1 1a — the density audit, and the one clean structural finding

`allperu.density.feature_density_audit` regresses every LightGBM feature on
`log(n_valid_obs)` **within parcel** (parcel FE, SEs clustered by parcel — the vectorised fit
reproduces statsmodels to 1e-9, pinned by a test), over all 25 panel years × 4.5 k parcels =
110,252 parcel-years. `coef_sd` is the slope in units of the feature's own within-parcel SD:
how far a feature moves for an e-fold change in observation count, **with the land held
fixed**.

| family | n | mean \|coef_sd\| | max | significant |
|---|---:|---:|---:|---:|
| static/meta (`n_dates`, `frac_l7`, `max_gap`) | 7 | 0.805 | 2.422 | 86 % |
| **order extremes** (`_min`, `_max`, `_amp`) | 33 | **0.347** | 0.635 | 100 % |
| order quantiles (`_p25`, `_p75`) | 22 | 0.314 | 0.470 | 100 % |
| central (`_median`, `_mean`, `_std`) | 44 | 0.243 | 0.479 | 98 % |
| **fitted** (`_slope`, `_h_mean`, `_h_cos`, `_h_sin`) | 33 | **0.080** | 0.151 | 94 % |

The prediction in the plan is confirmed: order statistics are the worst offenders (`NDWI_max`
0.635, `NDMI_min` −0.589, `BSI_max` 0.551 — the very feature §8.2 identified as the robust
one under the El Niño is one of the *least* robust to observation count). And the finding the
plan did not predict: **the harmonic and slope fits are almost immune — 4.3× less
density-sensitive than the extremes**. A max over 24 draws is a different estimator from a max
over 13; a least-squares harmonic coefficient is the same estimator at both.

Two things follow. The audit is a **structural** result about how to build features from an
archive whose depth is not stationary, independent of anything else here. And it is not
enough on its own: acting on it (`norder`) is the arm that *loses*.

Also measured, and corrected in code: **the SLC-off loss a parcel actually sees is ~11 pp,
not the nominal scene-level 22 %.** Mean per-date pixel count relative to a parcel's own best
date falls 0.968 (1999–2002) → 0.861 (2019–23). The gaps widen toward the swath edge and a
0.5 ha parcel samples one point on that gradient. `SLC_OFF_FRAC = 0.11` is the measured
number; using the literature figure would have over-degraded training twofold.

#### 9.1.2 1b — degradation augmentation: a real, small, marginally significant gain

`build_degraded_features` subsamples each training parcel's **acquisition dates** (never
observations within a date — an acquisition either happened and was clear or it did not) to a
target drawn from the endpoint's empirical `n_dates` distribution, then punches one contiguous
SLC-off-shaped lon-band out of each surviving acquisition. 54,438 → **54,052** parcels; 386
fall below the `n_valid_obs ≥ 4` gate at the degraded density and are **dropped, not imputed**.

The target must be **rank-matched**, not drawn i.i.d.: degradation can only ever remove dates,
so an i.i.d. pairing sends rich targets to poor parcels where they are wasted and the realised
distribution lands well below the endpoint. Rank-matched, the realised training density
reproduces the endpoint almost exactly — mean 12.52 vs 12.52, median 12 vs 12, p75 16 vs 16
(before: mean 15.28, p75 21). Pinned by a test.

Result, on paired per-cohort / per-department differences against the incumbent:

| arm | ΔLOYO mean | t (paired, 14 cohorts) | p | cohorts improved | ΔLODO mean | p |
|---|---:|---:|---:|---:|---:|---:|
| `aug` | **+0.0054** | 1.95 | 0.073 | 10/14 | +0.0005 | 0.89 |
| `norder` | −0.0026 | −0.62 | 0.54 | 7/14 | −0.0011 | 0.79 |
| `aug_norder` | +0.0017 | 0.30 | 0.77 | 7/14 | +0.0006 | 0.87 |

`aug` improves LOYO mean and pooled (+0.0054 / +0.0076), improves the substantive worst
cohort (2008: 0.406 → 0.421), flattens W2 by +0.008/decade, and costs **nothing** on LODO
(+0.0005) or CV (−0.0010). `aug_norder` gives the best worst-cohort (0.4269, +0.021) and the
lowest across-cohort spread (0.0488 vs 0.0568) but the worst W2 (−0.0698).

#### 9.1.3 1c — density-conditional calibration: the compression is **not** a calibration artefact

The clearest negative result in step 1, and it closes a route.

`calibration_ladder` refits fold 0 (the pipeline only persists the final refit, which has seen
every fold's validation parcels — calibrating on those measures nothing), then scores the same
8,605 held-out parcels at six artificial densities plus the undegraded store, and fits
temperature per density band:

| mean n_dates | n | fitted T | NLL | ECE |
|---:|---:|---:|---:|---:|
| 4.6 | 3,469 | **1.186** | 0.918 | **0.076** |
| 6.2 | 12,811 | 1.158 | 0.902 | 0.042 |
| 8.2 | 11,742 | 1.153 | 0.896 | 0.040 |
| 11.1 | 19,470 | 1.028 | 0.839 | 0.021 |
| 19.1 | 12,810 | **0.998** | 0.820 | **0.008** |

`T(log n) = 1.448 − 0.160·log n`, against a global scalar of 1.079.

**Temperature *falls* with density.** At the endpoint's density the model is **over**-confident
(T > 1, needs softening) and ECE is 9× worse than at high density; at 19 observations it is
already calibrated. So the compression is not "the model failing to soften as evidence
thins" — it is the opposite, and the correct fix pushes probabilities *further* toward the
base rate, i.e. in the same direction as the drift.

Applying it changes **nothing that matters**: W2 slope **−0.0591 vs −0.0592**. The
correction is real (T spans 1.00–1.19) but its effect on a share thresholded at 0.5 is
second-order, and it is not sign-reversing. **Probability compression is a genuine loss of
information as the archive thins, and no re-scaling of the same probabilities undoes it.**

#### 9.1.4 What the at-risk series did — the plan asked for this explicitly

Plan §1: "if the control flattens *and* the at-risk rise vanishes with it, that is a real
finding — the rise was the artefact." **It did not happen.** On `aug` the control flattens
(−0.0592 → −0.0512) while the at-risk net change more than *doubles* (+0.0178 → +0.0410).
The two move independently, which is weak evidence against the at-risk rise being purely the
mirror image of the control drift — but the control still moves 1.2–3× further than the
signal on every arm, so nothing is licensed either way.

W1 (the M1 result) is untouched everywhere: ≥2 window-state changes 0.086–0.092 against an
annual 3-class flicker of 0.75. Aggregation still works; it is the level that does not.

#### 9.1.5 A methodological correction to the LOYO criterion

The registered statistic is a **minimum over cohorts**, so it is decided by whichever cohort
is smallest — nationally that is **1996 with 464 parcels**, where macro-F1 moves ±0.03 on
resampling alone. Three of the four arms have their "worst cohort" set by 1996 or 2008, and
`aug` looks worse than the incumbent on the registered number (0.393 vs 0.406) while being
better on the same statistic restricted to cohorts with ≥1000 parcels (0.421 vs 0.406) and
better on 10 of 14 cohorts. `loyo.py` now reports
`worst_cohort_macro_f1_min_support` (≥1000) **alongside** the registered number, never
instead of it — the goalpost is not moved, a second one is added and labelled.

#### 9.1.6 What was NOT adopted, and why

* **`norder` (dropping the 33 order-statistic features) — NOT adopted.** It is the arm 1a's
  ranking most obviously implies, and LOYO rejects it: mean −0.0026 (7/14 cohorts), worst
  `PERENNIAL` recall 0.344 (the worst of any arm), LODO −0.0011, CV −0.0042. **Third
  independent confirmation that a feature-importance-style ranking generates candidates and
  cannot decide them** — after `frac_l7` (2nd by gain, worth +0.0013 to drop) and the
  12-class mission audit (29.8 % of gain worth 0.007). The mechanism is visible in the
  numbers: `BSI_max` is simultaneously the feature §8.2 found most robust to the El Niño and
  one of the most density-sensitive here, and dropping it costs more than the density
  sensitivity buys.
* **Density-conditional calibration — NOT adopted.** §9.1.3: measurable, correctly signed,
  and worth 0.0001/decade. Keeping it would add a moving part to every downstream number for
  no gain. `density_temperature.json` is written and left unused, exactly as
  `harmonization.py` was.
* **`aug_norder` — NOT adopted as primary.** Best worst-cohort and lowest spread, but the
  worst W2 slope of any arm (−0.0698) and its LOYO mean gain is not distinguishable from zero
  (p 0.77).
* **`aug` — adopted as a *change to how models are trained*, not as a new selected model.**
  It improves every temporal metric except the small-cohort minimum, costs nothing on CV or
  LODO, and is cheap. But step 1's acceptance failed, so it licenses nothing downstream and
  `runs/all_peru/selected_model.json` is **not** rewritten on this evidence alone — that
  decision belongs to step 2b, which has LOYO for every candidate.

### 9.2 Step 2 — select on temporal transfer, and remove year-leaking features

#### 9.2.1 ⭐ The headline: **LOYO inherits spatial CV's blind spot**, and only a joint test sees it

The finding that matters most in §9, and it was not anticipated by the plan.

Running LOYO over the existing candidates (plan §2b) put `lightgbm_nometa` — the arm carrying
`centroid_lat`, the one LODO disqualified in §6.2 — **far ahead of everything else**:

| | CV | LODO mean | **LOYO mean** |
|---|---:|---:|---:|
| `lightgbm_nometa` (has `centroid_lat`) | 0.6281 | 0.4169 | **0.5710** |
| `lightgbm_nometa_nolat` | 0.5805 | 0.4765 | 0.5236 |
| **Δ from `centroid_lat`** | **+0.0476** | **−0.0596** | **+0.0474** |

**The CV gain and the LOYO gain are the same number to three decimal places** (+0.0476 /
+0.0474), and only the LODO sign flips. That is not a coincidence — it is the mechanism.
`loyo.py` restricts to regions holding ≥2 cohorts precisely so that *place* is held
approximately fixed while *year* varies (§8.4). But a time-invariant lookup table is exactly
as available when place is fixed and time moves as when both are fixed. **LOYO was built to
test temporal transfer and it cannot distinguish temporal transfer from spatial
memorisation.**

The test that can is free, and it is now `allperu lodyo` (`loyo.lodo_by_cohort`): re-score the
**existing** LODO predictions **per label-year cohort**, so the department *and* the year are
out of distribution at once. No new fits — the per-department models already exist.

| arm | LOYO mean | **LODYO mean** | LODYO worst (n≥1000) |
|---|---:|---:|---:|
| `lightgbm_nometa` | **0.5710** | 0.5083 | 0.4031 |
| `lightgbm_nometa_nolat` | 0.5236 | **0.5133** | 0.4329 |
| `ltae` | — | 0.4965 | 0.4047 |

**`centroid_lat`'s +0.047 LOYO advantage becomes −0.005.** It evaporates the moment the
model must also read a place it has not seen. This is the fourth independent appearance of
the same lesson (`frac_l7`, `centroid_lat` on LODO, LTAE, now LOYO) and the sharpest form of
it: **an out-of-distribution evaluation is only blind-sided along the axis it holds fixed.**
Spatial CV cannot see space; LOYO cannot see space either; LODO cannot see time; LODYO sees
both and is the number to read when they disagree. Nothing in this project's design was ever
out-of-distribution in both axes at once until now.

`nometa` also has the **worst** worst-cohort `PERENNIAL` recall of any arm (0.281 against
`nolat`'s 0.370), which is one of T-D2's own deciding metrics. T-D3 ("a change that buys LOYO
by wrecking LODO is not an improvement") settles it correctly and independently.
**`runs/all_peru/selected_model.json` is not reopened in favour of `centroid_lat`.**

#### 9.2.2 2a — the year-leak audit: cohort is predictable at **0.508** from "land" features

`allperu/yearleak.py`, over 47,597 parcels in regions holding ≥2 cohorts, 14 cohorts, 134
features (`meta` and `location` already withheld). Two rankings: `eta2_year` (one-way ANOVA
R² of each feature on the cohort, computed **within region** so geography is absorbed) and
LightGBM gain on a model whose target *is* the cohort.

**A model that sees only the spectral summary features predicts which of 14 label years a
parcel belongs to at 0.508 accuracy against a 0.155 majority baseline — held out by
region.** That is the generalised `frac_l7` problem quantified for the first time: the
features carry year identity at 3.3× the base rate even in departments the cohort model never
saw.

The top of the within-region ranking is `R_amp`, `G_amp`, `SWIR2_amp`, `NIR_amp`, `R_max`,
`G_max`, `SWIR1_amp`, `R_std`, `B_amp`, `NDWI_amp` — **amplitudes, maxima and standard
deviations**, i.e. very nearly the same set 1a's density audit flagged, arrived at from a
completely different direction. Two independent diagnostics agree that the order statistics
are where the trouble is.

Dropping the top-k and re-running LOYO (the plan's sweep):

| k dropped | LOYO mean | LOYO sd | worst (n≥1000) | worst `PERENNIAL` recall |
|---:|---:|---:|---:|---:|
| 0 (incumbent) | 0.5236 | 0.0568 | 0.4063 | 0.3696 |
| 5 | 0.5245 | 0.0502 | 0.4240 | 0.3841 |
| **10** | **0.5267** | **0.0500** | 0.4253 | 0.3768 |
| 20 | 0.5209 | 0.0510 | 0.4275 | 0.2935 |

k = 10 is the optimum and the curve turns over after it — dropping 20 costs LOYO mean and
collapses `PERENNIAL` recall. The gain is small (+0.003) but the **spread** falls sharply
(0.0568 → 0.0500), which is the shape of a genuine reduction in cohort sensitivity rather
than a level shift.

#### 9.2.3 2c — per-year quantile alignment: ⛔ **rejected, and it is much worse than doing nothing**

The plan flagged this as "cheap and powerful — and dangerous". It is dangerous.

Quantile-mapping every panel year's feature marginals onto the training-year distribution
(`density.write_aligned_panel_bundles`, bundles in `<panel>/<year>_qmap/`) makes the control
pool drift **nearly twice as fast**: W2 slope **−0.1023/decade** against the baseline's
−0.0592, control share 0.501 → 0.277 over W99→W19. The at-risk series is simultaneously
flattened to nothing (+0.0031, from +0.0178).

The mechanism is the one the plan warned about, working in an unexpected direction: forcing
each year's *marginal* onto the training marginal does not preserve the *joint*, and the
class contrast the perennial signal rests on (§8.2 — it is a contrast with annual neighbours,
not a level) is exactly what a marginal alignment destroys. **Not adopted.** The bundles are
kept for reproducibility and the code is documented as a sensitivity arm only.

#### 9.2.4 The step-2b selection table

All LightGBM unless noted; `aug` = degradation augmentation (§9.1.2), `yleak10` = the top-10
year-leaking features dropped (§9.2.2).

| arm | CV | LODO | LOYO | LOYO sd | LOYO worst¹ | worst `PER` recall | **LODYO** | LODYO worst¹ | W2/decade |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `nolat` (incumbent) | 0.5805 | 0.4765 | 0.5236 | 0.0568 | 0.4063 | 0.3696 | 0.5133 | 0.4329 | −0.0592 |
| `nometa` | **0.6281** | 0.4169 | **0.5710** | 0.0782 | **0.4352** | 0.2813 | 0.5083 | 0.4031 | **−0.0437** |
| `ltae` | 0.6021 | 0.4422 | — | — | — | — | 0.4965 | 0.4047 | −0.1025 |
| `nolat_aug` | 0.5795 | 0.4770 | 0.5291 | 0.0576 | 0.4206 | 0.3636 | **0.5149** | 0.4301 | −0.0512 |
| `nolat_norder` | 0.5763 | 0.4754 | 0.5210 | 0.0558 | 0.4222 | 0.3442 | 0.5110 | 0.4277 | −0.0596 |
| `nolat_aug_norder` | 0.5776 | 0.4770 | 0.5253 | **0.0488** | 0.4269 | 0.3659 | 0.5111 | **0.4348** | −0.0698 |
| `nolat_yleak5` | — | — | 0.5245 | 0.0502 | 0.4240 | 0.3841 | — | — | — |
| `nolat_yleak10` | 0.5815 | **0.4772** | 0.5267 | 0.0500 | 0.4253 | 0.3768 | 0.5135 | 0.4329 | −0.0679 |
| `nolat_yleak20` | — | — | 0.5209 | 0.0510 | 0.4275 | 0.2935 | — | — | — |

¹ minimum over cohorts with ≥1000 parcels, excluding 1998 (§9.1.5).

### 9.3 Step 3 — admit OLI, with harmonisation ⛔ **3a FAILS**, and the harmonisation is *harmful*

`PANEL_MISSIONS = {"L5","L7"}` was chosen so every panel year is inferred on radiometry the
model trained on. It is also what creates the endpoint density collapse. Step 3a tests
whether that trade is available by scoring the **same parcel-year** twice.

**Cost, measured before spending anything** (`allperu oli probe`, plan §3a): 0.97–1.17
s/parcel-year, projecting **1.2–1.5 h per full panel year**. Extracted 2015, 2019 and 2022
(2022 also has L9) into `<panel>/oli/`, kept separate from the audited panel store.
**Verified by counting outputs, not by "the process ended"**: 1,467,266 / 1,402,733 /
3,087,714 pixel-obs over 4,473 / 4,441 / 4,529 parcels, and the assembled bundle row counts
match the parcels-with-pixels counts exactly. Gate pass 98.7 / 98.1 / 100 %.

**The density case for OLI is confirmed:** OLI *alone* returns **21.8** clear observations
per parcel-year against L7-only's **13.1** — a 66 % increase from the sensor the panel
refuses to use, before L7 is even added back.

#### 9.3.1 The result, and the control that makes it readable

12,940 paired parcel-years, scored by the selected model on both feature sets:

| arm | median \|Δp\| | p95 \|Δp\| | class agreement | **control-pool mean Δp** |
|---|---:|---:|---:|---:|
| criterion (registered) | < 0.05 | — | ≥ 0.95 | \|·\| < 0.02 |
| **split-half control** (L7 vs L7) | **0.0309** | 0.588 | **0.6551** | **−0.0035** |
| OLI, **un**harmonised | 0.0293 | 0.655 | 0.6309 | −0.0423 |
| OLI, Roy-harmonised | 0.0311 | 0.654 | 0.6164 | **−0.1073** |

The control is `allperu oli split-half`: the same model, same sensor, same year, same
parcel, scored on two **disjoint halves** of the L7 acquisitions. It is the ceiling any
cross-sensor comparison can reach — and it is **0.655**, not 0.95.

**So the registered agreement criterion was never achievable, and that is a criterion error
on our side, not a fact about OLI.** Read against the ceiling, OLI reaches 96 %
(unharmonised) and 94 % (harmonised) of the model's own self-agreement, and its median
per-parcel |Δp| (0.029/0.031) is **indistinguishable from resampling L7's own
observations** (0.031). Per parcel, OLI adds essentially no disagreement beyond the noise the
classifier already has at this observation density.

**The one leg that genuinely fails is the control-pool shift, and it is unambiguous.** The
split-half control's mean Δp is −0.0035, i.e. zero by symmetry as it must be. Unharmonised
OLI shifts the PETT-`PERENNIAL` pool by **−0.042**, twelve times the self-noise. That is a
real, systematic, class-specific radiometric effect: OLI reads the control pool as less
perennial than L7 does, on the same parcels in the same years. Admitting OLI from 2013 would
inject that step directly into the series at exactly the point where an export-crop expansion
would also appear — the failure mode `PANEL_MISSIONS` was created to prevent. **The mission
policy stands and step 3 stops here**, which is the plan's own stop rule.

#### 9.3.2 ⚠️ The Roy harmonisation makes it 2.5× worse — it has never been validated here

`perennial/harmonization.py` has been implemented since the panel was designed, documented as
the safe path to admitting OLI, and **never once run**. Running it:

* control-pool shift **−0.0423 → −0.1073** (2.5× worse);
* class agreement **0.6309 → 0.6164** (worse);
* median |Δp| 0.0293 → 0.0311 (worse).

**On every measured axis the correction is worse than no correction at all.** The mechanism
was measured in §9.5 and it is **not** the one hypothesised here first: raw OLI reads
*greener* than L7 on the same parcel on the same day (**mean NDVI +0.055**), and the Roy
correction pushes it **further** away (**+0.069**) rather than back. It over-corrects in the
visible bands — residual blue bias 0.025 with the correction against 0.015 without it.

The lesson generalises past this project: **an unused correction is an untested correction.**
It sat in the codebase for months as the reassuring answer to "what about OLI?", and the first
time it was measured it was the worse of the two options. It is left implemented, still
unused, now with a measurement attached.

#### 9.3.3 3b — not run

3a's stop rule. No panel was re-extracted with OLI; the mission policy is unchanged. What
this leaves on the table is quantified above: 21.8 vs 13.1 observations per parcel-year at
the endpoint, which is the largest single lever anyone has found on the density mechanism —
and it is unusable until the −0.042 class-specific offset is either explained or corrected
with coefficients fitted on *this* data rather than borrowed.

The combination of the two step-1/step-2 changes was then run as its own arm, and it is
**not** the sum of its parts:

| arm | CV | LODO | LOYO | LODYO | **W2/decade** |
|---|---:|---:|---:|---:|---:|
| `nolat` (incumbent) | 0.5805 | 0.4765 | 0.5236 | 0.5133 | −0.0592 |
| `nolat_aug` | 0.5795 | 0.4770 | **0.5291** | 0.5149 | −0.0512 |
| `nolat_yleak10` | 0.5815 | 0.4772 | 0.5267 | 0.5135 | −0.0679 |
| **`nolat_aug_yleak10`** ⭐ | **0.5811** | **0.4789** | 0.5261 | **0.5173** | **−0.0382** |

Each change alone moves W2 by less than the two together, and `yleak10` alone moves it the
*wrong* way (−0.0679). Combined, the control slope flattens **35 %** — much the largest move
any arm produced, and the only one that also improves CV, LODO, LOYO and LODYO simultaneously.

#### 9.2.5 Selection — `lightgbm_nometa_nolat_aug_yleak10` becomes primary

Adopted under T-D1 (out-of-distribution-year skill improves), T-D3 (no LODO regression) and
§9.2.1's LODYO tie-break. Against the incumbent: **CV +0.0006, LODO +0.0024, LOYO +0.0025,
LODYO +0.0040, W2 +0.0210 toward flat.** It costs nothing anywhere.

**⚠️ None of the labelled differences is individually significant** — paired over cohorts or
departments, p 0.14–0.54 for every one of them (`aug` alone is the closest at p 0.073 on
LOYO). The case for adoption is *consistency in sign across four independent criteria* plus
the W2 move, not a demonstrated effect. Anyone reading this should not treat the new arm as
measurably better at anything; it is better-or-equal everywhere at zero cost, which is enough
to prefer it and not enough to conclude anything.

`runs/all_peru/selected_model.json` is rewritten; the previous record is preserved verbatim
at `runs/all_peru/selected_model_20260809.json`.

**Not adopted, and why** (T-D6):

* **`lightgbm_nometa`** — the LOYO leader by +0.047, disqualified by §9.2.1: the gain equals
  its CV gain to three decimals and reverses on both LODO (−0.0596) and LODYO (−0.005). It
  also has the worst worst-cohort `PERENNIAL` recall of any arm (0.281 vs 0.370).
* **`ltae`** — now measured on the temporal axis for the first time and it is the **worst arm
  there too**: LOYO mean **0.4784** (against 0.5236), LODYO 0.4965, W2 −0.1025. §6.2c found a
  structurally static-free architecture does not transfer better across *space*; it does not
  transfer better across *time* either. The negative result is now two-dimensional.
* **`nolat_norder`** — §9.1.6.
* **`nolat_yleak20`** — LOYO mean falls below the incumbent and worst-cohort `PERENNIAL`
  recall collapses to 0.294. The k-sweep turns over after 10.
* **Quantile alignment (2c)** — §9.2.3.
* **Density-conditional calibration (1c)** — §9.1.3.

### 9.4 Where §9 leaves the project

**Nothing here licenses a 2019–23 estimate, and the gap is not close to closed.** Step 1's
acceptance failed on every arm; the best W2 control slope is −0.0382 against a criterion of
0.01, and the best LOYO worst-cohort gap is 0.151 against 0.10. Three steps of mitigation
recovered roughly a third of one artefact. **No trajectory, transition or area estimate
exists or should be produced, and the national locked test is still unspent.**

What §9 did produce is worth more than the increment it bought:

1. **An evaluation the project did not have.** LODYO is free, it is the only test here that is
   out of distribution in space *and* time, and it overturns what LOYO alone would have
   concluded. Every future candidate should report it.
2. **A structural feature result.** Harmonic and slope fits are 4.3× less sensitive to
   observation count than order statistics — a property of building features from a
   non-stationary archive, not of this dataset.
3. **Two closed routes**, each of which looked promising and each of which is now measured
   rather than assumed: recalibration cannot undo probability compression (§9.1.3), and
   quantile alignment makes the drift worse (§9.2.3).
4. **A quantified, unusable lever.** OLI would give 66 % more observations at the endpoint and
   cannot be admitted until its −0.042 class-specific offset is explained or refitted — for
   which the regression sample (12,940 paired parcel-years) is now on disk.
5. **A caution about unused code.** The Roy harmonisation was the reassuring answer to "what
   about OLI?" for months and is, measured, worse than doing nothing (§9.3.2).

The cheapest next moves are in [`temporal_ood_plan.md`](temporal_ood_plan.md) §6.5. The
conclusion that has now survived four estimands is unchanged: **the project needs endpoint
labels**, and no amount of mitigation substitutes for them.

### 9.5 Refitting the OLI correction on our own data — ⛔ **the route is closed, and now we know why**

Follow-up to §9.3.2, run 2026-08-11 on the direct question *"can a locally-fitted correction
rescue step 3?"* Answer: **no**, and the reason is structural rather than a tuning failure.

#### 9.5.1 The paired sample exists and is the right design

Landsat 7 and 8 fly 8 days apart, but adjacent WRS paths overlap, so a parcel in a sidelap is
imaged by **both sensors on the same day**. Over 2015/2019/2022 the panel yields **27,573
same-day parcel-date pairs across 3,583 parcels** (`allperu oli-refit`,
`allperu/oli_refit.py`) — the same observational design Roy et al. used, on our own imagery.

#### 9.5.2 ⚠️ The sample cannot identify a *slope* — the noise exceeds the signal

| band | corr(L7, OLI) | SD OLI | SD L7 | **SD of the difference** | OLS slope | binned slope | Roy slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| B | 0.116 | 0.077 | 0.077 | **0.102** | 0.116 | 0.263 | 0.847 |
| G | 0.180 | 0.073 | 0.077 | **0.096** | 0.189 | 0.352 | 0.848 |
| R | 0.271 | 0.079 | 0.085 | **0.099** | 0.288 | 0.454 | 0.905 |
| NIR | 0.506 | 0.085 | 0.088 | 0.086 | 0.524 | 0.607 | 0.846 |
| SWIR1 | 0.516 | 0.071 | 0.083 | 0.076 | 0.601 | 0.652 | 0.894 |
| SWIR2 | 0.596 | 0.065 | 0.075 | 0.064 | 0.692 | 0.718 | 0.907 |

**On the same parcel on the same day, the two sensors disagree more than parcels differ from
each other** — the SD of the difference (0.064–0.102) exceeds the SD of either sensor's own
values (0.065–0.088) in the visible bands. A blue-band correlation of 0.116 is not a
calibration offset; it is noise domination, from sidelap geometry (extreme view angle, and L7
SLC-off gaps are widest at scene edges), residual thin cloud, and small parcels whose median
rests on few pixels.

Consequences: OLS is diluted to physically impossible slopes (0.116 in blue against ~0.85
expected). Binning into 40 quantiles and regressing bin *means* removes most of the dilution
and still returns 0.26–0.72 — **still not credible**. So the slope is not identified even
after de-dilution, and only the *mean difference* is estimable (n = 27,573, SE ≈ 0.0006).
The only correction this data supports is **gain 1.0 plus a per-band offset**.

#### 9.5.3 The offset correction is by far the best — and it still fails

Assembled to `<panel>/<year>_olifit/` and put through the identical 3a comparison:

| arm | median \|Δp\| | class agreement | **control mean Δp** | **at-risk mean Δp** |
|---|---:|---:|---:|---:|
| split-half ceiling / target | 0.0309 | **0.6551** | −0.0035 | — |
| OLI + Roy (published) | 0.0311 | 0.6164 | **−0.1073** | −0.0347 |
| OLI, no correction | 0.0293 | 0.6309 | −0.0423 | −0.0126 |
| **OLI + offset refitted here** | **0.0298** | **0.6452** | **+0.0360** | **−0.0003** |

It is the best arm on every axis: agreement reaches **98.5 % of the model's own split-half
ceiling**, and the at-risk pool becomes **essentially unbiased (−0.0003)**. But the control
pool **overshoots from −0.0423 to +0.0360** — it flips sign at nearly the same magnitude, and
3a still fails.

#### 9.5.4 ⭐ Why no global correction can work: the sensor difference is **cover-type dependent**

Mean NDVI error against same-day L7, split by the parcel's PETT label:

| correction | `ANNUAL` | `PASTURE_FALLOW` | `PERENNIAL` | **between-class spread** |
|---|---:|---:|---:|---:|
| none (raw OLI) | +0.0542 | +0.0452 | +0.0704 | **0.0252** |
| Roy (published) | +0.0681 | +0.0620 | +0.0818 | 0.0198 |
| offset (refitted) | −0.0101 | −0.0243 | +0.0006 | **0.0249** |
| slope+intercept (binned) | −0.0194 | −0.0268 | −0.0348 | 0.0154 |

The refitted offset cuts the *average* error roughly fivefold (+0.055 → −0.010). **It does not
touch the spread between classes**, which stays at 0.015–0.025 NDVI in every arm. That is the
whole problem: a global band-level linear map — offset, slope, or both — moves all three
classes **together**. It cannot remove a difference that lies **between** them.

And a difference between them is what the data shows: raw OLI reads +0.070 greener on
perennial parcels and +0.045 on pasture, a 0.025 gap. Physically this is expected — OLI and
ETM+ integrate different parts of the spectrum, and vegetation has a steep red edge, so the
cross-sensor difference depends on what is on the ground. That is exactly why the published
form is a slope *and* intercept — and the slope is the thing this sample cannot identify.

**The only correction that would work is one conditioned on cover type, and cover type is
what the model is trying to predict.** The route is circular, and therefore closed.

#### 9.5.5 What is now settled

* **Step 3 stays closed and the mission policy stands** (`PANEL_MISSIONS = {"L5","L7"}`). The
  66 % observation gain from OLI (21.8 vs 13.1 per parcel-year) remains real and remains
  unusable.
* **The published Roy coefficients must not be used on this data**, and the reason is now
  measured rather than guessed: raw OLI is *greener* than L7 here and Roy makes it greener
  still. `perennial/harmonization.py` gains a warning pointing at this section.
* **`allperu/oli_refit.py` is kept** with its offset coefficients
  (`oli_refit_coefficients.json`) and its paired sample (`oli_refit_pairs.parquet`). It is
  the best available correction and it is still not good enough; anyone tempted to admit OLI
  should read §9.5.4 first.
* **What would actually reopen step 3:** a paired sample clean enough to identify a slope —
  which means larger parcels or pixel-level co-registration rather than parcel medians in
  scene sidelaps — *and* evidence that a slope+intercept map closes the between-class spread.
  The binned fit says it closes about 40 % of it (0.025 → 0.015), so even that is not assured.

### 9.6 A free pilot of the two-period tenure DiD — the drift *does* cancel, but the placebo does not clear

Run at the end of this session on the **existing** panel predictions (new primary arm,
`panel_predictions_nolat_aug_yleak10.parquet`), no new data. It is a **feasibility check, not
an estimate** — see the sample sizes.

**The design.** The two dated tenure observations (§8.7) supply the *treatment*; the
classifier still supplies the *outcome* at both dates. So the drift is still there — the
point is that it is now **common to both groups** and differences out. Restricted to the
at-risk pool (PETT-`ANNUAL`), split by whether the parcel moved NO INSCRITO → REGISTERED by
the ~2011–12 cadastre cut:

| window | never registered | became registered | difference |
|---|---:|---:|---:|
| W99 | 0.0649 | 0.0629 | −0.0020 |
| W04 | 0.0680 | 0.0746 | +0.0066 |
| W09 | 0.0678 | 0.0601 | −0.0077 |
| W14 | 0.0746 | 0.0601 | −0.0145 |
| W19 | 0.0934 | 0.0988 | +0.0054 |
| **slope/decade** | **+0.0127** | **+0.0115** | **−0.0013** |

**Both groups drift at the same rate (+0.0127 vs +0.0115) and their difference is flat
(−0.0013/decade).** That is the design's central claim, demonstrated. Contrast the pool that
broke M2 — the PETT-`PERENNIAL` control, drifting −0.0163/decade against an at-risk pool
moving the other way, because it is a *different kind of parcel* and the compression is
class-specific. Here both groups are annual-declared parcels starting at the same ~6 %
baseline, so they sit in the same part of the classifier's bias.

**The estimator and the power news.** Parcel FE + window FE, SEs clustered by 5 km region,
outcome = window-mean `prob_PERENNIAL` (W-D9):

| contrast | coefficient | se | p |
|---|---:|---:|---:|
| **PLACEBO W99 → W04** (both pre-treatment) | **−0.0169** | 0.0132 | 0.201 |
| W99 → W09 (straddles treatment) | −0.0179 | 0.0094 | 0.057 |
| W99 → W14 | −0.0197 | 0.0103 | 0.057 |
| W99 → W19 | −0.0569 | 0.0201 | 0.005 |
| W99 → W14+W19 (headline shape) | −0.0383 | 0.0141 | 0.006 |

Because the parcel fixed effect removes most of the variance, the standard error is far
better than the design effect alone predicted: **~1,300 treated parcels detect a 2 pp effect
at 80 % power**, and **44,957 at-risk parcels nationally became registered**. The extraction
is therefore *affordable* — roughly panel-sized, not the 12,500-per-group the window plan
feared.

**⚠️ And the placebo does not clear.** W99 → W04 is entirely pre-treatment and should be zero;
it is **−0.0169, which is 44 % of the headline coefficient and points the same way**. It is
not significant (p 0.20) — but with 334 treated parcels the placebo cannot detect a pre-trend
the size of the effect, which is exactly the situation in which a DiD produces a confident
wrong answer. The whole −0.017 → −0.018 → −0.020 → −0.057 sequence is a smooth divergence
that **begins before treatment could have occurred**, and its largest step lands in W19 where
the density artefact is worst.

**Read this as: the mechanism works, the identification is unproven, and the pilot is exactly
powered to mislead.** Do not quote −0.038 as a result. The properly-sized study, its
registered gates (the placebo first), and the sample design are in
[`tenure_did_plan.md`](tenure_did_plan.md).

---

## 10. The two-period tenure DiD (v2) — ⛔ **NOT FUNDABLE. Stopped before extraction, at N3.**

> **⚖️ SUPERSEDED IN PART by [§11](#11-the-tenure-did-reopened--pre-trend-correction-instead-of-a-pre-trend-gate)
> (2026-08-12).** The study was reopened with a corrected gate and **ran to completion**; an
> estimate exists. §10 stays exactly as written — it is the record of the v2 decision, and its
> two substantive findings still stand: the pilot's headline reverses sign under a valid
> pre-period (§10.2), and 6,559 treated parcels is all of Peru holds (§10.4). What did **not**
> stand is the *gate*: §11.7 shows G1 v2 was unpassable at any sample size.

[`tenure_did_plan.md`](tenure_did_plan.md) v2, tasks N1–N5. **N1 and N2 ran in full; N3
returned infeasible; N4 (the only GEE spend) and N5 were therefore NOT run**, which is the
plan's own stop rule. No estimate is produced and the national locked test is still unspent.

Code: `allperu/tenure_did.py` (`restrict`, `window_means`, `did`, `placebo`, `integrity`,
`precision`, `population_ceiling`, `feasibility`), CLI `allperu tenure-did` /
`allperu tenure-ceiling`. Tests: `tests/test_tenure_did.py` (17).

### 10.0 Scoreboard

| gate | registered criterion | measured | verdict |
|---|---|---|---|
| **G1** placebo | whole 95 % CI inside ±0.005/5 y (0.010/decade) | +0.0049, CI [−0.026, +0.036] | **INCONCLUSIVE** |
| **G2** integrity | 4 checks (see §10.3) | 3 of 4 fail | ⛔ **FAIL** |
| **G3a** precision | expected placebo SE ≤ 0.0025 | needs **25,202/arm**; **6,559 exist** | ⛔ **FAIL (2.41×)** |
| G3b | ≥80 % power at 2 pp | se 0.0248 → MDE 6.9 pp | ⛔ FAIL |

### 10.1 N1 — the pilot reproduces exactly, and its "sign disagreement" was a labelling error

Under the pilot's own specification (no R2–R4, control = everyone not treated, no department
time effects) every §9.6 coefficient reproduces to four decimals: **placebo −0.0169
(se 0.0132), headline −0.0383 (se 0.0141)**, n_treated 335. Pinned by a test, so §9.6 stays on
the record (N-D7).

**The §9.6 table and its regression do not actually disagree.** The tabulated series is the
**thresholded share** (the secondary outcome); the coefficients are on the **probability** (the
primary, W-D9). The two were printed side by side unlabelled. On the probability the arm gap
runs **+0.0650 (W99) → +0.0076 (W19)**, i.e. −0.057, which is the regression's −0.0569. The
estimator was never wrong; the table was mislabelled. *Two outcomes in one table need two
labels.*

**What the reconciliation exposes is worse than the discrepancy.** The treated arm starts
**6.5 pp higher** in predicted perennial probability than the control at baseline, and the
entire "effect" is that gap **closing**. A headline built from convergence between two arms
that were already far apart is the textbook signature of regression to the mean — and the
baseline gap is itself T1's differential false-positive rate (§8.5), because the pilot's
control mixed already-`INSCRITO` parcels (FPR 0.111) into a treated arm that is `NO INSCRITO`
by construction (FPR 0.155).

### 10.2 ⭐ N1 — the pilot's headline REVERSES SIGN once the pre-period is made valid

The two observations are snapshots of a rolling titling programme, not two shared dates
(plan §1.1, measured here for the first time): declarations spread 1996–2009, `fech_tran` is a
per-record transaction date (3–112 distinct dates *per department*, La Libertad spanning
1999–2015), **24.6 % of parcels carry a status with no date at all**, and the registration rate
is **non-monotone** in the gap between observations (0.28 at 0–3 y, 0.13 at 6–9 y, 0.31 at
12–15 y) — so the gap marks a departmental campaign wave, not a duration.

So W99 is a valid pre-period only for parcels declared *after* it (R4). Applying that:

| arm | R4 | control | n treated | placebo | G1 | headline | |
|---|---|---|---|---:|---:|---|---:|---|
| pilot repro | no | any | 335 | −0.0169 (0.0132) | INCONCLUSIVE | **−0.0383** | p 0.007 |
| **primary** | yes | NO INSCRITO | 60 | +0.0049 (0.0158) | INCONCLUSIVE | **+0.0338** | p 0.173 |
| sens: control=any | yes | any | 60 | +0.0066 (0.0181) | INCONCLUSIVE | +0.0156 | p 0.447 |
| sens: `nolat` | yes | NO INSCRITO | 60 | +0.0179 (0.0161) | INCONCLUSIVE | +0.0227 | p 0.335 |
| sens: `ltae` | yes | NO INSCRITO | 60 | **−0.0878** (0.0378) | ⛔ **FAIL** | +0.0649 | p 0.109 |

**The pilot's significant −0.038 becomes an insignificant +0.034 — a sign reversal — as soon as
the pre-period is required to precede the parcel's own declaration.** The placebo simultaneously
collapses from −0.0169 to +0.0049. Both facts point the same way: the pilot's headline was
measuring a pre-existing divergence, not a treatment effect. **N-D7 applies — §9.6's numbers
stay on the record, and this is what reverses them.**

⚠️ **And `ltae` fails the placebo outright** (−0.0878, CI excluding zero). Plan §5 registered
that if the sign depends on the architecture it is not a finding. Here even the *pre-treatment*
coefficient depends on it.

### 10.3 N2 — G2 sample integrity: 3 of 4 checks fail, and two had never been run

| check | criterion | pilot spec | primary spec |
|---|---|---:|---:|
| common support (max SMD) | < 0.25 | 0.750 ⛔ | 0.609 ⛔ |
| differential attrition | < 0.02 | 0.002 ✅ | 0.000 ✅ |
| training-set membership gap | < 0.02 | −0.080 ⛔ | −0.062 ⛔ |
| arms share regions | ≥ 0.80 | 0.405 ⛔ | 0.366 ⛔ |

Attrition — the check the plan added because window qualification depends on observation
density — is the **one clean pass**, and it is clean everywhere. The other three are new
failures:

* **Training-set membership** differs by 6–8 pp between arms. The model was trained on some of
  these parcels *labelled* `ANNUAL`, which holds their predicted perennial probability down in
  every window. That is a direct, previously unmeasured contamination of the outcome, and it
  is differential.
* **Arms do not share regions** (0.37–0.41 against a 0.80 criterion) in the *existing* panel,
  so treatment is largely collinear with the clustering unit. This is fixable by design in N3
  — it is a property of a sample drawn for a different purpose — but it means the pilot's
  region-clustered SEs were never doing what they appeared to.
* **Common support** fails mostly on department: treated parcels are 75 % La Libertad +
  Cajamarca (§10.4).

### 10.4 ⛔ N3 — the sample the placebo gate needs does not exist in Peru

This is the decisive result, and it is a fact about the archive rather than about the budget.

**Required**, from measured variance (`precision`, no assumption imported from the pilot):
SD of the within-parcel change 0.1225, intra-region design effect 1.37, placebo horizon 2.5 y
⇒ band ±0.0025 ⇒ **25,202 parcels per arm**.

**Available**, after R1–R4 over all 14 linkable departments (`population_ceiling`):

| step | parcels |
|---|---:|
| R1 at-risk (PETT `ANNUAL`, with tenure) | 363,529 |
| R2 both observations dated | 254,209 |
| R3 cadastre after declaration, before the post-windows | 254,033 |
| **R4 pre-window entirely before declaration** | **80,868** |
| ⇒ **treated** | **6,559** |
| ⇒ control (`NO INSCRITO` at both) | 25,840 |
| ⇒ **effective n** | **5,231** |

**Shortfall 2.41×.** Treated parcels are also concentrated — La Libertad 2,859, Cajamarca
2,054 (75 % between them), Piura 542 — so R5's within-department requirement would bite hard
on top.

**The trade-off is structural and cannot be bought out.** The panel starts in 1999 (Landsat +
El Niño) and declarations concentrate in 1997–2003. A pre-period requires the panel to precede
the declaration, so:

* **cohort ≥ 2004** — one clean pre-window (W99), 6,559 treated, but a pre-trend test needs two
  pre points, so W99 must be split into 2- and 3-year halves, which raises SD(delta)
  0.1204 → 0.1606 and halves the horizon;
* **cohort ≥ 2009** — two clean 5-year pre-windows (W99, W04), but **380 treated nationally**,
  against 11,479/arm required.

Either the pre-period is clean or it is powered. The data does not offer both.

**A recorded judgement call.** The registered band is expressed **per decade**
(`G1_BAND_PER_DECADE = 0.010`, equal to ±0.005 over a 5-year step) so contrasts of different
lengths are held to equal stringency. Applied literally instead, ±0.005 on a 2.5-year contrast
requires only 6,300/arm and *would* be feasible — but it is **half as strict as plan v1
intended**, because a pre-trend's contamination of the headline is (rate × horizon): a
2.5-year contrast of 0.005 implies twice the annual drift of a 5-year contrast of 0.005, and
extrapolated over the 17.5-year headline it permits ±0.035 of contamination — 92 % of the
pilot's own headline, i.e. no bound at all. Loosening the band to reach feasibility would be a
goalpost move in the permissive direction. **Both numbers are recorded in
`did_feasibility_cohort2004.json`; the horizon-normalised one is primary.**

### 10.5 N4 / N5 — NOT RUN

N4 is the only step that spends Earth Engine. G3a fails against the population, so no sample
can meet the gate that decides the study, and the plan's stop rule applies before
`timing_probe` is worth running. **Nothing was extracted, nothing was inferred, no DiD estimate
exists, and the locked test is untouched.** Budget saved: ~15–24 h of GEE at the sizes the plan
contemplated.

### 10.6 What was NOT adopted, and why

* **The pilot's −0.0383 headline — withdrawn as an estimate.** It reverses sign under a valid
  pre-period (§10.2). N-D7 keeps it on the record as the pilot's number, reproduced by a test.
* **The literal ±0.005 band on a 2.5-year placebo — rejected** (§10.4): feasible, but half the
  registered stringency.
* **Relaxing R4 — rejected.** It is the restriction that makes "before" mean before, and
  removing it is precisely what produced the reversed sign.
* **Dropping the at-risk restriction to gain sample — never considered.** It is what makes the
  drift cancel (§8.2/§8.3).
* **`ltae` as a sensitivity arm — it fails the placebo**, so it cannot corroborate anything.
* **Trimming on baseline `p_mean` to fix common support — rejected by design.** Selecting on
  the pre-period outcome induces regression to the mean, which is the artefact §10.1 already
  identifies in the pilot.

### 10.7 What this leaves

**A fifth estimand has failed, and for a new reason.** The previous four failed because the
classifier could not deliver a defensible level or trend. This one does not need it to — the
design is sound and its drift-cancellation premise held (§9.6, and the arms' baseline is
common by construction under R1). **It failed on the treatment side instead: the titling
observations are snapshots of a rolling programme, so the interval in which treatment could
have occurred overlaps the only pre-period the panel has.**

Two things are worth carrying forward:

1. **⭐ A pre-period must be defined per unit, not per calendar.** The pilot's headline was
   significant, sizeable, and reversed sign when this was enforced. Any future design here
   that uses these titling snapshots must apply R4 first and check its surviving n *before*
   anything else.
2. **The feasibility question is answerable before the budget question.** `population_ceiling`
   + `precision` + `feasibility` took minutes and closed the study. That pattern —
   measure the required n from variance, measure the available n from the archive, compare —
   should precede every extraction this project ever funds again.

The route that does not depend on any of this is unchanged and is now the only one left:
[`endpoint_labels_plan.md`](endpoint_labels_plan.md). Every accuracy number in this project is
still measured at the label year, and not one is measured at the endpoint.

---

## 11. The tenure DiD, REOPENED — pre-trend correction instead of a pre-trend gate

[`tenure_did_plan.md`](tenure_did_plan.md) **§9**. §10 stopped the study because G1 demanded
*proof* that the pre-trend was negligible — an equivalence test needing 25,202 parcels per
arm when Peru holds 6,559. **That was the wrong instrument, not the wrong design.** The
standard alternative is to **measure the pre-trend and subtract it**, carrying its
uncertainty into the final interval. This section is that study, run end to end.

**R1–R5 and N-D1…N-D9 are unchanged.** In particular **R4 is applied exactly as in §10** —
a parcel's pre-window must end before that parcel's own declaration. §8.4's prohibition
(*do not reopen by relaxing R4, widening the band, or dropping the at-risk restriction*) is
respected: the gate changed, the design did not.

Code: `allperu/tenure_did.py` (`amplification_factor`, `corrected_effect`,
`sensitivity_curve`, `decision`, `bootstrap_covariance`, `registration` /
`write_registration`, `run_corrected`; `did()` gained an opt-in `extra_post_fe`),
`allperu/did_sample.py` (new), `perennial/panel.py` (`rebuild_year_stores`, `verify_years`;
`timing_probe` takes `out=`). CLI: `allperu did-sample | tenure-register | tenure-did2`,
`perennial panel rebuild | verify`. Tests: `tests/test_tenure_did.py` (31),
`tests/test_coverage_years.py` (+2).

### 11.1 The arithmetic, and what it costs

The placebo spans **2.5 years**, the headline **17.5**, so **M = 7**: a persisting pre-trend
contaminates the headline by seven times the placebo coefficient, and its standard error is
amplified by seven too.

```
corrected = headline − M × placebo
se        = sqrt(se_headline² + M² × se_placebo²)
```

**M is derived from window midpoints in code, never hard-coded** — move a window and M moves
with it. Pinned by a test, along with the correction, the error propagation, and the decision
rule on both sides of its threshold.

The price is registered in advance and is unflattering. From variance measured on the
existing panel (SD of the within-parcel change 0.1225 on the placebo contrast, 0.1411 on the
headline; cluster design effects 1.37 / 1.70), the achieved sample projects to
**se_placebo ≈ 0.0024, se_headline ≈ 0.0031 ⇒ se_corrected ≈ 0.0169**. So **only an effect
larger than ~3.3 pp can survive the correction** (3.0 pp at a hypothetical 2:1 control
ratio). Anything smaller is genuinely not separable from pre-existing drift, and the
registered rule says so rather than reporting it.

### 11.2 What was registered before extraction

`data/processed/all_peru_did/did2_registration.json`, written before N4 and not editable
afterwards (`write_registration` compares and raises). Primary rule: **report only if the
95 % CI of `headline − 7 × placebo` excludes zero**, otherwise **NOT-SEPARABLE**. Primary
outcome: window-mean `prob_PERENNIAL`. Primary contrast: W99 → W14+W19, both post-windows
also reported separately. Placebo: P1 (1999–2000) vs P2 (2001–2003). Sensitivity curve over
**M ∈ {0, 1, 3, 5, 7}**, always reported. G2 is a **diagnostic, not a gate**.

### 11.3 The sample — every treated parcel that exists

`allperu did-sample`, into a new workspace (`CC_PROC=data/processed/all_peru_did`,
`CC_FEAT=…/features`, `CC_RUNS=runs/all_peru`). The national locked test is untouched.

| step | parcels |
|---|---:|
| R1 at-risk (PETT `ANNUAL`, tenure resolved) | 363,529 |
| R2 both observations dated | 254,209 |
| R3 cadastre after declaration, before the post-windows | 254,033 |
| **R4 pre-window entirely before declaration** | **80,868** |
| after the N-D9 control definition | 32,399 |
| ⇒ **drawn: 6,559 treated (a census) + 8,066 control** | **14,625** |

Every one of the 6,559 qualifying treated parcels is taken — there is no larger pool, and
their sampling weight is exactly 1 by construction. Controls are drawn **region-first**,
offering regions that already hold treated parcels of the same `department × declaration-year
cohort` stratum first.

**⭐ That fixes G2's worst failure by design: 0.851 of sampled parcels sit in a region holding
both arms, against 0.366 in the panel §10.3 measured.** The region-clustered standard errors
are now doing what they appear to do; in the old sample treatment was largely collinear with
the clustering unit.

**⚠️ The 2:1 control target is not reachable, and topping up elsewhere would be waste rather
than rescue.** The achieved ratio is **1.23:1**. LA_LIBERTAD 2006 alone holds 2,849 treated
against 2,193 eligible controls in the same department-cohort, and the department is
exhausted across all cohorts. Controls drawn from a stratum with **no** treated parcels are
absorbed by the department × POST fixed effects and contribute nothing to the treatment
coefficient, so the shortfall (5,052 controls) is taken as-is. The cost is arithmetic and
small: the detection threshold moves 3.0 pp → 3.3 pp. Design effect from the weights **1.65**.

### 11.4 Extraction — 15 years, and the completeness check earned its place twice

Five workers on disjoint year ranges, `CC_PROC=data/processed/all_peru_did`, L5+L7 only.

| | |
|---|---|
| Parcels × years | 14,625 × 15 (1999–2003, 2014–2023) |
| Pixel chunks | 5,410 (+ 660 coverage chunks) |
| **Pixel-observations** | **63,634,522** |
| Parcel-years in the pixel stores | **211,567** |
| Wall clock | ~13 h 30 m, 5 concurrent workers |
| Measured rate (`timing_probe`, contended) | 1.32 / 2.79 / 0.83 / 1.43 s per parcel-year for 1999 / 2003 / 2016 / 2022 |

Coverage-gate pass by year: 1999 **99.3** · 2000 **99.3** · 2001 **97.4** · 2002 **93.6** ·
2003 **98.4** · 2014 **95.3** · 2015 **97.4** · 2016 **98.8** · 2017 **98.5** · 2018 **98.6** ·
2019 **96.0** · 2020 **99.1** · 2021 **98.2** · 2022 **98.3** · **2023 86.4 %**.

**2023 is the one thin year** — L7's last — and it sits inside W19. W19 needs only 3 of 5
years so qualification is safe, but it is a second reason to read W19 separately.

**⭐ GEE throttling is now measured, not inferred.** Five workers produced
*"Exceeded Earth Engine concurrency limit. Your project is in Restricted Mode."* Throughput
fell from 11.5 to ~7 chunks/min. The important part: **most of that throttling arrived as
8 silent 900-second hangs, not as errors** (of 11 transient events in total). `_retry`'s wall-clock deadline converted every
one into a retry; zero workers were lost. Without it this run would have stalled permanently
eight times. The §7.2 fix has now paid for itself twice, in two independent runs.

**⛔ And the per-year store hazard fired again, visibly.** `pixels_2021.parquet` was written
five times — at 6,164 → 9,161 → 9,480 → 12,630 → 14,284 parcels — because each finishing
worker re-globs every chunk on disk. Every one of those snapshots is well-formed. This is why
`rebuild_year_stores` + `verify_years` run after **all** workers exit.

#### 11.4.1 ⭐ The verify tolerance was wrong, and the way it was wrong is the lesson

`verify_years` failed 13 of 15 years at its first run — every year at **0.9941–0.9963** of its
gate survivors, against a tolerance of 0.995 I had picked by taste.

The pattern was the diagnosis. A truncated year is an *outlier* (the real 2023 incident was
62 % against ~99 % elsewhere); a uniform 99.4 % across 13 years extracted by 5 independent
workers at different times is a *floor*. Measured directly: the missing parcels are **the same
~85 parcels in every year** — Jaccard **0.95** between 1999 and 2020, only 86 in the union over
six years — and they are **sub-pixel**: median **0.11 ha** and **1.23** estimated pixels,
against 1.00 ha / 11.06 for the sample. A parcel smaller than a Landsat pixel passes the
*coverage* gate, which counts scene observations, and still returns no pixel rows.

So the tolerance is now **0.99, set from that measurement**, and `verify_years` reports a
second statistic that does not depend on it: `deficit_ratio_to_median`, which flags a
*year-specific* deficit regardless of how large the structural floor is. On the final run all
15 years sit at 0.96–1.02× the median deficit. Both behaviours are pinned by tests.

**The generalisable point:** a completeness check whose threshold is guessed will either miss
a real truncation or cry wolf on a complete run — and I did the second. The fix is not a
looser number, it is a statistic that measures *consistency across years* instead of an
absolute ratio.

### 11.5 ⭐ The result

Assembled per year, inferred with the selected model
(`lightgbm_nometa_nolat_aug_yleak10`) — **219,375 parcel-years**. 14,473 of 14,625 parcels
survive abstention; **6,520 treated and 7,953 control**. The placebo was estimated **first**,
and the code path enforces that order.

| arm | placebo (2.5 y) | per decade | headline W99→W14+W19 | W14 | W19 | corrected M=7 | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| **`nolat_aug_yleak10`** ⭐ | **−0.0029** (0.0037) | −0.0118 | **−0.0011** (0.0059, p 0.85) | −0.0022 | −0.0006 | **+0.0195** (0.0268) | **NOT-SEPARABLE** |
| sens: `nolat` | −0.0009 (0.0042) | −0.0035 | −0.0020 (0.0056, p 0.72) | −0.0034 | −0.0010 | +0.0042 (0.0301) | NOT-SEPARABLE |
| neg. control: `ltae` | −0.0026 (0.0101) | −0.0106 | **−0.0154** (0.0080, p 0.055) | **−0.0252** (p 0.006) | −0.0053 | +0.0031 (0.0709) | NOT-SEPARABLE |

**The registered sensitivity curve (primary arm), reported in full as required:**

| M | corrected | se | 95 % CI |
|---:|---:|---:|---|
| 0 (uncorrected headline) | −0.0011 | 0.0059 | [−0.0126, +0.0104] |
| 1 | +0.0018 | 0.0070 | [−0.0118, +0.0155] |
| 3 | +0.0077 | 0.0126 | [−0.0171, +0.0325] |
| 5 | +0.0136 | 0.0196 | [−0.0248, +0.0520] |
| **7 (derived, pessimistic)** | **+0.0195** | **0.0268** | **[−0.0330, +0.0720]** |

**Decision under the registered rule: NOT-SEPARABLE, at every M.**

**But the headline is a *tight* null, and that is the substantive finding.** The uncorrected
estimate is **−0.0011 with a 95 % CI of [−0.0126, +0.0104]** — titling moves the predicted
perennial probability by less than **±1.3 pp**, on the entire national population of parcels
that qualify. The correction does not rescue a large effect from a pre-trend; it widens a null
from ±1.3 pp to ±5.3 pp. Both LightGBM arms agree, and all three arms agree in sign.

**Export-discounted** (§6.4: PERENNIAL is 57.3 % export): the corrected coefficient becomes
**+0.0112, CI [−0.0189, +0.0413]**; the uncorrected becomes −0.0006, CI [−0.0072, +0.0060].
No sentence about export crops may use the undiscounted number.

### 11.6 What the placebo says, substantively

Registered in advance: *interpret the placebo whichever way it lands.* It landed **near zero
and negative**: −0.0029 over 2.5 years, p 0.43, and −0.0009 on the second LightGBM arm.

* **There is no evidence of anticipation.** The alternative causal story — that crop change
  *leads* the paperwork, farmers who intend to invest getting titled — would show as a
  **positive** pre-trend. It is absent. Combined with a baseline gap of only **−0.0028** (the
  pilot's was **+0.065**), the two arms are genuinely comparable before treatment, which is
  what N-D9 and R4 were for.
* **The pilot's placebo was noise.** −0.0169 (se 0.0132) becomes −0.0029 (se 0.0037): the
  point estimate moved 5.8× closer to zero as the standard error fell 3.6×. Exactly what a
  noise term does, and not what a real pre-trend does.
* **⚠️ The correction's *sign* is not stable across outcomes.** On the probability the placebo
  is −0.0029; on the thresholded share it is **+0.0029**. A real anticipation trend would point
  the same way in both. That the two disagree is further evidence the pre-trend is noise around
  zero — and it is also why the corrected point estimate flips sign between the primary
  (+0.0195) and the secondary (−0.0230). **Neither corrected point estimate should be read as
  a direction.**

### 11.7 ⭐ The v2 gate was unpassable at ANY sample size — and would have failed for nothing

This is the sharpest methodological result of the reopening, and it is measurable only now
that the study has been run.

G1 v2 required the whole 95 % CI inside ±0.0025 (the horizon-normalised band on a 2.5-year
contrast). The measured point estimate is **−0.00295 — already outside the band**. The
standard error that would let its CI fit inside is therefore **negative**: no sample, at any
budget, in any country, could have passed it. With enough precision the verdict flips not to
PASS but to **FAIL**.

And what it would have failed on is **−0.0118 per decade** — a pre-trend that cannot change
the conclusion, because the conclusion is a null with a ±1.3 pp interval and stays a null at
every M from 0 to 7.

§10 recorded that the v1 gate *rewarded imprecision*. §11 records the complementary defect:
**the v2 gate punished a point estimate for landing 0.0005 outside an arbitrary band, on a
question where the answer does not depend on it.** A pre-trend gate should ask *"could this
pre-trend overturn my conclusion?"* — which is what the correction and its sensitivity curve
compute directly — not *"is this pre-trend smaller than a number I chose in advance?"*

### 11.8 G2 integrity — measured, not remedied, and 3 of 4 now pass

Reported as a diagnostic per the registration, and the sample was **not** redesigned to make
them pass. Two nevertheless improved by construction:

| check | criterion | §10.3 (old panel) | §11 (this sample) |
|---|---|---:|---:|
| common support (max SMD) | < 0.25 | 0.609 ⛔ | **0.409 ⛔** |
| differential attrition | < 0.02 | 0.002 ✅ | **0.014 ✅** |
| training-set membership gap | < 0.02 | −0.062 ⛔ | **+0.0026 ✅** |
| arms share regions | ≥ 0.80 | 0.366 ⛔ | **0.849 ✅** |

* **Training contamination is gone, and not by luck.** The old sample was drawn for model
  training, so a large and *differential* share of both arms had been fitted on. This sample is
  drawn from the full national table, so only ~5 % of parcels are in the model's training set
  and the arms differ by 0.26 pp. The outcome is no longer anchored differentially.
* **Region sharing was fixed by drawing controls region-first** (§11.3). The region-clustered
  SEs now mean what they appear to.
* **Common support still fails, on department and parcel size**, and this is structural: 75 %
  of qualifying treated parcels are in La Libertad and Cajamarca because that is where the
  titling campaigns ran. Parcel FE absorbs every time-invariant difference, so this is a
  common-support caveat on external validity, not a bias in the within-parcel contrast. **It is
  not remedied and the estimate should be read as applying to the north-coast/highland
  departments where titling actually happened.**

### 11.9 Robustness, and one thing that does depend on the architecture

* **R5 in regression form** (cohort × POST FE on top of department × POST): headline
  −0.0006 (se 0.0060) against the primary's −0.0011. The campaign-wave confound is doing
  nothing once the sample is stratified on it.
* **The bootstrap validates the registered formula.** A 40-replicate cluster bootstrap over
  whole regions gives headline/placebo correlation **0.083** — so assuming independence
  inflates the corrected SE from 0.0261 to 0.0266, i.e. the registered number is **2 %
  conservative**. Bootstrap SEs (0.0053 headline, 0.0038 placebo) match the analytic
  cluster-robust ones (0.0059, 0.0037).
* **⚠️ `ltae`'s headline is 14× the primary's** (−0.0154 vs −0.0011, and −0.0252 on W14 at
  p 0.006). The registered rule is *"if the sign depends on the architecture it is not a
  finding"* — the **sign does not** (all three arms negative), but the **magnitude plainly
  does**. Any future design here that needs a *level* rather than a null must treat the
  architecture as a first-order uncertainty.
* **`ltae` no longer fails the placebo.** §10.2 measured −0.0878 with a CI excluding zero on
  **60** treated parcels; at 6,520 it is −0.0026 (se 0.0101). That failure was small-sample
  noise, and recording it here corrects the earlier reading.
* **The W19 density artefact is large, visible, and common to both arms** — exactly the
  condition the design needs. Weighted mean `prob_PERENNIAL` runs 0.084 → 0.084 → 0.094 →
  **0.180** for controls across W99 → P2 → W14 → W19, and 0.074 → 0.072 → 0.078 → **0.141**
  for treated. Both roughly double into W19 (thin L7, 2023 at 86.4 % coverage). That doubling
  is the artefact that killed M2 when it was measured against a *different kind of parcel*;
  here it lands on both arms and differences out.

### 11.10 What was NOT adopted, and why

* **The 2:1 control ratio — not reached, and not chased.** 1.23:1 achieved. Topping up from
  strata holding no treated parcels would have added parcels absorbed by the department × POST
  fixed effects, contributing nothing to the coefficient (§11.3).
* **A looser `verify` tolerance to make the run pass — rejected in that form.** The tolerance
  moved only after the structural floor was *measured*, and a floor-independent consistency
  statistic was added alongside it (§11.4.1).
* **The measured headline/placebo covariance — computed, reported, and deliberately not used.**
  The registered formula assumes independence; a registered formula that moves after the data
  are seen is not registered. It is conservative by 2 % anyway.
* **Reading a direction from the corrected point estimate — refused.** It is +0.0195 on the
  primary outcome and −0.0230 on the secondary, because the placebo's sign flips between them.
* **Hunting for a variant that clears the threshold — not done.** Three arms were run because
  the registration named them; none was selected on its result.

### 11.11 ⚠️ The caveat that binds every number above

**The last observation of tenure is the cadastre cut (~2011–12); the outcome runs to 2023.**
Peru's titling programme did not stop. An unknown share of the control arm was titled between
2012 and 2023 and this design cannot see it — treated parcels sitting in the control group.
That attenuates any real effect **toward zero**.

Two consequences, and both must travel with the result: a positive finding here would be an
**underestimate**, and this null is **partly attributable to that contamination** and is
therefore **not** evidence that titling has no effect. Bounding it needs a third dated
observation, which this project does not have. Added to `tenure_did_plan.md` §6 as a listed
risk, where it was missing.

### 11.12 Where this leaves it

**A fifth estimand did not fail.** It returned an answer, and the answer is a bounded null:
over 1999→2014-23, becoming registered moves a parcel's predicted perennial probability by
**−0.0011 [−0.0126, +0.0104]** uncorrected, and by **+0.0195 [−0.0330, +0.0720]** under the
most pessimistic pre-trend assumption. Under the registered rule that is **NOT-SEPARABLE**,
which is the outcome the registration anticipated and is published as such.

What the study *does* establish, none of which existed before:

1. **The anticipation trend is measured for the first time** and it is ≈0 (−0.0029 ± 0.0073).
   The reverse-causality story — investment intent driving titling — has no support in the
   pre-period.
2. **The titling effect is bounded**: |effect| < 1.3 pp before correction, < 5.3 pp after.
   Against a ~7 % baseline perennial share, the first of those is a genuinely informative
   bound and the second is not.
3. **The design's premise held.** Both arms start 0.28 pp apart, both roughly double into W19,
   and the difference is flat. Drift common to both arms cancels, as §9.6 claimed and §10 could
   not test at n = 60.
4. **The locked national test is still UNSPENT.**

The honest limit is precision, and it is now a *measured* limit rather than an assumed one:
at the whole national population of 6,559 qualifying treated parcels, the 7× amplification of
a 2.5-year placebo onto a 17.5-year headline costs a factor of 4.5 in the width of the final
interval. **The only ways past it are a longer pre-period (the Landsat archive says no) or a
dated registration *event* rather than two snapshots (§10.7) — not a bigger sample.**

### 11.13 The descriptive cross-sectional companion — computed, and it is not an estimate

The design this project started with, and does not use: compare parcels **already registered
at declaration** (`INSCRITO`) against those not (`NO INSCRITO`), and read the perennial share.
Asked for directly, so it is computed and reported — and *why* it is uninterpretable turns out
to be the most useful part.

`allperu tenure-xsec` (`cross_sectional_contrast`, two tests). ⚠️ **It cannot be run on the
DiD sample**: N-D9 restricts both arms to `NO INSCRITO` at declaration, so already-titled
parcels are absent by construction. It runs on the **existing national panel** (§7.1) —
1,990 at-risk parcels carrying tenure, 1,069 `INSCRITO` / 921 `NO INSCRITO`, weighted.

| window | tenure | n | weighted `prob_PERENNIAL` | weighted thresholded share |
|---|---|---:|---:|---:|
| W99 | `INSCRITO` | 1,063 | **0.0614** | 0.0145 |
| W99 | `NO INSCRITO` | 920 | **0.1043** | 0.0229 |
| W19 | `INSCRITO` | 1,063 | **0.1232** | 0.0539 |
| W19 | `NO INSCRITO` | 920 | **0.1449** | 0.0744 |

**Gap (`INSCRITO` − `NO INSCRITO`): −0.0429 at W99 → −0.0216 at W19, change +0.0212.** On the
thresholded share it moves the *other* way (−0.0085 → −0.0204, change −0.0119). The parcel-FE
version of the change — i.e. a DiD on tenure-*at-declaration* — is **+0.0055 (se 0.0122,
p 0.65)**, a null.

**⭐ Three things this establishes, none of which is a treatment effect.**

**1. The baseline gap IS the classifier's false-positive rate, measured a second time and
independently.** Among at-risk parcels the true perennial share at the label year is ≈0 by
construction, so the W99 gap is pure error. It comes out at **−0.0429**, against T1's
independently measured **−0.044 raw** (§8.5) — from a completely different direction
(window-aggregated panel probabilities vs. held-out CV predictions on a different sample). The
cross-sectional "association" at baseline is a property of the instrument, not of the land.

**2. The sign is department-specific, so the pooled number describes a quantity that does not
exist.** `INSCRITO` reads higher in **5 of 14** departments at W99 and **4 of 14** at W19, and
the gap moves toward `INSCRITO` in **7 of 14** — a coin flip. The spread is enormous: Lima
+0.110, Tumbes +0.076, against Ayacucho −0.186, Moquegua −0.181, Tacna −0.164. Per-department
figures are written to `xsec_tenure_by_dept_*.csv` and the function returns them always.

**3. ⚠️ The Piura figure does not replicate, and the direction reverses.**
`../perennial/RESULTS.md` §7.5 reports a cross-sectional **24.9 % vs 10.9 %** in Piura,
`INSCRITO` **higher**. In this national panel Piura's W99 gap is **−0.081** — `INSCRITO`
**lower**. Different panel, different sample, different model, one department; the two are not
comparable and the older figure must not be carried forward as a national fact. This is the
same lesson LODO taught (§6.2): a quantity measured inside one department is not a quantity.

**What it would take to make this design usable.** Not a bigger sample — the at-risk pool with
tenure is **363,529 parcels**, ~55× the DiD's 6,559. The missing ingredient is an **endpoint
error matrix broken out by tenure group**, which is exactly what
[`endpoint_labels_plan.md`](endpoint_labels_plan.md) would produce. With it the endpoint gap
becomes correctable instead of confounded with measurement. It would still not be causal —
registration at declaration is not random, and it is entangled with declaration year (1998
cohort 0.4 % perennial vs 2000 cohort 33.8 %), with region, and with reverse causality.

**Filed as descriptive. It is not quoted anywhere as an effect, and the artifact carries an
`IS_NOT_CAUSAL` field so it cannot be lifted out of context.**
