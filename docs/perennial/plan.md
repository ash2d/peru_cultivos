# Perennial vs Annual vs Pasture/Fallow — implementation plan

> **Audience:** a coding agent picking this up cold. Read `CLAUDE.md` and
> [`docs/PIPELINE.md`](../PIPELINE.md) first — this plan reuses that pipeline almost
> entirely and only says what is *new* or *changed*. Where this plan and PIPELINE.md
> disagree, PIPELINE.md describes what exists today and this plan describes the target.
>
> Status when written: 2026-08-04. Nothing in this plan has been implemented yet.

---

## 1. What we are building and why

The existing classifier answers *"which of 12 crops is on this parcel in its titling
year?"*. That is a one-shot, single-year question, and it is the wrong shape for the
research question the project actually needs to answer:

> **Over time, has land in these Piura parcels shifted from non-export crops to export
> crops?**

The user's operationalisation: **perennial ≈ export** (mango, lime, coffee, banana,
cacao, avocado — Piura's export basket) and **annual ≈ domestic** (rice, maize, cotton,
beans, wheat). So the deliverable is a **3-class classifier**

| class | meaning | proxy for |
|---|---|---|
| `PERENNIAL` | woody/multi-year crops, canopy persists across the dry season | export crops |
| `ANNUAL` | sown-and-harvested crops with a bare-soil phase each cycle | domestic crops |
| `PASTURE_FALLOW` | grazing land, resting land, prepared-but-unsown ground | neither |

…run as **annual inference over every parcel for every year**, producing a per-parcel
class trajectory. The headline outputs are (a) the perennial/annual/pasture **area share
per year**, and (b) the set of parcels that **transitioned** annual→perennial or back.

Two classifiers are required, on purpose:

1. **A rule-based classifier** — explicit, auditable phenology thresholds. It is the
   scientific control: if a transparent NDVI-shape rule does nearly as well as a neural
   network, the paper should say so, and the rule is far easier to defend when applied to
   years with no ground truth.
2. **The three existing ML architectures** — LightGBM, LTAE, PSE-LTAE — retrained on the
   3-class label space, unchanged in architecture.

Validation is against **PETT labels** (internal, the same spatially-blocked protocol as
the 12-class work) and against **MapBiomas Peru** (external, independent-ish benchmark
covering 1985–2024 — the only comparison available for years where PETT has nothing).

### 1.1 Why this is much more tractable than the 12-class problem

* The 12-class tuned state of the art is macro-F1 **0.427** — genuinely hard. Three
  well-separated classes with strong phenological contrast should do substantially
  better. **If the 3-class macro-F1 is not comfortably above 0.65, something is wrong**
  — treat that as a debugging signal, not a result.
* The 3-class map absorbs *every* crop, so the rare-class drops and the unmerged-intercrop
  drops that cost the 12-class table ~16k parcels mostly go away (§3.2). Expect **~60k
  parcels vs 49,648**.
* **Phase 1–4 need zero new GEE work.** The full pixel store already on disk
  (`data/processed/features/`, 4.53M pixel-obs, 322 MB) is label-year data for 47,851
  parcels and is exactly what the 3-class model trains on. Only Phase 5 (the multi-year
  panel) costs GEE time.

### 1.2 Success criteria (state these in the final report, pass or fail)

| # | criterion | threshold |
|---|---|---|
| S1 | 3-class model beats the majority-class baseline on spatial CV | macro-F1 > baseline by ≥ 0.15 |
| S2 | Best ML model beats the rule-based classifier | any positive margin; report it honestly if not |
| S3 | Model beats MapBiomas Peru when both are scored against PETT labels | see §8.4 go/no-go |
| S4 | Temporal transfer degrades gracefully | accuracy at label-year ±3 within 0.10 of label-year accuracy (§7.3) |
| S5 | Trajectories are not noise | flicker rate (§9.2) under 15% for parcels labelled PERENNIAL |
| S6 | Aggregate trend agrees in *direction* with MapBiomas | sign of the perennial-share trend matches |

**S3 is the one that matters most.** If off-the-shelf MapBiomas predicts PETT labels
better than our model does, the honest conclusion is to use MapBiomas and drop the
custom model. Do not bury that outcome.

---

## 2. Design decisions

Decisions marked **[LOCKED]** are settled — implement as written. Decisions marked
**[FLAG]** need a judgement call: implement the recommendation, make it a config key, and
surface the choice explicitly in the report.

### D1 — Class definition **[LOCKED with one FLAG]**
Three classes as in §1. Built by mapping every normalised crop token (from
`crop_normalization.py`) to a group via an explicit lexicon in config (§3.1).

**[FLAG] Sugarcane (`CAÑA DE AZUCAR`, 938 records).** It is a multi-year ratoon crop
(spectrally perennial: no annual bare-soil phase) but is not an export perennial in the
sense the user means, and MapBiomas classes sugarcane under *temporary* crop.
*Recommendation:* assign it `ANNUAL` so our class definition matches MapBiomas's, and run
a sensitivity check (§9.5) reporting how the headline trend moves if it is `PERENNIAL`
instead. Config key `caña_policy: annual | perennial`.

**[FLAG] Woody non-crop tokens** — `ALGARROBO`, `FAIQUE`, `HUALTACO`, `ROBLE`,
`EUCALIPTO`, `COBERTURA ARBOREA`, `REFORESTACION` (~350 records total). Spectrally these
look exactly like perennial orchards but they are *not* export crops, so folding them into
`PERENNIAL` would corrupt the interpretation.
*Recommendation:* **exclude from training** (new exclusion reason `woody_noncrop`), and
document that at inference time such parcels will be predicted `PERENNIAL` and are a known
false-positive source for the export-crop reading. They are too few to train a 4th class.

### D2 — Pasture and fallow merged **[LOCKED, with a diagnostic]**
The user asked for one `PASTURE_FALLOW` class. Implement that as primary. **Also build a
4-class variant** (`PERENNIAL` / `ANNUAL` / `PASTURE` / `FALLOW`) behind a config flag and
train the leading model on it once — the 12-class results show PASTURE (F1 ~0.48) and
FALLOW (~0.50) are separately learnable and they are spectrally *opposite* (green vs
bare), so merging them creates a deliberately heterogeneous class. Report whether the
4-class variant, collapsed post-hoc to 3, beats the directly-trained 3-class model. Costs
one extra training run; worth it.

`land_prep` tokens (`MECANIZADO`, `GRADEO`, `ARADO`, `NIVELACION`, ~1,838 records) are
currently dropped by `labels.py`. **Change:** map them to `PASTURE_FALLOW` — prepared bare
ground is spectrally fallow and the user's class definition is land-state, not crop
identity. Config key `land_prep_policy: pasture_fallow | drop`, default `pasture_fallow`.

### D3 — Multi-crop parcels resolved at group level **[LOCKED]**
The 12-class pipeline needs an explicit 24-entry `merge` map and still drops 4,577 parcels
as `multicrop_unmerged`. At the group level most of that disappears:

* all crops in the set map to the **same** group → that group (e.g. `CAFE+PLATANO` → both
  `PERENNIAL`);
* mixed groups → **`PERENNIAL` wins over `ANNUAL` wins over `PASTURE_FALLOW`**. Rationale:
  a tree canopy dominates a 30 m pixel's reflectance regardless of what is intercropped
  beneath it, and this matches the existing map's "orchard+annual → the perennial"
  decision. Encode as an ordered priority list in config (`group_priority`), not as
  hard-coded logic.

This means the 12-class `merge` map is **not used** by the 3-class build. Do not try to
reuse it.

### D4 — Rule-based classifier is a registered model **[LOCKED]**
Implement the rule model as a fourth entry in the existing model registry
(`@register("rules")`, `input_kind="flat"`), satisfying the same `CropModel` protocol
(`fit` / `predict_proba` / `save` / `load`). Then `train.py`, `evaluate.py` and `infer.py`
work on it **unchanged**, it goes through the identical spatial-CV protocol, and it is
directly comparable to the other three. `fit()` calibrates its thresholds on the training
fold only — never on validation data. It imports only numpy/pandas, so it is safe to add
to `_LAZY_MODULES` (no libomp clash).

### D5 — Separate workspace, shared feature store **[LOCKED]**
The 3-class work needs its own `modeling_parcels.parquet` / `label_map.json` / splits, but
the **pixel feature store is identical** (same parcels, same years, same pixels — only the
label column differs). Do not duplicate 322 MB of features and do not overwrite the
12-class tables (they are a completed result with an unspent test set).

Solution: a small `paths.py` refactor (§3.0) giving an env-var-switchable *tables*
directory while the *features* directory stays shared and fixed.

### D6 — Panel scope is calibrated, not guessed **[FLAG]**
Full annual inference over all ~60k parcels × 35 years = 2.1M parcel-years ≈ **110 hours**
of GEE at the measured rate (§10). That is not a sensible first run. Take a **stratified
sample panel** and choose its size from a measured rate (§7.1). *Recommendation:* 12,000
parcels × 1990–2024 (35 years) = 420k parcel-years ≈ 22 h, run in the background,
resumable. A sample is statistically fine for area shares — expand to population with area
weights and report confidence intervals.

### D7 — Sensor harmonisation is mandatory for OLI years **[LOCKED]**
The audit (`docs/LANDSAT_MISSION_AUDIT.md`, memory: *Landsat mission-mixing audit*) found
L5/L7 mixing radiometrically fine. **That finding does not extend to L8/L9 (OLI).** OLI has
different band centres and response functions from TM/ETM+; training on 1998 TM data and
predicting 2020 OLI data without correction injects a spurious trend directly into the
headline result. Apply published OLI→ETM+ transformation coefficients (Roy et al. 2016,
*RSE* 185:57–70) to the OLI bands at assembly time, and verify empirically (§7.4). Also
**add L9** (`LANDSAT/LC09/C02/T1_L2`, 2021+) to `missions_for_year` — without it, 2022–2024
run on L8+L7 alone.

### D8 — Probability-weighted, bias-corrected area estimates **[LOCKED]**
The headline number is an *area share per year*. A naive count of argmax predictions is
biased by classifier error and flickers year to year. Produce three estimates and show all
of them: (a) naive argmax area, (b) probability-weighted area (`Σ P(class) × area_ha`),
(c) **Olofsson et al. (2014) bias-corrected area with 95% CIs**, using the locked-test
confusion matrix as the error matrix. (c) is the defensible one for a paper.

This requires calibrated probabilities → **temperature scaling** on a held-out fold
(already an open item in PIPELINE.md §7). Implement it here.

---

## 3. Phase 1 — 3-class label build (no GEE, ~half a day)

### 3.0 Prerequisite: the `paths.py` refactor

Create `src/crop_classifier/paths.py`:

```python
"""Path resolution. Tables are workspace-switchable; the feature store is shared."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC_SHARED = ROOT / "data" / "processed"      # inputs + the 12-class workspace
FEAT = PROC_SHARED / "features"                # SHARED pixel store — never switched

def proc() -> Path:
    """Workspace for label/split tables. Override with CC_PROC."""
    p = Path(os.environ.get("CC_PROC", PROC_SHARED))
    p.mkdir(parents=True, exist_ok=True)
    return p

def runs() -> Path:
    p = Path(os.environ.get("CC_RUNS", ROOT / "runs"))
    p.mkdir(parents=True, exist_ok=True)
    return p
```

Then update the modules that hard-code `PROC`/`RUNS` to call `proc()` / `runs()` **at call
time, not import time** (an import-time constant defeats the env var):

| file | change |
|---|---|
| `labels.py` | `F_OUT`/`F_MAP`/`F_EXCL` resolved inside `build()` and `apply_coverage_gate()` from `proc()`. **`F_POLY` keeps pointing at `PROC_SHARED`** — the raw label input is shared. |
| `splits.py` | `F_PARCELS`/`F_META`/`F_BLOCKS` resolved inside `assign()` from `proc()`. |
| `data.py` | `load_parcels()` reads `proc()/"modeling_parcels.parquet"`; `F_LGBM`/`F_PERDATE`/`F_PIXELSET` come from `FEAT` (unchanged, shared). |
| `train.py` | `PROC` → `proc()`, `RUNS` → `runs()`, inside `train()`/`sweep()`. |
| `infer.py` | already takes an explicit `run_dir`; only the `load_parcels()` call is affected (automatic). |
| `features/landsat_gee.py` | `F_PARCELS` → `proc()/"modeling_parcels.parquet"` inside `run_coverage()`/`run_pixels()`. |
| `features/assemble.py` | parcels table from `proc()`; outputs stay in `FEAT` by default (see §7.2 for the per-year override). |

Verification: `uv run pytest tests/` must stay green, and re-running
`uv run python -m crop_classifier.cli labels build` with no env var must reproduce the
existing 12-class table byte-for-byte in content (49,648 parcels, 12 classes). **Do that
check before going further** — this refactor touches the code path that produced every
existing result.

Everything after this point runs with:

```bash
export CC_PROC=data/processed/perennial
export CC_RUNS=runs/perennial
```

### 3.1 The class lexicon

New config `src/crop_classifier/config/perennial.yaml`. Structure:

```yaml
classes: [PERENNIAL, ANNUAL, PASTURE_FALLOW]
group_priority: [PERENNIAL, ANNUAL, PASTURE_FALLOW]   # D3 tie-break, first wins

caña_policy: annual          # D1 flag
land_prep_policy: pasture_fallow   # D2
woody_noncrop_policy: exclude      # D1 flag

# category-level defaults from crop_normalization (applied when no explicit crop entry)
category_default:
  pasture: PASTURE_FALLOW
  fallow: PASTURE_FALLOW
  land_prep: PASTURE_FALLOW
  unspecified: null            # null = exclude the parcel

perennial: [MANGO, LIMON, NARANJA, CAFE, CACAO, PLATANO, COCO, TAMARINDO, PALTA,
            CIRUELA, GUABO, MARACUYA, PAPAYA, GRANADILLA, MAMEY, GUANABANA, TUNA,
            FRUTALES, LUCUMO, MANDARINA, UVA, OLIVO, PACAE, ...]
annual:    [ARROZ, MAIZ, ALGODON, TRIGO, FRIJOL, "FRIJOL DE PALO", "FRIJOL CAUPI",
            "FRIJOL CASTILLA", ZARANDAJA, ARVEJA, PAPA, CAMOTE, YUCA, GIRASOL, CEBADA,
            HABA, SOYA, TOMATE, OLLUCO, OCA, AJO, CEBOLLA, ZANAHORIA, MARIGOLD, CHOCLO,
            MENESTRAS, REPOLLO, HORTALIZAS, "MAIZ AMARILLO DURO", "CAÑA DE AZUCAR", ...]
woody_noncrop: [ALGARROBO, FAIQUE, HUALTACO, ROBLE, EUCALIPTO, "COBERTURA ARBOREA",
                REFORESTACION, HIGUERON, ...]
min_group_parcels: 0     # no rare-class filtering: every token lands in a group
```

**How to build the lists — do not hand-wave this.** Run:

```bash
uv run python -c "
import pandas as pd
d = pd.read_parquet('data/processed/training_crop_records.parquet')
print(d.groupby(['crop','category']).size().sort_values(ascending=False).to_string())"
```

There are ~1,647 canonical tokens; the top 88 (≥15 records) cover 79,164 of 80,618
records — **assign those 88 by hand**, they are the ones that matter. The long tail is
mostly single-occurrence OCR noise. For the tail, apply `category_default` (a `crop`-category
token with no explicit assignment → `ANNUAL`, since annuals dominate the tail) and **write
every unassigned token plus its count to `unassigned_tokens.csv`** so the coverage of the
hand map is auditable. Add an assertion: unassigned tokens must account for **< 2% of
records**; fail the build loudly if not.

Known counts to sanity-check against (from `training_crop_records.parquet`, 80,618 rows):
ARROZ 26,422 · DESCANSO 9,867 · MAIZ 9,009 · PLATANO 5,087 · CAFE 4,395 · ALGODON 4,049 ·
MANGO 2,459 · LIMON 1,889 · TRIGO 1,736 · PASTO ELEFANTE 1,515 · FRIJOL 1,472 · PASTO
1,327 · NARANJA 1,183 · CAÑA DE AZUCAR 938.

### 3.2 `perennial/labels3.py`

New module `src/crop_classifier/perennial/labels3.py`, modelled on `labels.py` but
simpler (no merge map, no rare-class policy):

```python
def assign_group(crops: list[str], cats: list[str], cfg) -> tuple[str | None, str]:
    """Return (class, reason). None = exclude."""
```

Logic, in order:
1. map each `(crop, category)` token → a group via the explicit lexicon, else via
   `category_default`, else `None`;
2. if any token is `woody_noncrop` and policy is `exclude` → exclude, reason
   `woody_noncrop`;
3. drop `None` tokens; if nothing remains → exclude, reason `unmappable`;
4. take the highest-priority remaining group per `group_priority`; reason `single` or
   `mixed_priority`.

Then apply the **same hard gates as `labels.py`**: area ∈ [0.09, 50] ha, year present and
in [1990, 2020]. Reuse `labels.apply_coverage_gate` verbatim (it is label-space agnostic)
— import it, do not copy it.

Writes to `proc()`: `modeling_parcels.parquet`, `label_map.json`,
`label_exclusions.csv`, plus the new `unassigned_tokens.csv` and
`class_lexicon_resolved.csv` (every token → its assigned group → record count: the audit
trail, mirroring `crop_normalization_map.csv`).

**Expected output: ~58–62k parcels, 3 classes.** Print the class balance. Anticipate
roughly ARROZ+MAIZ+ALGODON+TRIGO+beans dominating `ANNUAL` (~50–55%), `PASTURE_FALLOW`
~20%, `PERENNIAL` ~20–25%. If `PERENNIAL` comes out under 10%, the lexicon has a bug.

CLI: add `perennial labels` to `cli.py` (new `perennial_app` typer sub-app).

### 3.3 Splits

```bash
CC_PROC=data/processed/perennial uv run python -m crop_classifier.cli splits assign
```

No code change beyond §3.0. Note explicitly in the report: **this produces a different
locked test set from the 12-class work** (different parcel set → different region
assignment). That is fine and not contamination — the 12-class test set was never spent,
and the 3-class question is a different question. Re-check the autocorrelation audit
output: with 3 broad classes, spatial agreement will be *higher* than the 12-class 0.75/0.37
figures (neighbouring parcels are even more likely to share a coarse class than an exact
crop), so the leakage warning is *more* important here, not less. Consider raising
`buffer_m` for this task and report CV at both 1500 m and 3000 m — cheap, and it bounds
how optimistic the numbers are.

### 3.4 Coverage gate

The full extraction already covers every parcel that has a label year in the store. Run:

```bash
CC_PROC=data/processed/perennial uv run python -c "
import pandas as pd
from crop_classifier.labels import apply_coverage_gate
from crop_classifier.paths import FEAT
apply_coverage_gate(pd.read_parquet(FEAT/'coverage.parquet'))"
```

Parcels newly admitted by the 3-class map (rare crops, unmerged intercrops) that were
never extracted will show `quality_ok = NA`. **Count them.** If it is a few thousand,
extract them (§7 machinery, label-year only) before training — otherwise the 3-class model
silently trains on a subset and the "more parcels" advantage evaporates. Budget: at the
measured 0.19 s/parcel-year, 10k new parcels ≈ 32 min. Do it.

---

## 4. Phase 2 — the rule-based classifier (~half a day)

New module `src/crop_classifier/perennial/rules.py`.

### 4.1 The discriminating physics

Over one year of Landsat, for a Piura parcel:

* **Perennial** — canopy present all year. High NDVI **minimum** (the single strongest
  signal: an orchard never goes bare), moderate mean, **low seasonal amplitude**, low BSI
  maximum.
* **Annual** — one or two green peaks separated by bare soil. **Low NDVI minimum, high
  amplitude**, high BSI maximum, strong order-1 harmonic component relative to the mean.
* **Pasture/fallow** — low-to-moderate NDVI throughout, low amplitude, but a *low* mean
  (this is what separates it from perennial, which has low amplitude and a *high* mean).

So the two-dimensional core is **(NDVI level, NDVI amplitude)**, and the natural rule is a
depth-2 decision tree:

```
if NDVI_p25 >= t_hi  and  NDVI_amp <= t_amp:   PERENNIAL
elif NDVI_amp > t_amp or NDVI_max >= t_peak:   ANNUAL
else:                                          PASTURE_FALLOW
```

Use **NDVI p25 rather than min** — the minimum is one cloud-edge pixel away from garbage.
All of these features already exist in `features_lightgbm.parquet`: `NDVI_p25`,
`NDVI_median`, `NDVI_max`, `NDVI_amp`, `NDVI_std`, `NDVI_h_cos`/`NDVI_h_sin` (harmonic
amplitude = `hypot(h_cos, h_sin)`, and the ratio `hypot(h_cos,h_sin)/h_mean` is a clean
scale-free seasonality index), `BSI_max`, `NDWI_median`. **Add no new feature extraction.**

### 4.2 Implementation

```python
@register("rules")
class RuleModel:
    input_kind = "flat"
    def __init__(self, features=("NDVI_p25","NDVI_amp","NDVI_max"),
                 grid_steps=25, seed=42, **kw): ...
    def set_n_classes(self, n): ...
    def fit(self, train_ds, val_ds, class_weight, run_dir):
        """Grid-search thresholds on TRAIN ONLY, maximising class-weighted macro-F1.
        Writes thresholds.json + a decision-boundary PNG into run_dir."""
    def predict_proba(self, ds) -> np.ndarray:
        """Soft scores, not one-hot — evaluate.py's reliability curve needs a spread.
        Use a logistic squash of the margin to each threshold."""
```

Design notes the implementer must respect:

* **Thresholds are fitted, not guessed.** A 3-parameter grid search over quantiles of the
  training distribution (25 steps each ≈ 15.6k combinations, sub-second on 30k rows with
  vectorised numpy) maximising class-weighted macro-F1 on the *training* fold. Report the
  fitted thresholds per fold — **stability of thresholds across the 5 spatial folds is
  itself a result** (stable ⇒ the rule generalises; wildly varying ⇒ it does not).
* `predict_proba` must return calibratable soft scores. Do not return hard one-hot: it
  makes the reliability curve and the probability-weighted area estimate meaningless.
* Register in `models/base.py`: `_LAZY_MODULES["rules"] = "crop_classifier.perennial.rules"`
  and `INPUT_KIND["rules"] = "flat"`.
* Rows with `NaN` in the rule features (fewer than 4 observations → no harmonic fit) must
  fall through to a documented default (the training-majority class) and be **counted** —
  the rule model cannot abstain via NaN silently.

### 4.3 Run it

```bash
CC_PROC=data/processed/perennial CC_RUNS=runs/perennial \
  uv run python -m crop_classifier.cli train --model rules --run-name rules_3c
CC_RUNS=runs/perennial uv run python -m crop_classifier.cli eval runs/perennial/rules_3c \
  --preds preds_cv.parquet
```

(Pool `fold*/preds_val.parquet` into `preds_cv.parquet` first — same trick as PIPELINE.md
§5.2; write it once as a helper `perennial/pool_cv.py` since it is needed for every model
here, and stop copy-pasting it.)

---

## 5. Phase 3 — the three ML models on 3 classes (~1 day, mostly waiting)

No architecture changes. With `CC_PROC`/`CC_RUNS` set:

```bash
uv run python -m crop_classifier.cli train --model lightgbm --run-name lightgbm_3c
uv run python -m crop_classifier.cli train --model ltae     --run-name ltae_3c
uv run python -m crop_classifier.cli train --model psetae   --run-name psetae_3c
```

Then Optuna sweeps (3-fold objective, 30 trials each, as before) and 5-fold refits at the
best parameters, named `*_3c_tuned`. Reuse the existing search spaces in `train.py::sweep`.

Expectations and what they mean:

* **Class weighting matters less here** than at 12 classes (no 300-parcel classes), but
  keep `class_weights` as-is for comparability.
* The gap that made LTAE win at 12 classes was rare-class rescue. With 3 balanced-ish
  classes **LightGBM may well win**, and it is 100× cheaper to run 420k inferences with.
  Take that seriously in model selection (§6) — do not default to the fanciest model.
* **Remember the macOS libomp gotcha**: never import torch and lightgbm in one process.
  The lazy registry handles it; keep the rule model torch-free.

Also train the **4-class diagnostic variant** (D2) with the leading architecture only,
collapse `PASTURE`+`FALLOW` post-hoc, and compare its 3-class macro-F1 against the
directly-trained model.

---

## 6. Phase 4 — validation, calibration, model selection (~half a day)

1. **Pooled spatial-CV comparison** across all four models: macro-F1, balanced accuracy,
   per-class F1, kappa, majority baseline. Produce the same comparison figures as
   `runs/model_comparison_12c/`.
2. **Temperature scaling** (D8): fit a single scalar `T` on held-out fold-0 validation
   logits/probabilities minimising NLL; store `T` in the run dir; apply in `infer.py`.
   Verify with the reliability curve + Brier score that `evaluate.full_report` already
   produces. Do this for the selected model at minimum.
3. **Select one model** on pooled CV macro-F1, with a documented tie-break toward the
   cheaper model when the margin is under ~0.02 (a 420k-parcel-year inference run makes
   inference cost a real criterion).
4. **Spend the locked test set exactly once**, on the selected model:
   `train --model <sel> --eval-test --run-name <sel>_3c_final`. Everything after this is
   reported, not tuned. The resulting confusion matrix is the error matrix for the
   Olofsson correction (D8) — save it explicitly as `test_confusion.csv`.

---

## 7. Phase 5 — the multi-year panel (GEE, the expensive phase)

This is the phase that turns a classifier into the actual deliverable. It is also where
most of the new bugs will be, because the existing extraction assumes **one year per
parcel**.

### 7.0 Three landmines in the existing code — fix before running anything

1. **`assemble.per_date_medians` groups on `(COD_PREDIO, doy)` with no year.**
   `load_pixels()` globs *all* `pixels_*.parquet` and concatenates. Today each parcel
   appears in exactly one year file so this is safe. Once a parcel has 35 years of pixels,
   this silently **merges observations from different years into one "date"** — producing
   plausible-looking, completely wrong features. Fix: give `assemble()` a `years` filter
   and per-year output directory (§7.2). Add a regression test with a synthetic two-year
   parcel asserting the year-1 tensor is unaffected by year-2 rows.
2. **`run_coverage` does `.drop_duplicates("COD_PREDIO")`** and writes one
   `coverage.parquet` — it will collapse a multi-year panel to one row per parcel. Fix:
   call it **once per year** with an explicit `out=FEAT/"panel"/f"coverage_{Y}.parquet"`
   (the signature already supports `parcels=`, `out=`, `years=`; no change needed, just
   call it correctly).
3. **`run_pixels` hard-codes `FEAT`** for both its chunk dir and its per-year output, so a
   panel run would write into the training store and `pixels_1998.parquet` would grow to
   include panel parcels. That is *arguably* fine (the combine dedupes on
   `(COD_PREDIO, doy, lon, lat, mission)`) but it mixes an audited training artifact with
   an experimental one. **Add a `feat_dir: Path = FEAT` parameter** to `run_pixels` and
   point the panel at `FEAT/"panel"`. Two-line change; keeps the training store immutable.

Also apply **D7** in `features/landsat_gee.py`: add L9 to `missions_for_year`
(`if year >= 2021: m["L9"] = (L9, _OLI_BANDS)` with
`L9 = "LANDSAT/LC09/C02/T1_L2"`), and note that OLI harmonisation happens downstream at
assembly (§7.4), not here — the raw store stays raw.

### 7.1 Calibrate the cost before committing

Measured baseline from the full run: 4.53M pixel-obs / 47,851 parcel-years in ~2.5 h ⇒
**~0.19 s per parcel-year**, ~322 MB of parquet. Landsat availability grows over time
(more missions after 2013 ⇒ more observations per parcel-year ⇒ slower and bigger), so
**do not extrapolate from 1998**.

Run a timing probe first: 500 parcels × {1995, 2005, 2015, 2023}, measure s/parcel-year
and MB/parcel-year for each era, then solve for the panel size that fits a stated budget.
Write the numbers into `docs/perennial/panel_budget.md` before launching. Expect the
2013+ years to cost 1.5–2× the 1998 rate and the panel to land near **2–4 GB**.

### 7.2 Panel definition

`src/crop_classifier/perennial/panel.py`:

* `build_panel(n=12000, seed=42)` — stratified sample of the 3-class parcel table by
  (`label` × `region_id`), **plus all locked-test parcels forced in** (needed for §7.3),
  written to `proc()/"panel_parcels.parquet"` with the sampling weights. Weights are
  essential: area shares must be expanded to the population, not reported for the sample.
* `extract_panel(years=range(1990, 2025))` — for each year Y: copy the panel table with
  `year = Y`, call `run_coverage(parcels=…, out=FEAT/"panel"/f"coverage_{Y}.parquet",
  years=[Y])`, apply the gate **per year** (do not write into the label-year
  `modeling_parcels.parquet` — keep a `panel_coverage_{Y}` table), then
  `run_pixels(parcels=survivors, years=[Y], feat_dir=FEAT/"panel", only_quality_ok=False)`.
  Fully resumable via the existing content-addressed chunk names. Log per-year gate pass
  rates.
* `assemble_panel(years=…)` — per year, call the modified `assemble(years=[Y],
  feat_dir=FEAT/"panel", out_dir=FEAT/"panel"/str(Y))` producing
  `features_lightgbm.parquet` / `tensor_perdate.npz` / `tensor_pixelset.npz` **per year**.
* `infer_panel(run_dir, years=…)` — loop `infer.infer()` per year against the per-year
  feature dir, concatenate to `proc()/"panel_predictions.parquet"` with columns
  `COD_PREDIO, year, pred_label, pred_proba, prob_*, abstained, abstain_reason,
  n_valid_obs, max_gap, sample_weight`. `infer.py` will need a `feat_dir` argument threaded
  through `make_dataset` — smallest clean change is an optional `feat_dir` on
  `data.make_dataset`/`make_flat`/`SeqDataset` defaulting to the shared `FEAT`.

Run `extract_panel` **in the background** (`run_in_background: true`) — it is a 20+ hour
job. Check in on it with the per-year progress log; do not poll every minute.

### 7.3 Temporal transfer validation — the key internal check

We have labels only for ~1996–2007 (bulk 1998–99) but will predict 2024. Before trusting
any trend, measure how fast accuracy decays with temporal distance:

For every locked-test parcel with label year `y0`, the panel gives predictions at
`y0 + k` for `k ∈ [-5, +5]`. A parcel labelled `PERENNIAL` in 1998 was almost certainly
perennial in 1997 and 2000 (orchards do not appear or vanish annually), and a parcel
labelled `ANNUAL` was very likely still annual. So **accuracy against the y0 label as a
function of k is a genuine measure of temporal transfer** — with the caveat, which must be
stated, that real land-use change also contributes to the decay, so this is a *lower
bound* on model stability.

Plot accuracy and macro-F1 vs `k`. **Criterion S4:** within 0.10 of the `k=0` value at
`|k| ≤ 3`. A cliff at a specific year (especially 2012, or the L8 boundary at 2013) is a
sensor artefact, not land-use change — investigate before proceeding.

### 7.4 Sensor-shift diagnostics

Three checks, all cheap, all necessary before believing a trend:

1. **Feature drift** — plot the panel-wide distribution of `NDVI_median`, `NDVI_amp`,
   `BSI_max` per year, 1990–2024. Step changes at 1999 (L7 arrives), 2012 (L7 SLC-off
   only) and 2013 (L8) are sensor artefacts. Apply the Roy et al. (2016) OLI→ETM+
   coefficients in `assemble` for missions 8/9 and re-plot: the steps should shrink
   materially. Keep the coefficients in `perennial/harmonization.py` with the citation in
   a docstring.
2. **Prediction drift on stable parcels** — for parcels predicted `PERENNIAL` with high
   confidence in *every* year (definitionally stable), the class share should be flat.
   Any year where it dips is a data-quality year, not a land-use year.
3. **2012 is the known weak year**: L5 ended Nov 2011, L8 started Apr 2013, so 2012 is
   L7-SLC-off only — expect high abstention and striping. Flag it in every figure rather
   than quietly interpolating over it.

---

## 8. Phase 6 — MapBiomas Peru benchmark

### 8.1 Get the asset right

Candidate asset (verify before use — collection numbering changes):

```
projects/mapbiomas-public/assets/peru/collection3/mapbiomas_peru_collection3_integration_v1
```

MapBiomas Peru Collection 3 covers **1985–2024** — the full period we need. Bands are
named `classification_<year>`. Verify by listing band names in GEE and printing a
`frequencyHistogram` of class codes over the Piura AOI; **if the asset id 404s, resolve the
current one** from the MapBiomas Peru toolkit
(<https://peru.mapbiomas.org/en/google-earth-engine/>) or the user-toolkit repo
(<https://github.com/mapbiomas/user-toolkit>) rather than guessing collection numbers.

Note: Piura is coastal/Andean, **not** Amazon — use the MapBiomas *Peru* national product,
not the RAISG Pan-Amazonia collection.

### 8.2 Class mapping

MapBiomas uses the shared MapBiomas legend. The codes that matter (**verify against the
official legend CSV linked from the Peru site — do not trust this table blind**):

| MapBiomas code | name | → our class |
|---|---|---|
| 36 | Perennial crop | `PERENNIAL` |
| 46 / 47 / 35 / 48 | Coffee / Citrus / Oil palm / Other perennial | `PERENNIAL` |
| 19 / 39 / 20 / 40 / 41 / 62 | Temporary crop and its sub-classes (soy, sugarcane, rice, other, cotton) | `ANNUAL` |
| 15 | Pasture | `PASTURE_FALLOW` |
| 21 | Mosaic of uses | *ambiguous* — see below |
| 9 | Forest plantation | exclude (D1 woody non-crop) |
| 3/4/5/6/11/12/13 | natural formations | exclude |
| 24/25/30/33 | urban / other non-vegetated / mining / water | exclude |
| 27 | not observed | treat as missing |

**Print the actual code histogram over the parcels first** and build the map from what is
present — many national collections do not use every code. Class 21 (mosaic of
agriculture and pasture) is likely to be *large* in Piura smallholder landscapes and has no
clean mapping; treat it as its own `MOSAIC` category, report agreement both with it
excluded and with it counted as whichever class maximises agreement (and say which you did).

### 8.3 Extraction

`src/crop_classifier/perennial/mapbiomas.py`, mirroring `landsat_gee.py`'s chunked,
resumable, content-addressed pattern (reuse `_retry`, `_run_chunk`, `_chunk_id` — import
them, don't reimplement):

* per parcel per year: the **mode** class within the polygon *and* the class **fraction
  vector** (via `reduceRegions` with a `frequencyHistogram` reducer). The fraction vector
  matters: our parcels are ~0.5 ha ≈ 5 Landsat pixels, so a single mixed pixel flips the
  mode. Purity (`max fraction`) is the right filter for a fair comparison.
* one `getInfo` per chunk over all 35 years at once (all years are bands of one image —
  far cheaper than the Landsat extraction; the whole thing should run in well under an
  hour).
* output `proc()/"mapbiomas_panel.parquet"`: `COD_PREDIO, year, mb_code, mb_class,
  mb_purity`.

### 8.4 The comparisons (and the go/no-go)

1. **Against PETT, at the label year** — three-way: PETT label vs our prediction vs
   MapBiomas class, over the locked-test parcels. Report macro-F1 of *both* our model and
   MapBiomas against PETT. **This is criterion S3 and the project's go/no-go**: if
   MapBiomas scores higher, the honest recommendation is to use MapBiomas directly for the
   trend analysis and report the custom model as a negative result. Say so in the report.
   Stratify by `mb_purity` — MapBiomas should look much better on parcels ≥ 3 ha, and that
   stratification *is* the answer to "why build a parcel-level model at all".
2. **Trajectory agreement, 1990–2024** — per-parcel-year agreement rate; per-year
   agreement (does it degrade in early years?); the transition-count comparison.
3. **Aggregate trend** — perennial area share per year, ours vs MapBiomas, on one figure.
   Criterion S6 is agreement in *direction*, not level; a level offset is expected because
   the class definitions differ.

**State the independence caveat prominently:** MapBiomas Peru is itself Landsat-derived at
30 m, so it shares sensors, cloud regimes and mixed-pixel problems with our model. It is a
benchmark, not ground truth, and agreement between the two is not evidence that either is
correct.

---

## 9. Phase 7 — trajectories, change detection, area estimation

`src/crop_classifier/perennial/trajectories.py`.

### 9.1 Series construction
From `panel_predictions.parquet`, build a per-parcel year-indexed series of class +
probability, with explicit gaps where the parcel abstained. **Never interpolate through a
gap silently** — carry an explicit `observed` mask through every downstream computation.

### 9.2 Smoothing (raw argmax will flicker — plan for it)
Apply, and report both raw and smoothed:

* **mode filter**, centred 3-year window, gap-aware;
* **minimum-duration rule**: a transition is recorded only if the new class persists
  **≥ 3 consecutive observed years**. This is the primary change definition — orchard
  establishment or removal is a multi-year event, so a 1-year excursion is noise by
  construction.

**Flicker rate** = fraction of parcels whose raw series changes class more than
`n_years / 5` times. Report it per class. Criterion S5: under 15% for parcels whose PETT
label is `PERENNIAL`. A high flicker rate means the per-year features are too weak and the
trend analysis should not proceed on raw predictions.

### 9.3 Transitions
Emit `transitions.parquet`: `COD_PREDIO, from_class, to_class, year_of_change,
confidence_before, confidence_after, n_years_before, n_years_after`. Aggregate into a
transition matrix per 5-year period.

**Realism checks** (these catch a broken model faster than any metric):
* `ANNUAL → PERENNIAL` should substantially exceed `PERENNIAL → ANNUAL` if the export-crop
  expansion narrative holds; a symmetric matrix means you are measuring noise.
* Perennial establishment is gradual — a new orchard takes 2–3 years to close canopy, so
  transitions should look like ramps, not steps. Plot mean `P(PERENNIAL)` in the ±5 years
  around detected `ANNUAL→PERENNIAL` transitions; a sigmoid is the expected shape.
* `PERENNIAL → ANNUAL` should be rare and, where it occurs, spatially clustered (orchard
  clearance happens in blocks).

### 9.4 Area estimation (D8)
Three estimators per year per class, on one figure:
(a) naive argmax area, (b) probability-weighted area, (c) **Olofsson et al. (2014)
bias-corrected area with 95% CIs** from the locked-test error matrix. Expand from the
sample to the population using the §7.2 sampling weights. The CIs are the point — an
uncertainty-free trend line from a macro-F1 ≈ 0.7 classifier is not publishable.

### 9.5 Sensitivity analyses
Re-run the headline trend under: sugarcane as `PERENNIAL` (D1); `PASTURE_FALLOW` split
(D2); rule-based vs ML model; raw vs smoothed; MapBiomas class-21 handling. A trend that
survives all five is a finding; a trend that flips under any of them is an artefact.

---

## 10. Phase 8 — reporting

* `docs/perennial/RESULTS.md` — the equivalent of `docs/SUMMARY.md`: what was built, the
  model comparison table, the MapBiomas comparison, the headline trend with CIs, and an
  explicit limitations section.
* `notebooks/05_perennial_trends.ipynb` — figures, following the conventions of the
  existing notebooks (relative paths from `notebooks/`, i.e. `../data/processed/…`).
* Figures: model comparison bars; per-class F1; confusion matrices (model, rule,
  MapBiomas); rule decision boundary in (NDVI level, amplitude) space; temporal-transfer
  curve; per-year feature drift with mission boundaries marked; area share 1990–2024 with
  CIs, ours vs MapBiomas; transition matrix heatmap; a map of `ANNUAL→PERENNIAL` parcels;
  a handful of example parcel trajectories with their Landsat NDVI series.
* Update `CLAUDE.md` §8 and `docs/PIPELINE.md` §1 to point at this work.

---

## 11. Effort and cost budget

| phase | work | wall-clock | GEE |
|---|---|---|---|
| 1 — labels + `paths.py` | code | ~4 h + 30 min gap-fill extraction | small |
| 2 — rule model | code | ~4 h | none |
| 3 — 3 ML models + sweeps | mostly waiting | ~6–10 h | none |
| 4 — validation, calibration, locked test | code + analysis | ~4 h | none |
| 5 — panel extraction | **background, resumable** | **~20–30 h** | yes, the bulk |
| 6 — MapBiomas | code | ~3 h | ~1 h |
| 7 — trajectories | code + analysis | ~6 h | none |
| 8 — reporting | writing | ~4 h | none |

Disk: panel pixel store ~2–4 GB (§7.1), on top of the existing 322 MB. Check free space
before launching Phase 5.

**Order matters:** phases 1–4 are self-contained and need no GEE, and they answer "does
this classifier work at all". **Do not start the 20-hour Phase 5 extraction until Phase 4
has cleared criterion S1.** If the 3-class model cannot beat its baseline on the label
year, it will not produce a meaningful 35-year trend.

---

## 12. Risks

| risk | likelihood | mitigation |
|---|---|---|
| **Sensor shift creates a spurious trend** | high | D7 harmonisation, §7.4 drift diagnostics, mission boundaries marked on every time-series figure |
| **Multi-year assemble merges years** (§7.0.1) | high without the fix | explicit `years` filter + regression test before any panel run |
| Spatial leakage inflates CV more than at 12 classes | high | report CV at `buffer_m` 1500 **and** 3000; the locked test is contiguous-region held out |
| Trends are flicker, not land-use change | medium | minimum-duration rule, flicker-rate criterion S5, ramp-shape check §9.3 |
| MapBiomas beats our model | medium | that is an *answer*, not a failure — §8.4 go/no-go; report it |
| Panel extraction exceeds budget | medium | timing probe §7.1 **before** committing; resumable chunks let you stop anywhere |
| 0.5 ha parcels are ~5 Landsat pixels | inherent | stratify every metric by `area_ha`; report the ≥ 3 ha subset separately |
| Labels are titling dates, not verified observations | inherent | already a known project caveat (`CLAUDE.md` §6); restate it in the results |
| 2012 data hole (L7 SLC-off only) | certain | flag rather than interpolate |

---

## 13. Testing checklist

Extend `tests/` (47 tests currently green — keep them that way):

* `test_labels3.py` — group assignment for single crops, same-group multi-crop,
  mixed-group priority, woody-noncrop exclusion, `land_prep` policy, unassigned-token
  budget assertion.
* `test_rules.py` — protocol conformance (`fit`/`predict_proba`/`save`/`load` round-trip),
  probabilities sum to 1 and are not one-hot, NaN-feature fallback path, threshold search
  finds a known-separable synthetic boundary.
* `test_assemble_years.py` — **the §7.0.1 regression**: a synthetic parcel with pixels in
  two years yields year-isolated tensors.
* `test_trajectories.py` — mode filter on a synthetic flickering series; minimum-duration
  rule does not fire on a 2-year excursion but does on a 3-year one; gap handling.
* `test_area_estimation.py` — Olofsson estimator reproduces a hand-computed worked example
  (use the worked example from the 2014 paper).
* `test_mapbiomas_map.py` — code→class mapping, unknown codes raise rather than silently
  passing through.
* `test_paths.py` — `CC_PROC` switches the tables directory and leaves `FEAT` shared.

Plus the non-negotiable manual check in §3.0: the refactor must reproduce the existing
12-class label table exactly.

---

## 14. Gotchas

Inherited from `CLAUDE.md` §9 / `docs/PIPELINE.md` §6 — all still apply:

* Always `uv run …` (project venv, Python 3.11); add deps with `uv add`, never edit
  `pyproject.toml` by hand.
* **Never import torch and lightgbm in the same process on macOS** (duplicate libomp →
  segfault, exit 139). The lazy registry protects this; keep `rules.py` torch-free.
* GEE needs `uv run earthengine authenticate` + GCP project `peru-crop-classifier`.
* `quality_ok` is a **nullable** boolean — compare with `== True` / `.isna()`, never
  truthiness.
* Parcels are stored EPSG:4326; metric operations go through EPSG:32717.
* `data/` is gitignored and local-only.

New to this work:

* `paths.py` helpers must be called **at call time**, not bound to module-level constants
  at import — otherwise `CC_PROC` silently does nothing and you will overwrite the
  12-class tables.
* The **12-class locked test set is unspent**. Do not run anything with `--eval-test`
  against `CC_PROC` unset.
* `run_coverage` dedupes on `COD_PREDIO` — always give it an explicit per-year `out=`
  path for panel work (§7.0.2).
* The panel writes into `FEAT/"panel"`, never into the audited training store (§7.0.3).
* MapBiomas asset ids change between collections — verify, don't assume (§8.1).
