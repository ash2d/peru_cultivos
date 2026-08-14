# Landsat multi-mission review — is mixing L5/L7/L8 safe here?

> Audit of how `src/crop_classifier/features/` selects, harmonises and merges Landsat missions,
> measured against the actual extracted pixel store (`data/processed/features/pixels_*.parquet`,
> 4.53 M pixel-obs) and against published cross-sensor best practice. Written 2026-08-03.

## 0. Verdict

**The multi-mission handling is sound, and no result currently on the books is materially distorted
by it.** Two things needed checking and both came back clean once measured rather than argued from
first principles.

1. **Radiometry.** The Collection-2 handling (collections, band aliasing, SR scale/offset, QA
   masking) is correct. The measured L5↔L7 step in *this* dataset is 0.005–0.012 reflectance —
   3–6% of the between-class spread on the vegetation indices. Real, second-order, and consistent
   with the literature. Cross-sensor calibration is **not** the main risk. The one channel where it
   might matter is green (~18% of class spread).
2. **Sensor-as-shortcut.** Which mission saw a parcel *is* a strong label proxy —
   class-conditional `frac_l7` runs 0.20 (ARROZ) to 0.72 (CAFE), driven by
   `crop → district → titling year → which satellites were flying` — and LightGBM leans on it hard:
   `frac_l7` is 3rd of 140 features by gain, and the six non-spectral acquisition/geography statics
   carry **29.8% of total gain**. **But gain is not predictive contribution.** Ablating all six
   costs **0.007 macro-F1** (0.3777 → 0.3705, §6) — inside the noise. The information is redundant
   with the spectral features, so the reported 0.379 is not propped up by it.

So the honest summary is: **there is no calibration error and no inflated score.** What remains is a
set of transfer and interpretation risks, the largest of which is not about sensors at all — it is
that the locked test set is 46% 1998/L5-only against 36% in trainval.

| Pitfall | Severity | Status after measurement |
|---|---|---|
| P1 `frac_l7` / acquisition metadata as shortcut features | **Low** (was: High) | Heavily *used* (29.8% gain) but worth only **0.007 macro-F1**. Not inflating results. Still worth dropping: costs nothing and **halves fold variance** |
| P2 Era shift into the locked test set | **Medium–High** | **Certain** — test is 46% 1998 vs 36% trainval, and 1998 is the sparsest + El Niño cohort |
| P3 No TM→ETM+ radiometric harmonisation | Low–Med | Small (≤6% of class spread on indices; ~18% on green) |
| P4 L7 SLC-off (2003+) | Low | Smaller than the textbook figure — measured 2–6% pixel loss/date, not 22% |
| P5 L5 orbit drift / solar-zenith BRDF | Low–Med | Unquantified here; partly folded into the §3 offset |
| P6 Unharmonised OLI/L9 in a future run | Low now, **High later** | Negligible today (203 obs); a live bug for any post-2013 inference |
| P7 1998 = L5-only, sparsest coverage, El Niño year | Medium | Certain, and it is 38% of the training data |
| P8 Minor code issues (fill bit, DN clipping, no L9) | Low | 0.02% of observations |

---

## 1. What the code actually does

[`features/landsat_gee.py`](../src/crop_classifier/features/landsat_gee.py) merges up to three
Collection-2 Level-2 collections per crop year:

```python
L5 = "LANDSAT/LT05/C02/T1_L2"     # missions_for_year: year <= 2013
L7 = "LANDSAT/LE07/C02/T1_L2"     #                    year >= 1999
L8 = "LANDSAT/LC08/C02/T1_L2"     #                    year >= 2013
```

Bands are aliased to a common set (`SR_B1..5,7` for TM/ETM+, `SR_B2..7` for OLI →
`B,G,R,NIR,SWIR1,SWIR2`), a **single** C2 scale/offset (`DN*2.75e-5 − 0.2`) is applied to all
missions in [`features/indices.py`](../src/crop_classifier/features/indices.py), and QA_PIXEL bits
1–4 plus QA_RADSAT are masked. The merged collection is sampled per pixel per date; `mission` is
carried through to the pixel store.

### Audit findings

| Check | Status |
|---|---|
| Collection ids, tier, level | ✅ T1_L2 for all three — the inter-calibrated tier |
| Band aliasing TM/ETM+ vs OLI | ✅ correct band numbers for both instrument families |
| C2 SR scale/offset shared across missions | ✅ correct — C2 L2 uses identical scaling for TM/ETM+/OLI |
| QA_PIXEL bits 1–4 (dilated cloud, cirrus, cloud, shadow) | ✅ correct C2 bit definitions. Bit 2 (cirrus) is a no-op for TM/ETM+ (no cirrus band) — harmless |
| QA_RADSAT saturation mask | ✅ present |
| **Radiometric harmonisation between missions** | ❌ **none applied** — reflectance is used as-is |
| Fill-pixel (QA_PIXEL bit 0) mask | ⚠️ not masked; measured 0.000% of obs have a zero band, so GEE's own no-data mask is covering it |
| Out-of-range DN handling | ⚠️ `scale_sr` **clips** to [0,1]; a DN of 65535 becomes reflectance 1.0 rather than NaN. 0.02% of obs |
| Same-DOY cross-mission pooling in `per_date_medians` | ⚠️ latent — groups by `(COD_PREDIO, doy)` only, so two missions on one DOY would be silently averaged and tagged `mission="first"`. **Measured occurrences: 0 of 376,998 parcel-dates** (L5/L7 are 8 days offset on a path) |
| Landsat 9 (`LC09`) | ❌ absent from `missions_for_year` — irrelevant for 1998–2010 labels, but any future inference on recent imagery would silently drop half the modern archive |

## 2. What is actually in the extracted data

The mission-selection *rule* spans L5–L8, but the *data* is a two-sensor problem:

| Mission | pixel-obs | share |
|---|---:|---:|
| L5 (TM) | 2,514,082 | 55.5% |
| L7 (ETM+) | 2,014,126 | 44.5% |
| L8 (OLI) | **203** | 0.004% |

L8 contributes 2 parcels (crop years 2015 and 2019). **So "Landsat 5–8" is effectively "Landsat 5
and 7."** Per year:

| year | parcels | L5 | L7 | era |
|---|---:|---:|---:|---|
| 1998 | 17,958 | 586,125 | 0 | L7 not yet launched (Apr 1999) |
| 1999 | 18,097 | 1,370,220 | 982,757 | both, SLC-on |
| 2000–2002 | 6,581 | 552,917 | 703,010 | both, SLC-on |
| 2003–2010 | 4,778 | 4,756 | 328,217 | essentially L7-only, **SLC-off** |

**51% of parcels (24,362) mix L5 and L7 inside their own single-year time series.** So any
uncorrected inter-sensor step appears *within* a parcel's sequence — which is exactly what the LTAE
and PSE-LTAE read.

## 3. Measured L5↔L7 offset in this dataset

Paired same-parcel L5/L7 acquisitions ≤1 day apart in 1999–2000 (n = 3,794 parcel-date pairs,
1,580 parcels), median over each parcel's clear pixels:

| channel | L7 − L5 (mean) | (median) | sd | as % of between-class spread |
|---|---:|---:|---:|---:|
| Blue | −0.0085 | −0.0116 | 0.019 | — |
| Green | −0.0114 | −0.0155 | 0.021 | **17.7%** |
| Red | −0.0052 | −0.0097 | 0.022 | — |
| NIR | −0.0052 | −0.0098 | 0.042 | — |
| SWIR1 | +0.0045 | +0.0072 | 0.037 | — |
| SWIR2 | +0.0043 | +0.0027 | 0.027 | — |
| NDVI | **+0.0113** | +0.0062 | 0.035 | 3.0% |
| NDMI | −0.0191 | −0.0170 | 0.028 | 6.2% |
| BSI | +0.0182 | +0.0170 | 0.023 | 6.0% |

Two caveats on these numbers: (a) 1-day pairs are **sidelap** pairs (adjacent WRS paths), so opposing
view-zenith angles inflate the difference via BRDF — this is an **upper bound** on the pure sensor
term; (b) it is not constant. Stratified by greenness, the NDVI offset grows with vegetation:

| L5 NDVI bin | n | mean ΔNDVI (L7−L5) |
|---|---:|---:|
| ≤0.2 | 156 | −0.005 |
| 0.2–0.4 | 2,045 | +0.007 |
| 0.4–0.6 | 1,512 | +0.017 |
| >0.6 | 81 | +0.032 |

**Reading:** for the vegetation indices the step is 3–6% of the spread between the 12 class medians —
real but second-order. For the raw **green band it is ~18%**, and `G_mean`/`G_p75` are the 4th and
8th most important LightGBM features. That is the one channel where the uncorrected sensor step is
plausibly doing damage.

## 4. Pitfalls, ranked

### P1 — `frac_l7` is a shortcut feature, but a cheap one (Low, after measurement)

`build_lightgbm_features` emits `frac_l7 = mean(mission == 7)` per parcel. Class-conditional means:

```
ARROZ 0.20 · FALLOW 0.26 · FRIJOL 0.36 · MAIZ 0.38 · ZARANDAJA 0.43 · ALGODON 0.45
PASTURE 0.55 · CAÑA 0.56 · PLATANO 0.57 · MANGO_LIMON 0.59 · TRIGO 0.62 · CAFE 0.72
```

That 3.6× range is not a spectral property of coffee. It is a chain:
`crop → district → titling campaign year → which missions were flying`. LightGBM importance
confirms the model uses it — `frac_l7` is 3rd of 140 by gain (4.1%), and the six non-spectral
acquisition/geography features together carry **29.8% of total gain**, led by `centroid_lat` (14.4%)
and `n_valid_obs` (7.1%).

**The ablation (§6) shows this matters far less than the gain figures suggest.** Removing all six
costs 0.007 macro-F1. Gain measures how often the tree found a feature convenient to split on, not
how much unique predictive signal it carried; here the metadata is largely redundant with the
spectral summaries, so the tree simply re-routes through correlated features when it is removed
(dropping `frac_l7` alone: −0.007; dropping all five acquisition features: −0.008; dropping those
plus `centroid_lat`: −0.007 — all the same, all within the ±0.02–0.04 fold spread).

So this is **not** an inflated-score problem. What it is:

- **A transfer risk.** "Recover the titling year and latitude" is a strategy that works inside this
  archive and nowhere else — not in a new region, not on a present-day image. It contributes
  nothing here, and would contribute negatively under deployment.
- **A variance problem.** Dropping the six *halves* the fold spread (±0.039 → ±0.022) and lifts the
  worst fold (0.304 → 0.331) while lowering the best (0.415 → 0.384). The model becomes more
  regionally consistent — which is what a crop map actually needs.
- **Not** a distortion of the model comparison. Verified against the saved tensors: LTAE and
  PSE-LTAE see **only the 11 spectral channels plus DOY and a validity mask** — no `mission`, no
  latitude, no observation counts. LightGBM gets six extra features **and still loses** (0.379 vs
  0.409). Removing them widens the gap slightly, so the "tuned LTAE wins" conclusion is *more*
  robust, not less.

Verdict: drop them because they are free to drop and buy stability, not because they are corrupting
anything.

### P2 — Mission/era is confounded with the splits (High, certain)

Sensor-era composition varies sharply across the spatial CV folds and, more importantly, between
trainval and the locked test set:

| | 1998 (L5 only) | 1999–2002 (L5+L7) | 2003+ (L7 SLC-off) |
|---|---:|---:|---:|
| trainval | 36.3% | 53.4% | 10.3% |
| **test** | **46.0%** | 42.8% | 11.2% |
| fold 0 | 38.3% | 55.9% | 5.9% |
| fold 3 | **12.4%** | 77.5% | 10.1% |
| fold 4 | 52.8% | 36.4% | 10.8% |

Fold 0 and fold 3 are near-opposite sensor regimes. **This does not, however, explain the fold
spread**: across the 5 folds, macro-F1 is uncorrelated with the 1998/L5 share (r = 0.01 LightGBM,
0.10 LTAE) — so [`SUMMARY.md`](SUMMARY.md)'s reading of fold 0 as genuine geographic difficulty
survives this check. Fold score does correlate positively with SLC-off share (r = 0.71–0.86), but
with n = 5 and SLC-off years carrying the easy CAFE/TRIGO classes, that is far more likely a
class-mix effect than a sensor effect. **The fold-level evidence for sensor-driven variance is
weak; the split-level evidence below is not.**

**Practical consequence for the one remaining locked-test evaluation:** the test set is
1998/L5-heavy relative to what the model trained on, and 1998 is simultaneously the L5-only, the
sparsest-coverage (median 6 valid obs vs 8–13 later), and the El Niño cohort. Expect the locked-test
number to come in **below** pooled CV, and do not interpret that gap as overfitting without
stratifying by era first.

### P3 — No TM→ETM+ harmonisation (Low–Medium)

No cross-sensor coefficients are applied anywhere in the pipeline. Best practice is split:

- USGS positions Collection 2 Tier 1 as **inter-calibrated across instruments**, and the 2016 TM
  recalibration explicitly tied L4/5 reflectance to OLI. Google's own harmonisation tutorial now
  opens with "outdated and **not recommended or necessary** when working with Landsat Collection 2
  surface reflectance data."
- But the time-series literature disagrees for *trend* work: Zhang & Roy and the `LandsatTS`
  authors report that uncorrected inter-sensor bias injects spurious NDVI drift, and recommend
  calibrating L5 and L8 onto L7 as the reference.

**For this project the distinction matters less than usual**, because the labels are single-year
snapshots — there is no multi-decadal trend to corrupt. The risk is confined to the ~51% of parcels
whose within-year series steps between sensors mid-sequence. Given the measured magnitude (3–6% of
class spread on indices) this is a **second-order** effect, with the green band the exception.

If you do harmonise, note the Roy et al. (2016) coefficients are ETM+→OLI and, per the GEE tutorial,
**apply identically to TM** — i.e. they do not give you a TM→ETM+ correction. You would need to fit
your own from the tandem overlap, and you already have 3,794 paired observations to do it with.

### P4 — L7 SLC-off (Low–Medium; smaller than the textbook 22%)

The scan-line corrector failed 2003-05-31; uncorrected scenes lose ~22% of pixels in wedge-shaped
gaps. 10.4% of modelled parcels (5,179) have crop years ≥2003 and are almost purely L7.

Measured impact here is much milder than the nominal figure, because parcels are small (median ~3–11
Landsat pixels) — a parcel tends to fall either wholly inside a gap (the date simply disappears) or
wholly outside it, rather than being partially striped:

| year | era | median fraction of parcel sampled per date |
|---|---|---:|
| 1999–2001 | SLC-on | 0.989–0.993 |
| 2003 before DOY 151 | SLC-on | 0.987 |
| 2003 after DOY 151 | SLC-off | 0.948 |
| 2005 / 2006 / 2007 | SLC-off | 0.980 / 0.955 / 0.934 |

So SLC-off costs 2–6% of within-date pixels, not 22%. It shows up instead as **missing dates**,
already captured by `n_valid_obs`/`max_gap` (2003 has the worst median max_gap, 5 months). The
literature is reassuring here: object-based multitemporal crop classification on un-gap-filled
SLC-off imagery still reaches >85% overall accuracy, and gap-filling buys only 0.3–1.3% kappa.

The concentration is the concern, not the gaps: **51.5% of all CAFE parcels come from SLC-off
years**, versus 2.2% of FALLOW. CAFE is also the highest-`frac_l7` class (0.72) and is currently
well classified (F1 ~0.70). The §6 ablation makes an "it's just reading the sensor" explanation
unlikely — CAFE survives without the metadata features — but the era stratification in
recommendation 2 is what would settle it properly.

### P5 — Landsat 5 orbit drift and solar-zenith/BRDF (Low–Medium, unquantified)

L5's orbit was maintained only by sparse station-keeping manoeuvres; across the mission, overpass
time varied by up to ~1 hour and solar zenith by >10°, which combines with surface reflectance
anisotropy to produce reflectance and NDVI inconsistencies independent of any sensor calibration.
The L5 block here (1998–2001) is short, so within-project drift is limited, but this compounds with
P3: part of the L7−L5 offset measured in §3 is illumination/view geometry, not radiometry. It also
means the 1998 L5-only cohort has a systematically different acquisition geometry from the 1999+
mixed cohort.

### P6 — OLI would enter unharmonised if the pipeline is re-run forward (Low today, High later)

`missions_for_year` will happily merge OLI into the same unharmonised pool. Today that is 203
observations and irrelevant. But OLI's differences from TM/ETM+ are the ones that genuinely do need
coefficients — different atmospheric correction (LaSRC vs LEDAPS, ~2% SR difference), a narrower NIR
band that avoids the 825 nm water-vapour feature, and OLI NDVI systematically **above** ETM+ NDVI
over vegetated surfaces (Roy et al. report ~4.7% mean relative TOA difference). **Any inference on
modern imagery with a model trained on this 1998–2001 TM/ETM+ archive is a cross-sensor domain shift
that is currently uncorrected and undetected.** L9 being absent from the code compounds this.

### P7 — The 1998 cohort (Medium, certain)

1998 is 38% of modelled parcels and is simultaneously: L5-only; the sparsest coverage (median 6
valid observations, vs 8–13 for 1999–2001); and the crop year of the catastrophic 1997–98 El Niño in
Piura. The pre-1999 Landsat archive is also globally thinner — before Landsat 7 introduced
systematic global acquisition, coverage depended on International Cooperator ground stations with
uneven archiving, and roughly two-thirds of pre-1999 data sat unrecovered at those stations. So
"1998 = L5" is inseparable from "1998 = sparse + anomalous weather." Any per-mission performance
breakdown will attribute all three to the sensor.

### P8 — Minor (Low)

- 0.02% of observations have a band outside the C2 valid range (7273–43636), including DN 65535.
  `scale_sr` **clips** these to 1.0 instead of nulling them — a plausible-looking wrong value rather
  than an honest NaN. All on L5.
- Fill (QA_PIXEL bit 0) is not explicitly masked; measured leakage is 0, so GEE's no-data mask is
  doing the job, but the guarantee is implicit.
- `per_date_medians` keys on `(COD_PREDIO, doy)` without `mission`, so two missions on one DOY would
  be averaged into a single "date". Currently 0 of 376,998 parcel-dates — but it would silently
  activate if L8/L9 (different repeat cycle) were ever added.

---

## 5. Recommendations, in priority order

1. **Drop the six metadata features** (`frac_l7`, `n_valid_obs`, `max_gap`, `n_dates`,
   `n_valid_pixels`, `centroid_lat`) from the LightGBM feature set. Measured cost: 0.007 macro-F1;
   measured benefit: fold sd 0.039 → 0.022 and a more transferable model (§6). Low priority in
   absolute terms — it changes no conclusion — but it is nearly free.
2. **Stratify every reported metric by sensor era** (1998 L5-only / 1999–2002 L5+L7 / 2003+ SLC-off)
   before the locked-test run. `plan.md` §13 already asks for this; it has not been done, and it is
   the cheapest way to separate P1/P2 from genuine spectral skill.
3. **Expect and pre-announce a test-set drop.** The test split is 46% 1998 vs 36% in trainval. Decide
   *now*, before touching the test set, that an era-stratified breakdown will accompany the headline
   number.
4. **Fit your own TM→ETM+ correction** from the 3,794 tandem pairs already in the pixel store, apply
   it to the L5 side, and re-run one model as a sensitivity check. Cheap, and it settles P3
   empirically rather than by appeal to authority. Prioritise the visible bands (green especially).
5. **Guard the forward path**: add `LC09`, and either apply Roy et al. (2016) ETM+↔OLI coefficients
   or refuse to mix OLI/OLI-2 with TM/ETM+ at inference time. Add `mission` to the
   `per_date_medians` group key while you are there.
6. **Null, don't clip**, out-of-range DN in `scale_sr`; add the QA fill bit explicitly.
7. Note for the write-up: SLC-off is *not* the headline risk here (measured 2–6% pixel loss), but
   its concentration in CAFE is. Say so explicitly rather than leaving `plan.md`'s
   "SLC-off striping injects gaps/artifacts" as the standing assessment.

## 6. Ablation result

LightGBM, tuned hyperparameters (`num_leaves=98, lr=0.077, min_child_samples=44, n_estimators=400`),
same 5-fold spatially-blocked CV and class weighting as the production run, differing only in which
feature columns are supplied. The "ALL features" row reproduces the saved
`lightgbm_12c_tuned` fold-mean (0.379), confirming the replication is faithful.

| feature set | macro-F1 | sd | per-fold |
|---|---:|---:|---|
| ALL features (140) | **0.3777** | 0.039 | 0.304 · 0.384 · 0.378 · 0.408 · 0.415 |
| − `frac_l7` | 0.3703 | 0.043 | 0.290 · 0.384 · 0.370 · 0.410 · 0.397 |
| − 5 sensor/acquisition (`frac_l7`, `n_valid_obs`, `max_gap`, `n_dates`, `n_valid_pixels`) | 0.3698 | 0.038 | 0.298 · 0.389 · 0.364 · 0.396 · 0.401 |
| − those 5 **and** `centroid_lat` | 0.3705 | **0.022** | 0.331 · 0.396 · 0.375 · 0.367 · 0.384 |

**Reading.** The entire 29.8%-of-gain metadata block is worth **0.007 macro-F1** — well inside the
fold spread, i.e. indistinguishable from noise. Two things follow:

1. **The published LightGBM number is not inflated by sensor metadata.** The pre-ablation worry
   (that `frac_l7`'s rank-3 gain implied a substantial shortcut) does not survive measurement.
   High gain ≠ high unique contribution when features are correlated.
2. **Removing them is still the right call**, for a different reason than expected: fold sd drops
   from 0.039 to **0.022** and the hardest fold improves by +0.027. Trading 0.007 of mean for half
   the regional variance — and for features that will actually exist in a new region — is a good
   trade.

Reproduce with `/private/tmp/.../scratchpad/ablate.py` (logic is ~20 lines against
`crop_classifier.data`; worth folding into the repo as a proper `--drop-features` flag on
`cli.train` if this is to be rerun).

---

## Sources

- [Landsat Collection 2 Surface Reflectance — USGS](https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-reflectance)
- [Landsat Collection 2 Level-2 Science Products — USGS](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products)
- [Landsat 4-5 TM Calibration Notices — USGS](https://www.usgs.gov/landsat-missions/landsat-4-5-tm-calibration-notices) (2016 recalibration tying TM reflectance to OLI)
- [Landsat Collection 2 Known Issues — USGS](https://www.usgs.gov/landsat-missions/landsat-collection-2-known-issues)
- [Landsat 5/7/8/9 Cross-Sensor Harmonization — ClimateEngine](https://support.climateengine.org/article/177-landsat-5-7-8-9-harmonization)
- [Landsat ETM+ to OLI harmonization tutorial — Google Earth Engine Community](https://github.com/google/earthengine-community/blob/master/tutorials/landsat-etm-to-oli-harmonization/index.md) ("outdated and not recommended or necessary" for Collection 2; Roy et al. Table 2 coefficients; TM treated as ETM+)
- Roy et al. (2016), [Characterization of Landsat-7 to Landsat-8 reflective wavelength and NDVI continuity](https://www.sciencedirect.com/science/article/pii/S0034425715302455)
- Teillet et al. (2001), [Radiometric cross-calibration of ETM+ and TM based on tandem data sets](https://www.sciencedirect.com/science/article/abs/pii/S0034425701002486) (the June 1999 tandem orbit)
- Zhang & Roy (2016), [Landsat 5 TM reflectance and NDVI 27-year time series inconsistencies due to satellite orbit change](https://www.sciencedirect.com/science/article/pii/S003442571630325X)
- [A CONUS analysis of the impact of Landsat 5 orbit drift on temporal consistency](https://www.sciencedirect.com/science/article/pii/S0034425720300705)
- Berner et al., [`LandsatTS`: retrieval, cleaning, cross-calibration and phenological modeling of Landsat time series](https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.06768) (recommends calibrating L5/L8 onto L7)
- Li et al. (2013), [Object-oriented crop classification using multitemporal ETM+ SLC-off imagery and random forest](https://www.tandfonline.com/doi/full/10.1080/15481603.2013.817150) (>85% OA without gap-filling)
- [Use of Landsat ETM+ SLC-off segment-based gap-filled imagery for crop type mapping — USGS](https://pubs.usgs.gov/publication/70000481) (gap-filling gains 0.3–1.3% kappa)
- Wulder et al. (2016), [The global Landsat archive: status, consolidation, and direction](https://landsat.usgs.gov/sites/default/files/documents/1-s2.0-S0034425715302194-main.pdf) (pre-1999 IC ground-station coverage gaps)
- [Global 30 m seamless data cube (2000–2022) from Landsat 5/7/8/9 and MODIS — ESSD](https://essd.copernicus.org/articles/16/5449/2024/)
