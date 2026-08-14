# Landsat mission audit (Landsat 5–8)

**Scope.** Read-only audit of the extraction and modelling code, the existing feature store,
and USGS guidance. The question is whether the project can pool Landsat 5, 7, and 8 as
one source for crop classification, and how material the remaining differences are.

## Conclusion

The implemented **Landsat 5 / Landsat 7** workflow is technically sound for this project:
it uses Collection 2 Tier 1 Level-2 surface reflectance, maps equivalent reflective bands to
a common six-band schema, applies the correct Collection-2 scale and offset, and masks the
important cloud, cloud-shadow, dilated-cloud, cirrus, and radiometric-saturation flags.
There is no evidence of a wrong scaling factor, reversed band mapping, or inclusion of a
non-comparable product.

The material caveat is that this is not actually a four-mission training set. The production
records are overwhelmingly 1998–2007, hence Landsat 5 TM and Landsat 7 ETM+ dominate.
The extracted store contains only two L7+L8 parcel-years (2015 and 2019). Therefore the
current classifier is not meaningfully validated for Landsat-8 inputs. Its results should be
described as **L5/L7**, not L5–L8, unless a separate L8 validation is added.

## What the project does

| Check | Finding | Assessment |
|---|---|---|
| Product | `C02/T1_L2` SR for L5, L7, L8 | Correct and recommended baseline |
| Bands | TM/ETM+ `B1,2,3,4,5,7`; OLI `B2,3,4,5,6,7` -> blue, green, red, NIR, SWIR1, SWIR2 | Correct equivalence mapping |
| Scaling | `reflectance = DN * 0.0000275 - 0.2` | Correct for Collection-2 L2 SR |
| QA | masks QA_PIXEL bits 1–4 and all QA_RADSAT flags | Good; see small hardening item below |
| L7 SLC-off | leaves missing pixels masked; does not spatially fill them | Appropriate for irregular-date ML |
| Model inputs | LTAE/PSE-LTAE receive no mission indicator; LightGBM has only `frac_l7` | Main modelling gap |

USGS specifies the same Collection-2 SR scale/offset and carries `QA_PIXEL` and `QA_RADSAT`
into Level-2 products. Surface reflectance removes much of the atmosphere-driven variation,
but not all cross-sensor spectral-response variation.

## Evidence from the existing extraction

Among parcel-years with extracted pixels, 18,341 are L5-only, 24,360 combine L5+L7, 5,148
are L7-only, and **2** combine L7+L8. L8 supplies only 203 pixel observations in 2015 and
10 in 2019. The deployed training problem is consequently dominated by TM/ETM+.

The mission regime is strongly associated with year and class composition. For example,
69.3% of quality-passing L5-only parcels are rice, compared with 28.6% in the 1999–2002
L5+L7 regime and 22.6% in the mainly-L7 later regime. The best tuned LTAE's spatial-CV
macro-F1 is 0.146 in 1998 (L5 only) versus 0.374 in 1999 (mixed L5/L7). This does **not**
demonstrate a calibration error: 1998 also has fewer clear dates (median 6 versus 8), severe
El Niño cloud effects, different geography, and different class prevalence. It does show
that sensor/year effects cannot presently be separated from coverage and label-distribution
effects.

## Pitfalls and likely effect on the classifier

| Pitfall | Current handling | Likely effect |
|---|---|---|
| TM vs ETM+ radiometric differences | Collection-2 calibration and common SR product; no explicit transform | **Low** for L5/L7 classification within this dataset; residual bias is possible in raw bands and indices |
| OLI spectral-bandpass and atmospheric-correction differences | Only renamed to common bands; no TM/ETM+ ↔ OLI adjustment | **Moderate to high if L8 is used at scale**; currently negligible because there are two L8 parcel-years |
| Sensor/year confounding | Spatial CV, but no mission feature in temporal models | **Moderate**, and likely more important than residual calibration; model may learn era/sensor proxies for crop prevalence |
| L7 SLC-off after 31 May 2003 | Missing pixels are naturally excluded by the mask; observation count/gap are retained | **Moderate** for 2003+ sparse parcels, mostly loss of phenology rather than biased reflectance |
| Clouds and cloud adjacency | Cloud, shadow, cirrus, dilation and saturation masked | **Moderate** in the 1998 El Niño cohort; residual cloud/haze can be more damaging than cross-calibration |
| Fill/snow QA flags | Fill bit 0 and snow bit 5 are not explicitly tested | **Low** here, but add the fill test defensively; snow is immaterial in Piura |
| Small/mixed parcels | Area gate, medians, spatial CV, pixel-set ablation | **High overall**, but this is a spatial-resolution issue rather than a mission-calibration issue |

## Best-practice recommendation

1. Keep the current Collection-2 Level-2 SR + QA workflow; do not mix Collection 1,
   TOA reflectance, or unscaled DN with it.
2. Treat L5/L7 pooling as acceptable, but retain `mission` per observation. Add either a
   one-hot mission channel to LTAE/PSE-LTAE or per-mission normalisation statistics fit on
   each training fold. Keep the raw SR values as well; do not use global histogram matching.
3. Before admitting L8 at useful volume, run a mission-transfer test: use 1999–2002 L5/L7
   near-date pairs over stable targets (or pseudo-invariant sites) to quantify per-band and
   index offsets; then validate a classifier trained on one mission regime against another.
   For L8, use published spectral-adjustment coefficients or derive site-specific coefficients
   only if that test finds a material degradation.
4. Add QA_PIXEL bit 0 (fill) to `_clear_mask` and unit-test it. This is inexpensive defence
   against no-data values; the current band mask may already exclude them, but the code should
   state the rule explicitly. Consider masking snow as well for portability.
5. Report accuracy/macro-F1 by year, mission mixture, number of clear dates, and L7 SLC-off
   status. Do not interpret the weak 1998 scores as a sensor-calibration result without a
   controlled ablation.

## Sources

- [USGS Collection 2 Level-2 science products](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products) — SR processing and scaling.
- [USGS QA bands](https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands) — QA_PIXEL/QA_RADSAT meanings and L4–7 vs L8–9 SR QA differences.
- [USGS Landsat 4–5 TM calibration notices](https://www.usgs.gov/landsat-missions/landsat-4-5-tm-calibration-notices) and [Landsat 7 ETM+ calibration notices](https://www.usgs.gov/land-resources/nli/landsat/landsat-7-etm-calibration-notices) — cross-calibration improvements, including the tie to OLI calibration.
- [USGS Level-1 product guidance](https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product) — consistently calibrated sensors can still differ because of spectral bandpass.
- [USGS Landsat 7 SLC-off guidance](https://www.usgs.gov/faqs/what-landsat-7-etm-slc-data) — post-31-May-2003 gaps retain radiometric and geometric corrections; [USGS SLC-off product development](https://www.usgs.gov/publications/landsat-7-scan-line-corrector-gap-filled-product-development) quantifies the approximate 22% scene-area loss.

## Files inspected

- `src/crop_classifier/features/landsat_gee.py`
- `src/crop_classifier/features/indices.py`
- `src/crop_classifier/features/assemble.py`
- `src/crop_classifier/data.py`
- `src/crop_classifier/train.py`
- `runs/*/stratified.csv` and the cached feature store
