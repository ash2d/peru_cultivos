# Panel extraction budget — measured, not extrapolated (plan §7.1, D6)

Measured 2026-08-04 with `crop_classifier.perennial.panel.timing_probe` (500 parcels per
era, both extraction stages) plus a follow-up coverage-only sweep over the early years.
Raw numbers: [`panel_budget.csv`](panel_budget.csv).

## 1. Rate by era

| year | s / parcel-year | kB / parcel-year | gate pass (n≥4 clear obs) |
|---|---|---|---|
| 1995 | 0.069 | 0.2 | **5.0 %** |
| 2005 | 0.301 | 3.5 | 99.8 % |
| 2015 | 0.520 | 5.5 | 100 % |
| 2023 | 0.774 | 8.7 | 100 % |

The plan warned not to extrapolate the 1998-derived 0.19 s/parcel-year rate, and that was
right: **2023 costs 11× what 1995 does** and 2.6× what 2005 does. More missions after 2013
means more observations per parcel-year — slower and bigger, exactly as anticipated.

## 2. The finding that changes the panel design: Landsat over Piura starts in 1996

Coverage-only probe, one identical 300-parcel sample across years:

| year | gate pass | median clear obs |
|---|---|---|
| 1990 | 3.7 % | 2 |
| **1992** | **0.0 %** | **0** |
| 1995 | 4.7 % | 2 |
| **1996** | **100 %** | **8** |
| 1998 | 95.0 % | 6 |
| 2005 | 100 % | 8 |
| 2015 | 100 % | 13 |

There is a **hard archive boundary at 1996**. Before it the Landsat 5 archive over Piura is
essentially empty — 1992 has *zero* clear acquisitions over the sample — and after it the
record is dense. This is the well-known pre-1999 international-ground-station gap in the
Landsat 5 archive, and it is decisive here:

* **the panel must start in 1996, not 1990.** Extracting 1990–95 would spend budget on
  years that yield nothing, and — worse — a near-empty early series would appear in the
  headline figure as a spurious rise in *every* class from 1996 onward, which is precisely
  the artefact the whole D7/§7.4 apparatus exists to prevent;
* `DEFAULT_YEARS` is therefore `1996–2024` (**29 years**), and the trend must be reported
  over that window with the pre-1996 gap stated, not silently cropped.

### Two bugs this probe exposed (both fixed, both regression-tested)

1. **Cross-year coverage contamination.** `run_coverage` globbed every `cov_*.parquet` in
   its chunk directory and deduped on `COD_PREDIO` alone, so two years whose `out=` files
   share a parent directory returned the *first* year's numbers for every parcel. The
   probe's first pass reported an identical 4.7 % gate pass for 1995, 1996 **and** 2005.
   The plan's stated fix ("call it once per year with an explicit `out=`") is not enough,
   because the chunk directory is derived from `out.parent`. Now the combine filters by the
   requested years and dedupes on `(COD_PREDIO, year)`.
   Regression: `tests/test_coverage_years.py`.
2. **Empty-year crash.** A year with no acquisitions produced a band-less count image and
   `unmask` raised *"If one image has no bands, the other must also have no bands"* —
   which is real for the early-1990s panel years. The zero base image is now merged in
   first, so an empty year is a legitimate count of 0.

## 3. Budget

Per-parcel cost over 1996–2024, interpolating the measured rates:

| span | years | s/parcel-year | s/parcel |
|---|---|---|---|
| 1996–2010 | 15 | ~0.30 | 4.5 |
| 2011–2018 | 8 | ~0.50 | 4.0 |
| 2019–2024 | 6 | ~0.77 | 4.6 |
| **total** | **29** | — | **~13.1** |

| panel size | wall-clock | disk |
|---|---|---|
| 12,125 parcels (first build) | **~44 h** | ~1.8 GB |
| **7,500 parcels (chosen)** | **~27 h** | ~1.1 GB |
| 5,000 parcels | ~18 h | ~0.7 GB |

**Chosen: 7,500 parcels ≈ 27 h**, inside the plan's 20–30 h envelope. The first build
(12,125) was sized before the rate was known and would have cost 44 h.

Free disk at time of writing: 181 GB — not a constraint.

## 4. Panel composition

`build_panel(n=7500, n_test_forced=2500)`:

* **2,500 locked-test parcels** forced in, stratified by label — enough for the §7.3
  temporal-transfer curve (accuracy vs `k`);
* the rest sampled proportionally by `(label × region_id)` with at least one per non-empty
  stratum, so the trend has spatial coverage across all 256 regions rather than only the 29
  test regions;
* `sample_weight = stratum population / stratum sample` is written alongside — **every area
  share must be expanded to the population with these weights**, never reported as a raw
  sample share.

*Deviation from the plan:* it said to force in *every* locked-test parcel. This test set is
9,894 parcels, which at n=12,000 would make the panel 81 % test parcels and confine the
trend sample to 29 contiguous regions. Capping the forced subset buys spatial coverage for
the area-share trend, which is the actual deliverable, at no cost to the temporal-transfer
check.
