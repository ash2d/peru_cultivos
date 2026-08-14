# Mission policy: TM/ETM+ only — measured coverage by year

> Decision (2026-08-04, user): **do not use L8/L9.** This documents the measurement that
> settles which years the panel can therefore cover. Raw numbers:
> [`l7_coverage.csv`](l7_coverage.csv).

## 1. Why drop OLI

The classifier's training data is:

| mission | observations | share |
|---|---|---|
| L5 (TM) | 2,982,419 | 52.8 % |
| L7 (ETM+) | 2,662,545 | 47.2 % |
| L8 (OLI) | 530 | **0.0 %** |

It has effectively **never seen OLI**. Under the previous 1996–2024 policy, everything from
2013 on would have been inferred on radiometry absent from training, with the sensor step
landing in 2013 — exactly where an export-crop expansion would also appear. The Roy et al.
(2016) harmonisation was implemented but never validated, so the trend would have rested on
an unchecked correction at precisely the point of maximum confounding.

Restricting to **L5 + L7** means every panel year is inferred on radiometry the model was
trained on. The cost is 2024 onwards.

## 2. L7-only coverage by year (300 parcels, gate = ≥4 clear acquisitions)

| year | gate pass | median obs | | year | gate pass | median obs |
|---|---|---|---|---|---|---|
| 1999 | 0.500 | 3.5 | | 2012 | 0.803 | 5 |
| 2000 | 0.970 | 4 | | 2013 | 0.973 | 7 |
| 2001 | 1.000 | 7 | | 2014 | 0.990 | 8 |
| 2002 | 0.927 | 5 | | 2015 | 0.883 | 5 |
| 2003 | 0.983 | 5 | | 2016 | 0.993 | 10 |
| 2004 | 0.900 | 5 | | 2017 | 0.857 | 5 |
| 2005 | 0.997 | 8 | | 2018 | 0.980 | 8 |
| 2006 | 1.000 | 8 | | 2019 | 1.000 | 9 |
| 2007 | 0.963 | 6 | | 2020 | 0.963 | 10 |
| 2008 | 0.837 | 5 | | 2021 | 0.990 | 9 |
| **2009** | **0.470** | 3 | | 2022 | 0.967 | 7 |
| 2010 | 0.970 | 7 | | 2023 | 0.913 | 6 |
| **2011** | **0.373** | 2 | | **2024** | **0.000** | **0** |
| | | | | **2025** | **0.000** | **0** |

**L7 is usable 2000–2023 and stops dead at 2024** (mean 0.07 clear observations per parcel
in 2024, 0.00 in 2025). 1999 is a half-year — L7 launched April 1999.

## 3. Adding L5 back does not rescue the weak years

L5 + L7 on the identical sample:

| year | L7 only | L5 + L7 |
|---|---|---|
| 2008 | 0.837 | 0.870 |
| **2009** | 0.470 | **0.470** |
| 2010 | 0.970 | 0.970 |
| **2011** | 0.373 | **0.373** |
| 2012 | 0.803 | 0.803 |
| 2013 | 0.973 | 0.973 |

Landsat 5 contributed essentially **nothing** over Piura in 2008–2011 — its degraded final
years before the November 2011 decommissioning. So 2009 and 2011 are thin under *any*
TM/ETM+ policy; they are a property of the archive, not of the mission choice.

L5 is kept anyway because it carries 1996–1998 (before L7 existed) and roughly half the
observations in 1999–2007.

## 4. SLC-off costs less than feared

Landsat 7's scan-line corrector failed in May 2003, losing ~22 % of pixels per scene in
wedge-shaped gaps. Since our parcels are ~0.5 ha ≈ 5 Landsat pixels, the worry was that
they would fall into gaps. Measured on 120 parcels, SLC-on vs SLC-off:

| | 2001 (SLC-on) | 2015 (SLC-off) |
|---|---|---|
| parcels with any pixel | 120 / 120 | 120 / 120 |
| median pixels per parcel-date | 6.0 | **5.0** |
| parcel-dates with only 1 pixel | 7.9 % | **10.9 %** |
| median distinct dates per parcel | 7 | 6 |

A ~17 % loss of pixels per parcel-date and a few more single-pixel dates — a real but modest
degradation, and **no parcel is systematically lost**. Post-2003 per-date medians rest on
~5 pixels rather than ~6, so they are noisier; that noise should be visible in the §7.4
drift diagnostics and is a reason to prefer the probability-weighted and bias-corrected
area estimators over raw argmax counts.

## 5. Resulting panel configuration

```python
PANEL_MISSIONS = {"L5", "L7"}      # perennial/panel.py
DEFAULT_YEARS  = 1996 … 2023       # 28 years
```

Mission composition by era: **1996–1998** L5 only · **1999–2011** L5 + L7 · **2012–2023**
L7 only.

**Years to flag in every figure rather than interpolate over:**

| year | gate pass | cause |
|---|---|---|
| 1997 | 48.3 % | the 1997–98 El Niño — catastrophic rain and persistent cloud over Piura |
| 2009 | 47.0 % | archive thin; L5 degraded |
| 2011 | 37.3 % | archive thin; L5 decommissioned November 2011 |
| 2012 | 80.3 % | L7 SLC-off alone (L5 dead, no OLI admitted) |

## 6. Trade-off, stated plainly

| | with OLI (1996–2024) | TM/ETM+ only (1996–2023) |
|---|---|---|
| years | 29 | 28 |
| radiometry at inference | **unseen by the model from 2013** | always seen in training |
| sensor step mid-series | yes, at 2013 | none |
| relies on an unvalidated correction | yes (Roy et al.) | no |
| 2024 onward | available | **not available** |

Losing 2024 is a small price for removing the confound. If the analysis later needs 2024+,
the honest route is to validate the OLI harmonisation empirically first (§7.4 drift
diagnostics) — the code is still there, just switched off.
