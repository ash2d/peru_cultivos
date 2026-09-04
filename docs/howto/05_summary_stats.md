# 5. Summary numbers

Crop mix, land registration, and how both changed. The numbers come from the titling records
(1997–2006) and the 2012 agricultural census. One command, a few seconds. No satellite images
and no model are involved.

```bash
uv run cc -w national analysis summary                      # all of Peru
uv run cc -w national analysis summary --by-dept            # and every department
uv run cc -w national analysis summary --dept PIURA         # one department only
uv run cc -w national analysis summary --by-dept --out summary.csv
```

```
         dept  n_pett  pett_perennial_pct  ...  inscrito_pct  newly_registered_pct  perennial_change_pp  tenure_gap_pp
PERU (all 14)  726808                 9.9  ...          60.3                   8.6                  9.9           -2.1
       ANCASH  154263                 2.9  ...          78.2                   2.0                  8.2           -1.0
          ICA   31274                31.0  ...          58.6                  15.2                  9.0           -0.8
        PIURA   56422                14.4  ...          16.0                  25.5                 11.6           -3.0
```

---

## The columns

| column | what it is |
|---|---|
| `n_pett` | how many parcels the farmers declared a crop for |
| `pett_perennial_pct`, `pett_annual_pct`, `pett_pasture_pct` | what share of those were tree crops, annual crops, and pasture or fallow |
| `n_tenure` | how many parcels have both records of registration |
| `inscrito_pct` | share already registered when the crop was declared |
| `registered_2011_pct` | share registered in the 2011 land survey |
| `newly_registered_pct` | share that went from unregistered to registered between the two. This is the **change in registration** |
| `n_linked` | how many parcels could be matched to a 2012 census record |
| `perennial_before_pct`, `perennial_after_pct` | share of parcels in tree crops at the declaration and in 2012 |
| `perennial_change_pp`, `change_ci95_pp` | the **change in crop type**, in percentage points, and its margin of error |
| `tenure_gap_pp` | that change for registered parcels minus the same for unregistered parcels |

## The rows cover three different sets of parcels

They are not the same parcels. The `n_` column next to each group says how many:

| group of columns | which parcels | how many |
|---|---|---|
| `pett_*` | every parcel with a declared crop and a boundary | 726,808 |
| `tenure_*` | every parcel with both records of registration | 1,780,580 |
| `n_linked` and the change columns | parcels matched to a 2012 census record by the farmer's name | 63,766 |

## Four things to read correctly

- **The change columns only count parcels with a crop in both records.** The census asks which
  crop is grown, so a parcel lying fallow does not appear at all. Comparing every parcel would
  make fallow land look like it fell from 30.8 % to 6.3 %, which is a difference between the two
  surveys, not a change on the ground.
- **The national row is adjusted, the department rows are not.** Matching by name requires a
  name in both records and a matching district, which favours larger, better-documented parcels.
  The matched parcels are 16.6 % tree crops against 9.9 % for the country. Reweighting them so
  each department and crop type carries its true national share raises the national change from
  +8.4 to +9.9 points, so the mismatch was hiding part of the effect. Within a single department
  there is nothing left to reweight.
- **`tenure_gap_pp` describes a difference, it does not prove a cause.** Titles were not handed
  out at random, so a gap here is a fact about the two groups of farmers, not the effect of a
  title. The formal estimate is close to zero: −0.0011, between −0.0126 and +0.0104
  ([`../RESULTS.md`](../RESULTS.md) section 7). The gap also changes sign from one department to
  another, so read the department rows before quoting the national one.
- **Registration was a rolling programme, not two fixed dates.** Almost a quarter of
  declarations carry no date, and the rate of registration does not move steadily with time.

## The full comparison

`analysis summary` is the short version. To see the whole comparison from the declaration to
2012 to 2019-and-later images, with the crop-name audit, the area-weighted version, a check on
match quality, and the figure:

```bash
uv run cc -w national analysis perennial-shift        # about 30 seconds, also draws the figure
uv run cc reproduce perennial-shift                   # just check the three headline numbers
```

The headline: tree and vine crops rose **9.9 points as a share of parcels and 12.5 points as a
share of land area** between about 1999 and 2012, across the country. Parcels with a title
shifted slightly less (−2.1 points, give or take 0.6), and measured by area that difference
disappears (+0.3 points).

Rebuilding the name-matching underneath these numbers needs the restricted files
([`../DATA_ACCESS.md`](../DATA_ACCESS.md)):

```bash
uv run cc -w national data cenagro-link      # match by farmer name, 14 departments, ~6 min
```

For the parcel-by-parcel data behind all of this, see
[`02_parcel_table.md`](02_parcel_table.md). For the land-title study and the questions that were
abandoned, see [`06_reference.md`](06_reference.md).
