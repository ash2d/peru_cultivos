# 5. Get satellite imagery

Pull imagery from Google Earth Engine and turn it into model features. Needs an Earth Engine
account ([`01_setup.md`](01_setup.md)) and parcel polygons.

**You may not need this.** Extraction is the expensive step — hours to days. If the features
already exist, go to [`03_train_and_evaluate.md`](03_train_and_evaluate.md).

---

## The two commands

```bash
uv run cc -w national satellite extract --stage all
uv run cc -w national satellite assemble
```

`extract` runs two stages on purpose:

1. **Coverage.** A cheap count of how many clear dates each parcel has in its year, and the
   longest gap between them. Parcels that cannot support a measurement are dropped here,
   before anyone pays for their pixels.
2. **Pixels.** For the survivors, every clear pixel observation is saved. This happens once —
   both feature formats are rebuilt from it offline, so trying new features does not mean
   extracting again.

`assemble` turns pixels into features: per channel median, mean, std, min, max, p25, p75,
amplitude, a slope and a simple seasonal fit. No filling in of gaps. Fewer than four
observations leaves the columns empty, which LightGBM handles.

| flag | what it does |
|---|---|
| `--stage coverage` / `--stage pixels` | run one stage only |
| `--years '1999-2003'` | restrict to those crop years |
| `--max-chunks N` | stop after N chunks — a partial run, safe to resume |
| `--chunk-size`, `--pixel-chunk-size` | parcels per request |

Extraction is **resumable**: chunk filenames are a hash of the parcels in them, so stopping
and restarting is safe even if the parcel set or chunk size changed. You can run two processes
on disjoint year ranges; five trips Earth Engine's Restricted Mode, two is stable.

---

## Earth Engine fails quietly, in three ways

None of these raise an error, so none of them look like failure.

1. **It hangs and returns nothing.** Retrying never triggers, because nothing failed. The code
   puts a 900-second deadline on each request; it fired 25 times during the national
   extraction and lost no work.
2. **Throttling arrives both as an error message and as a hang.** Throttle messages are
   retried; the deadline catches the silent half.
3. **The cache keys on which parcels are in a chunk, not on what the chunk computes.** Add a
   column to an extraction and every existing chunk still looks complete, silently without it.
   Clear the cache by hand when you change what is computed.

There is a fourth, about cost: **a chunk's price grows with how far apart its parcels are.**
Grouping by year alone can span a third of the country, and the job then sits at 0 % CPU with
no error. Chunks are packed for locality. If you write a new extractor, copy the packing, not
just the structure.

## Check the result by counting it

Never conclude an extraction worked because the process ended. Count what is on disk, and
count it **against other years**: there is a genuine ~0.6 % floor of parcels too small to
return a pixel, so "98 % complete" proves nothing on its own. A year that looks wrong next to
its neighbours is a real gap.

```bash
uv run cc -w national advanced density-audit
```

## What it costs

- Landsat over Piura starts in **1996**; 1992 has no clear acquisitions and 1997 is half lost
  to El Niño. Cost rises about 11× from 1996 to 2023 as the archive deepens.
- Landsat gives a median of **13–24 clear dates** per parcel-year; Sentinel-2 gives **47**.
  That difference is large enough that a model comparison decided on one has to be re-run on
  the other.
- The national multi-year extraction was a 20+ hour job. Plan for it and run it resumably.

Next: [`06_label_more_parcels.md`](06_label_more_parcels.md).
