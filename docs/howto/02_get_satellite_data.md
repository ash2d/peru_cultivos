# 2. Get the satellite data

Pulling imagery from Google Earth Engine and turning it into model features. Needs a working
Earth Engine account ([`01_setup.md`](01_setup.md)) and the parcel polygons.

**You may not need this at all.** Extraction is the expensive step — hours to days. If someone
has already given you a `features/` directory, skip to
[`04_train_and_evaluate.md`](04_train_and_evaluate.md).

---

## The shape of it

```bash
uv run cc -w national satellite extract --stage all
uv run cc -w national satellite assemble
```

Two stages, on purpose:

**Stage 1 — coverage.** A cheap per-chunk count: how many clear acquisitions does each parcel
have in its crop year (`n_valid_obs`), and what is the longest empty run of months (`max_gap`).
This feeds a gate, so parcels that cannot support a measurement are dropped *before* anyone
pays for their pixels.

**Stage 2 — pixels.** For gate survivors only, every clear pixel observation is written to
`pixels_<year>.parquet`. Extracted **once**; both model representations (the LightGBM summary
table and the LTAE sequence tensor) are re-assembled from it offline, so trying a new feature
does not mean re-extracting.

Then `assemble` turns pixels into features: per channel median / mean / std / min / max /
p25 / p75 / amplitude, a linear slope, and an order-1 harmonic fit. No binning, no
interpolation. Under four observations the columns are NaN, which LightGBM handles natively.

Sentinel-2 instead of Landsat: the same commands against the `national_s2` workspace.

## Useful flags

| flag | what it does |
|---|---|
| `--stage coverage` / `--stage pixels` | run one stage only |
| `--years '1999,2000'` | restrict to these crop years; ranges like `1999-2003` work |
| `--max-chunks N` | stop after N new chunks — a partial, resumable run |
| `--chunk-size`, `--pixel-chunk-size` | parcels per request |

**Extraction is resumable.** Chunk filenames are a hash of the chunk's parcel ids, so stopping
and restarting is safe even if the parcel set, its order, or the chunk size changed.

**It parallelises across processes on disjoint year ranges:**

```bash
uv run cc -w national satellite extract --stage pixels --years 1999-2003 &
uv run cc -w national satellite extract --stage pixels --years 2004-2008 &
```

The ranges must genuinely be disjoint — a process covering *all* years just duplicates the
others' work. ⚠️ Five workers trips Earth Engine's Restricted Mode; **two is stable** for the
Sentinel-2 store.

---

## ⚠️ Earth Engine fails silently, in three different ways

This is the section to read twice. None of these raise an exception.

**1. It hangs with no error.** A request can simply never return. Plain retry-with-backoff
never fires, because nothing failed. The fix in the code is a **900-second wall-clock
deadline** on a worker thread — it fired 25 times during the national extraction and lost no
workers.

⚠️ That thread must be a **daemon**. An earlier version used `ThreadPoolExecutor`, whose
non-daemon threads are joined by an `atexit` hook; every worker then sat alive at 0 % CPU for
hours after its work was done. `cancel_futures=True` does not help.

**2. Throttling arrives as both an error string and a hang.** So throttle messages are
classified as transient and retried, and the deadline catches the silent half.

**3. The chunk cache is content-addressed on the parcel ids — not on what the chunk
computes.** If you add a column to an extraction, every existing chunk still looks complete and
your new column is silently absent from all of them. **Invalidate the cache by hand** when you
change what is computed.

There is a fourth, which is about cost rather than correctness: **a chunk's price scales with
its bounding-box extent.** Grouping parcels by year alone can span 110 deg² nationally and the
job stalls at 0 % CPU with no error. Chunks are packed for locality under a 4 deg² cap. If you
write a new extractor, copy the *packing*, not just the structure — this was learned twice,
because the second module copied the shape and not the reason.

---

## ⭐ Verify a finished job by counting its output

**Never conclude an extraction worked because the process ended.** Count what is on disk.

And count it *comparatively*. A guessed completeness threshold cries wolf: there is a genuine
~0.6 % sub-pixel floor of parcels too small to ever return a pixel, so "98 % complete" is not
evidence of anything by itself. Compare the deficit **across years** — a year that is anomalous
against its neighbours is a real gap; a uniform small deficit is the floor.

```bash
uv run cc -w national advanced density-audit
```

---

## What it costs

* Landsat over Piura starts in **1996** — 1992 has zero clear acquisitions, and 1997 is half
  lost to El Niño. Cost rises about 11× from 1996 to 2023 as the archive deepens.
* Landsat gives a median of **13–24 clear dates** per parcel-year. Sentinel-2 gives **47**,
  which is why an architecture verdict measured on one store has to be re-asked on the other.
* The national panel extraction was a 20+ hour job. Budget accordingly and run it resumably.

Next: [`03_new_label_set.md`](03_new_label_set.md).
