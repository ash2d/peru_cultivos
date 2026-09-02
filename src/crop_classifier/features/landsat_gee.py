"""GEE Landsat extraction: stage-1 coverage pass + stage-2 raw dated pixels (plan.md §6, A4).

Two stages, both chunked and resumable (chunk parquets under ``data/processed/features/``;
re-running skips finished chunks):

* **Stage 1 — coverage**: for every eligible parcel (from ``modeling_parcels.parquet``),
  count QA-clear Landsat acquisitions in its crop year (``n_valid_obs``, best-pixel) and
  the longest run of empty months (``max_gap``) -> ``coverage.parquet``. Cheap: one
  ``reduceRegions`` per chunk. The §7 abstain gate (``n_valid_obs >= 4``) is applied to
  this table.
* **Stage 2 — pixels**: for gate *survivors only*, export every clear pixel observation
  (raw DN + pixel lon/lat + DOY/year/mission) -> ``pixels_<year>.parquet``. This is the
  raw store both assemblies rebuild from offline.

Mission selection by year: L5 (<=2013), L7 (>=1999), L8 (>=2013); bands harmonised to
``B,G,R,NIR,SWIR1,SWIR2``. QA_PIXEL bits 1-4 (dilated cloud, cirrus, cloud, shadow) and
QA_RADSAT saturation are masked out.

Run with::

    uv run python -m crop_classifier.cli features extract            # both stages
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from crop_classifier.paths import feat, proc

# The Earth Engine project id lives in workspaces.yaml, resolved at call time by `init_ee`
# (EE bills a project, so no shared default is ever right).

L5 = "LANDSAT/LT05/C02/T1_L2"
L7 = "LANDSAT/LE07/C02/T1_L2"
L8 = "LANDSAT/LC08/C02/T1_L2"
L9 = "LANDSAT/LC09/C02/T1_L2"


def f_coverage() -> Path:
    """Stage-1 coverage table, in the current feature store (``CC_FEAT``).

    A function, not a constant: binding it at import would pin the Piura store (see paths.py).
    """
    return feat() / "coverage.parquet"


def f_parcels() -> Path:
    """The current workspace's parcel table (resolved at call time — see paths.py)."""
    return proc() / "modeling_parcels.parquet"


# mission -> (collection id, band mapping to harmonised names)
_TM_BANDS = {"SR_B1": "B", "SR_B2": "G", "SR_B3": "R", "SR_B4": "NIR",
             "SR_B5": "SWIR1", "SR_B7": "SWIR2"}
_OLI_BANDS = {"SR_B2": "B", "SR_B3": "G", "SR_B4": "R", "SR_B5": "NIR",
              "SR_B6": "SWIR1", "SR_B7": "SWIR2"}

_ee = None  # module-level handle set by init_ee


def init_ee(project: str | None = None):
    """Initialise Earth Engine once; a 120 s socket timeout stops any call hanging.

    ``project`` defaults to ``gee_project`` in ``workspaces.yaml`` (or ``GEE_PROJECT``),
    resolved here so importing this module never requires a configured project.
    """
    global _ee
    if _ee is None:
        from crop_classifier.workspace import gee_project
        socket.setdefaulttimeout(120)
        import ee
        ee.Initialize(project=project or gee_project())
        _ee = ee
    return _ee


# Optional global restriction on missions, e.g. {"L5", "L7"} for a TM/ETM+-only panel;
# ``None`` = every mission available in the year. The training data is 52.8 % L5 / 47.2 % L7
# with ~530 OLI obs of 5.65 M, so restricting to TM/ETM+ removes the radiometric transfer
# risk at the cost of SLC-off L7 after May 2003 (docs/DATA.md §7.1).
MISSION_FILTER: set[str] | None = None


def missions_for_year(year: int,
                      only: set[str] | None = None
                      ) -> dict[str, tuple[str, dict[str, str]]]:
    """Missions with data in ``year``, optionally restricted to ``only``.

    OLI (L8/L9) radiometry differs from TM/ETM+; harmonisation happens downstream at assembly
    so this raw store stays raw.
    """
    m: dict[str, tuple[str, dict[str, str]]] = {}
    if year <= 2013:
        m["L5"] = (L5, _TM_BANDS)
    if year >= 1999:
        m["L7"] = (L7, _TM_BANDS)
    if year >= 2013:
        m["L8"] = (L8, _OLI_BANDS)
    if year >= 2021:
        m["L9"] = (L9, _OLI_BANDS)
    only = only if only is not None else MISSION_FILTER
    if only is not None:
        m = {k: v for k, v in m.items() if k in only}
    return m


def _clear_mask(img):
    """QA_PIXEL bits 1-4 clear + no radiometric saturation."""
    qa = img.select("QA_PIXEL")
    clear = (qa.bitwiseAnd(1 << 1).eq(0)
             .And(qa.bitwiseAnd(1 << 2).eq(0))
             .And(qa.bitwiseAnd(1 << 3).eq(0))
             .And(qa.bitwiseAnd(1 << 4).eq(0)))
    sat = img.select("QA_RADSAT").eq(0)
    return clear.And(sat)


def masked_collection(year: int, region):
    """Merged, harmonised, QA-masked collection for one calendar year over ``region``.

    Each image keeps the 6 harmonised raw-DN bands, fully masked to clear pixels, plus a
    ``mission`` int band; ``doy``/``yr`` arrive per-image at sample time.
    """
    ee = _ee
    parts = []
    for name, (cid, bmap) in missions_for_year(year).items():
        mnum = int(name[1:])

        def _prep(img, bmap=bmap, mnum=mnum):
            mask = _clear_mask(img)
            out = img.select(list(bmap)).rename(list(bmap.values()))
            out = out.addBands(ee.Image.constant(mnum).toInt16().rename("mission"))
            return (out.updateMask(mask)
                       .copyProperties(img, ["system:time_start"]))

        coll = (ee.ImageCollection(cid)
                .filterBounds(region)
                .filterDate(ee.Date.fromYMD(year, 1, 1),
                            ee.Date.fromYMD(year + 1, 1, 1))
                .map(_prep))
        parts.append(coll)
    merged = parts[0]
    for p in parts[1:]:
        merged = merged.merge(p)
    return merged


def _to_fc(chunk: gpd.GeoDataFrame):
    ee = _ee
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry(geom.__geo_interface__), {"cid": cid})
        for cid, geom in zip(chunk["COD_PREDIO"], chunk.geometry)
    ])


# --- Fault tolerance: retry transient errors; split a chunk the server says is too big ---
# deterministic "computation too big" errors — retrying is useless, split the chunk
_SPLIT_MSGS = ("computation timed out", "user memory limit exceeded", "too many pixels",
               "computation is too large", "request payload size")
# transient server/network errors — worth a backoff + retry. "too many concurrent
# aggregations" and "restricted mode" are GEE throttles, not failures: the same request
# succeeds a minute later. Missing them once killed a running S2 extraction at 144/215.
_TRANSIENT_MSGS = ("timed out", "timeout", "rate limit", "too many requests", "quota",
                   "unavailable", "internal error", "backend", "connection", "reset by peer",
                   "broken pipe", "bad gateway", "429", "502", "503",
                   "too many concurrent", "concurrent aggregations", "restricted mode",
                   "concurrency limit")


def _is_split_error(e: Exception) -> bool:
    return any(s in str(e).lower() for s in _SPLIT_MSGS)


# Wall-clock ceiling for a single GEE call, in seconds (a dense pixel chunk can take
# minutes). See `_retry` for why this exists.
CHUNK_DEADLINE_S = 900.0


class ChunkTimeout(TimeoutError):
    """A GEE call blew its wall-clock deadline without raising anything itself."""


def _call_with_deadline(fn: Callable, deadline_s: float):
    """Run ``fn`` on a **daemon** thread and give up on the result after ``deadline_s``.

    Hand-rolled rather than a ``ThreadPoolExecutor`` for the daemon flag: executor threads
    are non-daemon and joined by an ``atexit`` hook, and a thread parked in a hung GEE
    socket read is never joinable. Getting this wrong once left five panel workers at 0 %
    CPU for ~3 h in interpreter shutdown, data already complete. A daemon thread is
    abandoned at exit.
    """
    box: dict[str, object] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller's thread
            box["error"] = exc

    t = threading.Thread(target=target, name="gee-chunk", daemon=True)
    t.start()
    t.join(deadline_s)
    if t.is_alive():
        raise ChunkTimeout(
            f"no response from GEE in {deadline_s:.0f}s — treating the hang as a "
            f"transient error and retrying")
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box.get("value")


def _retry(fn: Callable, tries: int = 5, base_wait: float = 15.0,
           deadline_s: float | None = CHUNK_DEADLINE_S):
    """Run ``fn`` under a wall-clock deadline; back off and retry transient failures.

    Split-class errors raise immediately (the chunk runner halves the chunk instead).

    The deadline is the point of this function, not the retries. Silent GEE hangs are a
    recurring measured failure here (three in one run, one 13.4 h) where the process stays
    alive with no output or exception, so neither ``socket.setdefaulttimeout`` nor the
    backoff fires. Running the call on a daemon thread and giving up on the result converts a
    hang into a ``ChunkTimeout``, which is transient-classified and retried. The abandoned
    thread leaks but is a daemon, so the interpreter drops it at exit.
    """
    for attempt in range(tries):
        try:
            if deadline_s is None:
                return fn()
            return _call_with_deadline(fn, deadline_s)
        except Exception as e:  # noqa: BLE001 — ee raises plain EEException/socket errors
            msg = str(e).lower()
            if _is_split_error(e) or attempt == tries - 1 \
                    or not (isinstance(e, ChunkTimeout)
                            or any(s in msg for s in _TRANSIENT_MSGS)):
                raise
            wait = base_wait * 2**attempt
            print(f"    transient GEE error ({str(e)[:120]}) — retry {attempt + 1}/"
                  f"{tries - 1} in {wait:.0f}s", flush=True)
            time.sleep(wait)


def _run_chunk(fn: Callable[[gpd.GeoDataFrame, int], pd.DataFrame],
               chunk: gpd.GeoDataFrame, year: int, min_size: int = 5) -> pd.DataFrame:
    """Run one chunk with retries; on a "too big" error, recursively halve the chunk."""
    try:
        return _retry(lambda: fn(chunk, year))
    except Exception as e:  # noqa: BLE001
        if _is_split_error(e) and len(chunk) > min_size:
            mid = len(chunk) // 2
            print(f"    chunk too big for GEE ({str(e)[:80]}) — "
                  f"splitting {len(chunk)} -> {mid}+{len(chunk) - mid}", flush=True)
            return pd.concat([_run_chunk(fn, chunk.iloc[:mid], year, min_size),
                              _run_chunk(fn, chunk.iloc[mid:], year, min_size)],
                             ignore_index=True)
        raise


def _chunk_id(chunk: gpd.GeoDataFrame) -> str:
    """Content-addressed chunk name: same parcels+year -> same file, so resuming survives a
    change in parcel set, ordering or chunk_size."""
    import hashlib
    key = ",".join(sorted(chunk["COD_PREDIO"].astype(str)))
    return hashlib.md5(key.encode()).hexdigest()[:10]


def _max_gap(presence: np.ndarray) -> int:
    """Longest run of consecutive absent months in a 12-long 0/1 vector."""
    best = cur = 0
    for v in presence:
        cur = cur + 1 if v < 0.5 else 0
        best = max(best, cur)
    return best


# --- Stage 1 — coverage ---
def coverage_chunk(chunk: gpd.GeoDataFrame, year: int) -> pd.DataFrame:
    """n_valid_obs (best-pixel clear count) + monthly presence -> max_gap, per parcel."""
    ee = _ee
    region = ee.Geometry.Rectangle(list(chunk.total_bounds))
    coll = masked_collection(year, region)
    val = coll.map(lambda im: ee.Image.constant(1).float()
                   .updateMask(im.select("B").mask())
                   .rename("val").copyProperties(im, ["system:time_start"]))
    base = ee.ImageCollection(
        [ee.Image.constant(0).float().rename("val").updateMask(ee.Image.constant(0))])

    # A year with zero acquisitions yields a band-less count image, and `unmask` then fails
    # ("one image has no bands..."). Real for the early-1990s Piura panel years, so merge the
    # zero base image first — the count is then a legitimate 0.
    obs = (ee.ImageCollection(val).select("val").merge(base).count()
           .unmask(0).rename("valid_obs"))
    months = []
    for m in range(1, 13):
        m0 = ee.Date.fromYMD(year, m, 1)
        months.append(ee.ImageCollection(val).filterDate(m0, m0.advance(1, "month"))
                      .select("val").merge(base).sum().unmask(0).gt(0)
                      .rename(f"m{m:02d}"))
    stack = obs.addBands(ee.Image.cat(months))
    red = stack.reduceRegions(collection=_to_fc(chunk), reducer=ee.Reducer.max(),
                              scale=30, tileScale=2).getInfo()
    rows = []
    for f in red["features"]:
        p = f["properties"]
        pres = np.array([p.get(f"m{m:02d}", 0) or 0 for m in range(1, 13)], float)
        rows.append({"COD_PREDIO": p["cid"], "year": year,
                     "n_valid_obs": int(p.get("valid_obs", 0) or 0),
                     "max_gap": _max_gap(pres)})
    return pd.DataFrame(rows)


# Largest bbox (deg²) a chunk may span. The bbox becomes the `ee.Geometry.Rectangle` every
# `filterBounds` runs against, so extent drives cost and timeout risk. ~4 deg² is roughly a
# large Peruvian department.
MAX_CHUNK_BBOX_DEG2 = 4.0


def _chunk_todo(parcels: gpd.GeoDataFrame, chunk_size: int,
                max_bbox: float = MAX_CHUNK_BBOX_DEG2) -> list:
    """``[(year, i, chunk), …]`` packed so each chunk is both full AND geographically tight.

    Grouping by year alone was fine for Piura but not nationally: one label year draws from
    all 14 departments, and lon/lat sorting then yields chunks up to 110 deg² and
    unaffordable. Splitting strictly by ``(year, dept)`` overshoots the other way (89 of 204
    groups under 50 parcels). So departments are visited in longitude order and packed
    greedily, cutting a chunk early whenever the next department would push its bbox past
    ``max_bbox``; within a department a Hilbert curve keeps consecutive chunks contiguous.
    """
    todo: list = []
    i = 0
    has_dept = "dept" in parcels.columns
    for year, ygrp in parcels.groupby("year", sort=True):
        if has_dept:
            order = ygrp.groupby("dept").centroid_lon.mean().sort_values().index
            blocks = [ygrp[ygrp.dept == d] for d in order]
        else:
            blocks = [ygrp]
        blocks = [_hilbert(b) for b in blocks]

        buf: list = []
        for block in blocks:
            for j in range(0, len(block), chunk_size):
                piece = block.iloc[j:j + chunk_size]
                cand = pd.concat(buf + [piece]) if buf else piece
                if buf and (len(cand) > chunk_size or _bbox_area(cand) > max_bbox):
                    todo.append((int(year), i, pd.concat(buf)))
                    i += 1
                    buf = [piece]
                else:
                    buf.append(piece)
                if len(pd.concat(buf)) >= chunk_size:
                    todo.append((int(year), i, pd.concat(buf)))
                    i += 1
                    buf = []
        if buf:
            todo.append((int(year), i, pd.concat(buf)))
            i += 1
    return todo


def _hilbert(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if len(g) < 2:
        return g
    try:
        return g.iloc[np.argsort(g.hilbert_distance().values)]
    except Exception:  # noqa: BLE001 — older geopandas: fall back to lon/lat
        return g.sort_values(["centroid_lon", "centroid_lat"])


def _bbox_area(g: gpd.GeoDataFrame) -> float:
    b = g.total_bounds
    return float((b[2] - b[0]) * (b[3] - b[1]))


def run_coverage(parcels: gpd.GeoDataFrame | None = None, chunk_size: int = 400,
                 out: Path | None = None, max_chunks: int | None = None,
                 years: list[int] | None = None) -> pd.DataFrame:
    """Stage 1 over all parcels (or a provided subset), chunked + resumable.

    ``max_chunks`` stops after N *newly computed* chunks (the combine below still runs on
    everything on disk — the partial-run path). ``years`` restricts to those crop years.
    """
    init_ee()
    out = out or f_coverage()
    if parcels is None:
        parcels = gpd.read_parquet(f_parcels())
    if years is not None:
        parcels = parcels[parcels["year"].isin(years)]
    chunk_dir = out.parent / "coverage_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    todo = _chunk_todo(parcels, chunk_size)
    print(f"stage 1 coverage: {len(parcels):,} parcels -> {len(todo)} chunks")

    done = 0
    for n, (year, _i, chunk) in enumerate(todo):
        f = chunk_dir / f"cov_{year}_{_chunk_id(chunk)}.parquet"
        if f.exists():
            continue
        df = _run_chunk(coverage_chunk, chunk, year)
        df.to_parquet(f, index=False)
        print(f"  [{n + 1}/{len(todo)}] {year}: {len(df)} parcels", flush=True)
        done += 1
        if max_chunks is not None and done >= max_chunks:
            print(f"  max_chunks={max_chunks} reached — stopping (resumable)")
            break

    # Combine only the requested years, dedupe on (COD_PREDIO, year). Deduping on
    # COD_PREDIO alone collapses a multi-year panel to one row per parcel — it produced
    # identical "coverage" for 1995/1996/2005 during the panel timing probe.
    globs = [f"cov_{y}_*.parquet" for y in years] if years else ["cov_*.parquet"]
    files = sorted({f for g in globs for f in chunk_dir.glob(g)})
    cov = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    key = ["COD_PREDIO", "year"] if "year" in cov.columns else ["COD_PREDIO"]
    cov = cov.drop_duplicates(key)
    cov.to_parquet(out, index=False)
    print(f"coverage.parquet: {len(cov):,} parcels "
          f"(median n_valid_obs {cov.n_valid_obs.median():.0f})")
    return cov


# --- Stage 2 — raw dated pixels (survivors only) ---
def _fetch_all_features(fc) -> list[dict]:
    """Paginated computeFeatures fetch — plain ``getInfo`` caps at 5000 elements."""
    ee = _ee
    rows, token = [], None
    while True:
        params = {"expression": fc, "pageSize": 5000}
        if token:
            params["pageToken"] = token
        page = _retry(lambda: ee.data.computeFeatures(params))
        rows.extend(f["properties"] for f in page.get("features", []))
        token = page.get("nextPageToken")
        if not token:
            return rows


def pixels_chunk(chunk: gpd.GeoDataFrame, year: int) -> pd.DataFrame:
    """Every clear pixel observation for a chunk of parcels in one crop year."""
    ee = _ee
    region = ee.Geometry.Rectangle(list(chunk.total_bounds))
    coll = masked_collection(year, region)
    fc = _to_fc(chunk)

    def _sample(img):
        d = ee.Date(img.get("system:time_start"))
        img = img.addBands(ee.Image.pixelLonLat())
        img = img.addBands(ee.Image.constant(d.getRelative("day", "year").add(1))
                           .toInt16().rename("doy"))
        return img.sampleRegions(collection=fc, properties=["cid"], scale=30,
                                 tileScale=4, geometries=False)

    rows = _fetch_all_features(coll.map(_sample).flatten())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={"cid": "COD_PREDIO",
                                            "longitude": "lon", "latitude": "lat"})
    df["year"] = year
    keep = ["COD_PREDIO", "year", "doy", "mission", "lon", "lat",
            "B", "G", "R", "NIR", "SWIR1", "SWIR2"]
    return df[[c for c in keep if c in df.columns]]


def run_pixels(parcels: gpd.GeoDataFrame | None = None, chunk_size: int = 40,
               only_quality_ok: bool = True, max_chunks: int | None = None,
               years: list[int] | None = None, feat_dir: Path | None = None) -> None:
    """Stage 2 raw pixel export for gate survivors, chunked + resumable.

    ``max_chunks`` stops after N newly computed chunks; the per-year combine always runs on
    every chunk on disk. ``feat_dir`` defaults to the shared training store; the panel points
    it at ``FEAT/"panel"`` so an experimental run never mutates the audited artifact.
    """
    init_ee()
    feat_dir = feat_dir or feat()
    if parcels is None:
        parcels = gpd.read_parquet(f_parcels())
    if only_quality_ok and "quality_ok" in parcels:
        parcels = parcels[parcels["quality_ok"] == True]  # noqa: E712  (NA-safe)
    if years is not None:
        parcels = parcels[parcels["year"].isin(years)]
    chunk_dir = feat_dir / "pixels_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    todo = _chunk_todo(parcels, chunk_size)
    print(f"stage 2 pixels: {len(parcels):,} survivors -> {len(todo)} chunks")

    done = 0
    for n, (year, _i, chunk) in enumerate(todo):
        f = chunk_dir / f"px_{year}_{_chunk_id(chunk)}.parquet"
        if f.exists():
            continue
        df = _run_chunk(pixels_chunk, chunk, year)
        df.to_parquet(f, index=False)
        print(f"  [{n + 1}/{len(todo)}] {year}: {len(df):,} pixel-obs", flush=True)
        done += 1
        if max_chunks is not None and done >= max_chunks:
            print(f"  max_chunks={max_chunks} reached — stopping (resumable)")
            break

    # combine every chunk on disk into per-year stores (dedup guards against a parcel in two
    # chunk files after a boundary change)
    years_on_disk = sorted({f.stem.split("_")[1] for f in chunk_dir.glob("px_*.parquet")})
    for year in years_on_disk:
        files = sorted(chunk_dir.glob(f"px_{year}_*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df.drop_duplicates(subset=[c for c in ("COD_PREDIO", "doy", "lon", "lat",
                                                    "mission") if c in df.columns])
        df.to_parquet(feat_dir / f"pixels_{year}.parquet", index=False)
        print(f"pixels_{year}.parquet: {len(df):,} pixel-obs, "
              f"{df.COD_PREDIO.nunique():,} parcels")
