"""Esri World Imagery chips for the labelling HTML (s2_labelling_plan.md §6).

Two panels per parcel, no S2 true-colour composite — at 10 m it is worse than Esri and
would only crowd the page:

1. **context** — the whole parcel plus a margin of its surroundings, ~320 px;
2. **zoom** — 200 m across on the same centre, so canopy texture is legible on a parcel
   too large for the context panel to show it.

On both, the target parcel is outlined in a bright colour and its **neighbours in a thin
contrasting outline**. That is not decoration: perennial detection is a *contrast* with
neighbouring annual fields (RESULTS.md §8.2), and drawing the neighbours is the cheapest
way to put that contrast in front of the labeller.

Tiles come through a persistent on-disk cache — ~2,000 chips is a lot of requests and a
re-run should not repeat them.
"""

from __future__ import annotations

import io
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np

# ---- the left panel: the parcel **in its setting** ----------------------------------
# It was previously the parcel at 1.24x its own extent, which is a picture of the parcel
# and almost nothing else. The same crop reads differently in different geographies — rice
# beside a river, orchard on a town edge, pasture against forest — so the panel now pads by
# a fraction of the parcel's own size rather than showing it edge to edge.
#
# Padding is proportional (so a big field is not swamped by margin and a small one is not
# starved of it), capped in absolute metres (so the largest parcel in the draw, 2.1 km
# across, does not demand a 4 km tile mosaic), and floored (so a 66 m parcel — the 5th
# percentile — still gets a real landscape around it rather than 12 % of one).
#
# Realised over the draw: 400 m for everything up to the median parcel, ~530 m at q75,
# ~1.2 km at q95, 2.7 km for the largest. The old panel gave 195 m at the median.
CONTEXT_MIN_M = 400.0
CONTEXT_PAD_FRAC = 0.5      # margin each side, as a fraction of the parcel's own extent
CONTEXT_PAD_MAX_M = 300.0   # ... but never more than this, so tile counts stay bounded

# The right-hand panel is a **zoom-in**, not a second context view. The call the labeller
# has to make is "crowns or no crowns", and on a large parcel the whole-parcel panel is too
# small a scale to see them.
#
# ⚠️ For the ~89 % of the campaign served at 1.2 m this is display magnification, not more
# information: 200 m is ~187 native pixels upsampled to 320. It makes crowns easier to
# *see*; it cannot make them resolvable where the source does not resolve them.
ZOOM_M = 200.0
CHIP_PX = 320
JPEG_QUALITY = 72

C_TARGET = "#ffdd33"      # bright yellow — the parcel being labelled
C_NEIGHBOUR = "#38e0ff"   # thin cyan — everything else with a cadastral boundary

# Esri stops serving tiles at the zoom matching the source resolution and returns a flat
# grey "Map data not yet available" placeholder above it. That placeholder is *not* an
# error — `bounds2img` succeeds and hands back a perfectly valid uniform image — so asking
# for `zoom="auto"` on a 120 m extent silently produced blank chips on the first run.
# Measured over four departments: the placeholder is exactly RGB (205, 205, 205) with
# chroma ~0.03, against >15 for any real scene.
_PLACEHOLDER_CHROMA = 3.0
# probed source resolution -> highest zoom Esri actually serves there
ZOOM_FOR_RES = {"30cm": 19, "60cm": 18, "1.2m": 17}
MIN_ZOOM = 15

_PROVIDER = None


def provider():
    global _PROVIDER
    if _PROVIDER is None:
        import contextily as cx
        _PROVIDER = cx.providers.Esri.WorldImagery
    return _PROVIDER


def set_cache(path: Path | str) -> None:
    """Persistent tile cache. Called once by :func:`render_all`."""
    import contextily as cx
    Path(path).mkdir(parents=True, exist_ok=True)
    cx.set_cache_dir(str(path))


def _merc_scale(lat: float) -> float:
    """Web-Mercator metres per true metre at ``lat`` — 1/cos(lat).

    Mercator y-units are only true metres at the equator. At Peru's latitudes the error is
    ~1.2 %, which is invisible in an outline but would silently mis-size the scale bar, so
    it is applied rather than ignored.
    """
    return 1.0 / math.cos(math.radians(lat))


def context_extent_m(span_m: float, min_m: float = CONTEXT_MIN_M,
                     pad_frac: float = CONTEXT_PAD_FRAC,
                     pad_max_m: float = CONTEXT_PAD_MAX_M) -> float:
    """Width of the context panel, in **true metres**, for a parcel ``span_m`` across.

    ``span + 2 * min(pad_frac * span, pad_max_m)``, floored at ``min_m``. Shared by the
    renderer and the neighbour lookup so the panel can never be wider than the box
    neighbours were fetched from — which would draw a view with unoutlined fields in it and
    read as "this parcel has no neighbours".
    """
    return max(span_m + 2 * min(pad_frac * span_m, pad_max_m), min_m)


def is_placeholder(img: np.ndarray) -> bool:
    """True if a fetched mosaic is Esri's flat grey "not yet available" tile."""
    a = img[..., :3].astype(float)
    chroma = float(np.mean(np.abs(a[..., 0] - a[..., 1])
                           + np.abs(a[..., 1] - a[..., 2])))
    return chroma < _PLACEHOLDER_CHROMA


def _fetch(w, s, e, n, start_zoom: int):
    """Tile mosaic at the highest zoom that is not a placeholder, stepping down."""
    import contextily as cx
    last = None
    for z in range(start_zoom, MIN_ZOOM - 1, -1):
        img, ext = cx.bounds2img(w, s, e, n, zoom=z, source=provider(), ll=False)
        last = (img, ext, z)
        if not is_placeholder(img):
            return last
    return last


def _draw(ax, img, img_ext, view, target_m, neigh_m, scale_bar_m: float | None) -> None:
    """``img_ext`` and ``view`` are both matplotlib extents: ``(left, right, bottom, top)``.

    contextily returns the tile mosaic in that order too; keeping one convention here is
    the whole reason this is a named argument rather than a positional bounds tuple — the
    first version mixed ``(w, s, e, n)`` into it and asked matplotlib for a 10^7-pixel image.
    """
    ax.imshow(img, extent=img_ext, interpolation="bilinear")
    if neigh_m is not None and len(neigh_m):
        neigh_m.boundary.plot(ax=ax, color=C_NEIGHBOUR, linewidth=0.7, alpha=0.85)
    target_m.boundary.plot(ax=ax, color=C_TARGET, linewidth=1.8)
    ax.set_xlim(view[0], view[1])
    ax.set_ylim(view[2], view[3])
    ax.set_axis_off()
    if scale_bar_m:
        bar_m, label = scale_bar_m
        x0, x1, y0, y1 = view
        w = x1 - x0
        bx = x0 + 0.06 * w
        by = y0 + 0.10 * (y1 - y0)
        ax.plot([bx, bx + bar_m], [by, by], color="white", lw=2.6,
                solid_capstyle="butt")
        ax.plot([bx, bx + bar_m], [by, by], color="black", lw=1.0,
                solid_capstyle="butt")
        ax.text(bx, by + 0.022 * (y1 - y0), label, color="white", fontsize=6.5,
                ha="left", va="bottom")


def _nice_bar(extent_m: float) -> tuple[float, str]:
    """A round scale-bar length about a quarter of the panel width, and its label.

    A fixed 100 m bar spans the entire 100 m zoom panel, which reads as a border rather
    than a scale, so the length is picked from the extent instead of hard-coded.
    """
    target = extent_m / 4.0
    for m in (10, 20, 25, 50, 100, 200, 250, 500):
        if m >= target:
            return float(m), f"{m} m"
    return 1000.0, "1 km"


def _new_figure(px: int):
    """A standalone Agg figure + full-bleed axes — deliberately **not** via ``pyplot``.

    ``plt.subplots`` registers the figure in pyplot's global figure manager, which is
    process-wide mutable state and not thread-safe; the renderer here runs several parcels
    at once, so two threads sharing that registry is a race waiting to be hit. Constructing
    ``Figure`` directly and attaching an Agg canvas gives a figure no other thread can see,
    and removes the ``plt.close`` bookkeeping entirely.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(px / 100, px / 100), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    return fig, ax


def _fig_to_jpeg(fig, quality: int = JPEG_QUALITY) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, pad_inches=0, bbox_inches="tight")
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()


def chip_pair(target: gpd.GeoDataFrame, neighbours: gpd.GeoDataFrame | None,
              px: int = CHIP_PX, context_min_m: float = CONTEXT_MIN_M,
              zoom_m: float = ZOOM_M,
              imagery_res: str | None = None) -> tuple[bytes, bytes, dict]:
    """``(context_jpeg, zoom_jpeg, meta)`` for a single-row ``target`` GeoDataFrame.

    * **context** — the whole parcel *and its surroundings*: its extent padded by
      :func:`context_extent_m`, so what the parcel sits in is visible;
    * **zoom** — a fixed ``zoom_m`` window on the same centre, for canopy texture.

    ``imagery_res`` is the probed Esri source resolution ("30cm"/"60cm"/"1.2m"); it sets
    the starting zoom, and the placeholder check steps down from there if the service
    disagrees.
    """
    tm = target.to_crs(3857)
    nm = neighbours.to_crs(3857) if neighbours is not None and len(neighbours) else None
    lat = float(target.to_crs(4326).geometry.representative_point().y.iloc[0])
    k = _merc_scale(lat)

    cx_, cy_ = tm.geometry.representative_point().iloc[0].coords[0]
    b = tm.total_bounds
    span = max(b[2] - b[0], b[3] - b[1])
    half_context = context_extent_m(span / k, min_m=context_min_m) * k / 2
    half_zoom = zoom_m * k / 2

    start = ZOOM_FOR_RES.get(imagery_res or "1.2m", 17)
    out, zooms = [], []
    for half in (half_context, half_zoom):
        extent_m = 2 * half / k
        bar_m, label = _nice_bar(extent_m)
        w, s, e, n = cx_ - half, cy_ - half, cx_ + half, cy_ + half
        img, iext, z = _fetch(w, s, e, n, start)
        zooms.append(z)
        fig, ax = _new_figure(px)
        _draw(ax, img, iext, (w, e, s, n), tm, nm, (bar_m * k, label))
        out.append(_fig_to_jpeg(fig))
    meta = {"zoom_context": zooms[0], "zoom_zoom": zooms[1],
            "context_extent_m": round(2 * half_context / k, 1),
            "zoom_extent_m": round(zoom_m, 1)}
    return out[0], out[1], meta


# ------------------------------------------------------------------------------------
# Batch renderer
# ------------------------------------------------------------------------------------
def render_all(sample: gpd.GeoDataFrame, cadastre: gpd.GeoDataFrame,
               out_dir: Path, cache_dir: Path | None = None,
               workers: int = 3, px: int = CHIP_PX,
               overwrite: bool = False) -> dict:
    """Render both panels for every row of ``sample`` into ``out_dir`` as JPEGs.

    ``cadastre`` is the parcel layer neighbours are taken from — the full national table,
    spatially indexed once. Files are ``<item_id>_context.jpg`` / ``<item_id>_zoom.jpg``
    so a re-run skips what exists.

    ⚠️ **A panel's file name is not versioned, so changing what a panel shows means
    deleting the old files, not overwriting them.** An earlier revision of this campaign
    wrote ``_context.jpg`` for a 600 m fixed-width view; a leftover from it would load
    silently into the page and be *wrong*, not missing. Deleting is the only safe order.

    ``workers`` is deliberately small (2-4): the tiles come from Esri's public World
    Imagery service and this is a research use, so the run is throttled rather than
    parallel-maximal.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        set_cache(cache_dir)

    cad = cadastre.to_crs(4326)
    sindex = cad.sindex
    todo = [r for _, r in sample.iterrows()
            if overwrite or not (out_dir / f"{r.item_id}_zoom.jpg").exists()]
    print(f"chips: {len(todo)} of {len(sample)} parcels to render "
          f"({len(sample) - len(todo)} cached)")

    stats = {"ok": 0, "failed": []}
    zooms: list[dict] = []

    def _one(row):
        try:
            tgt = gpd.GeoDataFrame([row], geometry="geometry", crs=sample.crs).to_crs(4326)
            # Neighbours: every cadastral parcel the context panel can reach, target
            # excluded. The box is derived from `context_extent_m` rather than fixed, so
            # widening the panel can never outrun it — a fixed 600 m box under a 2.7 km
            # panel would leave the outer fields unoutlined, which reads to a labeller as
            # "this parcel has no neighbours" rather than as a missing lookup.
            b = tgt.to_crs(3857).total_bounds
            lat = float(tgt.geometry.representative_point().y.iloc[0])
            span_m = max(b[2] - b[0], b[3] - b[1]) / _merc_scale(lat)
            deg = (context_extent_m(span_m) / 2 + 100.0) / 111_000.0
            c = tgt.geometry.representative_point().iloc[0]
            box = (c.x - deg, c.y - deg, c.x + deg, c.y + deg)
            idx = list(sindex.intersection(box))
            nb = cad.iloc[idx]
            nb = nb[nb["COD_PREDIO"].astype(str) != str(row.COD_PREDIO)]
            ctx, zoom, meta = chip_pair(tgt, nb, px=px,
                                        imagery_res=getattr(row, "imagery_res", None))
            (out_dir / f"{row.item_id}_context.jpg").write_bytes(ctx)
            (out_dir / f"{row.item_id}_zoom.jpg").write_bytes(zoom)
            zooms.append({"item_id": row.item_id, **meta})
            return None
        except Exception as e:  # noqa: BLE001 — one bad chip must not kill the batch
            return (row.item_id, str(e)[:160])

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, res in enumerate(ex.map(_one, todo), 1):
            if res is None:
                stats["ok"] += 1
            else:
                stats["failed"].append(res)
            if n % 50 == 0:
                print(f"  {n}/{len(todo)} rendered ({len(stats['failed'])} failed)",
                      flush=True)

    sizes = [p.stat().st_size for p in out_dir.glob("*.jpg")]
    stats["n_files"] = len(sizes)
    stats["median_bytes"] = int(np.median(sizes)) if sizes else 0
    stats["total_mb"] = round(sum(sizes) / 1e6, 1)
    print(f"chips done: {stats['ok']} rendered, {len(stats['failed'])} failed; "
          f"{stats['n_files']} files, median {stats['median_bytes'] / 1024:.0f} KB, "
          f"{stats['total_mb']} MB total")
    if zooms:
        import pandas as pd
        z = pd.DataFrame(zooms)
        z.to_csv(out_dir / "chip_zooms.csv", index=False)
        stats["zoom_context"] = z["zoom_context"].value_counts().to_dict()
        stats["context_extent_m"] = {
            "median": float(z["context_extent_m"].median()),
            "min": float(z["context_extent_m"].min()),
            "max": float(z["context_extent_m"].max()),
        }
        print("  realised context zoom:", stats["zoom_context"])
        print("  context extent (m):", stats["context_extent_m"])
    if stats["failed"]:
        print("  failures:", stats["failed"][:5])
    return stats
