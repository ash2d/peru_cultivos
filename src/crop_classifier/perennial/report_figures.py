"""Figures for the summary report (reports/REPORT.md).

Two questions these exist to answer, both of which are prior to any model:

1. **Are the classes separable at all in the imagery?** Per-class seasonal profiles of the
   *actual model inputs* (the per-date tensors), so "can a human tell perennial from annual?"
   is answered with the same data the classifier sees, not a re-derivation.
2. **What did the 1997-98 El Nino do to the imagery?** Monthly panel-wide index trajectories
   for the El Nino years against the envelope of normal years.

Both are read straight from the feature store — no model is loaded, so this module imports
neither torch nor lightgbm and is safe to run in any process.

Bands/indices are the 11 assembly channels in ``features.indices.CHANNELS`` order:
``B G R NIR SWIR1 SWIR2 NDVI EVI NDWI NDMI BSI``.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from crop_classifier.paths import FEAT, proc
from crop_classifier.perennial.figures import C_FLAG, C_GRID, C_INK, C_SERIES

MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

# Perennial/annual grouping of the 12 crop classes, for colouring the small multiples.
# Verified against data/processed/perennial/class_lexicon_resolved.csv, not assumed. Note
# CAÑA DE AZUCAR is ANNUAL by an explicit `cana_policy` (plan D1) even though it looks
# spectrally perennial-like — that is a deliberate decision and a registered §9.5
# sensitivity arm, not a lexicon slip.
GROUP_12 = {
    "CAFE": "PERENNIAL", "MANGO_LIMON": "PERENNIAL", "PLATANO": "PERENNIAL",
    "ARROZ": "ANNUAL", "MAIZ": "ANNUAL", "ALGODON": "ANNUAL", "FRIJOL": "ANNUAL",
    "TRIGO": "ANNUAL", "ZARANDAJA": "ANNUAL", "CAÑA DE AZUCAR": "ANNUAL",
    "PASTURE": "PASTURE_FALLOW", "FALLOW": "PASTURE_FALLOW",
}
GROUP_COLOR = {"PERENNIAL": "#1baf7a", "ANNUAL": "#eb6834",
               "PASTURE_FALLOW": "#4a3aa7"}


# ------------------------------------------------------------------------------------
# monthly profiles from the per-date tensors
# ------------------------------------------------------------------------------------
def monthly_profiles(tensor_path: Path, channel: str = "NDVI"
                     ) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-parcel monthly mean of one channel. Returns ``(long df, cod_predio)``.

    Averaged **per parcel first**, so a parcel with 20 clear dates does not outweigh one
    with 4. Unobserved parcel-months stay NaN and are never interpolated.
    """
    z = np.load(tensor_path, allow_pickle=True)
    ci = list(z["channels"]).index(channel)
    x, doy, mask = z["X"][:, :, ci], z["doy"], z["mask"]
    month = np.clip((doy.astype(int) - 1) // 30.44, 0, 11).astype(int)

    n = x.shape[0]
    out = np.full((n, 12), np.nan, dtype=np.float32)
    for m in range(12):
        sel = mask & (month == m)
        cnt = sel.sum(axis=1)
        tot = np.where(sel, x, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, m] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return pd.DataFrame(out, columns=range(12)), z["cod_predio"]


def _class_bands(prof: pd.DataFrame, cods: np.ndarray, labels: pd.Series
                 ) -> dict[str, dict]:
    """Per-class monthly median and inter-quartile range **across parcels**.

    The IQR, not a CI of the mean, is what answers "can these be told apart?" — with tens of
    thousands of parcels a CI is invisibly narrow and would imply a separation that the
    parcel-level distributions do not have.
    """
    lab = pd.Series(cods).map(labels).to_numpy()
    out = {}
    for c in pd.unique(lab[pd.notna(lab)]):
        sub = prof[lab == c].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            out[str(c)] = {
                "n": int((lab == c).sum()),
                "median": np.nanmedian(sub, axis=0),
                "p25": np.nanpercentile(sub, 25, axis=0),
                "p75": np.nanpercentile(sub, 75, axis=0),
            }
    return out


def class_profile_data(workspace_proc: Path, channel: str = "NDVI") -> dict[str, dict]:
    """Per-class seasonal profile for one workspace's label set."""
    prof, cods = monthly_profiles(FEAT / "tensor_perdate.npz", channel)
    parcels = gpd.read_parquet(workspace_proc / "modeling_parcels.parquet")
    labels = parcels.set_index("COD_PREDIO")["label"]
    return _class_bands(prof, cods, labels)


# ------------------------------------------------------------------------------------
# figure 1 — 3-class separability
# ------------------------------------------------------------------------------------
def fig_3class_profiles(path: Path, channels=("NDVI", "NDMI", "BSI")) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["ANNUAL", "PASTURE_FALLOW", "PERENNIAL"]
    fig, axes = plt.subplots(1, len(channels), figsize=(4.6 * len(channels), 3.8),
                             squeeze=False)
    stats = {}
    for ax, ch in zip(axes[0], channels):
        d = class_profile_data(proc(), ch)
        stats[ch] = {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                         for kk, vv in v.items()} for k, v in d.items()}
        for c in order:
            if c not in d:
                continue
            col = GROUP_COLOR[c]
            ax.fill_between(range(12), d[c]["p25"], d[c]["p75"], color=col, alpha=0.13,
                            lw=0)
            ax.plot(range(12), d[c]["median"], "-o", ms=4, color=col, lw=2,
                    label=f"{c} (n={d[c]['n']:,})")
        ax.set_xticks(range(12), MONTHS, fontsize=7)
        ax.set_xlabel("month", fontsize=8, color=C_INK)
        ax.set_ylabel(ch, fontsize=9, color=C_INK)
        ax.grid(alpha=0.25, color=C_GRID, lw=0.6)
        ax.tick_params(labelsize=7, colors=C_INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0][0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("3-class seasonal profiles — line = median parcel, band = inter-quartile "
                 "range\n(the bands are what the classifier has to separate)",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return stats


# ------------------------------------------------------------------------------------
# figure 2 — 12-class separability, small multiples
# ------------------------------------------------------------------------------------
def fig_12class_profiles(path: Path, channel: str = "NDVI") -> dict:
    """One panel per crop. Small multiples rather than 12 lines on one axis: twelve
    categorical hues cannot be told apart reliably, and the question is per-class anyway."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = class_profile_data(Path("data/processed"), channel)
    prof, cods = monthly_profiles(FEAT / "tensor_perdate.npz", channel)
    grand = np.nanmedian(prof.to_numpy(dtype=float), axis=0)

    order = sorted(d, key=lambda c: (GROUP_12.get(c, "?"), c))
    ncols = 4
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.3 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for ax, c in zip(axes.ravel(), order):
        grp = GROUP_12.get(c, "?")
        col = GROUP_COLOR.get(grp, C_SERIES)
        ax.plot(range(12), grand, "-", color="#9a9a94", lw=1.4, zorder=1)
        ax.fill_between(range(12), d[c]["p25"], d[c]["p75"], color=col, alpha=0.16, lw=0)
        ax.plot(range(12), d[c]["median"], "-o", ms=3, color=col, lw=2, zorder=3)
        ax.set_title(f"{c}  (n={d[c]['n']:,})", fontsize=8, color=col)
        ax.grid(alpha=0.25, color=C_GRID, lw=0.6)
        ax.tick_params(labelsize=7, colors=C_INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes.ravel()[len(order):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xticks(range(12), MONTHS, fontsize=7)
    for r in range(nrows):
        axes[r][0].set_ylabel(channel, fontsize=8, color=C_INK)

    handles = [plt.Line2D([], [], color=GROUP_COLOR[g], lw=2, label=g)
               for g in ["ANNUAL", "PASTURE_FALLOW", "PERENNIAL"]]
    handles.append(plt.Line2D([], [], color="#9a9a94", lw=1.4,
                              label="all parcels (reference)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(f"12-class seasonal {channel} — median parcel (line) and inter-quartile "
                 f"range (band), against the all-parcel median (grey)\n"
                 f"Panel titles are coloured by the 3-class group each crop maps to.",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                for kk, vv in v.items()} for k, v in d.items()}


# ------------------------------------------------------------------------------------
# figure 3 — the 1997-98 El Nino in the imagery
# ------------------------------------------------------------------------------------
def elnino_year_profiles(years: list[int], channel: str = "NDVI",
                         parcels: set[str] | None = None) -> pd.DataFrame:
    """Panel-wide monthly median of one channel, per year, on a fixed parcel set."""
    from crop_classifier.perennial import panel as P
    rows = []
    for y in years:
        t = P.panel_dirs(y)[1] / "tensor_perdate.npz"
        if not t.exists():
            continue
        prof, cods = monthly_profiles(t, channel)
        if parcels is not None:
            keep = np.isin(cods, list(parcels))
            prof = prof[keep]
        arr = prof.to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            med = np.nanmedian(arr, axis=0)
            n_obs = np.sum(~np.isnan(arr), axis=0)
        for m in range(12):
            rows.append({"year": y, "month": m, "median": med[m], "n": int(n_obs[m])})
    return pd.DataFrame(rows)


def season_medians(years: list[int], channel: str, months: tuple[int, ...],
                   parcels: set[str] | None = None, min_parcels: int = 200
                   ) -> pd.DataFrame:
    """Per-year median over a set of months, per parcel first then across parcels."""
    from crop_classifier.perennial import panel as P
    rows = []
    for y in years:
        t = P.panel_dirs(y)[1] / "tensor_perdate.npz"
        if not t.exists():
            continue
        prof, cods = monthly_profiles(t, channel)
        if parcels is not None:
            prof = prof[np.isin(cods, list(parcels))]
        sub = prof[list(months)].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            per_parcel = np.nanmean(sub, axis=1)
        ok = np.isfinite(per_parcel)
        rows.append({"year": y, "n": int(ok.sum()),
                     "median": float(np.nanmedian(per_parcel)) if ok.sum() else np.nan,
                     "enough": bool(ok.sum() >= min_parcels)})
    return pd.DataFrame(rows)


def fig_elnino(path: Path, normal=(1999, 2010), elnino=(1997, 1998),
               min_parcels: int = 300) -> dict:
    """The 1997-98 El Nino signature in the model's own inputs.

    Piura's catastrophic rains ran **December 1997 to April 1998**, so the anomaly sits in
    *1998*'s early months, not 1997's. What it looks like spectrally is the opposite of
    intuition: the flooded desert **greens**, so NDVI goes *up* while SWIR1 goes sharply
    *down* (SWIR is absorbed by water, so low SWIR1 = wet soil and canopy).

    Two guards make this readable rather than misleading:

    * **Coverage filtering.** Monthly panel coverage is wildly uneven — 1998 April has 13
      parcels with a clear observation against ~3,000 in a good month — so months below
      ``min_parcels`` are dropped and left as visible gaps rather than plotted as noise.
    * **An envelope, not a comparison year.** Normal years are drawn as p10-p90 of their
      yearly monthly medians, so the anomaly is judged against real interannual variation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from crop_classifier.perennial import panel as P
    from crop_classifier.perennial.diagnostics import FLAG_YEARS, balanced_parcels
    bal = set(balanced_parcels())
    norm_years = [y for y in range(normal[0], normal[1] + 1) if y not in FLAG_YEARS]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))
    stats = {}
    for ax, ch, note in [(axes[0], "NDVI", "higher = greener"),
                         (axes[1], "SWIR1", "LOWER = wetter")]:
        df = elnino_year_profiles(sorted(set(norm_years) | set(elnino)), ch, bal)
        df.loc[df["n"] < min_parcels, "median"] = np.nan      # too thin to plot
        piv = df.pivot(index="year", columns="month", values="median")
        env = piv.loc[[y for y in norm_years if y in piv.index]]
        with np.errstate(all="ignore"):
            lo = np.nanpercentile(env, 10, axis=0)
            hi = np.nanpercentile(env, 90, axis=0)
            med = np.nanmedian(env, axis=0)
        ax.fill_between(range(12), lo, hi, color="#9a9a94", alpha=0.28, lw=0,
                        label=f"normal years {norm_years[0]}-{norm_years[-1]} (p10-p90)")
        ax.plot(range(12), med, "-", color="#5c5c57", lw=1.6, label="normal median")
        for y, c, ls in zip(elnino, ["#eda100", C_FLAG], ["--", "-"]):
            if y in piv.index:
                ax.plot(range(12), piv.loc[y], ls, marker="o", ms=5, color=c, lw=2.4,
                        label=f"{y}")
        stats[ch] = {"normal_median": [None if v != v else v for v in med],
                     **{str(y): [None if v != v else v for v in piv.loc[y]]
                        for y in elnino if y in piv.index}}
        ax.axvspan(-0.5, 3.5, color=C_SERIES, alpha=0.07, lw=0)
        ax.annotate("Dec 1997 - Apr 1998\nflood window", (1.5, 0.03),
                    xycoords=("data", "axes fraction"), fontsize=7, ha="center",
                    color=C_INK)
        ax.set_xticks(range(12), MONTHS, fontsize=7)
        ax.set_ylabel(f"{ch}  ({note})", fontsize=9, color=C_INK)
        ax.set_xlabel("month", fontsize=8, color=C_INK)
        ax.grid(alpha=0.25, color=C_GRID, lw=0.6)
        ax.tick_params(labelsize=7, colors=C_INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")

    # third panel: is 1998 an outlier across the WHOLE record, not just vs a few years?
    ax = axes[2]
    sm = season_medians(P.DEFAULT_YEARS, "SWIR1", (0, 1, 2), bal, min_parcels)
    sm = sm[sm["enough"]]
    cols = [C_FLAG if y == 1998 else ("#eda100" if y == 1997 else C_SERIES)
            for y in sm["year"]]
    ax.bar(sm["year"], sm["median"], color=cols, width=0.75)
    ax.axhline(sm.loc[sm.year > 1998, "median"].median(), color="#5c5c57", ls=":", lw=1.2)
    ax.annotate("median of 1999+", (2023, sm.loc[sm.year > 1998, "median"].median()),
                fontsize=7, ha="right", va="bottom", color=C_INK)
    for y, txt in ((1997, "1997\ndriest"), (1998, "1998\nwettest")):
        r = sm[sm.year == y]
        if len(r):
            v = float(r["median"].iloc[0])
            ax.annotate(txt, (y, v + 0.006), ha="center", va="bottom", fontsize=7.5,
                        color=C_FLAG if y == 1998 else "#a06e00", weight="bold")
    ax.set_ylim(0, sm["median"].max() * 1.35)
    ax.set(ylabel="SWIR1, Jan-Mar median (lower = wetter)", xlabel="year")
    ax.set_title("1998 is the wettest Jan-Mar, and 1997 the driest,\n"
                 "of the entire 28-year record", fontsize=8.5)
    ax.grid(axis="y", alpha=0.25, color=C_GRID, lw=0.6)
    ax.tick_params(labelsize=7, colors=C_INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    stats["swir1_jan_mar_by_year"] = sm.to_dict("records")

    fig.suptitle("The 1997-98 El Nino in the model's own inputs (balanced parcel set; "
                 "months with < 300 observed parcels dropped)\n"
                 "Jan-Mar 1998 is far wetter (SWIR1 ~0.14 vs ~0.23) AND far greener "
                 "(NDVI ~0.51-0.61 vs ~0.20-0.46) than any other year — the flooded desert "
                 "blooms.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return stats


# ------------------------------------------------------------------------------------
# figure 4 — flicker vs how much time-invariant information a model holds
# ------------------------------------------------------------------------------------
def fig_flicker_vs_statics(path: Path, rows: list[dict]) -> None:
    """Flicker against static-feature content, with per-year accuracy alongside.

    The point of pairing them: flicker falls a long way across these models while accuracy
    barely moves, which is what identifies the stability as an artefact of time-invariant
    inputs rather than better temporal discrimination.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    lbl = [r["label"] for r in rows]
    x = np.arange(len(rows))
    for ax, key, title, ylim in [
            (axes[0], "flicker", "S5 flicker rate (PERENNIAL parcels)", (0, 1.0)),
            (axes[1], "acc", "S4 accuracy at k = 0", (0, 1.0))]:
        vals = [r[key] for r in rows]
        ax.bar(x, vals, 0.55, color=[r["color"] for r in rows])
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.3f}", (xi, v + 0.02), ha="center", fontsize=8, color=C_INK)
        ax.set_xticks(x, lbl, fontsize=7.5)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.25, color=C_GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].axhline(0.15, color=C_FLAG, ls="--", lw=1.2)
    axes[0].annotate("S5 criterion: flicker < 0.15", (-0.45, 0.17), ha="left",
                     fontsize=7.5, color=C_FLAG, weight="bold")
    fig.suptitle("Time-invariant features buy stability, not skill — flicker falls by half "
                 "across these models while accuracy is flat", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def elnino_signature_collapse(features=("NDVI_p25", "NDVI_amp", "BSI_max")
                              ) -> pd.DataFrame:
    """Class separation on the discriminating features, 1998 imagery vs normal imagery.

    This is the *mechanism* behind the §6.3 confound test. The model has no absolute
    definition of "perennial": it learns thresholds (a higher NDVI floor, a flatter season)
    that work only while the two classes sit far apart on those features. So the quantity is
    the PERENNIAL-minus-ANNUAL gap within each cohort, in pooled-SD units, on the same
    regions, and the question is what the El Nino did to it.

    ⚠️ This is a **between-class separation measured across parcels after the fact**, not
    anything the model computes per parcel — no model in this project uses a neighbour
    feature. Earlier wording here said "contrast with its annual neighbours", which read as
    though adjacent parcels were an input; they are not. Regions are restricted to those
    present in both cohorts so that place is held roughly fixed.
    """
    from crop_classifier.data import FN_LGBM

    p = gpd.read_parquet(proc() / "modeling_parcels.parquet")
    p = p[p["quality_ok"] == True]                                     # noqa: E712
    r98 = set(p.loc[p["year"] == 1998, "region_id"])
    r90 = set(p.loc[p["year"].isin([1999, 2000]), "region_id"])
    sub = p[p["region_id"].isin(r98 & r90) & p["year"].isin([1998, 1999, 2000])]
    m = sub[["COD_PREDIO", "label", "year"]].merge(
        pd.read_parquet(FEAT / FN_LGBM), on="COD_PREDIO")
    m["cohort"] = np.where(m["year"] == 1998, "1998\n(El Nino)", "1999+2000\n(normal)")

    rows = []
    for coh, g in m.groupby("cohort"):
        for f in features:
            sd = g[f].std()
            for cls in ("ANNUAL", "PERENNIAL"):
                v = g.loc[g["label"] == cls, f]
                rows.append({"cohort": coh, "feature": f, "label": cls, "n": len(v),
                             "median": float(v.median()),
                             "p25": float(v.quantile(0.25)),
                             "p75": float(v.quantile(0.75))})
            gap = (g.loc[g["label"] == "PERENNIAL", f].median()
                   - g.loc[g["label"] == "ANNUAL", f].median())
            rows.append({"cohort": coh, "feature": f, "label": "_gap_sd",
                         "median": float(gap / sd), "n": len(g)})
    return pd.DataFrame(rows)


def fig_elnino_mechanism(path: Path) -> pd.DataFrame:
    """Why PERENNIAL recall goes to zero on 1998 imagery: the class separation collapses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = elnino_signature_collapse()
    feats = ["NDVI_p25", "NDVI_amp", "BSI_max"]
    sub = ["floor: does it stay green?", "swing: does it senesce?",
           "bare soil: is it ever exposed?"]
    order = ["1999+2000\n(normal)", "1998\n(El Nino)"]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1))
    for ax, f, s in zip(axes, feats, sub):
        for cls in ("ANNUAL", "PERENNIAL"):
            r = d[(d.feature == f) & (d.label == cls)].set_index("cohort").loc[order]
            y = r["median"].to_numpy()
            ax.plot([0, 1], y, "-o", ms=9, lw=2.5, color=GROUP_COLOR[cls], label=cls,
                    zorder=3)
            for xi, (v, lo, hi) in enumerate(zip(y, r["p25"], r["p75"])):
                ax.plot([xi, xi], [lo, hi], "-", lw=2, alpha=0.35,
                        color=GROUP_COLOR[cls], zorder=2)
        for xi, coh in enumerate(order):
            g = float(d[(d.feature == f) & (d.label == "_gap_sd")
                        & (d.cohort == coh)]["median"].iloc[0])
            ax.annotate(f"gap {g:+.2f} SD", (xi, 0.97), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=8.5,
                        color=C_FLAG if abs(g) < 0.5 else C_INK,
                        weight="bold" if abs(g) < 0.5 else "normal")
        ax.set_xlim(-0.45, 1.45)
        ax.set_xticks([0, 1], order, fontsize=8.5)
        ax.set_ylabel(f, fontsize=9.5, color=C_INK)
        ax.set_title(s, fontsize=8.5, color=C_INK)
        ax.grid(axis="y", alpha=0.25, color=C_GRID, lw=0.6)
        ax.tick_params(labelsize=8, colors=C_INK)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=8.5, frameon=False, loc="lower left")
    fig.suptitle("Why PERENNIAL recall collapses on 1998 imagery: the class separation "
                 "that identifies a perennial disappears\n"
                 "Same regions, same held-out parcels. Dot = class median, bar = "
                 "inter-quartile range. NDVI floor and seasonal swing both stop "
                 "separating the classes; only BSI survives.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return d


def build_all(out_dir: Path | str = "docs/figures") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    print("12-class profiles …")
    stats["profiles_12c_NDVI"] = fig_12class_profiles(out / "profiles_12class_ndvi.png")
    print("3-class profiles …")
    stats["profiles_3c"] = fig_3class_profiles(out / "profiles_3class.png")
    print("El Nino …")
    stats["elnino"] = fig_elnino(out / "elnino_signature.png")
    print("El Nino mechanism …")
    fig_elnino_mechanism(out / "elnino_mechanism.png").to_csv(
        out / "elnino_signature_collapse.csv", index=False)
    print("flicker vs statics …")
    fig_flicker_vs_statics(out / "flicker_vs_statics.png", [
        {"label": "LightGBM\n(+centroid_lat)", "flicker": 0.4282, "acc": 0.7456,
         "color": "#2a78d6"},
        {"label": "LightGBM\n(no centroid_lat)", "flicker": 0.5470, "acc": 0.7515,
         "color": "#7aa9e0"},
        {"label": "LTAE\n(no statics at all)", "flicker": 0.7799, "acc": 0.7124,
         "color": "#eb6834"},
    ])
    with open(out / "profile_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote figures + profile_stats.json to {out}")


if __name__ == "__main__":
    build_all()
