"""Report figures (plan §10). Matplotlib only, Agg backend, no torch/lightgbm imports.

Every time-series figure marks the Landsat mission boundaries, because a step at 1999
(L7 arrives), 2012 (L7 SLC-off only) or 2013 (L8) is a sensor artefact and must never be
read as land-use change. 2012 in particular is flagged rather than interpolated over.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MISSION_EVENTS = {
    1999: "L7 launch",
    2012: "L7 SLC-off only",
    2013: "L8 launch",
    2021: "L9 launch",
}

# Palette slots from the validated categorical set. Only two used: slot 1 (blue) carries the
# single data series, slot 8 (red) is the reserved *status* colour, only on data-quality
# years and always with a label, never colour alone.
C_SERIES = "#2a78d6"
C_FLAG = "#e34948"
C_INK = "#52514e"        # secondary text
C_GRID = "#c9c8c4"

# Mission composition of the L5+L7 panel, and the coverage-degraded years that must be
# flagged in every figure rather than interpolated over (RESULTS.md §6.2/§6.3, measured).
PANEL_ERAS = [("L5 only", 1996, 1998), ("L5 + L7", 1999, 2011), ("L7 only", 2012, 2023)]
PANEL_FLAG_YEARS = {1997: "El Nino", 2009: "thin", 2011: "thin", 2012: "SLC-off"}


def _ax_era_bands(ax, years, label: bool = False) -> None:
    """Shade the mission eras and mark the coverage-degraded years.

    Both are recessive background annotation, not data: eras are alternating neutral
    bands, flagged years a single reserved status colour.
    """
    lo, hi = min(years), max(years)
    for i, (name, e0, e1) in enumerate(PANEL_ERAS):
        if e1 < lo or e0 > hi:
            continue
        a, b = max(e0, lo) - 0.5, min(e1, hi) + 0.5
        if i % 2:
            ax.axvspan(a, b, color="#000000", alpha=0.035, lw=0, zorder=0)
        if label:
            ax.annotate(name, ((a + b) / 2, 1.02), xycoords=("data", "axes fraction"),
                        fontsize=7, ha="center", va="bottom", color=C_INK)
    for y in PANEL_FLAG_YEARS:
        if lo <= y <= hi:
            ax.axvline(y, color=C_FLAG, ls="--", lw=0.9, alpha=0.55, zorder=1)


def drift_panel_figure(drift: pd.DataFrame, path: Path, features: list[str],
                       title: str, ncols: int = 2) -> None:
    """Small multiples of per-year feature drift — one panel per feature (§7.4.1).

    Small multiples not one multi-series axis: the features are on different scales and the
    question ("is there a step at a mission boundary?") is per-feature, so a shared axis would
    only compress it.

    Each panel: panel-wide median + p10-p90 spread, mission eras as alternating bands, the
    four coverage-degraded years dashed. A step at an era boundary that is large relative to
    the year-to-year movement *inside* the eras is a sensor artefact; ``era_steps``
    quantifies it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    have = [f for f in features if f in set(drift["feature"])]
    if not have:
        return
    nrows = int(np.ceil(len(have) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 2.15 * nrows),
                             sharex=True, squeeze=False)
    years = sorted(drift["year"].unique())
    for ax, feat in zip(axes.ravel(), have):
        d = drift[drift["feature"] == feat].sort_values("year")
        _ax_era_bands(ax, years, label=(ax is axes.ravel()[0]))
        ax.fill_between(d["year"], d["p10"], d["p90"], color=C_SERIES, alpha=0.13, lw=0)
        ax.plot(d["year"], d["median"], "-", color=C_SERIES, lw=2, zorder=3)
        flg = d[d["flagged_year"]]
        ax.plot(flg["year"], flg["median"], "o", ms=5, mfc="white", mec=C_FLAG, mew=1.6,
                zorder=4)
        ax.set_ylabel(feat, fontsize=8, color=C_INK)
        ax.grid(alpha=0.25, color=C_GRID, lw=0.6)
        ax.tick_params(labelsize=7, colors=C_INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes.ravel()[len(have):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("year", fontsize=8, color=C_INK)
    handles = [plt.Line2D([], [], color=C_SERIES, lw=2, label="median (band = p10-p90)"),
               plt.Line2D([], [], color=C_FLAG, ls="--", lw=1,
                          marker="o", ms=5, mfc="white", mec=C_FLAG,
                          label="coverage-degraded year (1997, 2009, 2011, 2012)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.012))
    fig.suptitle(title, fontsize=9.5, y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def elnino_confound_figure(result: dict, path: Path) -> None:
    """Per-class recall, 1998 test arm vs the 1999+2000 control arm (RESULTS.md §8).

    The control arm is the point: the bars are interpretable only as a *pair*, since a low
    1998 recall alone can't be told from "1998 parcels are simply harder".
    """
    plt, (fig, ax) = _fig((6.4, 3.6))
    classes = [c for c in result["delta_recall_test_minus_control"]]
    test = [result["test_arm_1998"][c]["recall"] for c in classes]
    ctrl = [result["control_arm_1999_2000"][c]["recall"] for c in classes]
    x = np.arange(len(classes))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, ctrl, w, color=C_SERIES, label="control: held-out 1999+2000")
    ax.bar(x + w / 2 + 0.01, test, w, color="#eb6834", label="test: 1998")
    for xi, (c, t) in enumerate(zip(ctrl, test)):
        ax.annotate(f"{t - c:+.3f}", (xi, max(c, t) + 0.03), ha="center", fontsize=8,
                    color=C_INK)
    ax.set_xticks(x, classes, fontsize=8)
    ax.set(ylim=(0, 1.12), ylabel="recall")
    ax.set_title("train 1999+2000 -> test 1998, region held fixed\n"
                 "(labels = test minus control, the quantity of interest)", fontsize=9)
    ax.grid(axis="y", alpha=0.25, color=C_GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _ax_mission_marks(ax, years) -> None:
    lo, hi = min(years), max(years)
    for y, label in MISSION_EVENTS.items():
        if lo <= y <= hi:
            ax.axvline(y, color="grey", ls=":", lw=0.9, zorder=0)
            ax.annotate(label, (y, 1.005), xycoords=("data", "axes fraction"),
                        rotation=90, fontsize=6, va="bottom", ha="center", color="grey")
    if lo <= 2012 <= hi:
        ax.axvspan(2011.6, 2012.4, color="red", alpha=0.06, zorder=0)


def _fig(figsize=(7.5, 4)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt, plt.subplots(figsize=figsize)


def area_share_figure(share: pd.DataFrame, path: Path,
                      mapbiomas_share: pd.DataFrame | None = None,
                      ci: pd.DataFrame | None = None) -> None:
    """Per-year class share, optionally with Olofsson CIs and the MapBiomas comparison.

    Criterion S6 is agreement in *direction* with MapBiomas, not level: a level offset is
    expected because the class definitions differ.
    """
    plt, (fig, ax) = _fig((8, 4.5))
    years = sorted(share["year"].unique())
    for cls, sub in share.groupby("class"):
        sub = sub.sort_values("year")
        line, = ax.plot(sub["year"], sub["share"], "o-", ms=3, label=cls)
        if ci is not None:
            c = ci[(ci["class"] == cls) & (ci["method"] == "olofsson2014")]
            if len(c):
                c = c.sort_values("year")
                ax.fill_between(c["year"], c["share"] - c["share_ci95"],
                                c["share"] + c["share_ci95"], alpha=0.15,
                                color=line.get_color())
    if mapbiomas_share is not None:
        for cls, sub in mapbiomas_share.groupby("class"):
            ax.plot(sub["year"], sub["share"], "--", lw=1, alpha=0.7,
                    label=f"MapBiomas {cls}")
    _ax_mission_marks(ax, years)
    ax.set(xlabel="year", ylabel="area share", title="class area share by year")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def temporal_transfer_figure(tt: pd.DataFrame, path: Path) -> None:
    """Accuracy / macro-F1 vs distance from the label year (§7.3, criterion S4)."""
    plt, (fig, ax) = _fig((6, 4))
    ax.plot(tt["k"], tt["accuracy"], "o-", label="accuracy")
    ax.plot(tt["k"], tt["macro_f1"], "s-", label="macro-F1")
    k0 = tt.loc[tt["k"] == 0]
    if len(k0):
        base = float(k0["accuracy"].iloc[0])
        ax.axhspan(base - 0.10, base + 0.10, color="green", alpha=0.08,
                   label="S4 band (±0.10 of k=0)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="years from label year (k)", ylabel="score",
           title="temporal transfer (lower bound: real change also decays this)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def feature_drift_figure(drift: pd.DataFrame, path: Path,
                         cols=("NDVI_median", "NDVI_amp", "BSI_max")) -> None:
    """Per-year distribution of the drift-sensitive features (§7.4.1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    have = [c for c in cols if c in drift.columns]
    fig, axes = plt.subplots(len(have), 1, figsize=(7.5, 2.4 * len(have)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, have):
        ax.plot(drift["year"], drift[c], "o-", ms=3)
        _ax_mission_marks(ax, drift["year"])
        ax.set_ylabel(c, fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("year")
    axes[0].set_title("panel-wide feature drift — steps at mission boundaries are "
                      "sensor artefacts", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def transition_matrix_figure(tm: pd.DataFrame, path: Path) -> None:
    plt, (fig, ax) = _fig((5, 4.2))
    m = tm.to_numpy(dtype=float)
    im = ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(tm.columns)), tm.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(tm.index)), tm.index, fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{int(m[i, j]):,}", ha="center", va="center", fontsize=8,
                    color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set(xlabel="to", ylabel="from", title="transitions (min 3 observed years)")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def establishment_ramp_figure(panel_preds: pd.DataFrame, transitions: pd.DataFrame,
                              path: Path, window: int = 5) -> None:
    """Mean P(PERENNIAL) in the ±window years around detected ANNUAL->PERENNIAL changes.

    A new orchard takes 2-3 years to close canopy, so the expected shape is a sigmoid
    ramp. A step means the model is reacting to something other than canopy growth.
    """
    tr = transitions[(transitions["from_class"] == "ANNUAL")
                     & (transitions["to_class"] == "PERENNIAL")]
    if tr.empty:
        return
    p = panel_preds.set_index(["COD_PREDIO", "year"])["prob_PERENNIAL"]
    rows = []
    for _, r in tr.iterrows():
        y0 = r["year_of_change"]
        for k in range(-window, window + 1):
            v = p.get((r["COD_PREDIO"], y0 + k), np.nan)
            if not pd.isna(v):
                rows.append({"k": k, "p": float(v)})
    if not rows:
        return
    d = pd.DataFrame(rows).groupby("k")["p"].agg(["mean", "sem", "size"]).reset_index()
    plt, (fig, ax) = _fig((6, 4))
    ax.errorbar(d["k"], d["mean"], yerr=d["sem"], fmt="o-", capsize=2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="years from detected transition", ylabel="mean P(PERENNIAL)",
           title=f"orchard establishment ramp (n={len(tr):,} transitions)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def transition_map_figure(parcels, transitions: pd.DataFrame, path: Path) -> None:
    """Where ANNUAL->PERENNIAL parcels are. Clearance/planting happens in blocks, so
    a spatially scattered pattern is a warning sign."""
    tr = transitions[(transitions["from_class"] == "ANNUAL")
                     & (transitions["to_class"] == "PERENNIAL")]
    plt, (fig, ax) = _fig((6, 6))
    ax.scatter(parcels["centroid_lon"], parcels["centroid_lat"], s=1, c="lightgrey",
               label="panel parcels")
    sub = parcels[parcels["COD_PREDIO"].isin(tr["COD_PREDIO"])]
    ax.scatter(sub["centroid_lon"], sub["centroid_lat"], s=6, c="darkgreen",
               label=f"ANNUAL->PERENNIAL (n={len(sub):,})")
    ax.set(xlabel="lon", ylabel="lat", title="detected shifts to perennial")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
