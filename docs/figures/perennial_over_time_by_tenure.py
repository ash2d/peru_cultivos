#!/usr/bin/env python3
"""Reproduce `perennial_over_time_by_tenure.png` — standalone, matplotlib only.

    python docs/figures/perennial_over_time_by_tenure.py [out.png]

**Perennial share of cropped parcels at three observations of the same land, by tenure at the
PETT declaration.** Written up in `docs/RESULTS.md` §8.6–8.7.

⚠️ **The three points are three instruments, not one series.** Each arm is drawn from **its
own baseline**, not continued off the previous endpoint — one joined line would invent a
trend no instrument measured.

| x | observation | how | n |
|---|---|---|---|
| 2001 | PETT declared crop | farmer → titling clerk | — |
| 2012 | CENAGRO question 024 | farmer → census enumerator | 63,766 |
| 2025 | Sentinel-2 / Esri | human photo-interpretation | 214 / 364 |

`WOODY_NON_CROP` is the 2019+ codebook's hardest call and is left unmapped, so the imagery
endpoint is reported **both ways** — excluded (circles) and read as perennial (triangles).
The two readings use different parcel sets (214 vs 364) and so carry different baselines;
both are drawn.

⚠️ **Every share is weighted.** The census arm is post-stratified to the national PETT
population on department × declared class (the name link over-selects perennial parcels,
16.6 % vs 9.9 %). The imagery arm uses the campaign design weight N_h/n_h — unweighted its
perennial share is ~3× the population's. CIs use Kish's effective n, (Σw)²/Σw².

The numbers below are produced by
`uv run python -m crop_classifier.cli allperu cenagro-shift`, which re-derives them from
`data/processed/cenagro/national_panel.parquet` and warns if this file has drifted.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- the values, as (share %, 95 % half-width in pp) ---
# Regenerate with `allperu cenagro-shift`; it checks these against the data.
VALUES: dict[str, dict] = {
    "INSCRITO": {
        # census panel, post-stratified to the national PETT population
        "cen": [(15.07, 14.71, 15.44), (24.23, 23.80, 24.67)], "n_cen": 40161,
        # imagery arm, design-weighted, WOODY_NON_CROP excluded   (n_eff 27.6)
        "s2": [(10.53, 3.59, 27.10), (13.86, 5.42, 31.13)], "n_s2": 82,
        # imagery arm, design-weighted, WOODY_NON_CROP read as perennial   (n_eff 40.6)
        "woody": [(20.67, 11.03, 35.38), (39.61, 26.09, 54.91)], "n_woody": 182,
    },
    "NO INSCRITO": {
        "cen": [(19.20, 18.68, 19.73), (30.49, 29.89, 31.11)], "n_cen": 23605,
        "s2": [(13.91, 6.26, 28.10), (18.17, 9.09, 33.01)], "n_s2": 132,
        "woody": [(21.54, 12.20, 35.15), (27.22, 16.61, 41.25)], "n_woody": 182,
    },
}


def _err(level) -> list[list[float]]:
    """(pct, lo, hi) -> matplotlib's asymmetric [[below], [above]].

    Accepts either the 3-tuple baked in above or the dict `cenagro_shift.figure_values()`
    returns, so the pipeline can pass fresh numbers straight through.
    """
    p, lo, hi = ((level["pct"], level["lo"], level["hi"])
                 if isinstance(level, dict) else level)
    return [[max(p - lo, 0.0)], [max(hi - p, 0.0)]]


def _pct(level) -> float:
    return level["pct"] if isinstance(level, dict) else level[0]


# --- design tokens ---
# Slots 1–2 of the validated categorical theme. Checked, not eyeballed: all-pairs CVD
# ΔE 24.7 (target 8), normal-vision ΔE 33.6 (floor 15), both ≥ 3:1 on the light surface.
TENURE_COLOR = {"INSCRITO": "#2a78d6", "NO INSCRITO": "#eb6834"}
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e8e7e2"

T_PETT, T_CEN, T_S2 = 2001, 2012, 2025
DODGE = 1.45          # x-offset for the WOODY arm — legibility only, not a different date
DEFAULT_OUT = Path(__file__).with_suffix(".png")


def draw(values: dict[str, dict] | None = None, path: Path | str | None = None) -> Path:
    """Draw the figure. Returns the path written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    V = values or VALUES
    path = Path(path or DEFAULT_OUT)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11.4, 7.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes((0.075, 0.225, 0.70, 0.585))
    ax.set_facecolor(SURFACE)

    for tenure, s in V.items():
        col = TENURE_COLOR[tenure]

        # census panel: solid, the high-precision leg. CIs are ±0.4 pp, hence invisible.
        ax.plot([T_PETT, T_CEN], [_pct(s["cen"][0]), _pct(s["cen"][1])],
                color=col, lw=2.4, zorder=5)
        for x, lv in ((T_PETT, s["cen"][0]), (T_CEN, s["cen"][1])):
            ax.errorbar([x], [_pct(lv)], yerr=_err(lv), color=col, marker="o", ms=8.5,
                        mfc=col, mec=SURFACE, mew=1.6, capsize=3, elinewidth=1.2, zorder=5)

        # imagery arm, WOODY excluded: dotted, from its OWN lower baseline
        ax.plot([T_PETT, T_S2], [_pct(s["s2"][0]), _pct(s["s2"][1])],
                color=col, lw=1.5, ls=(0, (1.4, 2.4)), alpha=.85, zorder=2)
        ax.errorbar([T_PETT], [_pct(s["s2"][0])], yerr=_err(s["s2"][0]), color=col,
                    marker="o", ms=7, mfc=SURFACE, mec=col, mew=1.7, capsize=2.5,
                    elinewidth=1.0, alpha=.7, zorder=3)
        ax.errorbar([T_S2], [_pct(s["s2"][1])], yerr=_err(s["s2"][1]), color=col,
                    marker="o", ms=8.5, mfc=SURFACE, mec=col, mew=2.0, capsize=3,
                    elinewidth=1.2, zorder=5)

        # imagery arm, WOODY read as perennial: its own baseline too, dodged in x
        xa, xb = T_PETT + DODGE, T_S2 + DODGE
        ax.plot([xa, xb], [_pct(s["woody"][0]), _pct(s["woody"][1])],
                color=col, lw=1.2, ls=(0, (1, 2.2)), alpha=.65, zorder=2)
        ax.errorbar([xa], [_pct(s["woody"][0])], yerr=_err(s["woody"][0]), color=col,
                    marker="^", ms=7, mfc=SURFACE, mec=col, mew=1.5, capsize=2.5,
                    elinewidth=1.0, alpha=.6, zorder=3)
        ax.errorbar([xb], [_pct(s["woody"][1])], yerr=_err(s["woody"][1]), color=col,
                    marker="^", ms=9, mfc=SURFACE, mec=col, mew=1.9, capsize=3,
                    elinewidth=1.2, alpha=.9, zorder=5)

    # direct labels at the census line ends — ink not series colour; the marker to their
    # left carries identity
    for tenure, s in V.items():
        ax.annotate(tenure, xy=(T_CEN, _pct(s["cen"][1])), xytext=(9, -3),
                    textcoords="offset points", fontsize=10.5, color=INK,
                    fontweight="bold", ha="left", va="center")

    ax.set_xlim(1998.5, 2030.5)
    ax.set_ylim(0, 60)
    ax.set_yticks(range(0, 61, 10))
    ax.set_yticklabels([f"{v} %" for v in range(0, 61, 10)], fontsize=10, color=INK2)
    ax.set_xticks([T_PETT, T_CEN, T_S2])
    ax.set_xticklabels(["2001", "2012", "2025"], fontsize=11.5, color=INK)
    ax.grid(axis="y", color=GRID, lw=.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.axhline(0, color="#d5d4ce", lw=1.0, zorder=1)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", length=0, pad=8)

    for x, lab in [(T_PETT, "PETT declaration\ndeclared crop"),
                   (T_CEN, "CENAGRO census\ndeclared crop"),
                   (T_S2, "Sentinel-2 / Esri\nphoto-interpreted")]:
        ax.annotate(lab, xy=(x, 0), xytext=(0, -38), textcoords="offset points",
                    ha="center", fontsize=9, color=INK3, annotation_clip=False)

    fig.text(0.075, 0.958, "Perennial share of cropped parcels, by tenure at titling",
             fontsize=15.5, color=INK, fontweight="bold", ha="left", va="top")
    fig.text(0.075, 0.908,
             "Titled parcels start lower and gain LESS: +9.2 pp to 2012 against untitled "
             "parcels' +11.3 pp.\nBy cadastral area that gap disappears (+10.7 vs +11.0). "
             "Descriptive — title is not randomly assigned.",
             fontsize=10, color=INK2, ha="left", va="top", linespacing=1.5)

    n_cen = sum(s["n_cen"] for s in V.values())
    n_s2 = sum(s["n_s2"] for s in V.values())
    n_wd = sum(s["n_woody"] for s in V.values())
    handles = [
        Line2D([], [], color=TENURE_COLOR["INSCRITO"], lw=2.4, marker="o", ms=8.5,
               mec=SURFACE, label="INSCRITO (titled at declaration)"),
        Line2D([], [], color=TENURE_COLOR["NO INSCRITO"], lw=2.4, marker="o", ms=8.5,
               mec=SURFACE, label="NO INSCRITO (untitled)"),
        Line2D([], [], color=INK3, lw=2.4, marker="o", ms=8.5, mec=SURFACE,
               label=f"census panel, post-stratified to the nation  (n={n_cen:,})"),
        Line2D([], [], color=INK3, lw=1.5, ls=(0, (1.4, 2.4)), marker="o", ms=8.5,
               mfc=SURFACE, mew=2.0,
               label=f"imagery arm, WOODY_NON_CROP excluded  (n={n_s2:,})"),
        Line2D([], [], color=INK3, lw=1.2, ls=(0, (1, 2.2)), marker="^", ms=9,
               mfc=SURFACE, mew=1.9,
               label=f"imagery arm, WOODY_NON_CROP read as perennial  (n={n_wd:,})"),
    ]
    # a surface-coloured backing so the y gridlines do not run through the legend text
    leg = ax.legend(handles=handles, loc="upper left", fontsize=9, handlelength=2.6,
                    labelspacing=.5, bbox_to_anchor=(0.0, 1.03), frameon=True,
                    facecolor=SURFACE, edgecolor="none", framealpha=1.0, borderpad=0.6)
    leg.set_zorder(6)
    for txt in leg.get_texts():
        txt.set_color(INK2)

    fig.text(0.075, 0.113,
             "Whiskers are 95 % CIs on Kish's effective n. Both imagery readings carry their "
             "own baseline because each restricts to a different\nparcel set; the triangles "
             "are offset in x for legibility, not dated differently. Neither imagery reading "
             "separates itself from zero,\nfrom the other, or from the census estimate — "
             "one ambiguous class moves the 2025 point by more than the effect being measured.",
             fontsize=8.5, color=INK3, ha="left", va="top", linespacing=1.6)

    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


if __name__ == "__main__":
    out = draw(path=sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"wrote {out}")
