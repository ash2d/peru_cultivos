"""Report figures for the national strand — the window-pivot failure and the tenure DiD.

Two figures, both read only persisted artefacts (no model, no GEE, no torch/lightgbm), so
they can be regenerated from a clean checkout of `data/processed/`:

* ``window_control_drift.png`` — why a level or a trend cannot be read off this classifier:
  the internal control pool drifts further, and the other way, than the pool being measured.
* ``did_result.png`` — the tenure difference-in-differences: the two arms move together, and
  the corrected effect over the registered sensitivity grid.

Run: ``uv run python -m crop_classifier.allperu.report_figures``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from crop_classifier.perennial.figures import C_FLAG, C_GRID, C_INK, C_SERIES

# Third slot for the second data series; C_FLAG stays reserved for status and never carries data.
C_SERIES2 = "#e8961f"

WINDOW_LABEL = {"W99": "1999–03", "W04": "2004–08", "W09": "2009–13",
                "W14": "2014–18", "W19": "2019–23"}


def _despine(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25, color=C_GRID, lw=0.6)
    ax.tick_params(labelsize=8.5, colors=C_INK)


# --- figure 1: the window pivot's W2 failure ---
def fig_control_drift(path: Path, diag: Path | str | None = None) -> dict:
    """The control pool drifts more than the at-risk pool, in the opposite direction.

    The design reads a trend off the at-risk pool (PETT says ANNUAL) against the
    PETT-PERENNIAL pool as yardstick: land already perennial should stay perennial, so
    movement there is measurement not land-use change. It moves 5.6x further than the signal
    and downward — which closes the design.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from crop_classifier.paths import proc

    p = Path(diag) if diag else proc() / "window_diagnostic_nolat.json"
    d = json.loads(p.read_text())
    at = pd.DataFrame(d["at_risk_series"])
    ct = pd.DataFrame(d["control_series"])
    x = np.arange(len(at))
    lab = [WINDOW_LABEL[w] for w in at["window"]]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    # The W2 gate acts on the thresholded share, so both panels plot that not the mean
    # probability — else the figure and the documented ratio disagree.
    ax = axes[0]
    for df, c, name in ((ct, C_SERIES2, "control pool — PETT says PERENNIAL"),
                        (at, C_SERIES, "at-risk pool — PETT says ANNUAL")):
        ax.plot(x, df["share"], "-o", ms=7, lw=2.4, color=c, label=name, zorder=3)
    ax.set_ylim(0, 0.65)
    ax.set_xticks(x, lab, fontsize=8.5)
    ax.set_ylabel("share classified perennial", fontsize=9, color=C_INK)
    ax.set_title("Levels — the yardstick is falling", fontsize=9.5, color=C_INK)
    ax.legend(fontsize=8, frameon=False, loc="center left")
    _despine(ax)

    ax = axes[1]
    for df, c, name in ((ct, C_SERIES2, "control (should be flat)"),
                        (at, C_SERIES, "at-risk (the signal)")):
        y = (df["share"] - df["share"].iloc[0]) * 100
        ax.plot(x, y, "-o", ms=7, lw=2.4, color=c, label=name, zorder=3)
        ax.annotate(f"{y.iloc[-1]:+.1f} pp", (x[-1], y.iloc[-1]),
                    xytext=(-6, -6 if y.iloc[-1] < 0 else 8), textcoords="offset points",
                    ha="right", va="top" if y.iloc[-1] < 0 else "bottom",
                    fontsize=9.5, weight="bold", color=c)
    ax.axhline(0, color=C_INK, lw=1.0, alpha=0.6)
    ax.set_ylim(-15, 5)
    ax.set_xticks(x, lab, fontsize=8.5)
    ax.set_ylabel("change from 1999–03 (percentage points)", fontsize=9, color=C_INK)
    ax.set_title("Change — the yardstick moves 7x further, the other way",
                 fontsize=9.5, color=C_FLAG)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    _despine(ax)

    fig.suptitle("Why no trend can be read off this classifier (national, 2026-08-10)\n"
                 "Land already perennial in the titling record cannot become 12 pp less "
                 "perennial. That drift is the Landsat archive thinning, not the land.",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"control_slope_per_decade": d["control_slope_per_decade"],
            "at_risk_net_change": d["at_risk_net_change"],
            "control_net_change": float(ct["share"].iloc[-1] - ct["share"].iloc[0])}


# --- figure 2: the tenure DiD result ---
def fig_did_result(path: Path, tag: str = "did2_nolat_aug_yleak10_cohort2004",
                   proc_dir: Path | str | None = None) -> dict:
    """The two arms and the registered sensitivity curve.

    Left is the *raw* group means so the reader sees the arms moving together; the right-panel
    estimate conditions on parcel and department x period, which is why the widening raw gap
    is not in it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from crop_classifier.paths import proc

    d = Path(proc_dir) if proc_dir else proc()
    ser = pd.read_csv(d / f"did2_series_{tag}.csv")
    cur = pd.read_csv(d / f"did2_curve_{tag}.csv")
    gates = json.loads((d / f"did2_gates_{tag}.json").read_text())

    order = ["W99", "W14", "W19"]
    ser = ser[ser["window"].isin(order)].copy()
    ser["k"] = ser["window"].map({w: i for i, w in enumerate(order)})

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    ax = axes[0]
    for t, c, name in ((0.0, C_SERIES2, "control — never registered"),
                       (1.0, C_SERIES, "treated — became registered")):
        g = ser[ser["treat"] == t].sort_values("k")
        ax.plot(g["k"], g["p_mean"], "-o", ms=7, lw=2.4, color=c, label=name, zorder=3)
    ax.set_xticks(range(len(order)), [WINDOW_LABEL[w] for w in order], fontsize=8.5)
    ax.set_ylim(0, 0.22)
    ax.set_ylabel("mean predicted P(perennial)", fontsize=9, color=C_INK)
    ax.set_title("Both arms move together (raw group means)", fontsize=9.5, color=C_INK)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _despine(ax)

    ax = axes[1]
    ax.fill_between(cur["m"], cur["ci_low"], cur["ci_high"], color=C_SERIES, alpha=0.16,
                    lw=0, label="95 % CI")
    ax.plot(cur["m"], cur["coef"], "-o", ms=7, lw=2.4, color=C_SERIES, zorder=3,
            label="corrected effect")
    ax.axhline(0, color=C_INK, lw=1.0, alpha=0.6)
    for m, ha, dx in ((0.0, "left", 6), (7.0, "right", -6)):
        r = cur[cur["m"] == m].iloc[0]
        ax.annotate(f"{r['coef']:+.4f}", (m, r["coef"]), xytext=(dx, 12),
                    textcoords="offset points", ha=ha, fontsize=9, weight="bold",
                    color=C_SERIES)
    ax.set_xlabel("M — how many placebo horizons the headline spans", fontsize=9,
                  color=C_INK)
    ax.set_ylabel("effect of registration on P(perennial)", fontsize=9, color=C_INK)
    ax.set_title("Zero at every assumption about the pre-trend", fontsize=9.5, color=C_INK)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _despine(ax)

    fig.suptitle("The two-period tenure difference-in-differences — a bounded null "
                 "(national, 2026-08-12)\n"
                 "14,625 parcels, 15 years, 63.6 M pixel-observations. The CI never "
                 "excludes zero, so the registered decision is NOT-SEPARABLE.",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"headline": gates["headline"], "placebo": gates["placebo"],
            "decision": gates["decision"]["decision"]}


# The two figures live in different workspaces — the panel diagnostics under the national
# sample, the DiD under its own draw — so build_all names both rather than relying on CC_PROC.
NAT_PROC = Path("data/processed/all_peru")
DID_PROC = Path("data/processed/all_peru_did")


def build_all(out_dir: Path | str = "docs/figures") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    print("control drift …")
    stats["control_drift"] = fig_control_drift(
        out / "window_control_drift.png", NAT_PROC / "window_diagnostic_nolat.json")
    print("tenure DiD …")
    stats["did"] = fig_did_result(out / "did_result.png", proc_dir=DID_PROC)
    print(json.dumps(stats, indent=2, default=float))
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    build_all()
