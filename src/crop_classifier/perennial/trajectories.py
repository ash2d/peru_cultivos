"""Per-parcel class trajectories, smoothing and change detection (plan §9).

Turns ``panel_predictions.parquet`` (one row per parcel-year) into:

* a per-parcel year-indexed **series** with an explicit ``observed`` mask — gaps are never
  interpolated silently, since an abstained year and a genuinely unchanged year look
  identical once filled;
* a **smoothed** series (gap-aware 3-year mode filter);
* **transitions** under a minimum-duration rule — orchard establishment or removal is a
  multi-year event, so a 1-year excursion is noise *by construction*, not by tuning;
* a **flicker rate** per class: whether any of this is signal. Criterion S5 is flicker under
  15 % for PETT-PERENNIAL parcels; a high rate means the per-year features are too weak to
  run the trend analysis on raw predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GAP = -1          # sentinel for an unobserved (abstained) year


def to_series(preds: pd.DataFrame, years: list[int] | None = None,
              class_col: str = "pred_label", classes: list[str] | None = None
              ) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Wide ``[n_parcels, n_years]`` arrays of class ids and an ``observed`` mask.

    A parcel-year that is missing from ``preds`` or was abstained becomes ``GAP``.
    """
    df = preds.copy()
    if classes is None:
        classes = sorted(df[class_col].dropna().unique())
    cid = {c: i for i, c in enumerate(classes)}
    if years is None:
        years = sorted(df["year"].unique())
    parcels = sorted(df["COD_PREDIO"].unique())
    pidx = {p: i for i, p in enumerate(parcels)}
    yidx = {y: i for i, y in enumerate(years)}

    arr = np.full((len(parcels), len(years)), GAP, dtype=np.int64)
    abstained = df["abstained"].to_numpy(dtype=bool) if "abstained" in df else \
        np.zeros(len(df), dtype=bool)
    for (p, y, c), ab in zip(df[["COD_PREDIO", "year", class_col]].itertuples(index=False),
                             abstained):
        if ab or pd.isna(c) or y not in yidx:
            continue
        arr[pidx[p], yidx[y]] = cid[c]
    return arr, arr != GAP, classes, years


def mode_filter(series: np.ndarray, window: int = 3) -> np.ndarray:
    """Centred gap-aware mode filter. Unobserved years stay unobserved.

    Ties keep the parcel's own current class — a smoother must not invent a change.
    """
    out = series.copy()
    n_years = series.shape[1]
    half = window // 2
    for t in range(n_years):
        lo, hi = max(0, t - half), min(n_years, t + half + 1)
        win = series[:, lo:hi]
        for i in range(series.shape[0]):
            if series[i, t] == GAP:
                continue
            vals = win[i][win[i] != GAP]
            if len(vals) == 0:
                continue
            counts = np.bincount(vals)
            top = counts.max()
            winners = np.flatnonzero(counts == top)
            # tie -> keep the current class rather than inventing a change
            out[i, t] = series[i, t] if series[i, t] in winners else winners[0]
    return out


def flicker_rate(series: np.ndarray, threshold_div: float = 5.0) -> np.ndarray:
    """Per-parcel flag: does the observed series change class more than
    ``n_observed / threshold_div`` times? (§9.2)"""
    n_par = series.shape[0]
    out = np.zeros(n_par, dtype=bool)
    for i in range(n_par):
        obs = series[i][series[i] != GAP]
        if len(obs) < 2:
            continue
        changes = int((obs[1:] != obs[:-1]).sum())
        out[i] = changes > len(obs) / threshold_div
    return out


def detect_transitions(series: np.ndarray, years: list[int], classes: list[str],
                       parcels: list[str], min_duration: int = 3) -> pd.DataFrame:
    """Class changes that **persist** for ``min_duration`` consecutive observed years.

    The primary change definition (§9.2): a new orchard takes 2-3 years to close canopy and a
    cleared one does not come back next season, so a shorter excursion is not a land-use
    change.
    """
    rows = []
    for i in range(series.shape[0]):
        obs_t = np.flatnonzero(series[i] != GAP)
        if len(obs_t) < 2 * min_duration:
            continue
        vals = series[i][obs_t]
        j = 0
        while j < len(vals) - 1:
            if vals[j + 1] == vals[j]:
                j += 1
                continue
            before, after = vals[j], vals[j + 1]
            # how long did each state hold, in observed years?
            k = j
            while k > 0 and vals[k - 1] == before:
                k -= 1
            n_before = j - k + 1
            m = j + 1
            while m < len(vals) - 1 and vals[m + 1] == after:
                m += 1
            n_after = m - j
            if n_before >= min_duration and n_after >= min_duration:
                rows.append({
                    "COD_PREDIO": parcels[i],
                    "from_class": classes[before], "to_class": classes[after],
                    "year_of_change": years[obs_t[j + 1]],
                    "n_years_before": int(n_before), "n_years_after": int(n_after)})
            j = m if m > j else j + 1
    return pd.DataFrame(rows, columns=["COD_PREDIO", "from_class", "to_class",
                                       "year_of_change", "n_years_before",
                                       "n_years_after"])


def transition_matrix(transitions: pd.DataFrame, classes: list[str],
                      period: tuple[int, int] | None = None) -> pd.DataFrame:
    """Counts of ``from_class -> to_class``, optionally within a year range."""
    t = transitions
    if period is not None:
        t = t[t["year_of_change"].between(*period)]
    m = pd.DataFrame(0, index=classes, columns=classes, dtype=int)
    for _, r in t.iterrows():
        m.loc[r["from_class"], r["to_class"]] += 1
    return m


def add_confidence(transitions: pd.DataFrame, preds: pd.DataFrame,
                   window: int = 3) -> pd.DataFrame:
    """Attach mean predicted probability in the ``window`` years either side of a change."""
    if transitions.empty:
        return transitions.assign(confidence_before=[], confidence_after=[])
    p = preds.set_index(["COD_PREDIO", "year"])["pred_proba"]
    before, after = [], []
    for _, r in transitions.iterrows():
        y = r["year_of_change"]
        b = [p.get((r["COD_PREDIO"], yy), np.nan) for yy in range(y - window, y)]
        a = [p.get((r["COD_PREDIO"], yy), np.nan) for yy in range(y, y + window)]
        before.append(np.nanmean(b) if np.any(~np.isnan(b)) else np.nan)
        after.append(np.nanmean(a) if np.any(~np.isnan(a)) else np.nan)
    return transitions.assign(confidence_before=before, confidence_after=after)


def class_share_by_year(series: np.ndarray, years: list[int], classes: list[str],
                        area_ha: np.ndarray | None = None,
                        weights: np.ndarray | None = None) -> pd.DataFrame:
    """Per-year class share over *observed* parcel-years, optionally area/weight expanded."""
    w = np.ones(series.shape[0]) if weights is None else np.asarray(weights, float)
    a = np.ones(series.shape[0]) if area_ha is None else np.asarray(area_ha, float)
    val = a * w
    rows = []
    for t, y in enumerate(years):
        col = series[:, t]
        obs = col != GAP
        tot = val[obs].sum()
        for c, name in enumerate(classes):
            m = obs & (col == c)
            rows.append({"year": y, "class": name,
                         "area": float(val[m].sum()),
                         "share": float(val[m].sum() / tot) if tot else np.nan,
                         "n_observed": int(obs.sum())})
    return pd.DataFrame(rows)
