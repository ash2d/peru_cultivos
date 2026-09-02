"""The evaluation protocol, as one command: CV / LODO / LOYO / LODYO with their floor.

One command because reading only the first split produced three wrong conclusions:
``centroid_lat`` gains +0.047 macro-F1 on CV and LOYO and loses 0.060 when the department
changes (CV holds out 5 km cells inside seen departments; LOYO holds region roughly fixed);
``frac_l7`` manufactured *change* and static features *stability*; LTAE wins CV 8/8 on the
S2 store and loses LODO 4/4. An evaluation nobody remembers to run does not exist, so the
protocol is the default.

Every macro-F1 is printed beside its floor. It is not comparable across label spaces —
collapsing 4 classes to 2 raised it 0.672 -> 0.715 *and* raised the floor 0.171 -> 0.467.
The ``skill`` column is ``(macro_f1 - floor) / (1 - floor)`` — the one to compare across
label spaces. ``docs/RESULTS.md`` §4.6, §8.2c; ``docs/LESSONS.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

#: What each split holds out, printed under the table — the LOYO/LODYO difference is the
#: whole point of it.
SPLIT_MEANING = {
    "CV": "5 km blocks, held out INSIDE departments the model has seen",
    "LODO": "a whole department, unseen — the spatial generalisation test",
    "LOYO": "a whole label-year cohort, region held roughly fixed",
    "LODYO": "department AND year at once; free, re-scored from the LODO fits",
}


@dataclass
class Row:
    """One split's line.

    ``mean``/``sd`` are over held-out UNITS (departments, cohorts, folds); ``pooled`` is
    over every held-out parcel at once. The gap is information — LODO pools to 0.538,
    averages to 0.479, because departments differ in size and difficulty. The mean leads:
    it is what to expect from *a* new department; pooled weights toward the largest unit.
    """

    split: str
    mean: float | None          # over held-out units
    sd: float | None
    pooled: float | None        # over all held-out parcels
    floor: float | None
    n_units: int | None
    n_parcels: int | None
    units: str
    source: str


def majority_class_floor(y_true) -> float:
    """macro-F1 of the always-guess-the-largest-class predictor, on this exact label vector.
    Computed from the data, not a formula, so it is right for whatever label space is in play.
    """
    from sklearn.metrics import f1_score
    y = pd.Series(y_true)
    modal = y.value_counts().idxmax()
    labels = sorted(y.unique())
    return float(f1_score(y, pd.Series(modal, index=y.index), average="macro",
                          labels=labels, zero_division=0))


def _y_pred(d: pd.DataFrame, label_map: dict[str, int] | None) -> pd.Series | None:
    """The predicted class id, from either shape a prediction file comes in.

    LODO/LOYO store a hard ``y_pred``; a training run stores ``prob_<CLASS>`` columns. Take
    the argmax, mapped back through the run's label map so the ids match ``y_true`` rather
    than alphabetical column order.
    """
    if "y_pred" in d.columns:
        return d["y_pred"]
    prob_cols = [c for c in d.columns if c.startswith("prob_")]
    if not prob_cols:
        return None
    names = [c[len("prob_"):] for c in prob_cols]
    winner = d[prob_cols].to_numpy().argmax(axis=1)
    if label_map:
        ids = [label_map[n] for n in names]
    else:                                    # no map: column order is the only ordering left
        ids = list(range(len(names)))
    return pd.Series([ids[i] for i in winner], index=d.index)


def _pooled_and_floor(path: Path, label_map: dict[str, int] | None = None):
    """(pooled macro-F1, majority-class floor, n parcels) from a predictions file."""
    if not path.exists():
        return None, None, None
    from sklearn.metrics import f1_score
    d = pd.read_parquet(path)
    y_pred = _y_pred(d, label_map)
    if y_pred is None:
        return None, None, None
    labels = sorted(pd.unique(d.y_true))
    pooled = float(f1_score(d.y_true, y_pred, average="macro", labels=labels,
                            zero_division=0))
    return pooled, majority_class_floor(d.y_true), len(d)


def _pooled_folds(run: Path, label_map: dict[str, int] | None):
    """Pool ``fold*/preds_val.parquet`` — every parcel appears in exactly one fold's val
    set, so concatenating them is the out-of-fold prediction for the whole trainval set."""
    parts = sorted(run.glob("fold*/preds_val.parquet"))
    if not parts:
        return None, None, None
    d = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
    y_pred = _y_pred(d, label_map)
    if y_pred is None:
        return None, None, None
    from sklearn.metrics import f1_score
    labels = sorted(pd.unique(d.y_true))
    pooled = float(f1_score(d.y_true, y_pred, average="macro", labels=labels,
                            zero_division=0))
    return pooled, majority_class_floor(d.y_true), len(d)


def _cv_row(run: Path) -> Row | None:
    """CV from a run directory.

    mean/sd come from ``cv_metrics.json`` (the recorded fold stats), so this reports the
    same CV number as everything else. The predictions file is read only for the floor and
    pooled figure, which the JSON does not carry.
    """
    lm = run / "label_map.json"
    label_map = json.loads(lm.read_text()) if lm.exists() else None
    pooled, floor, n = _pooled_and_floor(run / "preds_cv.parquet", label_map)
    if pooled is None:
        # not every run has been through `pool-cv`; the per-fold val predictions are the
        # same rows, so pool them rather than leave the floor blank
        pooled, floor, n = _pooled_folds(run, label_map)
    j = run / "cv_metrics.json"
    if not j.exists():
        if pooled is None:
            return None
        return Row("CV", None, None, pooled, floor, None, n, "5 km blocks",
                   "preds_cv.parquet")
    m = json.loads(j.read_text())
    return Row("CV", m.get("cv_macro_f1_mean"), m.get("cv_macro_f1_std"), pooled, floor,
               len(m.get("folds", [])) or None, n, "CV folds", j.name)


def _summary_row(split: str, summary: Path, preds: Path, units: str,
                 n_units_key: str) -> Row | None:
    """LODO / LOYO: mean and sd over held-out units from the summary, floor from the
    predictions the same run wrote."""
    if not summary.exists():
        return None
    m = json.loads(summary.read_text())
    pooled, floor, n = _pooled_and_floor(preds)
    return Row(split, m.get("mean_macro_f1"), m.get("std_macro_f1"),
               m.get("pooled_macro_f1", pooled), floor, m.get(n_units_key), n,
               units, summary.name)


def collect(run: Path, tag: str = "", proc_dir: Path | None = None) -> tuple[list[Row], dict]:
    """Gather whatever of the four splits is already on disk. Reads, never computes — LODO
    refits per department and is hours, so this reports what exists and what does not."""
    from crop_classifier.paths import proc
    p = Path(proc_dir) if proc_dir else proc()
    suf = f"_{tag}" if tag else ""
    rows = [
        _cv_row(run),
        _summary_row("LODO", p / f"lodo_summary{suf}.json",
                     p / f"lodo_predictions{suf}.parquet", "departments", "n_departments"),
        _summary_row("LOYO", p / f"loyo_summary{suf}.json",
                     p / f"loyo_predictions{suf}.parquet", "label years", "n_cohorts"),
    ]
    # LODYO re-scores the LODO predictions by cohort, so its pooled score is LODO's; what it
    # adds is the per-cohort spread, which lives in its own summary.
    lodyo = p / f"lodyo_summary{suf}.json"
    if lodyo.exists():
        m = json.loads(lodyo.read_text())
        lodo_row = next((r for r in rows if r is not None and r.split == "LODO"), None)
        rows.append(Row("LODYO", m.get("mean_macro_f1"), m.get("std_macro_f1"),
                        m.get("pooled_macro_f1"),
                        lodo_row.floor if lodo_row else None,   # same parcels as LODO
                        m.get("n_cohorts"), None, "dept x year cells", lodyo.name))
    extra = {}
    for name, path in (("lodo", p / f"lodo_summary{suf}.json"),
                       ("loyo", p / f"loyo_summary{suf}.json"),
                       ("lodyo", lodyo)):
        if path.exists():
            extra[name] = json.loads(path.read_text())
    return [r for r in rows if r is not None], extra


def format_table(rows: list[Row], extra: dict, run: Path, tag: str) -> str:
    """The table, plus what each split holds out and how far the units spread around the mean."""
    out = [f"evaluation protocol — run={run.name}" + (f"  tag={tag}" if tag else ""), ""]
    head = (f"{'split':<7} {'mean±sd over units':>20} {'pooled':>8} {'floor':>7} "
            f"{'skill':>7} {'units':>18} {'n':>9}")
    out += [head, "-" * len(head)]
    for r in rows:
        if r.mean is None:
            mean = "                   -"
        elif r.sd is None:
            mean = f"{r.mean:>13.4f}       "
        else:
            mean = f"{r.mean:.4f} ± {r.sd:.4f}".rjust(20)
        pooled = f"{r.pooled:.4f}" if r.pooled is not None else "       -"
        floor = f"{r.floor:.3f}" if r.floor is not None else "      -"
        # skill = headline normalised by the floor; the only column comparable across label spaces
        head_f1 = r.mean if r.mean is not None else r.pooled
        if head_f1 is not None and r.floor is not None and r.floor < 1:
            skill = f"{(head_f1 - r.floor) / (1 - r.floor):.3f}"
        else:
            skill = "      -"
        n_units = f"{r.n_units} {r.units}" if r.n_units else r.units
        n = f"{r.n_parcels:,}" if r.n_parcels is not None else "-"
        out.append(f"{r.split:<7} {mean:>20} {pooled:>8} {floor:>7} {skill:>7} "
                   f"{n_units:>18} {n:>9}")

    missing = [k for k in SPLIT_MEANING if k not in {r.split for r in rows}]
    if missing:
        out += ["", f"⚠️  not computed: {', '.join(missing)}."]
        if "LODO" in missing:
            out.append("    LODO is the one that decides. `cc advanced lodo --tag <tag>` "
                       "(hours; LODYO comes free with it).")
        if "LOYO" in missing:
            out.append("    `cc advanced loyo --tag <tag>`.")

    out += ["", "what each split holds out:"]
    for k, v in SPLIT_MEANING.items():
        out.append(f"  {k:<6} {v}")

    if "lodyo" in extra and extra["lodyo"].get("worst_cohort_macro_f1_excl_elnino") is not None:
        out += ["", f"LODYO worst cohort (excluding the El Nino years): "
                    f"{extra['lodyo']['worst_cohort_macro_f1_excl_elnino']:.4f}"]

    out += ["",
            "Read the LODO row, not the CV row. CV holds out blocks INSIDE departments the",
            "model has seen, so it cannot separate signal from spatial memorisation.",
            "`mean ± sd` is over held-out units and is what to expect from a NEW unit;",
            "`pooled` weights toward whichever unit was largest.",
            "⚠️  an OOD estimate over 14 units is 14 numbers — the LTAE verdict reversed on",
            "    going from 4 held-out departments to 14.",
            "`skill` = (macro-F1 - floor) / (1 - floor), the ONLY column here comparable",
            "across label spaces: collapsing 4 classes to 2 raises macro-F1 and raises the",
            "floor further. docs/LESSONS.md, docs/RESULTS.md §8.2c."]
    return "\n".join(out)


def report(run: Path, tag: str = "", proc_dir: Path | None = None) -> list[Row]:
    rows, extra = collect(run, tag=tag, proc_dir=proc_dir)
    print(format_table(rows, extra, run, tag))
    return rows
