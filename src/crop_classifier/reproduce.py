"""``cc reproduce`` — re-derive the published headline numbers from the committed data.

One command, no raw archive, no Earth Engine account, no multi-hour job. Every check here
runs from files that are in the repository, and every published value it compares against is
quoted from ``docs/RESULTS.md`` with its section, so a mismatch points at one paragraph
rather than at the project.

Why this exists as code rather than as a page of commands: a reproduction that a reader has
to assemble by hand is one they will assemble differently. This fixes the workspace, the
flags and the tolerance for each number, and prints the published value beside the measured
one — the comparison is the output, not something the reader does afterwards.

⚠️ **Tolerances are not all the same, and the differences are the point.** Reading a
committed metric back off disk should be exact. Refitting LightGBM reproduces to a few
thousandths, not exactly: the library is not bit-deterministic across platforms and thread
counts, and a check that demanded equality there would fail on every machine but this one.
Each check states which kind it is.
"""

from __future__ import annotations

import contextlib
import io
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from crop_classifier.paths import ROOT


@dataclass
class Measure:
    """One number, as published and as measured here."""

    what: str
    published: float
    measured: float | None
    tol: float
    note: str = ""

    @property
    def ok(self) -> bool:
        return (self.measured is not None
                and abs(self.measured - self.published) <= self.tol)


@dataclass
class Check:
    name: str
    about: str
    where: str                       # the RESULTS.md section the numbers come from
    run: Callable[[], list[Measure]]
    cost: str = "seconds"
    needs: tuple[str, ...] = field(default_factory=tuple)

    def missing(self) -> list[str]:
        return [n for n in self.needs if not (ROOT / n).exists()]


# ── the checks ────────────────────────────────────────────────────────────────────────
def _demo() -> list[Measure]:
    """Train the quickstart model and compare with the CV score of record.

    Refit, so the tolerance is a few thousandths rather than zero. `--drop-features
    meta,location` is not optional and not cosmetic: without it the model may use
    acquisition metadata and latitude, and the recorded score is the one *with* it.
    """
    from crop_classifier import workspace as W
    from crop_classifier.train import train

    W.activate("demo")
    expected = json.loads((ROOT / "data/demo/expected_cv.json").read_text())
    run = train(model_name="lightgbm", run_name="reproduce",
                drop_features=expected["drop_features"])
    got = json.loads((run / "cv_metrics.json").read_text())
    return [Measure("demo quickstart, CV macro-F1",
                    expected["cv_macro_f1_mean"], got["cv_macro_f1_mean"], 0.01,
                    "refit; data/demo/expected_cv.json")]


def _national_lodo() -> list[Measure]:
    """Read the national Landsat evaluation record. No refit — this is a file read.

    The LODO record is committed because recomputing it is a model per department, hours.
    What matters is that the number a clone reads is the number that was published.
    """
    from crop_classifier import workspace as W
    from crop_classifier.protocol import collect

    W.activate("national")
    rows, _ = collect(Path("runs/all_peru/lightgbm_nometa_nolat_aug_yleak10"),
                      tag="nolat_aug_yleak10")
    by = {r.split: r for r in rows}
    lodo, loyo = by.get("LODO"), by.get("LOYO")
    return [
        Measure("national Landsat, LODO macro-F1 (mean over 14 departments)",
                0.4789, lodo.mean if lodo else None, 0.0005, "read from disk, §8.2"),
        Measure("national Landsat, LODO majority-class floor",
                0.201, lodo.floor if lodo else None, 0.0015, "read from disk"),
        Measure("national Landsat, LOYO macro-F1 (mean over 14 label years)",
                0.5261, loyo.mean if loyo else None, 0.0005, "read from disk"),
    ]


def _s2_model() -> list[Measure]:
    """Refit the selected Sentinel-2 model: LightGBM, 3 classes + woody, mean temperature.

    This is the model of record (``RESULTS.md`` §8.8b). It trains on 865 photo-interpreted
    parcels and takes about a minute.
    """
    from crop_classifier import workspace as W
    from crop_classifier.labelling.arms import run_step

    W.activate("national_s2")
    run_step("prep", target="t3w", climate="temp")
    run_step("fit", target="t3w", model="lightgbm", climate="temp")
    f = ROOT / "runs/s2_labels/ws_t3w__clim_temp/lightgbm/cv_metrics.json"
    m = json.loads(f.read_text()) if f.exists() else {}
    return [Measure("Sentinel-2 3-class (t3w, --climate temp), CV macro-F1",
                    0.762, m.get("cv_macro_f1_mean"), 0.02,
                    "refit on 865 parcels; §8.8b")]


def _perennial_shift() -> list[Measure]:
    """The headline. Two official declarations of the same land, no classifier in it.

    Runs the whole PETT → CENAGRO 2012 comparison and reads the three numbers the README
    leads with off the post-stratified tables.
    """
    from crop_classifier import workspace as W
    from crop_classifier.allperu import cenagro_shift as S

    W.activate("national")
    out = S.build(save=False)
    ps = out["poststratified"]
    ps_row = ps[ps["estimator"].str.startswith("post-stratified")].iloc[0]
    area = out["poststratified_area"]
    area_row = area[area["class"] == "PERENNIAL"].iloc[0]
    ten = out["poststratified_tenure"]
    diff = ten[ten["tenure"].str.startswith("difference")].iloc[0]
    return [
        Measure("perennial share of parcels, 1999 → 2012 (pp, post-stratified)",
                9.9, float(ps_row["change_pp"]), 0.15, "§8.5"),
        Measure("perennial share of cadastral area, 1999 → 2012 (pp)",
                12.5, float(area_row["change_pp"]), 0.15, "§8.5"),
        Measure("titled − untitled gap in that change (pp)",
                -2.1, float(diff["change_pp"]), 0.15, "§8.6; descriptive, not causal"),
        Measure("parcels in the like-for-like panel",
                63766, float(len(out["panel_crop_only"])), 0, "§8.5"),
    ]


def _tenure_did() -> list[Measure]:
    """The project's only causal estimate: a bounded null.

    Deterministic — it is a regression on committed predictions, so the tolerance is tight.
    """
    from crop_classifier import workspace as W
    from crop_classifier.allperu.tenure_did import run_corrected
    from crop_classifier.paths import proc, shared_input

    W.activate("tenure_did")
    v = run_corrected(
        proc() / "panel_predictions_nolat_aug_yleak10.parquet",
        proc() / "panel_parcels.parquet",
        shared_input("tenure_two_period.parquet"),
        # save=False: a real DiD run refuses to overwrite its own artifacts, which is the
        # right rule for an estimate of record and the wrong one for a check you re-run.
        tag="reproduce", save=False)
    h = v["headline"]
    # the headline estimate carries coef and a cluster-robust se; the published interval is
    # the usual normal one around it
    coef, se = float(h["coef"]), float(h["se"])
    return [
        Measure("tenure DiD headline effect on perennial probability",
                -0.0011, coef, 0.0005, "§7"),
        Measure("  its 95 % CI, lower bound", -0.0126, coef - 1.96 * se, 0.0010),
        Measure("  its 95 % CI, upper bound", +0.0104, coef + 1.96 * se, 0.0010),
    ]


CHECKS: tuple[Check, ...] = (
    Check("demo", "the quickstart model on the committed 1,302-parcel sample",
          "data/demo/expected_cv.json", _demo, "~10 s",
          ("data/demo/features/features_lightgbm.parquet",)),
    Check("national-lodo", "the national Landsat evaluation record, read back",
          "RESULTS.md §8.2", _national_lodo, "instant",
          ("data/processed/all_peru/lodo_summary_nolat_aug_yleak10.json",)),
    Check("s2-model", "the selected Sentinel-2 classifier, refitted",
          "RESULTS.md §8.8b", _s2_model, "~1 min",
          ("data/processed/all_peru/labels_s2/labelled_parcels.parquet",
           "data/processed/all_peru/features_s2/s2_features_lightgbm.parquet")),
    Check("perennial-shift", "the headline: PETT 1999 → CENAGRO 2012, by tenure",
          "RESULTS.md §8.5, §8.6", _perennial_shift, "~30 s",
          ("data/processed/cenagro/national_panel.parquet",
           "data/raw/Cenagro_IV/Piura.parquet")),
    Check("tenure-did", "the two-period difference-in-differences",
          "RESULTS.md §7", _tenure_did, "~10 s",
          ("data/processed/all_peru_did/panel_predictions_nolat_aug_yleak10.parquet",
           "data/processed/all_peru/tenure_two_period.parquet")),
)


# ── the runner ────────────────────────────────────────────────────────────────────────
def _fmt(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:+.4f}" if abs(v) < 1 else f"{v:.3f}"


def run(names: list[str] | None = None, verbose: bool = False) -> int:
    """Run the named checks (all of them by default). Returns the number that failed.

    Each check prints its own analysis — a few hundred lines for the census comparison — and
    that is the wrong thing to read here, so it is captured and only shown when the check
    raises, or when ``verbose`` asks for it. The table at the end is the output.
    """
    todo = [c for c in CHECKS if not names or c.name in names]
    if not todo:
        raise SystemExit(f"no such check; expected one of "
                         f"{', '.join(c.name for c in CHECKS)}")

    results: list[tuple[Check, list[Measure] | str]] = []
    for c in todo:
        gaps = c.missing()
        if gaps:
            results.append((c, f"missing {gaps[0]} — run `cc data verify`"))
            continue
        print(f"  running {c.name:<16} ({c.cost:<8}) {c.about}", flush=True)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(None if verbose else buf):
                results.append((c, c.run()))
        except Exception as e:                                   # noqa: BLE001
            tail = "\n".join(buf.getvalue().splitlines()[-15:])
            if tail:
                print(tail)
            results.append((c, f"{type(e).__name__}: {e}"))

    print("\n" + "=" * 92)
    print("REPRODUCTION — published value vs the value this clone just computed")
    print("=" * 92)
    print(f"{'check':<16}{'quantity':<48}{'published':>11}{'measured':>11}  ")
    print("-" * 92)
    failed = 0
    for c, res in results:
        if isinstance(res, str):
            print(f"{c.name:<16}{res[:70]:<48}{'':>11}{'':>11}  SKIP")
            failed += 1
            continue
        for i, m in enumerate(res):
            print(f"{c.name if i == 0 else '':<16}{m.what[:47]:<48}"
                  f"{_fmt(m.published):>11}{_fmt(m.measured):>11}  "
                  f"{'ok' if m.ok else 'DIFFERS'}")
            failed += not m.ok
    print("-" * 92)
    print(f"{len(results)} checks, {failed} line(s) not matching.")
    if failed:
        print("\nA refit line off by a few thousandths is the library, not the result — the\n"
              "tolerance is printed in `cc reproduce --list`. A line that is far off, or a\n"
              "SKIP, means an input is missing: `uv run cc data verify` says which.")
    else:
        print("Every published number re-derived from the committed data. docs/RESULTS.md\n"
              "has the intervals and the caveats that these single numbers do not carry.")
    return failed


def show() -> None:
    """`--list`: what each check does, what it costs, and how tight its tolerance is."""
    print("checks, in the order `cc reproduce` runs them:\n")
    for c in CHECKS:
        gaps = c.missing()
        print(f"  {c.name:<16} {c.cost:<8} {c.about}")
        print(f"  {'':<16} {'':<8} numbers of record: {c.where}")
        if gaps:
            print(f"  {'':<16} {'':<8} ⚠️ missing {gaps[0]}")
        print()
    print("Run one with `cc reproduce <name>`, or all of them with no argument.")
