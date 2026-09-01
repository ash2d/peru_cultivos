"""Drive one training arm of the S2 campaign: pick the workspace, then run the step.

This was ~120 lines inside the ``s2-train`` CLI command. It is not argument parsing — it is
the arm-selection policy, and three of its branches exist to *refuse* a combination rather
than run it. Policy that says no belongs where it can be read and tested, not in a command
body that only ever runs when someone types the command.

An "arm" is (label set) x (climate covariates) x (pilot folded in or held out). Each gets its
own workspace directory on disk, so every arm is a set of files that can be re-read rather
than a flag threaded through a training call.

⚠️ **One arm per process.** LightGBM and torch each bundle their own libomp and co-loading
them on macOS segfaults. The steps are separate commands for that reason; drive them from a
shell loop, never a Python one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_TARGET = "t4"
STEPS = ("prep", "fit", "lodo", "baseline", "report")


def resolve_workspace(target: str, climate: str, pilot: bool) -> Path:
    """The directory holding this arm's tables. Raises if it has not been built."""
    from crop_classifier.labelling import train_prep as P

    if climate != "none":
        from crop_classifier.labelling import climate_arms as C
        if climate not in C.CLIMATE_SETS:
            raise SystemExit(f"unknown climate arm {climate!r}; expected one of {C.ARMS}")
        ws = C.ws_dir(target, climate, pilot)
    else:
        ws = P.ws_dir(target, pilot)
    if not (ws / "modeling_parcels.parquet").exists():
        raise SystemExit(f"{ws} not built — run `cc labelling train prep` first")
    return ws


def check_model_is_runnable(step: str, model: str, target: str) -> None:
    """Refuse `rules` where it would return a number instead of an error.

    The rule maps three *semantic* groups (PERENNIAL / ANNUAL / PASTURE_FALLOW) onto label
    ids. In a two-class space its fallback resolves PASTURE_FALLOW to id 1, which is
    PERENNIAL. It would run to completion and the result would be meaningless — which is
    worse than a crash, because nothing about the output says so.
    """
    from crop_classifier.label_sets import rules_incompatible

    if step in ("fit", "lodo") and model == "rules" and target in rules_incompatible():
        raise SystemExit(
            f"--model rules cannot be run on --target {target}: in a two-class space its "
            f"fallback resolves PASTURE_FALLOW to PERENNIAL, so it would return a "
            f"meaningless number rather than an error. Use lightgbm or ltae.")


def run_step(step: str, *, model: str = "lightgbm", target: str = "",
             pilot: bool = False, run: Path | None = None, model_kw: str = "",
             climate: str = "none", eval_test: bool = False) -> None:
    """One step of one arm. See ``cc labelling train --help`` for what each step is."""
    from crop_classifier.labelling import train_prep as P

    target = target or DEFAULT_TARGET

    if step == "prep":
        if climate != "none":
            from crop_classifier.labelling import climate_arms as C
            C.build_all(target, include_pilot=pilot)
            return
        for t in P.TARGETS:
            for inc in (False, True):
                P.build_workspace(t, include_pilot=inc)
                print()
        return

    if step == "report":
        if climate != "none":
            from crop_classifier.labelling import climate_arms as C
            C.report(target, include_pilot=pilot)
            return
        P.report(target=target or None, include_pilot=pilot or None)
        return

    if step not in STEPS:
        raise SystemExit(f"unknown step {step!r}; expected one of {', '.join(STEPS)}")

    ws = resolve_workspace(target, climate, pilot)
    check_model_is_runnable(step, model, target)

    os.environ["CC_PROC"] = str(ws)
    os.environ["CC_FEAT"] = str(ws / "features")

    kw = json.loads(model_kw) if model_kw else {}
    if climate != "none" and model == "rules":
        # the rule has no coefficient to give a covariate; climate enters it as a median
        # split of the training set with its own thresholds either side
        from crop_classifier.labelling import climate_arms as C
        kw["climate_features"] = C.CLIMATE_SETS[climate]

    if step == "fit":
        from crop_classifier.train import train as _train
        if eval_test and model == "rules":
            # the rule is a floor exercise, not a candidate; §8.8/§8.8b never proposes it
            raise SystemExit("--eval-test is for a selected model; `rules` is a control.")
        if eval_test:
            print(f"⚠️  SPENDING THE LOCKED TEST on {model}/{target}/climate={climate}. "
                  f"This is one-way — record it in RESULTS.md.")
        os.environ["CC_RUNS"] = str(Path("runs/s2_labels") / ws.name)
        _train(model_name=model, run_name=model, eval_test=eval_test, model_kw=kw or None)

    elif step == "baseline":
        if run is None:
            raise SystemExit("`baseline` needs --run, the Landsat model run to transfer")
        out = P.landsat_baseline(run, target=target, include_pilot=pilot)
        P.report_baseline(out, ws)

    elif step == "lodo":
        # the climate arms address their workspace explicitly; the plain arm keeps the
        # historical behaviour (pilot always folded in) so its numbers stay comparable with
        # the LODO CSVs already on disk
        extra = {"ws": ws, "tag": f"_clim_{climate}"} if climate != "none" else {}
        P.dept_transfer(target=target, model_name=model, model_kw=kw or None, **extra)
