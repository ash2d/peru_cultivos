"""Drive one training arm of the S2 campaign: pick the workspace, then run the step.

This was ~120 lines inside the ``s2-train`` CLI command. It is not argument parsing — it is
arm-selection policy, three branches of which exist to *refuse* a combination. Policy that
says no belongs where it can be read and tested, not in a command body.

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
        # `prep` builds the plain label sets; climate arms only when asked, so name the
        # exact command.
        cmd = "cc labelling train prep" + (
            f" --target {target} --climate {climate}" if climate != "none" else "")
        raise SystemExit(f"{ws} not built — run `{cmd}` first")
    return ws


def check_model_is_runnable(step: str, model: str, target: str) -> None:
    """Refuse `rules` where it would return a number instead of an error.

    The rule maps three semantic groups (PERENNIAL / ANNUAL / PASTURE_FALLOW) onto label ids.
    In a two-class space its fallback resolves PASTURE_FALLOW to PERENNIAL, so it runs to
    completion and returns a meaningless number — worse than a crash, nothing says so.
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

    # the model/label-space check first: about the arm itself, so refuse an impossible
    # combination whether or not its workspace is built
    check_model_is_runnable(step, model, target)
    if step == "fit" and eval_test and model == "rules":
        # the rule is a floor exercise, not a candidate (§8.8/§8.8b never proposes it);
        # refuse the combination before any path is resolved.
        raise SystemExit("--eval-test is for a selected model; `rules` is a control.")
    ws = resolve_workspace(target, climate, pilot)

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
        if eval_test:
            print(f"⚠️  SPENDING THE LOCKED TEST on {model}/{target}/climate={climate}. "
                  f"This is one-way — record it in RESULTS.md.")
        # a round trains into its own run directory: the arm names (`ws_t3w`, ...) repeat
        # across rounds, and two rounds writing one run dir would overwrite each other
        from crop_classifier.paths import labels_dir
        ld = labels_dir().name
        base = Path("runs/s2_labels") if ld == "labels_s2" else Path("runs") / ld
        os.environ["CC_RUNS"] = str(base / ws.name)
        _train(model_name=model, run_name=model, eval_test=eval_test, model_kw=kw or None)

    elif step == "baseline":
        if run is None:
            raise SystemExit("`baseline` needs --run, the Landsat model run to transfer")
        out = P.landsat_baseline(run, target=target, include_pilot=pilot)
        P.report_baseline(out, ws)

    elif step == "lodo":
        # the climate arms address their workspace explicitly; the plain arm keeps the
        # historical behaviour (pilot always folded in) so its numbers stay comparable with
        # the LODO CSVs on disk
        extra = {"ws": ws, "tag": f"_clim_{climate}"} if climate != "none" else {}
        P.dept_transfer(target=target, model_name=model, model_kw=kw or None, **extra)
