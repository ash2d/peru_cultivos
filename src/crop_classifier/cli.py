"""CLI entrypoints. Run ``uv run python -m crop_classifier.cli --help``.

Command reference and cookbook: ``docs/PIPELINE.md`` §3 and §7. Commands whose estimand
was abandoned live under ``allperu closed`` (``docs/RESULTS.md`` §9).
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)
labels_app = typer.Typer(
    no_args_is_help=True,
    help="Turn declared crop text into model labels, under one label policy")
splits_app = typer.Typer(
    no_args_is_help=True,
    help="Spatially blocked train/val/test splits, with an autocorrelation audit")
features_app = typer.Typer(no_args_is_help=True, help="Deprecated: see `cc satellite`")
perennial_app = typer.Typer(no_args_is_help=True,
                            help="3-class perennial/annual/pasture work (docs/RESULTS.md §2)")
allperu_app = typer.Typer(no_args_is_help=True,
                          help="all-Peru extension: 14 linkable departments (docs/RESULTS.md §4)")
# ⛔ Commands whose ESTIMAND was abandoned after a pre-registered gate failed. The code is
# built, unit-tested and correct; it stays unrun (docs/RESULTS.md §9), and it stays in the
# tree because "we tried this and measured why it does not work" is a result.
#
# It sits at the top level as `cc archive`, beside the modules in
# src/crop_classifier/archive/ (see the README there for what each one tried and which gate
# killed it). Nothing about how they run has changed, only the prefix:
#   cc allperu windows ...  ->  cc archive windows ...
closed_app = typer.Typer(
    no_args_is_help=True,
    help="⛔ closed routes — the estimand failed its gate; kept for reproduction only "
         "(src/crop_classifier/archive/README.md, docs/RESULTS.md §5, §6.4, §7, §9)")
# The historical grouping: `features`, `perennial`, `allperu`. It groups commands by which
# *strand of the research* built them, which is the right axis for someone who worked on the
# project and the wrong one for someone arriving at it — `allperu` alone held 27 commands, of
# which about four are the pipeline.
#
# The task-shaped groups below (`data`, `satellite`, `analysis`, ...) re-register these same
# functions under names that say what you would be trying to do. Both work; only the new ones
# are listed in `--help`, so a shell history, a running script or a line in docs/PIPELINE.md
# keeps working while the docs migrate.
app.add_typer(labels_app, name="labels")
app.add_typer(splits_app, name="splits")
app.add_typer(features_app, name="features", hidden=True)
app.add_typer(perennial_app, name="perennial", hidden=True)
app.add_typer(allperu_app, name="allperu", hidden=True)
app.add_typer(closed_app, name="archive")


# ── the active workspace ─────────────────────────────────────────────────────────────────
# `-w NAME` replaces the three exports (CC_PROC / CC_FEAT / CC_RUNS) that every recipe used
# to start with. It resolves the name against `workspaces.yaml` and sets those same three
# variables, so nothing downstream changes: paths.py still resolves at call time, which is
# still the rule that keeps one workspace from overwriting another's tables.
#
# Omitting -w leaves the environment exactly as it was, so an explicit CC_PROC still wins.


@app.callback()
def main(
    workspace: str = typer.Option(
        None, "--workspace", "-w",
        help="named workspace from workspaces.yaml (e.g. demo, national, national_s2). "
             "Sets CC_PROC/CC_FEAT/CC_RUNS for this command. Run `cc workspaces` to list "
             "them. Omit to use whatever the environment already says."),
    quiet: bool = typer.Option(False, "--quiet", "-q",
                               help="do not print the resolved workspace banner"),
):
    """Peru crop classifier. Start with `cc workspaces`, then `docs/howto/`."""
    if workspace is None:
        return
    from crop_classifier import workspace as W
    ws = W.activate(workspace)
    if not quiet:
        # print it on stderr so piping a command's output stays clean, and print it at all
        # because silently targeting the wrong store is this project's oldest footgun
        typer.echo(f"workspace={ws.name}  proc={ws.proc}  feat={ws.feat}  runs={ws.runs}",
                   err=True)


def _use_round(name: str) -> None:
    """`--round NAME` -> ``CC_LABELS=<CC_PROC>/labels_s2_NAME``.

    A labelling round is a draw plus the labels that came back from it, and the two are
    joined by an ``item_id`` that only that draw's ``label_sample.parquet`` explains. So a
    second round gets its own directory rather than adding to the first — nothing of record
    is edited, and both rounds stay readable on their own.
    """
    if not name:
        return
    import os

    from crop_classifier.paths import proc
    d = proc() / f"labels_s2_{name}"
    f = proc() / f"features_s2_{name}"
    d.mkdir(parents=True, exist_ok=True)
    f.mkdir(parents=True, exist_ok=True)
    os.environ["CC_LABELS"] = str(d)
    # the feature store moves with the round too: `campaign assemble` REPLACES the feature
    # table for the sample it is given, so a round assembling into the store of record would
    # drop the 865 parcels that are already in it.
    os.environ["CC_FEAT"] = str(f)
    typer.echo(f"labelling round '{name}' -> labels {d}\n"
               f"{' ' * 22}features {f}", err=True)


@app.command("workspaces")
def workspaces_cmd(
    check: bool = typer.Option(True, help="also report which inputs exist on this machine"),
):
    """List the configured workspaces and what of each one is present locally.

    The first command to run on a new machine. A workspace is three directories; this shows
    where they resolve to, whether they exist, and whether two of them deliberately share a
    feature store.
    """
    from crop_classifier import workspace as W

    cfg = W.load_config()
    typer.echo(f"config     {W.config_path()}")
    typer.echo(f"gee project {cfg.get('gee_project') or '(unset)'}")
    typer.echo("")

    feat_users: dict[str, list[str]] = {}
    for name in W.names():
        feat_users.setdefault(str(W.resolve(name).feat), []).append(name)

    for name in W.names():
        ws = W.resolve(name)
        typer.echo(f"── {name} " + "─" * max(0, 60 - len(name)))
        if ws.about:
            for line in _wrap(ws.about, 76):
                typer.echo(f"   {line}")
        for role, path in (("proc", ws.proc), ("feat", ws.feat), ("runs", ws.runs)):
            mark = "" if not check else ("[ok]  " if path.exists() else "[--]  ")
            note = ""
            if check and path.exists():
                n = sum(1 for _ in path.iterdir())
                note = f"  ({n} item{'s' if n != 1 else ''})"
            if role == "feat":
                shared = [o for o in feat_users[str(path)] if o != name]
                if shared:
                    note += f"  shared with: {', '.join(shared)}"
            typer.echo(f"   {mark}{role}  {path}{note}")
        typer.echo("")

    env = W.active()
    if W.activated():
        typer.echo(f"active: {W.activated()} (selected with -w)")
    elif any(env.values()):
        typer.echo("active: set by the environment, NOT by workspaces.yaml —")
        for k, v in env.items():
            if v:
                typer.echo(f"   {k}={v}")
        typer.echo("   unset these, or pass -w, to use a named workspace.")
    else:
        typer.echo("active: none — commands will use the built-in default "
                   "(the `piura` layout). Pass -w to choose one.")


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(" ".join(text.split()), width=width)


@labels_app.command("build")
def labels_build(config: Path | None = None):
    """Label table: category policy + merge map + area/year gates (§7)."""
    from crop_classifier.labels import build
    build(config_path=config)


@labels_app.command("list")
def labels_list():
    """Every label set on disk, and what each one collapses.

    A label set is a file in `src/crop_classifier/config/labels/`. Adding one needs no code
    change — copy the nearest file, edit `collapse:`, and it appears here.
    """
    from crop_classifier.label_sets import CONFIG_DIR, available, load

    typer.echo(f"label sets in {CONFIG_DIR}\n")
    for name in available():
        ls = load(name)
        typer.echo(f"── {name}  ({len(ls.classes)} classes: {', '.join(ls.classes)})")
        if not ls.rules_compatible:
            typer.echo("   ⚠️  the `rules` baseline is refused here — it would return a "
                       "meaningless number rather than an error")
        for src, dst in sorted(ls.collapse.items()):
            typer.echo(f"   {src:<16} -> {'(dropped)' if dst is None else dst}")
        if not ls.collapse:
            typer.echo("   (no collapse — the full recorded label space)")
        if ls.about:
            for line in _wrap(ls.about, 74):
                typer.echo(f"   {line}")
        typer.echo("")
    typer.echo("⚠️  macro-F1 is NOT comparable across these — collapsing classes raises it "
               "and raises\n    the majority-class floor further. `cc evaluate` prints both. "
               "docs/RESULTS.md §8.2c.")


@splits_app.command("assign")
def splits_assign(config: Path | None = None):
    """1 km blocks, autocorrelation audit, locked test, CV folds, buffers (§5)."""
    from crop_classifier.splits import assign
    assign(config_path=config)


@features_app.command("extract")
def features_extract(stage: str = typer.Option("all", help="coverage | pixels | all"),
                     chunk_size: int = 400, pixel_chunk_size: int = 40,
                     max_chunks: int | None = typer.Option(
                         None, help="stop each stage after N new chunks (partial runs; "
                                    "resumable, still combines/gates what is on disk)"),
                     years: str | None = typer.Option(
                         None, help="comma-separated crop years, e.g. '1998,1999'")):
    """Stage-1 coverage (+gate) then stage-2 raw pixels for survivors (§6, A4)."""
    from crop_classifier.features import landsat_gee as lg
    from crop_classifier.labels import apply_coverage_gate
    yrs = [int(y) for y in years.split(",")] if years else None
    if stage in ("coverage", "all"):
        cov = lg.run_coverage(chunk_size=chunk_size, max_chunks=max_chunks, years=yrs)
        apply_coverage_gate(cov)
    if stage in ("pixels", "all"):
        lg.run_pixels(chunk_size=pixel_chunk_size, max_chunks=max_chunks, years=yrs)


@features_app.command("assemble")
def features_assemble(t_max: int = 64, p_max: int = 8):
    """Raw pixel store -> LightGBM summary features + per-date/pixel-set tensors (§6)."""
    from crop_classifier.features.assemble import assemble
    assemble(t_max=t_max, p_max=p_max)


@perennial_app.command("labels")
def perennial_labels(config: Path | None = None):
    """3-class label table (PERENNIAL/ANNUAL/PASTURE_FALLOW). Set CC_PROC first."""
    from crop_classifier.perennial.labels3 import build
    build(config_path=config)


def _years(spec: str) -> list[int]:
    """``"1996-2023"`` / ``"1999-2003,2019-2023"`` / ``"2023"`` -> year list.

    Delegates to ``data.parse_year_spec`` so a two-window extraction (RESULTS.md §5, gate T4) is
    one command instead of a loop — this used to accept only a single ``lo-hi`` range, which
    is why recovering a stalled panel meant replaying every cached year.
    """
    from crop_classifier.data import parse_year_spec
    return parse_year_spec(spec) or []


@perennial_app.command("panel")
def perennial_panel(step: str = typer.Argument(
                        ..., help="build|probe|extract|rebuild|verify|assemble|infer"),
                    years: str = "1996-2023", n: int = 7500,
                    run: Path | None = None,
                    out: Path | None = typer.Option(
                        None, help="infer: write predictions here instead of "
                                   "panel_predictions.parquet (a second model). "
                                   "probe: write the measured budget CSV here instead of "
                                   "docs/figures/panel_budget.csv — a second panel must "
                                   "pass its own path, never overwrite the first's record"),
                    bundle_suffix: str = typer.Option(
                        "", help="infer: read <panel>/<year><suffix>/ instead of the "
                                 "audited bundle (e.g. '_qmap' for the quantile-aligned "
                                 "sensitivity arm, RESULTS.md §6)")):
    """Multi-year panel steps (§7). `extract` is a 20+ hour resumable job."""
    from crop_classifier.perennial import panel as P
    if step == "build":
        P.build_panel(n=n)
    elif step == "probe":
        P.timing_probe(years=tuple(_years(years)) or (1995, 2005, 2015, 2023), out=out)
    elif step == "extract":
        P.extract_panel(years=_years(years))
    elif step == "rebuild":
        P.rebuild_year_stores(years=_years(years))
    elif step == "verify":
        v = P.verify_years(years=_years(years))
        raise typer.Exit(code=0 if bool(v["complete"].all()) else 1)
    elif step == "assemble":
        P.assemble_panel(years=_years(years))
    elif step == "infer":
        if run is None:
            raise typer.BadParameter("--run is required for the infer step")
        P.infer_panel(run, years=_years(years), out=out, bundle_suffix=bundle_suffix)
    else:
        raise typer.BadParameter(f"unknown step {step!r}")


@perennial_app.command("diagnostics")
def perennial_diagnostics(
        preds: Path | None = typer.Option(
            None, help="panel predictions to gate (default panel_predictions.parquet)"),
        tag: str = typer.Option(
            "", help="suffix for every artifact, so a second model's gate does not "
                     "overwrite the first (e.g. 'ltae')"),
        elnino_model: str = typer.Option(
            "lightgbm", help="architecture for the El Nino arm (lightgbm|ltae|psetae). "
                             "NEVER mix torch and lightgbm in one process on macOS.")):
    """The Phase-7 GATE: S4 temporal transfer, S5 flicker, sensor drift, El Nino test.

    S4 and S5 are pass/fail. If either fails, do not produce trajectories, transitions or
    area estimates — a trend from a panel that failed validation is not a weaker finding,
    it is a wrong one.
    """
    from crop_classifier.perennial.diagnostics import run_gate
    g = run_gate(preds_path=preds, tag=tag, elnino_model=elnino_model)
    raise typer.Exit(code=0 if g["gate_pass"] else 1)


@perennial_app.command("mapbiomas")
def perennial_mapbiomas(years: str = "1996-2024", chunk_size: int = 200,
                        max_chunks: int | None = None):
    """Extract the MapBiomas Peru benchmark panel (§8)."""
    from crop_classifier.perennial.mapbiomas import run
    run(years=_years(years), chunk_size=chunk_size, max_chunks=max_chunks)


@perennial_app.command("compare")
def perennial_compare(runs: list[Path], out: Path,
                      preds: str = "preds_cv.parquet"):
    """Pooled spatial-CV comparison table + figures across runs (§6)."""
    from crop_classifier.perennial.compare import compare
    compare(runs, out, preds)


@perennial_app.command("calibrate")
def perennial_calibrate(run: Path,
                        preds: str = typer.Option("fold0/preds_val.parquet",
                                                  help="held-out predictions to fit on")):
    """Fit the temperature scalar on held-out validation probabilities (D8)."""
    from crop_classifier.perennial.calibration import calibrate_run
    calibrate_run(run, preds)


@perennial_app.command("pool-cv")
def perennial_pool_cv(run: Path):
    """Pool fold*/preds_val.parquet into preds_cv.parquet for an unbiased CV report."""
    from crop_classifier.perennial.pool_cv import pool_cv
    pool_cv(run)


@allperu_app.command("labels")
def allperu_labels(only: list[str] | None = typer.Option(
        None, help="department folder names, e.g. PIURA TUMBES (default: all linkable)")):
    """Chain-A build over every linkable department -> data/processed/all_peru/."""
    from crop_classifier.allperu.build_labels import build
    build(only=list(only) if only else None)


@allperu_app.command("sample")
def allperu_sample(source: Path = typer.Option(
        ..., help="workspace holding the FULL all-Peru modeling_parcels.parquet"),
        target_n: int = 56419, max_per_region: int = 220, floor: int = 800,
        seed: int = 42):
    """Down-sample to Piura scale (whole regions, sqrt-proportional by department).

    Writes into ``CC_PROC``, so set that to the *sampled* workspace and point ``--source``
    at the full one.
    """
    from crop_classifier.allperu.sample import sample
    sample(source, target_n=target_n, max_per_region=max_per_region, floor=floor,
           seed=seed)


@allperu_app.command("lodo")
def allperu_lodo(model: str = "lightgbm", drop_features: str = "meta",
                 train_years: str | None = None, buffer_m: float = 1500.0,
                 min_parcels: int = 200,
                 augment_feat: Path | None = typer.Option(
                     None, help="degraded feature table appended to the TRAIN side "
                                "(allperu.density) — temporal_ood_plan §1b"),
                 tag: str = typer.Option(
                     "", help="suffix every artifact and nest the per-department fits "
                              "under runs/lodo/<tag>/, so a second architecture's LODO "
                              "does not overwrite the first (e.g. 'ltae')")):
    """Leave-one-department-out: the spatial-generalisation test Piura could not run."""
    from crop_classifier.allperu.lodo import run
    run(model_name=model, drop_features=drop_features, train_years=train_years,
        buffer_m=buffer_m, min_parcels=min_parcels, tag=tag,
        augment_feat=str(augment_feat) if augment_feat else None)


@allperu_app.command("tenure")
def allperu_tenure(policy: str = typer.Option("any", help="any | mode — how to reconcile a "
                                                         "parcel whose declarations disagree"),
                   two_period: bool = typer.Option(
                       False, help="also join the ~2011 cadastre status (window_plan §6.2)")):
    """ESTADO en RRPP per parcel, from the raw workbooks -> tenure_by_predio.parquet."""
    from crop_classifier.allperu import tenure as T
    from crop_classifier.paths import proc
    T.audit(policy=policy)
    if two_period:
        tp = T.tenure_two_period(policy=policy)
        tp.to_parquet(proc() / "tenure_two_period.parquet", index=False)
        print(f"\ntwo-period: {len(tp):,} parcels; "
              f"{tp.became_registered.mean():.1%} NO INSCRITO -> REGISTERED")


@closed_app.command("windows")
def allperu_windows(preds: Path = typer.Option(..., help="panel_predictions*.parquet"),
                    tag: str = "", tenure: Path | None = None,
                    min_years: int = 3, threshold: float = 0.5,
                    balanced: bool = typer.Option(
                        False, help="restrict to parcels qualifying in every window")):
    """T3: the 5-year-window diagnostic (W1-W4) — run this BEFORE funding any extraction."""
    import pandas as pd

    from crop_classifier.archive.windows import print_diagnostic, run_diagnostic
    ten = pd.read_parquet(tenure) if tenure else None
    v = run_diagnostic(preds, tag=tag, tenure=ten, min_years=min_years,
                       threshold=threshold, balanced=balanced)
    print_diagnostic(v)
    raise typer.Exit(code=0 if v["pass"] else 1)


@closed_app.command("tenure-did")
def allperu_tenure_did(
        preds: Path = typer.Option(..., help="panel_predictions*.parquet"),
        tag: str = typer.Option(..., help="names the CONFIGURATION, not the date"),
        parcels: Path | None = None, tenure: Path | None = None,
        cohort_min_year: int = typer.Option(
            2004, help="R4: keep parcels whose declaration follows the pre-window"),
        control: str = typer.Option("no_inscrito", help="no_inscrito (N-D9) | any (pilot)"),
        no_r4: bool = typer.Option(False, "--no-r4",
                                   help="reproduce the pilot — NOT a valid specification")):
    """Two-period tenure DiD + gates G1-G3 (docs/RESULTS.md §7). Exits 1 if G1 FAILS."""
    from crop_classifier.allperu.tenure_did import print_verdict, run
    from crop_classifier.paths import proc, shared_input

    v = run(preds, parcels or proc() / "panel_parcels.parquet",
            tenure or shared_input("tenure_two_period.parquet"), tag=tag,
            cohort_min_year=cohort_min_year, control=control, apply_r4=not no_r4)
    print_verdict(v)
    raise typer.Exit(code=1 if v["G1_placebo"]["verdict"] == "FAIL" else 0)


@allperu_app.command("did-sample")
def allperu_did_sample(
        source: Path = typer.Option(..., help="FULL all-Peru workspace (modeling_parcels)"),
        tenure: Path = typer.Option(..., help="tenure_two_period.parquet"),
        model_sample: Path | None = typer.Option(
            None, help="the model's own parcel table — supplies `split` for the "
                       "training-membership diagnostic"),
        cohort_min_year: int = 2004, control_ratio: float = 2.0,
        max_per_region: int = 220, seed: int = 42):
    """Draw the v3 DiD extraction sample into CC_PROC (whole 5 km regions, both arms).

    Takes **every** qualifying treated parcel and controls at ``--control-ratio`` within
    department x declaration-year cohort. Refuses to overwrite an existing draw.
    """
    from crop_classifier.allperu.did_sample import sample
    sample(source, tenure, model_sample=model_sample, cohort_min_year=cohort_min_year,
           control_ratio=control_ratio, max_per_region=max_per_region, seed=seed)


@allperu_app.command("tenure-register")
def allperu_tenure_register(
        out: Path | None = typer.Option(None, help="default CC_PROC/did2_registration.json"),
        cohort_min_year: int = 2004,
        control: str = typer.Option("no_inscrito", help="no_inscrito (N-D9) | any")):
    """Write the pre-registration for the v3 (pre-trend-corrected) DiD. Refuses to overwrite.

    ⛔ Run this **before** any extraction. The decision rule, the primary outcome, the
    contrast, the placebo and the sensitivity grid are fixed here and never edited after.
    """
    from crop_classifier.allperu.tenure_did import write_registration
    write_registration(out, cohort_min_year=cohort_min_year, control=control)


@allperu_app.command("tenure-did2")
def allperu_tenure_did2(
        preds: Path = typer.Option(..., help="panel_predictions*.parquet"),
        tag: str = typer.Option(..., help="names the CONFIGURATION, not the date"),
        parcels: Path | None = None, tenure: Path | None = None,
        cohort_min_year: int = 2004,
        control: str = typer.Option("no_inscrito", help="no_inscrito (N-D9) | any"),
        n_boot: int = typer.Option(0, help="cluster-bootstrap replicates for the "
                                           "headline/placebo covariance DIAGNOSTIC")):
    """v3 DiD: placebo first, then headline, then the corrected effect + sensitivity curve.

    Exits 1 only if the run could not produce an estimate at all. **NOT-SEPARABLE is a
    complete outcome and exits 0** — it bounds the effect and measures the anticipation
    trend, which is what the study is for.
    """
    from crop_classifier.allperu.tenure_did import print_corrected, run_corrected
    from crop_classifier.paths import proc, shared_input

    v = run_corrected(preds, parcels or proc() / "panel_parcels.parquet",
                      tenure or shared_input("tenure_two_period.parquet"), tag=tag,
                      cohort_min_year=cohort_min_year, control=control, n_boot=n_boot)
    print_corrected(v)
    raise typer.Exit(code=0 if v["corrected"]["se"] == v["corrected"]["se"] else 1)


@allperu_app.command("tenure-xsec")
def allperu_tenure_xsec(
        preds: Path = typer.Option(..., help="panel_predictions*.parquet"),
        tag: str = typer.Option(...), parcels: Path | None = None,
        tenure: Path | None = None,
        windows: str = typer.Option("W99,W19", help="two window names")):
    """⚠️ DESCRIPTIVE companion: INSCRITO vs NO INSCRITO perennial share, by window.

    **Not an estimate and never reported as one.** The baseline gap among at-risk parcels is
    the classifier's differential false-positive rate, not agronomy, and the contrast's sign
    is department-specific. Always read the per-department table.
    """
    from crop_classifier.allperu.tenure_did import (
        cross_sectional_contrast,
        print_cross_sectional,
    )
    from crop_classifier.paths import proc, shared_input

    w = tuple(x.strip() for x in windows.split(","))
    v = cross_sectional_contrast(preds, parcels or proc() / "panel_parcels.parquet",
                                 tenure or shared_input("tenure_two_period.parquet"),
                                 tag=tag, windows=w)  # type: ignore[arg-type]
    print_cross_sectional(v)


@allperu_app.command("tenure-ceiling")
def allperu_tenure_ceiling(
        source: Path = typer.Option(..., help="FULL all-Peru workspace (modeling_parcels)"),
        tenure: Path = typer.Option(...), cohort_min_year: int = 2004,
        tag: str = typer.Option("default")):
    """G3a feasibility: how many parcels in all of Peru could ever enter the DiD."""
    import json

    from crop_classifier.allperu.tenure_did import population_ceiling
    from crop_classifier.paths import proc

    c = population_ceiling(source, tenure, cohort_min_year=cohort_min_year)
    print(json.dumps({k: v for k, v in c.items() if k != "by_dept"}, indent=2))
    print("treated by department:", c["by_dept"])
    out = proc() / f"did_population_ceiling_{tag}.json"
    with open(out, "w") as f:
        json.dump(c, f, indent=2, default=float)
    print(f"wrote {out}")


@allperu_app.command("tenure-error")
def allperu_tenure_error(run: Path = typer.Option(..., help="model run dir with preds_cv"),
                         tenure: Path = typer.Option(...), tag: str = ""):
    """T1: is classifier error non-differential with respect to tenure?"""
    import geopandas as gpd
    import pandas as pd

    from crop_classifier.archive.windows import print_tenure_error, tenure_error_test
    from crop_classifier.paths import proc as _proc
    v = tenure_error_test(pd.read_parquet(run / "preds_cv.parquet"),
                          gpd.read_parquet(_proc() / "modeling_parcels.parquet"),
                          pd.read_parquet(tenure), tag=tag)
    print_tenure_error(v)
    raise typer.Exit(code=0 if v["pass"] else 1)


@allperu_app.command("loyo")
def allperu_loyo(model: str = "lightgbm", drop_features: str = "meta,location",
                 buffer_m: float = 1500.0, min_parcels: int = 300,
                 all_regions: bool = typer.Option(
                     False, help="do NOT restrict to regions holding >= 2 label-year "
                                 "cohorts (an upper bound: mixes place with time)"),
                 augment_feat: Path | None = typer.Option(
                     None, help="degraded feature table appended to the TRAIN side "
                                "(allperu.density) — temporal_ood_plan §1b"),
                 tag: str = ""):
    """T2: leave-one-YEAR-out cohort transfer — the temporal analogue of LODO."""
    from crop_classifier.allperu.loyo import run
    run(model_name=model, drop_features=drop_features, buffer_m=buffer_m,
        min_parcels=min_parcels, shared_regions_only=not all_regions, tag=tag,
        augment_feat=str(augment_feat) if augment_feat else None)


@allperu_app.command("lodyo")
def allperu_lodyo(tag: str = typer.Option("nolat", help="LODO artifact tag to re-score"),
                  min_parcels: int = 300):
    """Score an existing LODO run per label-year cohort: place AND year held out at once.

    Free — it re-scores predictions that already exist. Read this when CV, LODO and LOYO
    disagree: LOYO holds place approximately fixed, so it cannot see spatial memorisation
    any more than CV can (RESULTS.md §9.2).
    """
    from crop_classifier.allperu.loyo import lodo_by_cohort
    print(lodo_by_cohort(tag=tag, min_parcels=min_parcels).to_string(index=False))


@closed_app.command("window-sample")
def allperu_window_sample(source: Path = typer.Option(..., help="FULL all-Peru workspace"),
                          tenure: Path = typer.Option(...),
                          at_risk_per_tenure: int = 5000, n_perennial: int = 1500,
                          n_pasture: int = 1500, max_per_region: int = 220,
                          seed: int = 42):
    """Draw the dept x tenure x label stratified window sample (§4.5/§4.6). Writes CC_PROC."""
    from crop_classifier.archive.window_sample import power_table, sample
    print(power_table().to_string(index=False), "\n")
    sample(source, tenure, at_risk_per_tenure=at_risk_per_tenure,
           n_perennial=n_perennial, n_pasture=n_pasture, max_per_region=max_per_region,
           seed=seed)


@allperu_app.command("export-status")
def allperu_export_status():
    """§6.4/§6.3: how much of PERENNIAL is really export, and the woody non-crop bound."""
    from crop_classifier.allperu.export_crops import report, woody_noncrop_bound
    report()
    print()
    woody_noncrop_bound()


@closed_app.command("estimate")
def allperu_estimate(preds: Path = typer.Option(...), tenure: Path = typer.Option(...),
                     run: Path = typer.Option(..., help="model run dir with preds_cv"),
                     tag: str = "nolat",
                     force: bool = typer.Option(
                         False, help="run even though a gate failed — recorded in output")):
    """T5: the tenure contrast on the window estimand. Refuses to run unless T1-T3 passed."""
    import json as _json

    from crop_classifier.archive.estimate import run as _run
    print(_json.dumps(_run(preds, tenure, run, tag=tag, force=force), indent=2,
                      default=float))


@closed_app.command("external")
def allperu_external(preds: Path = typer.Option(...), siea: Path = typer.Option(
        ..., help="downloaded MIDAGRI/SIEA district-crop-year CSV (not redistributed)"),
        map: str | None = typer.Option(None, help="JSON dict renaming SIEA columns"),
        min_parcels: int = 30):
    """T6: rank-compare district perennial growth against MIDAGRI/SIEA statistics."""
    import json as _json

    from crop_classifier.archive.external import compare
    compare(preds, siea, _json.loads(map) if map else None, min_parcels=min_parcels)


@allperu_app.command("density-audit")
def allperu_density_audit(tag: str = "panel"):
    """Step 1a: rank every feature by how much it moves when only the *number of looks* does."""
    from crop_classifier.allperu.density import audit_summary, feature_density_audit
    a = feature_density_audit(tag=tag)
    print(audit_summary(a).to_string(index=False))
    print()
    print(a.head(20)[["feature", "family", "coef", "coef_sd", "t"]].to_string(index=False))


@allperu_app.command("degrade")
def allperu_degrade(out: Path = typer.Option(..., help="destination parquet"),
                    slc_gap_frac: float = typer.Option(
                        None, help="within-parcel SLC-off loss (default: the measured 0.11, "
                                   "NOT the nominal scene-level 0.22)"),
                    seed: int = 42):
    """Step 1b: assemble a feature table from a pixel store degraded to endpoint density."""
    from crop_classifier.allperu.density import SLC_OFF_FRAC, build_degraded_features
    build_degraded_features(out_path=out, seed=seed,
                            slc_gap_frac=SLC_OFF_FRAC if slc_gap_frac is None
                            else slc_gap_frac)


@allperu_app.command("density-calibrate")
def allperu_density_calibrate(
        run: Path = typer.Option(..., help="LightGBM run dir to calibrate"),
        preds: Path | None = typer.Option(
            None, help="panel predictions to recalibrate post hoc (optional)"),
        out: Path | None = typer.Option(None, help="destination for the recalibrated panel"),
        fold: int = 0):
    """Step 1c: fit temperature as a function of log(observations), not one scalar."""
    import json as _json

    import pandas as pd

    from crop_classifier.allperu import density as D
    from crop_classifier.perennial.calibration import load_temperature
    lad = D.calibration_ladder(run, fold=fold)
    prob = lad[[c for c in lad.columns if c.startswith("prob_")]].to_numpy()
    params = D.fit_density_temperature(prob, lad["y_true"].to_numpy(),
                                       lad["n_dates"].to_numpy())
    D.save_params(params, run / "density_temperature.json")
    print(_json.dumps({k: v for k, v in params.items() if k != "bands"}, indent=2))
    print(pd.DataFrame(params["bands"]).to_string(index=False))
    if preds is not None:
        p = D.attach_panel_n_dates(pd.read_parquet(preds))
        rec = D.recalibrate_predictions(p, params,
                                        base_temperature=load_temperature(run),
                                        density_col="n_dates")
        dest = out or preds.with_name(preds.stem + "_dcal.parquet")
        rec.to_parquet(dest, index=False)
        print(f"wrote {dest}: {len(rec):,} parcel-years")


@allperu_app.command("year-leak")
def allperu_year_leak(drop_features: str = "meta,location", tag: str = "",
                      all_regions: bool = typer.Option(
                          False, help="do NOT hold region approximately fixed (upper bound)")):
    """Step 2a: which features identify the label-year cohort? Candidates only — LOYO decides."""
    from crop_classifier.allperu.yearleak import run
    a = run(drop_features=drop_features, shared_regions_only=not all_regions, tag=tag)
    print(a.head(25).to_string(index=False))


@allperu_app.command("esri-dates")
def allperu_esri_dates(out: Path = typer.Option(Path("docs/figures/esri_imagery_dates.csv")),
                       n_per_dept: int = typer.Option(60, help="parcel centroids per dept"),
                       depts: str = typer.Option("", help="comma list; default the 4 targets")):
    """Gate zero for endpoint labelling: what imagery date/resolution would a labeller see?

    Probes real parcel centroids, not a bbox — one huge old tile and one small new tile count
    the same in a tile query, and only the parcel-weighted number decides anything.
    See docs/s2_labelling/plan.md.
    """
    from crop_classifier.allperu import esri_dates as ed
    if depts:
        ed.DEPTS = [d.strip() for d in depts.split(",")]
    ed.N_PER_DEPT = n_per_dept
    ed.main(str(out))


@allperu_app.command("label-budget")
def allperu_label_budget(out_dir: Path = typer.Option(Path("docs/figures"))):
    """How many labelled parcels does a campaign need? Learning curve on the existing store.

    Runs both draw protocols — whole 5 km regions vs stratified by class — because the gap
    between them is worth ~8x the labelling budget. docs/s2_labelling/plan.md.
    """
    from crop_classifier.allperu.label_budget import build
    print(build(out_dir).round(3).to_string(index=False))


@allperu_app.command("s2-labels")
def allperu_s2_labels(
        step: str = typer.Argument(
            ..., help="universe|pool|probe|draw|split|chips|extract|harmonisation|"
                      "html|assemble|ingest|combine|transitions"),
        source: Path = typer.Option(Path("data/processed/all_peru_full"),
                                    help="universe: workspace holding the FULL national "
                                         "modeling_parcels.parquet"),
        csv_dir: Path | None = typer.Option(None, help="ingest: returned label CSVs"),
        workers: int = typer.Option(3, help="chips/probe: parallel workers"),
        chunk_size: int = typer.Option(25, help="extract: parcels per GEE request"),
        max_chunks: int | None = typer.Option(None, help="extract: stop after N chunks"),
        supp_depts: str = typer.Option("", help="pool: comma list to supplement"),
        overwrite: bool = typer.Option(False, help="chips: re-render existing"),
        lang: str = typer.Option("en", help="html: interface + codebook language "
                                            "(en|es). Non-English writes to "
                                            "labels_s2/html_<lang>/"),
        round_: str = typer.Option(
            "", "--round", help="name a NEW labelling round; everything it writes goes to "
                                "`labels_s2_<name>/` instead of over the campaign of "
                                "record. Use this for any second batch of parcels "
                                "(docs/howto/06_label_more_parcels.md)")):
    """S2 endpoint-labelling campaign (docs/s2_labelling/plan.md).

    The steps run in the plan's §11 order, cheapest-that-can-kill-it first::

        universe -> pool -> probe -> draw -> split -> chips -> extract
                 -> harmonisation -> html -> (humans label) -> assemble -> ingest
                 -> transitions

    Needs ``CC_PROC=data/processed/all_peru`` and, for the S2 steps,
    ``CC_FEAT=data/processed/all_peru/features_s2``.
    """
    from crop_classifier.allperu import s2_campaign as C
    _use_round(round_)
    C.run_step(step, source=source, csv_dir=csv_dir, workers=workers,
               chunk_size=chunk_size, max_chunks=max_chunks, overwrite=overwrite,
               lang=lang,
               supp_depts=[d.strip() for d in supp_depts.split(",") if d.strip()])


@allperu_app.command("s2-train")
def allperu_s2_train(
        step: str = typer.Argument(..., help="prep|fit|lodo|baseline|report"),
        model: str = typer.Option("lightgbm", help="fit: lightgbm|ltae|rules"),
        target: str = typer.Option("", help="label target: t5|t4|t3|t3w|t2|t2w "
                                          "(default t4; `report` with no --target "
                                          "tabulates every arm). t2/t2w are "
                                          "PERENNIAL vs NON_PERENNIAL and are not "
                                          "runnable with --model rules"),
        pilot: bool = typer.Option(False, "--pilot",
                                   help="fold the 120-parcel pilot into trainval "
                                        "(NON-CANONICAL: the frozen split holds it out)"),
        run: Path = typer.Option(
            Path("runs/all_peru/lightgbm_nometa_nolat_aug_yleak10"),
            help="baseline: the Landsat model run to transfer"),
        model_kw: str = typer.Option(
            "", help="fit: JSON of model hyper-parameters, e.g. "
                     "'{\"batch_size\": 32, \"epochs\": 300}'. The registry defaults "
                     "were tuned on 50k Landsat parcels; this campaign trains on ~200"),
        climate: str = typer.Option(
            "none", help="climate covariate arm: none|temp|rain|both. Adds the WorldClim "
                         "normals (tmean_c / precip_mm_yr) as extra inputs — flat columns "
                         "for lightgbm, a static embedding for ltae, threshold strata for "
                         "rules. `prep` builds every arm; `report` with --climate all "
                         "tabulates them"),
        round_: str = typer.Option(
            "", "--round", help="train on a labelling round built with `campaign --round "
                                "<name>` instead of the campaign of record"),
        eval_test: bool = typer.Option(
            False, "--eval-test",
            help="⚠️ ONE-WAY: also score the LOCKED TEST (161 usable parcels, 14 depts) "
                 "after the refit. Every use contaminates it for model selection, so pass "
                 "it only for a configuration already chosen on CV/LODO, and record the "
                 "result in RESULTS.md. Refuses any model but the one you name — there is "
                 "no sweep behind this flag")):
    """Train and compare models on the returned S2 endpoint labels (docs/s2_labelling/plan.md).

    ⚠️ **One arm per process.** LightGBM and torch each bundle their own libomp and
    co-loading them on macOS segfaults, so `fit --model ltae` must not share a process with
    `fit --model lightgbm`. The steps are separate commands for that reason; drive them from
    a shell loop, not a Python one.

    ⚠️ The locked test set is untouched by `prep`, `lodo`, `baseline` and `report`, and by
    `fit` unless **`--eval-test`** is passed. That flag is one-way: 161 usable parcels over
    14 departments, so a macro-F1 from it carries roughly a +/-7 pp interval, and every
    configuration scored on it is one the test can no longer independently confirm. Spend it
    on an arm already selected on CV/LODO (RESULTS.md §8.8b), and write the number down.
    """
    from crop_classifier.labelling.arms import run_step
    _use_round(round_)
    run_step(step, model=model, target=target, pilot=pilot, run=run,
             model_kw=model_kw, climate=climate, eval_test=eval_test)


@allperu_app.command("cenagro")
def allperu_cenagro(
        config: Path | None = typer.Option(
            None, help="3-class lexicon to classify BOTH sides with "
                       "(default config/perennial.yaml — Piura, which is the only "
                       "department CENAGRO covers)")):
    """CENAGRO 2012 as the 'before' observation instead of the PETT declaration.

    Produces the paired **PETT declaration -> CENAGRO 2012** change on the same parcels, the
    like-for-like version restricted to parcels with a crop recorded on both sides, the
    area-weighted version, the same broken out by census link quality, and the handful of
    parcels that also carry a 2019+ photo-interpreted label.

    ⚠️ **Piura only**, and the census-to-parcel link is **farmer-level, not parcel-level** —
    the census carries no parcel key at all. See the module docstring and `DATA.md` Chain B.
    """
    from crop_classifier.allperu import cenagro as C
    C.build(config_path=config)


@allperu_app.command("cenagro-extract")
def allperu_cenagro_extract(
        dept: str = typer.Option("all", help="department stem (e.g. Piura) or 'all'"),
        overwrite: bool = typer.Option(False, help="rewrite files that already exist"),
        verify: bool = typer.Option(False, help="run the post-extraction audit instead")):
    """Slim the 25 OneDrive CENAGRO 2012 `.dta` files to `data/raw/Cenagro_IV/*.parquet`.

    409 columns -> 76, ~18 GB -> ~250 MB, one file per department, still **long** (one row
    per parcel x crop-order). Reads each `.dta` once, chunked, with an explicit `usecols`.

    ``--verify`` re-reads the written Parquet and reports: rows in/out, the `P009_01`
    non-blank rate (the farmer name is the only link to PETT), whether the column set is
    identical across all 25 files, whether `P024_03` resolves against the crop table, and
    the posesionario rate that tests what "sin posesionario" actually filtered.
    """
    from crop_classifier.allperu import cenagro_extract as E
    if verify:
        E.verify()
        return
    E.extract(None if dept == "all" else [dept], overwrite=overwrite)


@allperu_app.command("cenagro-link")
def allperu_cenagro_link(
        dept: str = typer.Option("all", help="department stem or 'all' (the 14 linkable)")):
    """Build the NATIONAL census⇄PETT crosswalk by farmer name (Chain B, 14 departments).

    Writes `data/processed/cenagro/cenagro_pett_link.parquet`. ⚠️ The census carries no parcel
    key, so the link is **farmer-level, not parcel-level**; every number built on it must be
    reported by `link_confidence`. `DATA.md` §2 Chain B.
    """
    from crop_classifier.allperu import cenagro_link as L
    L.build(None if dept == "all" else [dept])


@allperu_app.command("cenagro-shift")
def allperu_cenagro_shift(
        dept: str = typer.Option("all", help="restrict to one department, or 'all'"),
        figure: bool = typer.Option(False, help="only redraw the figure, skip the tables")):
    """⭐ PETT → CENAGRO 2012 → photo-interpreted 2019+, nationally, split by tenure.

    Perennial share of parcels and of **cadastral** area at each of the three observations,
    with the tenure (INSCRITO / NO INSCRITO) split and the difference between them. Needs
    `allperu cenagro-link` first.

    ⚠️ Descriptive, not causal — title is not randomly assigned. ⚠️ The S2 shares are
    design-weighted because the labelling campaign over-sampled PERENNIAL.
    """
    from crop_classifier.allperu import cenagro_shift as S
    if figure:
        S.figure()
        return
    S.build(None if dept == "all" else [dept])
    S.figure()


@allperu_app.command("climate")
def allperu_climate(
        step: str = typer.Argument(..., help="normals|rainfall|both"),
        years: str = typer.Option("1996-2024",
                                  help="rainfall: CHIRPS calendar years to sample"),
        parcels: Path | None = typer.Option(
            None, help="parcel table to key on (default: the FULL national "
                       "modeling_parcels.parquet, 726,808 parcels)")):
    """Per-parcel mean temperature and rainfall (docs/DATA.md, climate section).

    ``normals``  WorldClim 2.1, ~1 km, the 1970-2000 climatological normal -> annual mean
                 temperature (degC) + annual rainfall (mm/year) + the monthly profile.
                 **Time-invariant**: safe for a single-year classifier, not for a panel.
    ``rainfall`` CHIRPS 2.0, ~5.5 km, one *actual* rainfall total per parcel per calendar
                 year. Read remotely over Peru's bounding box; nothing global is stored.

    ``normals`` needs the two WorldClim zips in ``data/raw/worldclim/`` (the error message
    carries the exact curl commands). ``rainfall`` needs only a network connection.
    """
    from crop_classifier.allperu import climate as C
    if step in ("normals", "both"):
        C.build_normals(parcels_path=parcels)
    if step in ("rainfall", "both"):
        C.build_rainfall_annual(years=_years(years), parcels_path=parcels)
    if step not in ("normals", "rainfall", "both"):
        raise typer.BadParameter(f"unknown step {step!r}")


@closed_app.command("oli")
def allperu_oli(step: str = typer.Argument(
                    ..., help="probe|extract|assemble|verify|compare|split-half"),
                years: str = "2015,2019",
                run: Path | None = typer.Option(None, help="compare: model run dir"),
                l7_preds: Path | None = typer.Option(
                    None, help="compare: L7-only panel predictions "
                               "(default panel_predictions_nolat.parquet)"),
                harmonize: bool = typer.Option(
                    True, help="assemble: apply the Roy et al. OLI->ETM+ coefficients. "
                               "--no-harmonize writes <year>_oliraw/ instead, the arm that "
                               "attributes a 3a failure to the harmonisation or not"),
                suffix: str = typer.Option("_oli", help="compare: which bundle to score"),
                tag: str = ""):
    """Step 3a: does harmonised OLI agree with L7 on the SAME parcel-year? Probe first.

    `split-half` is the control: the same model, sensor, year and parcel scored on two
    disjoint halves of the L7 acquisitions. It is the ceiling any cross-sensor agreement can
    reach, and without it the 0.95 criterion has no referent.
    """
    from crop_classifier.archive import oli_overlap as O
    yrs = tuple(_years(years))
    if step == "probe":
        O.probe(yrs)
    elif step == "extract":
        O.extract(yrs)
    elif step == "assemble":
        O.assemble(yrs, harmonize=harmonize)
    elif step == "verify":
        O.verify(yrs)
    elif step == "split-half":
        if run is None:
            raise typer.BadParameter("--run is required for split-half")
        O.split_half_control(run, yrs, tag=tag)
    elif step == "compare":
        if run is None:
            raise typer.BadParameter("--run is required for compare")
        v = O.compare(run, yrs, l7_preds=l7_preds, tag=tag, suffix=suffix)
        O.print_compare(v)
        raise typer.Exit(code=0 if v["pass"] else 1)
    else:
        raise typer.BadParameter(f"unknown step {step!r}")


@closed_app.command("oli-refit")
def allperu_oli_refit(years: str = "2015,2019,2022", max_days: int = 1):
    """Refit the OLI->ETM+ coefficients on same-day L7/OLI parcel pairs from THIS data.

    The published Roy et al. coefficients were measured in step 3a and are worse than no
    correction at all (RESULTS.md §9.3.2). Landsat 7 and 8 image the same parcel on the same
    day in WRS sidelaps, which is the same observational design, on our own imagery.
    """
    from crop_classifier.archive.oli_refit import run
    run(tuple(_years(years)), max_days=max_days)


@allperu_app.command("inventory")
def allperu_inventory():
    """What links to what: the 15 usable departments and why the rest are excluded."""
    from crop_classifier.allperu.build_labels import SHP_VIEW
    from crop_classifier.allperu.sources import departments, shapefile_view, shp_info, unlinkable
    for d in departments():
        info = shp_info(shapefile_view(d, SHP_VIEW))
        print(f"{d.name:14s} {info['n_features']:>8,} polygons  {info['crs']:>10s}  "
              f"sset={d.sset.name}")
    print()
    for k, v in unlinkable().items():
        print(f"{k}: {', '.join(v) or '(none)'}")


@app.command()
def train(model: str = "lightgbm", folds: int | None = None,
          eval_test: bool = typer.Option(False, help="evaluate locked test ONCE"),
          run_name: str | None = None,
          drop_features: str | None = typer.Option(
              None, help="comma-separated feature columns to withhold from the flat "
                         "model, or the group aliases 'meta' (acquisition metadata: "
                         "frac_l7, n_valid_obs, n_dates, max_gap, n_valid_pixels) and "
                         "'location' (centroid_lat). See RESULTS.md §4.6"),
          train_years: str | None = typer.Option(
              None, "--train-years",
              help="restrict the TRAINING cohort to these PETT label years, e.g. "
                   "'1999-2023' or '1999,2000'. Validation/test membership is NOT "
                   "changed, so CV stays comparable. Use to exclude the 1997-98 El "
                   "Nino cohort (RESULTS.md §8.1)"),
          augment_feat: Path | None = typer.Option(
              None, "--augment-feat",
              help="a degraded copy of the feature store (allperu.density) appended to "
                   "the TRAIN side only, so each parcel is seen at both its own and the "
                   "endpoint observation density. RESULTS.md §6")):
    """Spatial CV + final refit for one of: lightgbm | ltae | psetae (§8/§9)."""
    from crop_classifier.train import train as _train
    _train(model_name=model, n_folds=folds, eval_test=eval_test, run_name=run_name,
           drop_features=drop_features, train_years=train_years,
           augment_feat=augment_feat)


@app.command(hidden=True)
def sweep(model: str, trials: int = 30, folds: int = 3):
    """Optuna sweep on CV macro-F1 (never touches the locked test) (§9)."""
    from crop_classifier.train import sweep as _sweep
    _sweep(model, n_trials=trials, n_folds=folds)


@app.command("evaluate")
def evaluate(run: Path,
             tag: str = typer.Option(
                 "", help="the LODO/LOYO artifact tag for this configuration, e.g. "
                          "'nolat_aug_yleak10'. Written by `cc advanced lodo --tag`"),
             ):
    """⭐ The evaluation protocol in one table: CV / LODO / LOYO / LODYO, each beside its floor.

    Reads what is already on disk; it never starts a LODO refit, which is hours. Anything
    missing is named, with the command that produces it.

    Read the LODO row, not the CV row. CV holds out 5 km blocks *inside* departments the
    model has already seen, so it cannot separate signal from spatial memorisation — which
    is how `centroid_lat` (+0.047 CV, -0.060 LODO) and an entire architecture both got as
    far as being adopted. See docs/LESSONS.md.
    """
    from crop_classifier.protocol import report
    report(run, tag=tag)


@app.command("reproduce")
def reproduce_cmd(
        check: str = typer.Argument("", help="one check name, or blank for all"),
        list_: bool = typer.Option(False, "--list", help="describe the checks, run none"),
        verbose: bool = typer.Option(False, "-v", help="also print each check's own output — "
                                                      "the full tables, not just the compare")):
    """⭐ Re-derive the published headline numbers from the committed data. Start here.

    Needs nothing but a clone: no licensed archive, no Earth Engine account, no multi-hour
    job. Each line prints the published value beside the one this machine just computed, so
    a disagreement is visible rather than something you have to go and look up.

    ⚠️ Refit checks (the demo, the Sentinel-2 model) will not match to the last digit —
    LightGBM is not bit-identical across platforms. Read-from-disk checks should match
    exactly. `--list` says which is which.
    """
    from crop_classifier.reproduce import run, show
    if list_:
        show()
        return
    raise typer.Exit(code=1 if run([check] if check else None, verbose=verbose) else 0)


@app.command("report")
def report_cmd(run: Path, preds: str = "preds_test.parquet"):
    """Write the full report bundle (metrics/confusion/per-class/stratified) for one run."""
    from crop_classifier.evaluate import full_report
    full_report(run, preds)


@app.command("eval", hidden=True)
def eval_cmd(run: Path, preds: str = "preds_test.parquet"):
    """Deprecated alias for `cc report`."""
    from crop_classifier.evaluate import full_report
    full_report(run, preds)


@app.command("predict")
def predict(run: Path, polygons: Path | None = None, out: Path | None = None,
            tau: float = 0.0):
    """Apply a trained model to a parcel set. `--tau` abstains below that probability."""
    from crop_classifier.infer import infer as _infer
    _infer(run, polygons, out, tau)


@app.command("infer", hidden=True)
def infer(run: Path, polygons: Path | None = None, out: Path | None = None,
          tau: float = 0.0):
    """Deprecated alias for `cc predict`."""
    from crop_classifier.infer import infer as _infer
    _infer(run, polygons, out, tau)


# ══════════════════════════════════════════════════════════════════════════════════════
#  The task-shaped command surface
# ══════════════════════════════════════════════════════════════════════════════════════
#  Same functions, same options, grouped by what someone is trying to *do* rather than by
#  which strand of the research produced them. The old names above still work and are
#  hidden rather than removed.
#
#  Deliberately NOT everything is promoted. A command that exists because one audit needed
#  it once lives under `advanced`, where it is findable without competing for attention
#  with the four commands that are the pipeline.

data_app = typer.Typer(
    no_args_is_help=True,
    help="Build the parcel, crop and tenure tables from the raw archive (docs/DATA.md)")
satellite_app = typer.Typer(
    no_args_is_help=True,
    help="Pull imagery from Google Earth Engine and turn it into model features")
analysis_app = typer.Typer(
    no_args_is_help=True,
    help="The research outputs: perennial change over time, and its relation to land tenure")
labelling_app = typer.Typer(
    no_args_is_help=True,
    help="The human photo-interpretation campaign that produced the Sentinel-2 label set")
advanced_app = typer.Typer(
    no_args_is_help=True,
    help="Audits, diagnostics and sensitivity arms — one-off tools, not the pipeline")

app.add_typer(data_app, name="data")
app.add_typer(satellite_app, name="satellite")
app.add_typer(analysis_app, name="analysis")
app.add_typer(labelling_app, name="labelling")
app.add_typer(advanced_app, name="advanced")

# ── data: raw archive -> linked parcels with a crop, a year and a tenure status ────────
@data_app.command("verify")
def data_verify(
    what: str = typer.Argument("", help="one capability name, or blank for all"),
    verbose: bool = typer.Option(False, "-v", help="list every file, not just the gaps"),
):
    """⭐ What can this clone actually do? Run this first, and after obtaining any data.

    A missing input usually does not crash here — four of the raw datasets return a
    plausible EMPTY or column-less result instead of an error (docs/DATA.md §4), so
    "start the job and see" is not a check. This is.

    Each capability is one of:

      ok          everything it needs is present
      derivable   the gap is rebuildable by a command, named in the output
      needs-data  the gap is the licensed archive or a fresh GEE extraction
    """
    from crop_classifier.manifest import CAPABILITIES, ROOT, check, status

    caps = [c for c in CAPABILITIES if not what or c.name == what]
    if not caps:
        raise typer.BadParameter(
            f"unknown capability {what!r}; expected one of "
            f"{', '.join(c.name for c in CAPABILITIES)}")

    mark = {"ok": "[ok]        ", "derivable": "[derivable] ", "needs-data": "[needs data]"}
    tally: dict[str, int] = {}
    for cap in caps:
        st = status(cap)
        tally[st] = tally.get(st, 0) + 1
        typer.echo(f"{mark[st]} {cap.name}")
        for line in _wrap(cap.about, 72):
            typer.echo(f"             {line}")
        for need, ok in check(cap):
            if ok and not verbose:
                continue
            size = ""
            if ok:
                path = ROOT / need.path
                n = (sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                     if path.is_dir() else path.stat().st_size)
                size = f"  ({n / 1e6:.1f} MB)"
            flag = "○" if (not ok and need.optional) else ("✓" if ok else "✗")
            typer.echo(f"    {flag} {need.path}{size}")
            if not ok:
                tag = f"{need.provenance}, optional" if need.optional else need.provenance
                typer.echo(f"        {tag}: {need.note or 'see docs/DATA_ACCESS.md'}")
        if st != "ok" and cap.doc:
            typer.echo(f"      -> {cap.doc}")
        typer.echo("")

    typer.echo("  ".join(f"{v} {k}" for k, v in sorted(tally.items())))
    if tally.get("needs-data"):
        typer.echo("`needs-data` gaps are the licensed raw archive or a fresh Earth Engine "
                   "extraction:\n  docs/DATA_ACCESS.md")


data_app.command("link")(allperu_labels)
data_app.command("sample")(allperu_sample)
data_app.command("tenure")(allperu_tenure)
data_app.command("climate")(allperu_climate)
data_app.command("inventory")(allperu_inventory)
data_app.command("cenagro-extract")(allperu_cenagro_extract)
data_app.command("cenagro")(allperu_cenagro)
data_app.command("cenagro-link")(allperu_cenagro_link)

# ── satellite: the two-stage GEE extraction, then assembly into features ──────────────
satellite_app.command("extract")(features_extract)
satellite_app.command("assemble")(features_assemble)

# ── analysis: what the project is actually for ────────────────────────────────────────
analysis_app.command("perennial-shift")(allperu_cenagro_shift)
analysis_app.command("tenure-gap")(allperu_tenure_xsec)
analysis_app.command("did")(allperu_tenure_did2)
analysis_app.command("did-sample")(allperu_did_sample)
analysis_app.command("did-register")(allperu_tenure_register)
analysis_app.command("did-feasibility")(allperu_tenure_ceiling)

# ── labelling: the S2 campaign ────────────────────────────────────────────────────────
labelling_app.command("campaign")(allperu_s2_labels)
labelling_app.command("budget")(allperu_label_budget)
labelling_app.command("train")(allperu_s2_train)

# ── advanced: real tools, but not the road anyone should start down ───────────────────
advanced_app.command("lodo")(allperu_lodo)
advanced_app.command("loyo")(allperu_loyo)
advanced_app.command("lodyo")(allperu_lodyo)
advanced_app.command("year-leak")(allperu_year_leak)
advanced_app.command("tenure-error")(allperu_tenure_error)
advanced_app.command("export-status")(allperu_export_status)
advanced_app.command("density-audit")(allperu_density_audit)
advanced_app.command("density-calibrate")(allperu_density_calibrate)
advanced_app.command("degrade")(allperu_degrade)
advanced_app.command("esri-dates")(allperu_esri_dates)
advanced_app.command("panel")(perennial_panel)
advanced_app.command("panel-gate")(perennial_diagnostics)
advanced_app.command("mapbiomas")(perennial_mapbiomas)
advanced_app.command("compare")(perennial_compare)
advanced_app.command("calibrate")(perennial_calibrate)
advanced_app.command("pool-cv")(perennial_pool_cv)
advanced_app.command("sweep")(sweep)


if __name__ == "__main__":
    app()
