"""Score parcels with the Sentinel-2 classifier, in one command.

Everything between "here are some polygons" and "here is a class per polygon": draw the
parcels, pull the imagery, build the features, add the climate column the model of record
was fitted with, predict. Each step is still its own command
(`docs/howto/03_score_parcels.md` lists them); this runs them in order against a scratch
workspace so nothing has to be added to ``workspaces.yaml``.

    uv run cc predict-s2 --n 500                       # a random sample of Peru
    uv run cc predict-s2 --parcels my_farms.shp        # polygons of your own

⚠️ Earth Engine is the slow step and it fails quietly — verify by counting rows, never by
"the command finished" (`docs/howto/06_reference.md`).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from crop_classifier.paths import ROOT

# The full national table is a census of the 726,808 linkable parcels, so a random draw from
# it is a sample of that population. `all_peru/modeling_parcels.parquet` is NOT: it doubles
# the perennial share by design, and shares computed off a draw from it are about the sample.
POPULATION = ROOT / "data" / "processed" / "all_peru_full" / "modeling_parcels.parquet"
FALLBACK = ROOT / "data" / "processed" / "all_peru" / "modeling_parcels.parquet"
LABELLED = ROOT / "data" / "processed" / "all_peru" / "labels_s2" / "labelled_parcels.parquet"
DEFAULT_RUN = ROOT / "runs" / "s2_labels" / "ws_t4_pilot__clim_temp" / "lightgbm"


def _source(parcels: Path | None) -> tuple[Path, bool]:
    """The polygon file to draw from, and whether it is the population table."""
    if parcels is not None:
        return Path(parcels), False
    if POPULATION.exists():
        return POPULATION, True
    if FALLBACK.exists():
        print(f"⚠️ {POPULATION.relative_to(ROOT)} is missing, drawing from "
              f"{FALLBACK.relative_to(ROOT)} instead — that table over-samples PERENNIAL "
              f"~2x by design, so shares off this draw are about the sample, not Peru")
        return FALLBACK, False
    raise SystemExit(f"no parcels to draw from: pass --parcels, or build "
                     f"{POPULATION.relative_to(ROOT)} with `cc data link` (docs/DATA_ACCESS.md)")


def _labelled_ids() -> set[str]:
    if not LABELLED.exists():
        return set()
    return set(pd.read_parquet(LABELLED, columns=["COD_PREDIO"])["COD_PREDIO"].astype(str))


def run(parcels: Path | None = None, n: int = 500, date: str = "2025-03-01",
        work: Path = Path("data/predict_s2"), model: Path = DEFAULT_RUN,
        climate: str = "temp", tau: float = 0.0, out: Path | None = None,
        seed: int = 0, id_col: str | None = None, chunk_size: int = 25,
        workers: int = 4, skip_extract: bool = False) -> pd.DataFrame:
    from crop_classifier.allperu import s2_campaign as C
    from crop_classifier.infer import infer
    from crop_classifier.prepare_parcels import attach_climate, prepare

    model = Path(model)
    if not (model / "model.bin").exists():
        raise SystemExit(f"no trained model at {model} — fit one with\n"
                         f"  uv run cc -w national_s2 labelling train prep --target t4 "
                         f"--pilot --climate temp\n"
                         f"  uv run cc -w national_s2 labelling train fit  --target t4 "
                         f"--pilot --climate temp --model lightgbm")
    src, population = _source(parcels)
    work = Path(work)
    out = Path(out) if out else work / "predictions.parquet"

    # a scratch workspace in the environment, not in workspaces.yaml: `paths.py` resolves
    # these at call time, so the campaign steps below read and write inside `work`
    os.environ["CC_PROC"] = str(work)
    os.environ["CC_FEAT"] = str(work / "features")
    os.environ["CC_LABELS"] = str(work / "labels_s2")

    if not skip_extract:
        prepare(src, store="s2", imagery_date=date, sample=n or None, out_dir=work,
                id_col=id_col, seed=seed,
                drop_ids=_labelled_ids() if parcels is None else None)
        C.step_extract(chunk_size=chunk_size, workers=workers)
        C.step_assemble()
        if climate != "none":
            attach_climate(work, climate)

    preds = infer(model, None, out, tau)
    got = preds.loc[~preds["abstained"].astype(bool), "pred_label"] \
        if "abstained" in preds.columns else preds["pred_label"]
    print(f"\n{len(got):,} of {len(preds):,} parcels scored "
          f"({int(len(preds) - len(got)):,} abstained)")
    if len(got):
        share = got.value_counts(normalize=True).mul(100).round(1)
        print(share.to_string())
        if not population:
            print("⚠️ these are shares of the parcels you scored, not of Peru")
    return preds
