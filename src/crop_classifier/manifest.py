"""What each thing this project can do actually needs on disk, and where to get it.

``cc data verify`` walks this. It exists because "do I have what I need?" was previously
answered by starting a job and seeing whether it crashed — and the four silent data traps
mean a missing input often does not crash, it returns a plausible empty result
(``docs/DATA.md`` §4).

Three provenances, and the distinction is the useful part of the output:

``repo``
    Committed to this repository. A clone has it. If it is missing, the clone is broken.
``derive``
    Not committed, but reproducible by a command named here. Excluded because it is large
    and either cheap to rebuild or slow-but-optional.
``obtain``
    Neither. It is the licensed raw archive or a fresh Earth Engine extraction, and no
    command in this repository can conjure it — ``docs/DATA_ACCESS.md`` says who to ask.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Need:
    path: str                     # relative to the repository root
    provenance: str               # repo | derive | obtain
    note: str = ""                # how to get it, when it is not `repo`
    #: Needed only to rebuild this capability's inputs from further upstream, not to run it.
    #: The DiD is the case: its predictions are committed so the estimate reproduces, and the
    #: 13.5-hour panel extraction behind them is optional. Without this the check says "needs
    #: data" for something that works.
    optional: bool = False


@dataclass(frozen=True)
class Capability:
    name: str
    about: str
    needs: tuple[Need, ...]
    doc: str = ""


CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        "demo",
        "The quickstart: train and evaluate with no raw data and no Earth Engine account.",
        (Need("data/demo/modeling_parcels.parquet", "repo"),
         Need("data/demo/features/features_lightgbm.parquet", "repo"),
         Need("data/demo/label_map.json", "repo"),
         Need("data/demo/lodo_summary_demo.json", "repo")),
        "docs/howto/01_setup.md"),

    Capability(
        "national-landsat",
        "Train and evaluate the national 3-class Landsat model over 14 departments.",
        (Need("data/processed/all_peru/modeling_parcels.parquet", "repo"),
         Need("data/processed/all_peru/features/features_lightgbm.parquet", "repo"),
         Need("data/processed/all_peru/features/features_lightgbm_degraded.parquet", "repo",
              "the density-augmented copy the primary model trains against"),
         Need("data/processed/all_peru/lodo_summary_nolat_aug_yleak10.json", "repo",
              "the LODO record; recompute with `cc advanced lodo` (hours)"),
         Need("data/processed/all_peru/lodo_predictions_nolat_aug_yleak10.parquet", "repo",
              "the held-out predictions; without them `cc evaluate` cannot print the "
              "majority-class floor")),
        "docs/howto/06_reference.md"),

    Capability(
        "s2-labels",
        "Train and evaluate on the 865 photo-interpreted Sentinel-2 endpoint labels.",
        (Need("data/processed/all_peru/labels_s2/labelled_parcels.parquet", "repo"),
         Need("data/processed/all_peru/labels_s2/label_sample.parquet", "repo"),
         Need("data/processed/all_peru/features_s2/s2_features_lightgbm.parquet", "repo"),
         Need("data/processed/all_peru/features_s2/tensor_perdate.npz", "repo",
              "the LTAE sequence input"),
         Need("data/processed/all_peru/labels_s2/ws_t3w_pilot", "derive",
              "cc labelling train prep  (add --target/--climate for a climate arm)")),
        "docs/howto/04_label_and_train.md"),

    Capability(
        "perennial-shift",
        "The PETT -> CENAGRO 2012 comparison and its tenure split — the headline result.",
        (Need("data/raw/Cenagro_IV/Piura.parquet", "repo",
              "the 14 linkable departments are committed; the other 11 are not"),
         Need("data/raw/Cenagro_IV/crop_code_table.csv", "repo",
              "the question-024 crop code list; without it every census crop is unresolved"),
         Need("data/processed/cenagro/national_panel.parquet", "repo",
              "the built panel. Rebuilding it needs the FULL national parcel table (437 MB, "
              "not committed) — so the comparison reads this instead, and reproduces"),
         Need("data/processed/cenagro/pett_population_by_dept_class.csv", "repo",
              "the department x class counts post-stratification reweights to"),
         Need("data/processed/pett_crop_year.parquet", "repo"),
         Need("data/processed/all_peru/tenure_by_predio.parquet", "repo")),
        "docs/howto/05_summary_stats.md"),

    Capability(
        "tenure-did",
        "The two-period difference-in-differences — the project's only causal estimate.",
        (Need("data/processed/all_peru_did/panel_parcels.parquet", "repo"),
         Need("data/processed/all_peru_did/panel_predictions_nolat_aug_yleak10.parquet",
              "repo", "the classifier outputs the DiD differences; the panel that produced "
                      "them is not committed"),
         Need("data/processed/all_peru/tenure_two_period.parquet", "repo"),
         Need("data/processed/all_peru_did/features/panel", "obtain",
              "the 15-year panel extraction, ~13.5 h on 5 GEE workers — needed only to "
              "rebuild the predictions above from pixels, not to run the DiD",
              optional=True)),
        "docs/howto/06_reference.md"),

    Capability(
        "piura",
        "The original single-department strand: 12-class (closed) and 3-class perennial.",
        (Need("data/processed/modeling_parcels.parquet", "repo"),
         Need("data/processed/features/features_lightgbm.parquet", "repo"),
         Need("data/processed/perennial/modeling_parcels.parquet", "repo")),
        "docs/RESULTS.md"),

    Capability(
        "new-extraction",
        "Pull NEW imagery from Earth Engine — other years, other parcels, another sensor.",
        (Need("data/processed/all_peru/modeling_parcels.parquet", "repo",
              "carries the real polygons, so an extraction can run from it alone"),
         Need("workspaces.yaml", "repo", "set `gee_project:` to your own GCP project"),
         Need("data/raw/QGIS", "obtain",
              "only if you need parcels beyond the 56,419 already linked — 3 GB, licensed",
              optional=True)),
        "docs/howto/06_reference.md"),

    Capability(
        "rebuild-from-raw",
        "Rebuild the label tables from the licensed raw archive, rather than using the "
        "committed derived tables.",
        (Need("data/raw/BD_SSET", "obtain", "the crop registry, 8 workbooks"),
         Need("data/raw/Grafica_Tabular", "obtain",
              "⭐ the mandatory bridge — the only file carrying both keys"),
         Need("data/raw/QGIS", "obtain", "the parcel polygons")),
        "docs/DATA_ACCESS.md"),
)


def check(cap: Capability) -> list[tuple[Need, bool]]:
    return [(n, (ROOT / n.path).exists()) for n in cap.needs]


def status(cap: Capability) -> str:
    """``ok`` | ``derivable`` | ``needs-data`` — the three answers worth distinguishing."""
    missing = [n for n, ok in check(cap) if not ok and not n.optional]
    if not missing:
        return "ok"
    return "derivable" if all(n.provenance == "derive" for n in missing) else "needs-data"
