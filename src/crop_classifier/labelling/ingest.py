"""Ingest returned label CSVs (docs/s2_labelling/plan.md).

Reads the CSVs the labelling HTML downloads, joins ``item_id`` -> ``COD_PREDIO``, computes
**Cohen's kappa on the double-labelled overlap** (over the called parcels, over all six
values, and perennial-vs-rest separately),
resolves disagreements by adjudication rather than majority — with two labellers a
majority does not exist — and writes the labelled parcel table.

**``label`` is stored exactly as the annotator recorded it.** Whether ``WOODY_NON_CROP``
trains as ``OTHER`` or is dropped, and how the 3-class target is formed, is a modelling
decision taken later against this table rather than baked into it. The only rows held out
of training regardless are ``boundary_mismatch=True``, ``label == "UNSURE"`` and parcels
with too few usable S2 pixels — the first and last are label noise, and the second carries
no class information at all. All three stay in the table, flagged in ``exclude_reason``.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# Key order, matching `build_html.LABELS` — `UNSURE` stays on 5 and `NON_AGRICULTURE`,
# added later, takes 6.
LABELS = ["PERENNIAL", "ANNUAL", "OTHER", "WOODY_NON_CROP", "UNSURE",
          "NON_AGRICULTURE"]
# The five that carry class information. `UNSURE` is an abstain, so it is stored verbatim
# but can never train anything and is not a class that has to meet a per-class gate.
#
# `NON_AGRICULTURE` was carved out of `OTHER`, which used to mean both "farmable land not
# currently cropped" and "not farmland at all". Those are different things for this project:
# a fallow field can convert to a perennial and a road cannot, so pooling them puts a
# structurally impossible outcome into the same class as the interesting one. The codebook's
# separating test is whether the ground could be sown next season as it stands.
CLASSES = ["PERENNIAL", "ANNUAL", "OTHER", "WOODY_NON_CROP", "NON_AGRICULTURE"]
ABSTAIN = "UNSURE"
# No `confidence`. It was a second, softer abstain beside UNSURE; two ways to record doubt
# split the signal, and the graded one was never moved off its default.
CSV_COLUMNS = ["item_id", "labeller", "label", "crop_guess",
               "boundary_mismatch", "seconds_spent", "timestamp"]
MIN_PIXELS = 5


def read_csvs(paths: list[Path] | Path) -> pd.DataFrame:
    """Concatenate returned CSVs, validating the label vocabulary as it goes."""
    if isinstance(paths, (str, Path)):
        p = Path(paths)
        paths = sorted(p.glob("*.csv")) if p.is_dir() else [p]
    frames = []
    for f in paths:
        df = pd.read_csv(f)
        missing = [c for c in ("item_id", "label") if c not in df.columns]
        if missing:
            raise ValueError(f"{f.name}: missing required column(s) {missing}")
        bad = sorted(set(df["label"].dropna()) - set(LABELS))
        if bad:
            raise ValueError(f"{f.name}: labels outside the label scheme: {bad}")
        df["source_file"] = f.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError("no label CSVs found")
    out = pd.concat(frames, ignore_index=True)
    for c in CSV_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    out["boundary_mismatch"] = (out["boundary_mismatch"].astype(str)
                                .str.lower().isin(["true", "1", "yes"]))
    return out


# ------------------------------------------------------------------------------------
# Agreement
# ------------------------------------------------------------------------------------
def cohens_kappa(a: pd.Series, b: pd.Series, classes: list[str] | None = None) -> float:
    """Cohen's kappa for two aligned label series.

    Hand-rolled rather than taken from sklearn so the kappa reported in a gate is the one
    the test suite pins against a worked example.
    """
    a, b = pd.Series(list(a)), pd.Series(list(b))
    classes = classes or sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a.values == b.values).mean())
    pe = sum((a == c).mean() * (b == c).mean() for c in classes)
    if np.isclose(pe, 1.0):
        return float("nan")
    return float((po - pe) / (1 - pe))


def agreement(labels: pd.DataFrame) -> dict:
    """Kappa on the parcels two labellers both saw — multi-class **and** perennial-vs-rest.

    The plan asks for both because the gap between them is essentially
    ``WOODY_NON_CROP``/``PERENNIAL`` confusion, the hardest call in the codebook, and it
    tells a later modelling decision how much to trust that distinction.
    """
    dup = labels.groupby("item_id")["labeller"].nunique()
    both = dup[dup >= 2].index
    sub = labels[labels.item_id.isin(both)].sort_values(["item_id", "labeller"])
    if sub.empty:
        return {"n_overlap": 0, "kappa_called": None, "kappa_perennial": None,
                "note": "no double-labelled parcels found"}
    wide = sub.pivot_table(index="item_id", columns="labeller", values="label",
                           aggfunc="first")
    cols = list(wide.columns)[:2]
    a, b = wide[cols[0]].dropna(), wide[cols[1]].dropna()
    idx = a.index.intersection(b.index)
    a, b = a[idx], b[idx]
    per_a, per_b = (a == "PERENNIAL").map({True: "PER", False: "REST"}), \
                   (b == "PERENNIAL").map({True: "PER", False: "REST"})
    cm = pd.crosstab(a, b)

    # Two kappas over the label vocabulary, and they answer different questions.
    # `kappa_all` includes UNSURE as a category, so it is depressed whenever the two
    # labellers abstain on different parcels — which is a real disagreement about
    # difficulty but not about land cover. `kappa_called` is computed on the parcels
    # **both** actually called, which is the number the codebook is responsible for and
    # the one gate G1 reads.
    both_called = idx[(a != ABSTAIN) & (b != ABSTAIN)]
    ac, bc = a[both_called], b[both_called]
    return {
        "n_overlap": int(len(idx)), "labellers": cols,
        "kappa_all": cohens_kappa(a, b, LABELS),
        "kappa_called": cohens_kappa(ac, bc, CLASSES) if len(both_called) else None,
        "n_both_called": int(len(both_called)),
        "kappa_perennial": cohens_kappa(per_a, per_b, ["PER", "REST"]),
        "abstain_rate": {c: float((wide[c] == ABSTAIN).mean()) for c in cols},
        "one_abstained_other_did_not": int(((a == ABSTAIN) ^ (b == ABSTAIN)).sum()),
        "raw_agreement": float((a.values == b.values).mean()),
        "confusion": cm.to_dict(),
        "disagreements": [{"item_id": i, cols[0]: a[i], cols[1]: b[i]}
                          for i in idx if a[i] != b[i]],
    }


def resolve(labels: pd.DataFrame, adjudication: dict[str, str] | None = None
            ) -> pd.DataFrame:
    """One row per ``item_id``.

    With two labellers there is no majority, so disagreements are settled by an explicit
    ``adjudication`` map (``item_id -> label``). An unadjudicated disagreement is **kept
    and flagged**, not silently resolved to the first labeller — it is a real measurement
    of ambiguity, and dropping it would flatter the campaign.
    """
    adjudication = adjudication or {}
    rows = []
    for item, g in labels.groupby("item_id"):
        vals = list(g["label"])
        called = [v for v in vals if v != ABSTAIN]
        if len(set(vals)) == 1:
            lab, how = vals[0], "agreed" if len(vals) > 1 else "single"
        elif item in adjudication:
            lab, how = adjudication[item], "adjudicated"
        elif len(set(called)) == 1:
            # one labeller abstained and the other called it: that is not a disagreement
            # about land cover, and discarding the call would waste a real observation
            lab, how = called[0], "one_abstained"
        else:
            lab, how = g.sort_values("labeller")["label"].iloc[0], "unresolved"
        rows.append({
            "item_id": item, "label": lab, "resolution": how,
            "n_labellers": int(g["labeller"].nunique()),
            "crop_guess": "; ".join(sorted({str(c) for c in g["crop_guess"].dropna()
                                            if str(c).strip()})),
            "boundary_mismatch": bool(g["boundary_mismatch"].any()),
            "seconds_spent": float(g["seconds_spent"].fillna(0).sum()),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------------
def ingest(csv_dir: Path, sample: gpd.GeoDataFrame, item_key: pd.DataFrame,
           s2_meta: pd.DataFrame | None = None,
           adjudication: dict[str, str] | None = None,
           out_dir: Path | None = None, min_pixels: int = MIN_PIXELS) -> gpd.GeoDataFrame:
    """CSVs + the frozen sample -> ``labelled_parcels.parquet`` + the kappa report."""
    out_dir = Path(out_dir) if out_dir else Path(csv_dir).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_csvs(Path(csv_dir))
    key = item_key[["item_id", "COD_PREDIO"]].drop_duplicates()
    n_key = key.groupby("item_id").size()
    if (n_key > 1).any():
        raise ValueError(f"{int((n_key > 1).sum())} item_ids map to >1 COD_PREDIO")
    unknown = sorted(set(raw.item_id) - set(key.item_id))
    if unknown:
        raise ValueError(f"{len(unknown)} item_ids in the CSVs are not in item_key "
                         f"(e.g. {unknown[:3]})")

    kappa = agreement(raw)
    resolved = resolve(raw, adjudication)
    # The sample carries its own `item_id`; so does the key. Merging on COD_PREDIO with
    # both present silently produces `item_id_x`/`item_id_y` and the next merge fails on a
    # missing column. The **key is authoritative** — it records what was actually emitted
    # into a shard — so the sample's copy is dropped rather than reconciled.
    out = (sample.drop(columns=["item_id"], errors="ignore")
           .merge(key, on="COD_PREDIO", how="inner")
           .merge(resolved, on="item_id", how="inner"))

    # ---- the exclusions that hold regardless of any later modelling decision ----
    out["exclude_reason"] = ""
    out.loc[out.boundary_mismatch, "exclude_reason"] = "boundary_mismatch"
    # UNSURE carries no class information, so it cannot train anything — but it is a real
    # measurement of what the imagery could not resolve, so the row is kept and flagged
    # rather than dropped. It is also the numerator of gate G2.
    m = (out["label"] == ABSTAIN) & (out.exclude_reason == "")
    out.loc[m, "exclude_reason"] = "unsure"
    if s2_meta is not None:
        thin = set(s2_meta.loc[s2_meta["n_px_median"] < min_pixels, "COD_PREDIO"]
                   .astype(str))
        m = out["COD_PREDIO"].astype(str).isin(thin) & (out.exclude_reason == "")
        out.loc[m, "exclude_reason"] = "sub_pixel_parcel"
    no_s2 = set()
    if s2_meta is not None:
        no_s2 = set(out["COD_PREDIO"].astype(str)) - set(s2_meta["COD_PREDIO"].astype(str))
        m = out["COD_PREDIO"].astype(str).isin(no_s2) & (out.exclude_reason == "")
        out.loc[m, "exclude_reason"] = "no_s2_observations"
    out["usable"] = out["exclude_reason"] == ""

    gates = _gates(out, kappa)
    _report(out, kappa, gates)

    keep = ["COD_PREDIO", "geometry", "item_id", "label", "crop_guess",
            "boundary_mismatch", "resolution", "n_labellers", "seconds_spent",
            "imagery_date", "imagery_res", "dept", "declared_class", "stratum",
            "weight", "region_id", "split", "fold", "batch", "area_ha",
            "exclude_reason", "usable"]
    out = gpd.GeoDataFrame(out[[c for c in keep if c in out.columns]],
                           geometry="geometry", crs=sample.crs)
    out.to_parquet(out_dir / "labelled_parcels.parquet", index=False)
    with open(out_dir / "kappa_report.json", "w") as f:
        json.dump({"kappa": kappa, "gates": gates}, f, indent=2, default=str)
    (out.groupby(["dept", "declared_class", "label"]).size()
     .rename("n").reset_index().to_csv(out_dir / "stratum_counts.csv", index=False))
    print(f"\nwrote labelled_parcels.parquet ({len(out):,} rows), kappa_report.json, "
          f"stratum_counts.csv to {out_dir}")
    return out


def _gates(out: pd.DataFrame, kappa: dict) -> dict:
    """G1/G2/G3 evaluated from the ingested labels (§10)."""
    usable = out[out.usable]
    per_dept = usable.groupby("dept").size()
    per_class = usable.groupby("label").size().reindex(CLASSES).fillna(0).astype(int)
    unsure = float((out["label"] == ABSTAIN).mean()) if len(out) else float("nan")
    k = kappa.get("kappa_called")
    return {
        "G1_kappa_called": {"value": k, "criterion": ">= 0.75",
                            "pass": bool(k is not None and k >= 0.75),
                            "note": "on parcels BOTH labellers called; kappa_all "
                                    "(abstains included) is in the kappa report"},
        # G2 is now a single number. It used to carry
        # `low_confidence_share_among_called` beside it as a softer reading of the same
        # thing; with the confidence control gone there is one measure of doubt and it is
        # the one the gate reads.
        "G2_unsure_share": {"value": round(unsure, 4), "criterion": "< 0.25",
                            "pass": bool(unsure < 0.25)},
        "G3_min_per_dept": {"value": int(per_dept.min()) if len(per_dept) else 0,
                            "criterion": ">= 35",
                            "pass": bool(len(per_dept) and per_dept.min() >= 35),
                            "departments_below": per_dept[per_dept < 35].to_dict()},
        "G3_min_per_class": {"value": int(per_class.min()) if len(per_class) else 0,
                             "criterion": ">= 150",
                             "pass": bool(len(per_class) and per_class.min() >= 150),
                             "classes_below": per_class[per_class < 150].to_dict(),
                             "note": "over the five real classes; UNSURE is an abstain, "
                                     "not a class with a quota. NON_AGRICULTURE was split "
                                     "out of OTHER after the draw was fixed, so it has no "
                                     "stratum of its own and may well land below the "
                                     "floor — `classes_below` says which, and a class "
                                     "under it is a decision to pool at training time, "
                                     "not a reason to re-draw"},
    }


def _report(out: pd.DataFrame, kappa: dict, gates: dict) -> None:
    print(f"\ningested {len(out):,} labelled parcels "
          f"({int(out.usable.sum()):,} usable)")
    print("\nlabel distribution:")
    print(out["label"].value_counts().to_string())
    if kappa.get("n_overlap"):
        kc = kappa.get("kappa_called")
        print(f"\nkappa on {kappa['n_overlap']} double-labelled parcels: "
              f"called-only {kc:.3f} (n={kappa['n_both_called']}), "
              f"all values {kappa['kappa_all']:.3f}, "
              f"perennial-vs-rest {kappa['kappa_perennial']:.3f} "
              f"(raw agreement {kappa['raw_agreement']:.1%})")
        print("  abstain rate per labeller: "
              + ", ".join(f"{k} {v:.1%}" for k, v in kappa['abstain_rate'].items())
              + f"; only one abstained on {kappa['one_abstained_other_did_not']}")
    print("\nexclusions:")
    print(out.loc[~out.usable, "exclude_reason"].value_counts().to_string()
          or "  none")
    print("\ngates:")
    for name, g in gates.items():
        print(f"  {'PASS' if g['pass'] else 'FAIL'}  {name}: "
              f"{g['value']} (criterion {g['criterion']})")


# ------------------------------------------------------------------------------------
# §11 step 10 — the campaign's second deliverable
# ------------------------------------------------------------------------------------
def transition_matrix(labelled: pd.DataFrame, weighted: bool = True) -> pd.DataFrame:
    """Weighted declared (1996-2006) -> observed (2019+) transition matrix.

    A descriptive conversion estimate read straight off the labels with **no classifier in
    it** — the question this project has never been able to answer. Rows are the declared
    PETT class, columns the photo-interpreted label, cells are weighted row shares with a
    binomial CI from the effective sample size.
    """
    # `usable` already drops UNSURE, boundary mismatches and sub-pixel parcels, so the
    # transition matrix describes the parcels that were actually called.
    d = labelled[labelled.usable].copy()
    d["w"] = d["weight"].fillna(1.0) if weighted else 1.0
    tab = d.pivot_table(index="declared_class", columns="label", values="w",
                        aggfunc="sum", fill_value=0.0)
    n = d.pivot_table(index="declared_class", columns="label", values="w",
                      aggfunc="size", fill_value=0)
    share = tab.div(tab.sum(axis=1), axis=0)
    # design effect from the weights, so the CI reflects the stratified draw
    deff = d.groupby("declared_class")["w"].apply(
        lambda w: 1 + (w.std() / w.mean()) ** 2 if w.mean() else 1.0)
    n_eff = n.sum(axis=1) / deff
    ci = share.copy()
    for c in share.columns:
        p = share[c]
        ci[c] = 1.96 * np.sqrt((p * (1 - p) / n_eff).clip(lower=0))
    out = share.round(4)
    out.columns = pd.MultiIndex.from_product([["share"], out.columns])
    ci.columns = pd.MultiIndex.from_product([["ci95"], ci.columns])
    return pd.concat([out, ci.round(4)], axis=1)
