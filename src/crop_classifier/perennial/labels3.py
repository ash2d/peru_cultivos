"""3-class label table: PERENNIAL / ANNUAL / PASTURE_FALLOW (plan §3.2).

Simpler than ``labels.py``: no intercrop ``merge`` map and no rare-class policy, because the
3-class map absorbs every crop token. Multi-crop parcels resolve by group priority (D3), so
most of the 4,577 the 12-class build drops as ``multicrop_unmerged`` survive here.

Writes into the current workspace (``CC_PROC``):

* ``modeling_parcels.parquet`` — same schema as the 12-class table;
* ``label_map.json``, ``label_exclusions.csv``;
* ``class_lexicon_resolved.csv`` — every crop token -> group, resolution, record count;
* ``unassigned_tokens.csv`` — ``crop``-category tokens that fell through to
  ``crop_fallback`` (a guess). The build fails if they exceed ``max_unassigned_frac``.

Run with::

    CC_PROC=data/processed/perennial \\
      uv run python -m crop_classifier.cli perennial labels
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from crop_classifier.config_loader import load_yaml_config
from crop_classifier.labels import PIXEL_HA, crop_set_key
from crop_classifier.paths import PROC_SHARED, proc

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def source_tables() -> tuple[Path, Path]:
    """``(training_crop_polygon, training_crop_records)`` for the current workspace.

    Prefers the workspace's own copies (``CC_PROC``), falls back to the shared Piura build.
    The all-Peru workspace writes its own, so it is picked up here without a flag.
    """
    for d in (proc(), PROC_SHARED):
        poly, rec = (d / "training_crop_polygon.parquet",
                     d / "training_crop_records.parquet")
        if poly.exists() and rec.exists():
            return poly, rec
    raise FileNotFoundError(
        f"no training_crop_polygon/records pair in {proc()} or {PROC_SHARED} — "
        f"run build_training_data (Piura) or allperu.build_labels first")


# sugarcane and its OCR variants; the group comes from cfg["cana_policy"] (D1 flag)
CANA_TOKENS = {"CAÑA DE AZUCAR", "CAÑA DEAZUCAR", "CAÑA DE AZCAR", "AZUCAR",
               "CANA DE AZUCAR"}


def load_config(path: Path | None = None) -> dict[str, Any]:
    return load_yaml_config(path or CONFIG_DIR / "perennial.yaml")


# --- token -> group resolution ---
def build_resolver(cfg: dict[str, Any]) -> dict[str, tuple[str | None, str]]:
    """Explicit ``token -> (group, source)`` map from the config lexicon.

    ``source`` records why a token got its group, for the audit CSV. A token listed twice
    raises rather than the later entry winning.
    """
    out: dict[str, tuple[str | None, str]] = {}

    def _put(token: str, group: str | None, source: str) -> None:
        t = token.strip().upper()
        if t in out and out[t][0] != group:
            raise ValueError(f"token {t!r} listed in two groups: "
                             f"{out[t][0]} and {group}")
        out[t] = (group, source)

    for token in cfg.get("woody_noncrop") or []:
        _put(token, None if cfg["woody_noncrop_policy"] == "exclude" else "PERENNIAL",
             "woody_noncrop")
    # In the 4-class diagnostic variant PASTURE_FALLOW does not exist: `pasture_fallow` is
    # bare/resting ground except for the grazing tokens named in `pasture_tokens`.
    four = bool(cfg.get("four_class"))
    names = cfg.get("four_class_names") or {}
    grazing = {t.strip().upper() for t in (cfg.get("pasture_tokens") or [])}
    pf_group = names.get("fallow", "FALLOW") if four else "PASTURE_FALLOW"
    # config key -> group, overridable so a variant label space can name its own lists.
    groups = cfg.get("lexicon_groups") or {
        "perennial": "PERENNIAL", "annual": "ANNUAL", "pasture_fallow": pf_group}
    for key, group in groups.items():
        for token in cfg.get(key) or []:
            g = group
            if four and key == "pasture_fallow" and token.strip().upper() in grazing:
                g = names.get("pasture", "PASTURE")
            _put(token, g, "lexicon")
    # `annual_class` lets a variant label space name its non-perennial class (2-class:
    # NON_PERENNIAL) without hard-coding "ANNUAL".
    annual_class = cfg.get("annual_class", "ANNUAL")
    cana = "PERENNIAL" if cfg["cana_policy"] == "perennial" else annual_class
    for token in CANA_TOKENS:
        _put(token, cana, "cana_policy")
    return out


def category_group(cat: str, cfg: dict[str, Any]) -> tuple[str | None, str]:
    """Group for a non-``crop`` category, honouring ``land_prep_policy`` and the 4-class
    diagnostic variant (D2)."""
    if cat == "land_prep" and cfg.get("land_prep_policy") == "drop":
        return None, "land_prep_dropped"
    if cfg.get("four_class"):
        names = cfg.get("four_class_names") or {}
        if cat in names:
            return names[cat], "category_default_4c"
    defaults = cfg.get("category_default") or {}
    return defaults.get(cat), "category_default"


# Leading noise phrases wrapping a crop name ("PLANTACION DE VID", "BOSQUE DE EUCALIPTOS");
# stripped before re-resolution.
_PREFIX_RE = re.compile(
    r"^(?:CONTIENE|TIENE|SEMBRADO|CULTIVADO)\s+|"
    r"^(?:ASOCIADO|ASOCIADA|ASOCIACION)\s+(?:A|DE|CON)\s+|"
    r"^(?:PLANTACION|PLANTACIONES|CULTIVO|CULTIVOS|SIEMBRA|BOSQUE|AREA|ZONA|TERRENO|"
    r"CAMPO|PARCELA|HUERTO|VIVERO)\s+(?:DE|CON|EN)\s+")


def _lookup(resolver: dict[str, tuple[str | None, str]],
            token: str) -> tuple[str | None, str] | None:
    """Lexicon lookup that also tries the singular form.

    The national registry is full of plurals Piura never had (``TUNAS``, ``PALTOS``,
    ``MELOCOTONES``); listing both forms of every token would double the lexicon for no gain.
    """
    hit = resolver.get(token)
    if hit is not None:
        return hit
    for suffix in ("ES", "S"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            hit = resolver.get(token[: -len(suffix)])
            if hit is not None:
                return hit
    return None


def _stage_regex(cfg: dict[str, Any]) -> re.Pattern | None:
    """Regex stripping growth-stage / association phrases off a crop token.

    Sierra registrars often wrote the crop plus its phenological stage (``MAIZ EN
    FLORACION``, ``PAPA EN FASE DE CRECIMIENTO``) — a productive any-crop x any-stage
    pattern that is ~8 % of national records across thousands of variants. Enumerating it is
    hopeless, so the phrase is stripped and the base token re-resolved. Stage is irrelevant
    here: the label is the year's land state.
    """
    words = [w.strip().upper() for w in (cfg.get("stage_words") or []) if w.strip()]
    if not words:
        return None
    alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    # "<CROP> EN|PARA|DE [FASE DE] <STAGE> [anything]"  /  "<CROP> ASOCIADO|MAS <anything>"
    return re.compile(
        rf"\s+(?:(?:EN|PARA|DE|CON)\s+(?:FASE\s+DE\s+|ESTADO\s+DE\s+)?(?:{alt})"
        rf"|(?:ASOCIADO|ASOCIADA|ASOCIACION|MAS|Y)\b).*$")


def resolve_token(crop: str, cat: str, cfg: dict[str, Any],
                  resolver: dict[str, tuple[str | None, str]],
                  stage_re: re.Pattern | None = None) -> tuple[str | None, str]:
    """One ``(crop, category)`` token -> ``(group, source)``; group None = contributes
    nothing (or, for ``woody_noncrop``, poisons the whole parcel — handled by the caller).
    """
    key = crop.strip().upper()
    hit = _lookup(resolver, key)
    if hit is not None:
        return hit
    if stage_re is not None:
        base = stage_re.sub("", key).strip()
        base = _PREFIX_RE.sub("", base).strip()
        if base and base != key:
            hit = _lookup(resolver, base)
            if hit is not None:
                return hit[0], f"stage_stripped:{hit[1]}"
        key = base or key
    # Last resort before guessing: resolve the token's individual WORDS and combine them by
    # `group_priority`, as for a multi-crop parcel ("PLANTACION DE VID" -> VID). Keeps the
    # national tail (~14,483 free-text tokens around a recognisable crop) off a blanket
    # ANNUAL guess. Recorded as `word_match` so the audit separates it from an exact hit.
    if cfg.get("word_match", False):
        found: list[str] = []
        for w in re.split(r"[^\wÑÁÉÍÓÚÜ]+", key):
            h = _lookup(resolver, w)
            if h is None:
                continue
            if h[1] == "woody_noncrop":
                return h                      # a eucalyptus plantation is still woody
            if h[0] is not None:
                found.append(h[0])
        if found:
            for g in cfg["group_priority"]:
                if g in set(found):
                    return g, "word_match"
    if cat == "crop":
        return cfg.get("crop_fallback"), "crop_fallback"
    return category_group(cat, cfg)


def assign_group(crops: list[str], cats: list[str], cfg: dict[str, Any],
                 resolver: dict[str, tuple[str | None, str]],
                 stage_re: re.Pattern | None = None) -> tuple[str | None, str]:
    """Return ``(class, reason)`` for one parcel; class None means excluded.

    Order (plan §3.2): woody-non-crop veto -> drop unmappable tokens -> highest-priority
    remaining group per ``group_priority``.
    """
    groups, sources = [], []
    for crop, cat in zip(crops, cats):
        g, src = resolve_token(crop, cat, cfg, resolver, stage_re)
        if src == "woody_noncrop" and cfg["woody_noncrop_policy"] == "exclude":
            return None, "woody_noncrop"
        if g is not None:
            groups.append(g)
            sources.append(src)
    if not groups:
        return None, "unmappable"
    distinct = set(groups)
    for g in cfg["group_priority"]:
        if g in distinct:
            return g, ("single" if len(distinct) == 1 else "mixed_priority")
    # group_priority not covering the lexicon is a config bug, not data
    raise ValueError(f"groups {sorted(distinct)} not covered by "
                     f"group_priority={cfg['group_priority']}")


# --- audit: which tokens resolved how, and how many records rest on a guess ---
def token_audit(cfg: dict[str, Any], resolver: dict[str, tuple[str | None, str]],
                records: pd.DataFrame,
                stage_re: re.Pattern | None = None) -> pd.DataFrame:
    """Every distinct token in the record table -> group, source, record count."""
    counts = (records.groupby(["crop", "category"]).size()
              .reset_index(name="n_records"))
    res = counts.apply(lambda r: resolve_token(r["crop"], r["category"], cfg, resolver,
                                              stage_re),
                       axis=1)
    counts["group"] = [g for g, _ in res]
    counts["source"] = [s for _, s in res]
    return counts.sort_values("n_records", ascending=False).reset_index(drop=True)


# --- build ---
def build(config_path: Path | None = None, save: bool = True) -> gpd.GeoDataFrame:
    cfg = load_config(config_path)
    resolver = build_resolver(cfg)
    stage_re = _stage_regex(cfg)
    out_dir = proc()
    f_poly, f_records = source_tables()
    gdf = gpd.read_parquet(f_poly)
    print(f"loaded {len(gdf):,} labelled polygons")

    # ---- token audit + the unassigned-token budget (fail loudly, plan §3.1) ----
    records = pd.read_parquet(f_records)
    audit = token_audit(cfg, resolver, records, stage_re)
    unassigned = audit[audit["source"] == "crop_fallback"]
    frac = unassigned["n_records"].sum() / len(records)
    print(f"lexicon: {len(audit):,} distinct tokens; "
          f"{len(unassigned):,} unassigned `crop` tokens -> {cfg['crop_fallback']} "
          f"({unassigned['n_records'].sum():,} records, {frac:.2%})")

    # ⚠️ Print the tail, most-frequent first, always — not only on failure. A catch-all is
    # never uniform and the budget check cannot see that: mapping the 2012 census left
    # 4.09 % of tokens on a blanket ANNUAL, 80 % of it the single token `VERGEL FRUTICOLA`
    # (a perennial), which moved a headline +2.4 pp -> +12.5 pp. The tail was already in
    # unassigned_tokens.csv but unread. (RESULTS.md §8.5)
    if len(unassigned):
        top = unassigned.sort_values("n_records", ascending=False).head(10)
        share = top["n_records"].to_numpy() / max(unassigned["n_records"].sum(), 1)
        print(f"  the tail, most frequent first -> all becoming {cfg['crop_fallback']}:")
        for (_, row), sh in zip(top.iterrows(), share, strict=False):
            print(f"    {str(row['crop'])[:32]:<32} {int(row['n_records']):>8,} records "
                  f"({sh:6.1%} of the tail)")
        if len(unassigned) > 10:
            print(f"    ... and {len(unassigned) - 10:,} more, in unassigned_tokens.csv")
        print("  ⚠️  read these. A catch-all is never uniformly distributed, and one token "
              "at\n      80 % of the tail once moved a headline from +2.4 pp to +12.5 pp.")

    if frac > cfg["max_unassigned_frac"]:
        raise AssertionError(
            f"unassigned tokens cover {frac:.2%} of records, over the "
            f"{cfg['max_unassigned_frac']:.0%} budget — extend the lexicon in "
            f"config/perennial.yaml (see unassigned_tokens.csv)")

    # ---- group per parcel ----
    lab = gdf.apply(lambda r: assign_group(list(r["crops"]), list(r["crop_categories"]),
                                           cfg, resolver, stage_re), axis=1)
    gdf["label_raw"] = [t[0] for t in lab]
    gdf["label_reason"] = [t[1] for t in lab]

    # ---- hard gates: same as labels.py, plus an explicit year range ----
    gdf["gate_area_ok"] = gdf["area_ha"].between(cfg["area_min_ha"], cfg["area_max_ha"])
    year_ok = gdf["year"].notna() & gdf["year"].between(cfg["year_min"], cfg["year_max"])
    gdf["gate_year_ok"] = year_ok if cfg.get("require_year", True) else True

    excl: list[dict[str, Any]] = []

    def _tally(mask: pd.Series, reason: str) -> None:
        excl.append({"reason": reason, "n_parcels": int(mask.sum())})

    _tally(gdf["label_reason"].eq("woody_noncrop"), "woody non-crop (not an export crop)")
    _tally(gdf["label_reason"].eq("unmappable"), "no mappable crop token")
    has_label = gdf["label_raw"].notna()
    _tally(has_label & ~gdf["gate_area_ok"],
           f"area outside {cfg['area_min_ha']}-{cfg['area_max_ha']} ha")
    _tally(has_label & gdf["gate_area_ok"] & ~gdf["gate_year_ok"],
           f"year missing or outside {cfg['year_min']}-{cfg['year_max']}")

    df = gdf[has_label & gdf["gate_area_ok"] & gdf["gate_year_ok"]].copy()
    df["label"] = df["label_raw"]

    classes = sorted(df["label"].unique())
    label_map = {c: i for i, c in enumerate(classes)}
    df["label_id"] = df["label"].map(label_map).astype(int)

    # ---- static columns (schema identical to labels.py — downstream is unchanged) ----
    cen = df.geometry.representative_point()
    df["centroid_lon"], df["centroid_lat"] = cen.x, cen.y
    df["n_pixels_est"] = df["area_ha"] / PIXEL_HA
    df["year"] = df["year"].astype(int)
    df["crop_set"] = df["crops"].map(crop_set_key)
    df["n_valid_obs"] = pd.array([pd.NA] * len(df), dtype="Int64")
    df["max_gap"] = pd.array([pd.NA] * len(df), dtype="Int64")
    df["quality_ok"] = pd.array([pd.NA] * len(df), dtype="boolean")

    cols = ["COD_PREDIO", "label", "label_id", "label_reason", "crop_set", "year",
            "area_ha", "n_pixels_est", "centroid_lon", "centroid_lat",
            "n_valid_obs", "max_gap", "quality_ok", "geometry"]
    # `dept` (all-Peru build only) is the sampling stratum and the LODO held-out unit
    if "dept" in df.columns:
        cols.insert(1, "dept")
    out = df[cols].reset_index(drop=True)

    # ---- report ----
    print(f"\neligible parcels: {len(out):,}  ({len(classes)} classes)")
    vc = out["label"].value_counts()
    for c, n in vc.items():
        print(f"  {c:<16} {n:>7,}  ({n / len(out):.1%})")
    print(f"\nmulti-crop parcels resolved by priority: "
          f"{int((df['label_reason'] == 'mixed_priority').sum()):,}")
    print("\nexclusions:")
    excl_df = pd.DataFrame(excl)
    print(excl_df.to_string(index=False))

    if save:
        out.to_parquet(out_dir / "modeling_parcels.parquet", index=False)
        with open(out_dir / "label_map.json", "w") as f:
            json.dump(label_map, f, indent=2, ensure_ascii=False)
        excl_df.to_csv(out_dir / "label_exclusions.csv", index=False)
        audit.to_csv(out_dir / "class_lexicon_resolved.csv", index=False)
        unassigned.to_csv(out_dir / "unassigned_tokens.csv", index=False)
        print(f"\nwrote {out_dir}/modeling_parcels.parquet, label_map.json, "
              f"label_exclusions.csv, class_lexicon_resolved.csv, unassigned_tokens.csv")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the 3-class label table (plan §3)")
    ap.add_argument("--config", type=Path, default=None, help="path to perennial.yaml")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    build(config_path=args.config, save=not args.no_save)


if __name__ == "__main__":
    main()
