"""3-class group assignment (plan §13). No GEE, no data files — pure policy logic."""

from __future__ import annotations

import pandas as pd
import pytest

from crop_classifier.perennial import labels3


@pytest.fixture
def cfg():
    """A miniature lexicon exercising every resolution path."""
    return {
        "classes": ["PERENNIAL", "ANNUAL", "PASTURE_FALLOW"],
        "group_priority": ["PERENNIAL", "ANNUAL", "PASTURE_FALLOW"],
        "cana_policy": "annual",
        "land_prep_policy": "pasture_fallow",
        "woody_noncrop_policy": "exclude",
        "category_default": {"pasture": "PASTURE_FALLOW", "fallow": "PASTURE_FALLOW",
                             "land_prep": "PASTURE_FALLOW", "unspecified": None},
        "crop_fallback": "ANNUAL",
        "max_unassigned_frac": 0.02,
        "perennial": ["MANGO", "CAFE"],
        "annual": ["ARROZ", "MAIZ"],
        "pasture_fallow": ["POTRERO"],
        "woody_noncrop": ["ALGARROBO"],
        "four_class": False,
    }


@pytest.fixture
def resolver(cfg):
    return labels3.build_resolver(cfg)


def assign(crops, cats, cfg, resolver):
    return labels3.assign_group(crops, cats, cfg, resolver)


# --- single tokens ---
def test_single_crop_per_group(cfg, resolver):
    assert assign(["MANGO"], ["crop"], cfg, resolver) == ("PERENNIAL", "single")
    assert assign(["ARROZ"], ["crop"], cfg, resolver) == ("ANNUAL", "single")
    assert assign(["POTRERO"], ["crop"], cfg, resolver) == ("PASTURE_FALLOW", "single")


def test_category_defaults(cfg, resolver):
    assert assign(["DESCANSO"], ["fallow"], cfg, resolver) == ("PASTURE_FALLOW", "single")
    assert assign(["PASTO"], ["pasture"], cfg, resolver) == ("PASTURE_FALLOW", "single")


def test_unspecified_alone_is_unmappable(cfg, resolver):
    assert assign(["SIN ESPECIFICAR"], ["unspecified"], cfg, resolver) \
        == (None, "unmappable")


def test_unlisted_crop_token_falls_back_to_annual(cfg, resolver):
    group, reason = assign(["QUINUA"], ["crop"], cfg, resolver)
    assert (group, reason) == ("ANNUAL", "single")
    assert labels3.resolve_token("QUINUA", "crop", cfg, resolver)[1] == "crop_fallback"


# --- multi-crop (D3) ---
def test_same_group_multicrop(cfg, resolver):
    assert assign(["CAFE", "MANGO"], ["crop", "crop"], cfg, resolver) \
        == ("PERENNIAL", "single")


def test_mixed_group_uses_priority(cfg, resolver):
    # a tree canopy dominates a 30 m pixel regardless of what grows beneath it
    assert assign(["CAFE", "ARROZ"], ["crop", "crop"], cfg, resolver) \
        == ("PERENNIAL", "mixed_priority")
    assert assign(["ARROZ", "PASTO"], ["crop", "pasture"], cfg, resolver) \
        == ("ANNUAL", "mixed_priority")


def test_priority_order_is_config_driven(cfg, resolver):
    cfg = {**cfg, "group_priority": ["ANNUAL", "PERENNIAL", "PASTURE_FALLOW"]}
    assert assign(["CAFE", "ARROZ"], ["crop", "crop"], cfg, resolver) \
        == ("ANNUAL", "mixed_priority")


# --- policy flags ---
def test_woody_noncrop_excludes_whole_parcel(cfg, resolver):
    assert assign(["ALGARROBO"], ["crop"], cfg, resolver) == (None, "woody_noncrop")
    # …even when a real crop is also declared: the parcel's class would be unreadable
    assert assign(["ALGARROBO", "ARROZ"], ["crop", "crop"], cfg, resolver) \
        == (None, "woody_noncrop")


def test_woody_noncrop_policy_perennial(cfg):
    cfg = {**cfg, "woody_noncrop_policy": "perennial"}
    r = labels3.build_resolver(cfg)
    assert assign(["ALGARROBO"], ["crop"], cfg, r) == ("PERENNIAL", "single")


def test_land_prep_policy(cfg, resolver):
    assert assign(["MECANIZADO"], ["land_prep"], cfg, resolver) \
        == ("PASTURE_FALLOW", "single")
    dropped = {**cfg, "land_prep_policy": "drop"}
    assert assign(["MECANIZADO"], ["land_prep"], dropped, resolver) \
        == (None, "unmappable")
    # a land-prep token alongside a crop must not change the crop's class
    assert assign(["ARROZ", "MECANIZADO"], ["crop", "land_prep"], dropped, resolver) \
        == ("ANNUAL", "single")


def test_cana_policy(cfg):
    r_annual = labels3.build_resolver(cfg)
    assert assign(["CAÑA DE AZUCAR"], ["crop"], cfg, r_annual) == ("ANNUAL", "single")
    per = {**cfg, "cana_policy": "perennial"}
    assert assign(["CAÑA DE AZUCAR"], ["crop"], per, labels3.build_resolver(per)) \
        == ("PERENNIAL", "single")


def test_four_class_variant(cfg):
    four = {**cfg, "four_class": True,
            "four_class_names": {"pasture": "PASTURE", "fallow": "FALLOW",
                                 "land_prep": "FALLOW"},
            "group_priority": ["PERENNIAL", "ANNUAL", "PASTURE", "FALLOW"]}
    r = labels3.build_resolver(four)
    assert assign(["PASTO"], ["pasture"], four, r) == ("PASTURE", "single")
    assert assign(["DESCANSO"], ["fallow"], four, r) == ("FALLOW", "single")


# --- config integrity ---
def test_duplicate_token_in_two_groups_raises(cfg):
    bad = {**cfg, "annual": ["ARROZ", "MANGO"]}
    with pytest.raises(ValueError, match="two groups"):
        labels3.build_resolver(bad)


def test_group_priority_must_cover_lexicon(cfg, resolver):
    bad = {**cfg, "group_priority": ["ANNUAL", "PASTURE_FALLOW"]}
    with pytest.raises(ValueError, match="not covered by"):
        assign(["MANGO"], ["crop"], bad, resolver)


def test_token_lookup_is_case_and_space_insensitive(cfg, resolver):
    assert labels3.resolve_token("  mango ", "crop", cfg, resolver)[0] == "PERENNIAL"


# --- the unassigned-token budget ---
def test_token_audit_reports_source_and_counts(cfg, resolver):
    records = pd.DataFrame({
        "crop": ["ARROZ"] * 90 + ["MANGO"] * 5 + ["QUINUA"] * 5,
        "category": ["crop"] * 100})
    audit = labels3.token_audit(cfg, resolver, records)
    by_token = audit.set_index("crop")
    assert by_token.loc["ARROZ", "source"] == "lexicon"
    assert by_token.loc["QUINUA", "source"] == "crop_fallback"
    assert by_token.loc["QUINUA", "group"] == "ANNUAL"
    unassigned = audit[audit["source"] == "crop_fallback"]["n_records"].sum()
    assert unassigned / len(records) == 0.05      # over the 2% budget -> build must fail
