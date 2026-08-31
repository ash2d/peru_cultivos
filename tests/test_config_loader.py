"""Guards on ``config/`` and the ``extends:`` resolution.

The configs used to be a copy-forward chain — ``perennial_cenagro.yaml`` was
``perennial_allperu.yaml`` pasted whole, which was ``perennial.yaml`` pasted whole, 559
lines deep, with two "Original header follows" markers in it. They are now diffs resolved
by ``crop_classifier.config_loader``.

Two things need pinning. The first is that the resolution is **correct** — every config
still loads to a usable policy. The second is the one that matters scientifically: the
derived lexicons claim to be **additive only**, because if a derived config reassigned an
inherited token then part of a measured "change in the land" would be a change of
definition (``RESULTS.md`` §8.5). That claim used to be a sentence in a header. Here it is
a test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from crop_classifier.config_loader import load_yaml_config
from crop_classifier.perennial.labels3 import build_resolver

CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "crop_classifier" / "config"

# child -> parent, as declared by the `extends:` keys. Kept here as well so that a
# silently-repointed parent is a test failure rather than a quiet change of label policy.
EXPECTED_CHAIN = {
    "perennial_4c.yaml": "perennial.yaml",
    "perennial_binary.yaml": "perennial.yaml",
    "perennial_allperu.yaml": "perennial.yaml",
    "perennial_cenagro.yaml": "perennial_allperu.yaml",
    "split_b3000.yaml": "split.yaml",
    "split_window.yaml": "split_allperu.yaml",
    "split_window_b3000.yaml": "split_window.yaml",
    "split_s2labels.yaml": "split_allperu.yaml",
}

# the lexicons whose token -> group assignment is a label of record
LEXICON_KEYS = ("perennial", "annual", "pasture_fallow", "woody_noncrop")


def _all_configs() -> list[Path]:
    return sorted(CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("path", _all_configs(), ids=lambda p: p.name)
def test_every_config_resolves(path):
    cfg = load_yaml_config(path)
    assert isinstance(cfg, dict) and cfg
    assert not (set(cfg) & {"extends", "add", "drop"}), (
        f"{path.name}: merge keys leaked into the resolved config"
    )


def test_the_extends_chain_is_the_expected_one():
    found = {}
    for path in _all_configs():
        raw = yaml.safe_load(path.read_text()) or {}
        if "extends" in raw:
            found[path.name] = raw["extends"]
    assert found == EXPECTED_CHAIN, (
        "an `extends:` parent changed; that changes a label policy, so update "
        "EXPECTED_CHAIN deliberately.\n"
        f"  found:    {found}\n  expected: {EXPECTED_CHAIN}"
    )


@pytest.mark.parametrize(
    "child,parent",
    [("perennial_allperu.yaml", "perennial.yaml"),
     ("perennial_cenagro.yaml", "perennial_allperu.yaml")],
)
def test_the_derived_lexicons_are_additive_only(child, parent):
    """No inherited token may change group, and none may be dropped.

    This is the discipline `RESULTS.md` §8.5 rests on: the PETT side and the census side
    are classified by the same rules, so a measured change is a change in the land.
    """
    kid, mum = load_yaml_config(CONFIG_DIR / child), load_yaml_config(CONFIG_DIR / parent)
    kid_group = {t: g for k in LEXICON_KEYS for t, g in ((t, k) for t in kid.get(k, []))}
    mum_group = {t: g for k in LEXICON_KEYS for t, g in ((t, k) for t in mum.get(k, []))}

    reassigned = {t: (g, kid_group.get(t)) for t, g in mum_group.items() if kid_group.get(t) != g}
    assert not reassigned, (
        f"{child} reassigns or drops tokens inherited from {parent}: {reassigned}. "
        "A derived lexicon must ADD only — otherwise part of any measured change is a "
        "change of definition (RESULTS.md §8.5)."
    )
    assert len(kid_group) > len(mum_group), f"{child} adds nothing to {parent}"


@pytest.mark.parametrize(
    "name", ["perennial.yaml", "perennial_allperu.yaml", "perennial_cenagro.yaml",
             "perennial_4c.yaml", "perennial_binary.yaml"])
def test_no_token_resolves_to_two_groups(name):
    """`build_resolver` raises on a duplicate; run it so a bad merge cannot ship."""
    resolver = build_resolver(load_yaml_config(CONFIG_DIR / name))
    assert resolver


def test_a_circular_extends_is_an_error(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\nx: 1\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\ny: 2\n")
    with pytest.raises(ValueError, match="circular"):
        load_yaml_config(tmp_path / "a.yaml")


def test_add_onto_a_missing_key_is_an_error(tmp_path):
    """A typo in an `add:` key must not silently create a new, unread group."""
    (tmp_path / "base.yaml").write_text("perennial: [MANGO]\n")
    (tmp_path / "kid.yaml").write_text("extends: base.yaml\nadd:\n  perenial: [LIMON]\n")
    with pytest.raises(KeyError, match="perenial"):
        load_yaml_config(tmp_path / "kid.yaml")


def test_add_appends_and_dedupes_and_drop_removes(tmp_path):
    (tmp_path / "base.yaml").write_text("perennial: [MANGO, LIMON]\nannual: [ARROZ]\nk: 1\n")
    (tmp_path / "kid.yaml").write_text(
        "extends: base.yaml\nadd:\n  perennial: [LIMON, CACAO]\ndrop: [annual]\nk: 2\n")
    cfg = load_yaml_config(tmp_path / "kid.yaml")
    assert cfg["perennial"] == ["MANGO", "LIMON", "CACAO"]   # appended, deduped, in order
    assert "annual" not in cfg
    assert cfg["k"] == 2                                     # child scalar replaces parent
