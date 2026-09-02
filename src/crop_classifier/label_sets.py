"""Label spaces, loaded from ``config/labels/*.yaml``.

**Why a directory and not a dict.** These six label spaces lived as a literal in
``labelling/train_prep.py``, so adding a seventh — the likeliest need when new labels arrive
— meant a Python edit in a module that must never be imported alongside torch. A label space
is a *modelling decision*, and this project's record is that such decisions must be readable
and diffable: the 2012 census mapping left a 4.09 % catch-all that was 80 % one token, and
the headline moved +2.4 pp → +12.5 pp when someone finally printed it.

A label set names how the five classes an annotator can record

    PERENNIAL   ANNUAL   OTHER   WOODY_NON_CROP   NON_AGRICULTURE

collapse into what a model predicts. ``UNSURE`` is an abstain and never a class — it is
dropped upstream, in ``labelling.ingest``, because a guess recorded at confidence 1 is
indistinguishable from a real label.

See ``config/labels/README.md`` for how to add one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "config" / "labels"

#: The vocabulary a label set may collapse *from*. Kept here so a config typo is an error
#: naming the valid classes, not a silently ignored key that changes every inheriting arm.
SOURCE_CLASSES = ("PERENNIAL", "ANNUAL", "OTHER", "WOODY_NON_CROP", "NON_AGRICULTURE")


@dataclass(frozen=True)
class LabelSet:
    """One label space."""

    name: str
    #: source class -> target class, or ``None`` to drop it from training entirely.
    collapse: dict[str, str | None] = field(default_factory=dict)
    about: str = ""
    #: whether the hand-written ``rules`` baseline can be scored in this space at all
    rules_compatible: bool = True

    @property
    def classes(self) -> list[str]:
        """The classes a model trained on this set predicts, sorted as the label map is."""
        out = {self.collapse.get(c, c) for c in SOURCE_CLASSES}
        return sorted(c for c in out if c is not None)

    def apply(self, labels):
        """Map a series of recorded labels into this space. Dropped classes become ``None``."""
        return labels.map(lambda v: self.collapse.get(v, v))


def _path(name: str) -> Path:
    return CONFIG_DIR / f"{name}.yaml"


@cache
def load(name: str) -> LabelSet:
    """Read one label set. Raises with the available names, and the directory to add to."""
    p = _path(name)
    if not p.exists():
        raise SystemExit(
            f"unknown label set {name!r}. Available: {', '.join(available())}.\n"
            f"Add one by copying a file in {CONFIG_DIR} — see the README there. No Python "
            f"change is needed."
        )
    cfg = yaml.safe_load(p.read_text()) or {}
    collapse = cfg.get("collapse") or {}
    unknown = sorted(set(collapse) - set(SOURCE_CLASSES))
    if unknown:
        raise SystemExit(
            f"{p.name}: `collapse` names {unknown}, which are not classes an annotator can "
            f"record. Valid sources: {', '.join(SOURCE_CLASSES)}."
        )
    declared = cfg.get("name", name)
    if declared != name:
        raise SystemExit(f"{p.name} declares name: {declared!r}; it must match the filename.")
    return LabelSet(name=name, collapse=dict(collapse),
                    about=str(cfg.get("about", "")).strip(),
                    rules_compatible=bool(cfg.get("rules_compatible", True)))


def available() -> list[str]:
    """Every label set on disk, by name."""
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def targets() -> dict[str, dict[str, str | None]]:
    """The collapse maps, keyed by name — the shape ``train_prep.TARGETS`` had as a literal."""
    return {n: load(n).collapse for n in available()}


def rules_incompatible() -> set[str]:
    """Label sets the ``rules`` baseline must refuse rather than score."""
    return {n for n in available() if not load(n).rules_compatible}
