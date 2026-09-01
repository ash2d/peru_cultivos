"""Named workspaces, read from ``workspaces.yaml`` at the repo root.

**Why this module exists.** Which dataset a command touches used to be decided by three
environment variables — ``CC_PROC``, ``CC_FEAT``, ``CC_RUNS``. Forget one ``export`` and the
command does not fail: it silently reads or writes the wrong store, and every cookbook recipe
in the docs began with an ``export`` line for exactly that reason. A workspace gives that
triple a *name*, checked against a file, so ``-w national`` replaces three exports that must
agree.

**How it plugs in.** ``activate()`` sets the same three environment variables. Every module
downstream keeps resolving paths through :mod:`crop_classifier.paths` at call time and is
unchanged — the call-time rule in that module still holds and still matters. This layer only
decides *what to put in the variables*, and makes the answer visible.

Omitting ``-w`` leaves the environment untouched, so an explicit ``CC_PROC`` still wins and
pre-existing shell sessions keep working. ``cc workspaces`` prints whichever is in effect.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_NAME = "workspaces.yaml"

#: The three environment variables a workspace sets. Kept as the mechanism deliberately:
#: it is what the whole codebase already reads, and it means a workspace and a hand-set
#: variable cannot disagree.
ENV_VARS = ("CC_PROC", "CC_FEAT", "CC_RUNS")


def config_path() -> Path:
    """The workspaces file. ``CC_WORKSPACES_FILE`` overrides it (used by the tests)."""
    return Path(os.environ.get("CC_WORKSPACES_FILE", ROOT / CONFIG_NAME))


def load_config() -> dict[str, Any]:
    """Parse ``workspaces.yaml``. Raises with the path to edit if it is missing."""
    p = config_path()
    if not p.exists():
        raise SystemExit(
            f"no {CONFIG_NAME} found at {p}.\n"
            f"Copy the one in the repository root and edit `data_root` to point at your data."
        )
    cfg = yaml.safe_load(p.read_text()) or {}
    if "workspaces" not in cfg:
        raise SystemExit(f"{p} has no `workspaces:` section.")
    return cfg


def _root(cfg: dict[str, Any], key: str, default: str) -> Path:
    """Resolve a root directory relative to the config file, so a relative path in the
    file means "next to the file" rather than "next to wherever you happened to cd"."""
    raw = Path(str(cfg.get(key) or default)).expanduser()
    return raw if raw.is_absolute() else (config_path().parent / raw).resolve()


@dataclass(frozen=True)
class Workspace:
    """One named set of directories."""

    name: str
    proc: Path
    feat: Path
    runs: Path
    about: str = ""

    def env(self) -> dict[str, str]:
        return {"CC_PROC": str(self.proc), "CC_FEAT": str(self.feat), "CC_RUNS": str(self.runs)}


def names() -> list[str]:
    """Workspace names, in the order the file lists them."""
    return list(load_config()["workspaces"])


def resolve(name: str) -> Workspace:
    """Look up one workspace and resolve its three directories to absolute paths."""
    cfg = load_config()
    spec = cfg["workspaces"].get(name)
    if spec is None:
        raise SystemExit(
            f"unknown workspace {name!r}. Known: {', '.join(cfg['workspaces'])}.\n"
            f"Add it to {config_path()} or run `cc workspaces` to see them."
        )
    data_root = _root(cfg, "data_root", "./data")
    runs_root = _root(cfg, "runs_root", "./runs")
    return Workspace(
        name=name,
        proc=data_root / str(spec["proc"]),
        feat=data_root / str(spec["feat"]),
        runs=runs_root / str(spec["runs"]),
        about=str(spec.get("about", "")).strip(),
    )


_ACTIVATED: str | None = None


def activated() -> str | None:
    """The workspace name ``activate()`` selected in this process, if any.

    Distinguishes "``-w`` put these variables here" from "the user exported them", which is
    the difference between a normal run and one that is overriding the config file.
    """
    return _ACTIVATED


def activate(name: str) -> Workspace:
    """Select ``name`` for this process by setting the three environment variables.

    Does **not** create the directories — that is ``paths.proc()`` and friends, at call
    time, only for the workspace a command actually touches. Creating all three here would
    scatter empty directories every time someone ran ``--help``.
    """
    global _ACTIVATED
    ws = resolve(name)
    os.environ.update(ws.env())
    _ACTIVATED = name
    return ws


def active() -> dict[str, str | None]:
    """What the three variables currently say, whatever set them."""
    return {v: os.environ.get(v) for v in ENV_VARS}


def gee_project() -> str:
    """The Google Cloud project id for Earth Engine.

    ``GEE_PROJECT`` in the environment wins, then ``workspaces.yaml``. There is no usable
    default: Earth Engine bills a *project*, and another user cannot use this one's.
    """
    from_env = os.environ.get("GEE_PROJECT")
    if from_env:
        return from_env
    proj = (load_config().get("gee_project") or "").strip()
    if not proj:
        raise SystemExit(
            f"no Earth Engine project configured. Set `gee_project:` in {config_path()} "
            f"(or the GEE_PROJECT environment variable) to your own Google Cloud project id, "
            f"then run `uv run earthengine authenticate`."
        )
    return proj


def cenagro_source_dir() -> Path:
    """Where the 25 CENAGRO 2012 ``.dta`` files live. Only ``cc data cenagro-extract``
    needs it, so it stays null for everyone who is not regenerating that extract."""
    raw = load_config().get("cenagro_source_dir")
    if not raw:
        raise SystemExit(
            f"`cenagro_source_dir:` is not set in {config_path()}.\n"
            f"It must point at the 25 CENAGRO 2012 department .dta files (the UDEP OneDrive "
            f"share, ~18 GB). See docs/DATA_ACCESS.md. Only this command needs it."
        )
    return Path(str(raw)).expanduser()
