"""Every `cc …` command line printed in the documentation actually resolves.

The how-tos are the entry point for someone who has never seen this project, and a command
that no longer exists costs them more than a missing page: they cannot tell whether they typed
it wrong, installed it wrong, or are missing data. This walks the real Typer app rather than a
list of names, so renaming a command breaks the test that names the doc to fix.

It checks that the command PATH exists. It does not run anything — most of these need data,
Earth Engine, or hours.
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import pytest
from typer.main import get_command

from crop_classifier.cli import app

ROOT = Path(__file__).resolve().parents[1]

# `uv run cc …` or the long form, up to the end of the line, a shell operator, or the
# closing backtick of an inline mention — without that last one, prose after a `cc …` span
# is read as arguments.
_CMD = re.compile(r"uv run (?:cc|python -m crop_classifier\.cli)\s+([^\n&|>#`]*)")

# Global options, stripped before resolution. The two that take a value must have the value
# stripped with them, or `-w national` leaves `national` looking like a subcommand.
_GLOBAL_FLAG = {"-q", "--quiet", "--help"}
_GLOBAL_VALUED = {"-w", "--workspace"}
_GLOBAL = _GLOBAL_FLAG | _GLOBAL_VALUED


def _docs() -> list[Path]:
    return [ROOT / "README.md", ROOT / "CLAUDE.md", ROOT / "CONTRIBUTING.md",
            *sorted(ROOT.glob("docs/**/*.md")),
            *sorted(ROOT.glob("src/crop_classifier/**/*.md")),
            *sorted(ROOT.glob("data/demo/README.md"))]


def _invocations() -> list[tuple[str, str]]:
    out = []
    for doc in _docs():
        for line in doc.read_text(encoding="utf-8").splitlines():
            for m in _CMD.finditer(line):
                out.append((str(doc.relative_to(ROOT)), m.group(1).strip()))
    return out


def _tokens(invocation: str) -> list[str]:
    """Shell-ish tokens, with the noise a documentation example carries stripped out."""
    raw = invocation.split(";")[0].replace("\\", " ").split()
    out: list[str] = []
    skip = False
    for tok in raw:
        if skip:
            skip = False
            continue
        if tok in _GLOBAL_VALUED:
            skip = True                      # ...and its value
            continue
        if tok in _GLOBAL_FLAG:
            continue
        # placeholders a reader is meant to substitute: <run>, $m, {a,b}, ...
        if tok.startswith(("<", "$", "{")) or tok in {"...", "…"}:
            continue
        out.append(tok)
    return out


def _resolve(tokens: list[str]) -> tuple[click.Command | None, list[str]]:
    """Walk the token list down the command tree; return (command, leftover tokens)."""
    cmd: click.Command = get_command(app)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-"):
            break
        # duck-typed, not isinstance: Typer's TyperGroup does not always register as a
        # click.Group under the installed click, and an isinstance check silently stops
        # descending — which makes every nested command "pass" for the wrong reason
        if not hasattr(cmd, "get_command"):
            break
        sub = cmd.get_command(click.Context(cmd), tok)
        if sub is None:
            return None, tokens[i:]
        cmd, i = sub, i + 1
    return cmd, tokens[i:]


def test_the_docs_contain_command_examples():
    """A guard on the guard: if the regex stops matching, this file silently passes."""
    assert len(_invocations()) >= 20


@pytest.mark.parametrize("doc,invocation",
                         _invocations(), ids=lambda v: v if isinstance(v, str) else str(v))
def test_every_documented_command_exists(doc, invocation):
    tokens = _tokens(invocation)
    if not tokens:
        return
    cmd, rest = _resolve(tokens)
    assert cmd is not None, (
        f"{doc}: `uv run cc {invocation}` — no such command "
        f"(failed at {rest[0]!r}). Fix the doc or restore the command.")

    # any long option named must be a real one on that command
    known = {o for p in cmd.params for o in getattr(p, "opts", [])} | _GLOBAL
    for tok in rest:
        name = tok.split("=", 1)[0]
        if name.startswith("--") and name not in known:
            pytest.fail(f"{doc}: `uv run cc {invocation}` — {name} is not an option of "
                        f"`{cmd.name}`. Known: {sorted(o for o in known if o.startswith('--'))}")
