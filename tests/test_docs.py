"""Guards on the documentation set itself.

The project once carried 23 markdown files / 13,336 lines, most of it results narrative
appended to ``CLAUDE.md`` — which is auto-loaded into every agent context. These tests are
what stop that coming back: they fail if a doc link dangles or if ``CLAUDE.md`` grows back
into a results document.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Orientation only. Results go in docs/RESULTS.md, status in docs/STATUS.md.
CLAUDE_MD_MAX_LINES = 250

# The whole documentation set. Adding a file here is a deliberate act, not an accident.
EXPECTED_DOCS = {
    "README.md",
    "CLAUDE.md",
    "docs/STATUS.md",
    "docs/RESULTS.md",
    "docs/LESSONS.md",
    "docs/DATA.md",
    "docs/PIPELINE.md",
    "docs/REPORT.md",
    "docs/s2_labelling/plan.md",
    "docs/s2_labelling/codebook.md",
}

_LINK = re.compile(r"\]\(([^)]+?\.md)(?:#[^)]*)?\)")


def _docs() -> list[Path]:
    out = [ROOT / "README.md", ROOT / "CLAUDE.md"]
    out += sorted(ROOT.glob("docs/**/*.md"))
    return [p for p in out if p.exists()]


def test_no_internal_markdown_link_is_broken():
    broken = []
    for doc in _docs():
        for target in _LINK.findall(doc.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://")):
                continue
            if not (doc.parent / target).resolve().exists():
                broken.append(f"{doc.relative_to(ROOT)} -> {target}")
    assert not broken, "broken markdown links:\n  " + "\n  ".join(broken)


def test_claude_md_stays_orientation_sized():
    n = len((ROOT / "CLAUDE.md").read_text(encoding="utf-8").splitlines())
    assert n <= CLAUDE_MD_MAX_LINES, (
        f"CLAUDE.md is {n} lines (limit {CLAUDE_MD_MAX_LINES}). It is auto-loaded into every "
        "agent context, so it must stay orientation: put results in docs/RESULTS.md and "
        "current state in docs/STATUS.md."
    )


def test_the_documentation_set_is_the_expected_one():
    found = {str(p.relative_to(ROOT)) for p in _docs()}
    assert found == EXPECTED_DOCS, (
        "the doc set changed; update EXPECTED_DOCS deliberately.\n"
        f"  added:   {sorted(found - EXPECTED_DOCS)}\n"
        f"  missing: {sorted(EXPECTED_DOCS - found)}"
    )


@pytest.mark.parametrize("name", sorted(EXPECTED_DOCS))
def test_every_doc_is_non_empty(name):
    assert (ROOT / name).stat().st_size > 0, f"{name} is empty"
