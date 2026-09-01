"""Guards on the documentation set itself.

The project once carried 23 markdown files / 13,336 lines, most of it results narrative
appended to ``CLAUDE.md`` — which is auto-loaded into every agent context. These tests are
what stop that coming back: they fail if a doc link dangles or if ``CLAUDE.md`` grows back
into a results document.

Three more were added after a clean-up on 2026-08-31, one per thing that had actually gone
wrong and that reading the docs would not have surfaced:

* a **code comment** pointing at a doc the consolidation deleted — 41 of them, invisible to
  a checker that only walks markdown;
* **24 orphan figures**, 11 MB, referenced by nothing, because no test asked;
* the ``RESULTS.md`` **scoreboard drifting away from its own body** — it said 412 of 1,112
  parcels were labelled while §8.1, in the same file, said 1,012.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Orientation only. Results go in docs/RESULTS.md, status in docs/STATUS.md.
CLAUDE_MD_MAX_LINES = 250

# A notebook carrying its rendered imagery is tens of MB of base64 in git. 01 keeps its
# TEXT outputs on purpose — they are the forensic record of what each raw file is — so the
# cap is per-notebook and generous rather than a blanket "strip everything".
NOTEBOOK_MAX_MB = 1.0

# The whole documentation set. Adding a file here is a deliberate act, not an accident.
EXPECTED_DOCS = {
    "README.md",
    "CLAUDE.md",
    "docs/STATUS.md",
    "docs/RESULTS.md",
    "docs/LESSONS.md",
    "docs/DATA.md",
    "docs/PIPELINE.md",
    "docs/cenagro_columns.md",
    "docs/DATA_ACCESS.md",
    "docs/howto/01_setup.md",
    "docs/howto/02_get_satellite_data.md",
    "docs/howto/03_new_label_set.md",
    "docs/howto/04_train_and_evaluate.md",
    "docs/howto/05_perennial_change_by_tenure.md",
    "docs/s2_labelling/plan.md",
    "docs/s2_labelling/codebook.md",
    "reports/archive/REPORT.md",
    "docs/repo_layout.md",
}

_LINK = re.compile(r"\]\(([^)]+?\.md)(?:#[^)]*)?\)")
# a doc path mentioned in a comment or docstring, e.g. ``docs/RESULTS.md`` or plan.md §3
_CODE_DOC_REF = re.compile(r"[\w/]*\b[\w]+_?[Pp]lan\.md|\bdocs/[\w/]+\.md|\b[A-Z_]{4,}\.md")


def _docs() -> list[Path]:
    out = [ROOT / "README.md", ROOT / "CLAUDE.md"]
    out += sorted(ROOT.glob("docs/**/*.md"))
    out += sorted(ROOT.glob("reports/**/*.md"))
    return [p for p in out if p.exists()]


def _code_files() -> list[Path]:
    return sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("src/**/*.yaml"),
                   *ROOT.glob("tests/**/*.py")])


def test_no_internal_markdown_link_is_broken():
    broken = []
    for doc in _docs():
        for target in _LINK.findall(doc.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://")):
                continue
            if not (doc.parent / target).resolve().exists():
                broken.append(f"{doc.relative_to(ROOT)} -> {target}")
    assert not broken, "broken markdown links:\n  " + "\n  ".join(broken)


def test_no_code_comment_points_at_a_doc_that_does_not_exist():
    """The consolidation deleted 13 planning docs and left 41 references to them.

    Comments are where a lesson gets recorded next to the code it is about, so a comment
    citing a file that no longer exists is a dead end at exactly the moment someone needs
    it. Markdown-only link checking cannot see them.
    """
    known = {p.name for p in _docs()} | {str(p.relative_to(ROOT)) for p in _docs()}
    dangling = []
    for path in _code_files():
        for ref in set(_CODE_DOC_REF.findall(path.read_text(encoding="utf-8"))):
            ref = ref.strip("`")
            if ref in known or (ROOT / ref).exists():
                continue
            if any(k.endswith("/" + ref) for k in known):
                continue
            dangling.append(f"{path.relative_to(ROOT)} -> {ref}")
    assert not dangling, (
        "code comments cite docs that do not exist:\n  " + "\n  ".join(sorted(dangling))
    )


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


def test_every_committed_figure_is_referenced_by_a_doc():
    """24 of 37 figures were orphans, 11 MB of them, because nothing asked.

    ``docs/figures/`` is checked in, so anything in it that no document cites is weight
    with no reader. Regenerate-and-forget is the normal way it happens.
    """
    # tracked files only: several CLI commands default to writing here, and a figure
    # someone regenerated locally is not repository weight until it is committed.
    tracked = subprocess.run(["git", "ls-files", "docs/figures"], cwd=ROOT,
                             capture_output=True, text=True)
    if tracked.returncode != 0:
        pytest.skip("not a git checkout")
    figures = [Path(line).name for line in tracked.stdout.split() if line]
    cited = [*_docs(), *sorted(ROOT.glob("reports/**/*.tex"))]
    prose = "\n".join(p.read_text(encoding="utf-8") for p in cited)
    orphans = [f for f in figures if f not in prose]
    assert not orphans, (
        "figures in docs/figures/ that no doc references:\n  " + "\n  ".join(orphans)
        + "\nEither cite them or delete them — data/ and runs/ are gitignored, so a "
          "figure nobody reads is the only copy of nothing."
    )


@pytest.mark.parametrize("nb", sorted(ROOT.glob("notebooks/**/*.ipynb")), ids=lambda p: p.name)
def test_notebooks_do_not_carry_their_rendered_imagery(nb):
    mb = nb.stat().st_size / 1e6
    assert mb <= NOTEBOOK_MAX_MB, (
        f"{nb.name} is {mb:.1f} MB (limit {NOTEBOOK_MAX_MB} MB) — it is almost certainly "
        "carrying base64 image outputs. Strip them; keep text outputs where they are the "
        "record of a finding."
    )


def test_notebook_01_keeps_its_text_outputs():
    """01 is the forensic exploration; its printed tables are the only record of them."""
    nb = json.loads((ROOT / "notebooks" / "exploratory" /
                     "01_explore_raw_datasets.ipynb").read_text())
    n = sum(len(c.get("outputs", [])) for c in nb["cells"])
    assert n >= 20, f"01_explore_raw_datasets.ipynb has only {n} outputs left — over-stripped"


def test_the_results_scoreboard_agrees_with_its_own_body():
    """§0's row for the live strand drifted to 412 while §8.1 said 1,012.

    A summary table that disagrees with the section it summarises is worse than no table:
    it is the part people quote.
    """
    text = (ROOT / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    heading = re.search(r"^### 8\.1 .*?(\d[\d,]*\d) of (\d[\d,]*\d)", text, re.M)
    assert heading, "could not find the §8.1 returned-label count"
    returned, drawn = heading.group(1), heading.group(2)

    row = next((ln for ln in text.splitlines()
                if ln.startswith("| 8 ") and "S2 endpoint labelling" in ln), None)
    assert row, "could not find the §0 scoreboard row for strand 8"
    assert returned in row and drawn in row, (
        f"scoreboard row says {row.strip()!r} but §8.1 says {returned} of {drawn} labelled"
    )
