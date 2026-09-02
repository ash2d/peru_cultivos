"""`cc data verify` — and the guarantee that what it calls `repo` is actually committed.

The manifest's whole value is the distinction between "your clone is broken", "run this
command" and "go and ask someone for data". If a `repo` entry is not tracked by git, the
first turns into a lie and a new collaborator is told their clone is broken when the file was
never shipped.

That is not hypothetical: a missing input in this project usually does not crash. Four of the
raw datasets return a plausible empty or column-less result instead of an error
(`docs/DATA.md` §4), which is the reason this command exists at all.
"""

from __future__ import annotations

import subprocess

import pytest

from crop_classifier.manifest import CAPABILITIES, ROOT, Need, check, status

PROVENANCES = {"repo", "derive", "obtain"}


def _tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    return set(out.stdout.split())


def test_capability_names_are_unique():
    names = [c.name for c in CAPABILITIES]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("cap", CAPABILITIES, ids=lambda c: c.name)
def test_every_need_declares_a_known_provenance(cap):
    for need in cap.needs:
        assert need.provenance in PROVENANCES, f"{need.path}: {need.provenance!r}"


@pytest.mark.parametrize("cap", CAPABILITIES, ids=lambda c: c.name)
def test_a_non_repo_need_says_how_to_get_it(cap):
    """`derive` without a command, or `obtain` without a pointer, leaves the reader exactly
    where they were before they ran the check."""
    for need in cap.needs:
        if need.provenance != "repo":
            assert need.note, f"{cap.name}: {need.path} is `{need.provenance}` with no note"


@pytest.mark.parametrize("cap", CAPABILITIES, ids=lambda c: c.name)
def test_everything_claimed_to_be_in_the_repo_is_tracked_by_git(cap):
    """⭐ The load-bearing one. `repo` means "a clone has this"."""
    tracked = _tracked()
    for need in cap.needs:
        if need.provenance != "repo":
            continue
        path = ROOT / need.path
        # a directory counts as tracked if anything under it is
        ok = (need.path in tracked
              or any(f.startswith(need.path.rstrip("/") + "/") for f in tracked))
        assert ok, (
            f"{cap.name}: {need.path} is declared `repo` but git does not track it. "
            f"Either commit it (check the allowlist in .gitignore) or change its "
            f"provenance. Exists locally: {path.exists()}")


@pytest.mark.parametrize("cap", CAPABILITIES, ids=lambda c: c.name)
def test_nothing_excluded_by_gitignore_is_claimed_as_repo(cap):
    """The subtler half: a file can exist locally, be listed as `repo`, and be ignored — in
    which case it works on the machine that wrote the manifest and nowhere else."""
    paths = [n.path for n in cap.needs if n.provenance == "repo"]
    if not paths:
        return
    out = subprocess.run(["git", "check-ignore", "-v", "--no-index", *paths],
                         cwd=ROOT, capture_output=True, text=True)
    # ⚠️ `check-ignore -v` prints the matching rule for NEGATION patterns too, and exits 0.
    # A line whose pattern begins with `!` means the path is explicitly re-included — the
    # opposite of ignored. Reading exit status alone marks every allowlisted file as ignored.
    ignored = [ln for ln in out.stdout.splitlines()
               if ln and not ln.split("\t")[0].rsplit(":", 1)[-1].startswith("!")]
    assert not ignored, (
        f"{cap.name}: declared `repo` but ignored by .gitignore:\n" + "\n".join(ignored))


def test_the_demo_capability_is_satisfied_by_a_bare_clone():
    """The one capability that must hold with no data obtained at all."""
    demo = next(c for c in CAPABILITIES if c.name == "demo")
    assert status(demo) == "ok"
    assert all(n.provenance == "repo" for n in demo.needs)


def test_status_distinguishes_the_three_answers(tmp_path, monkeypatch):
    from crop_classifier import manifest as M
    from crop_classifier.manifest import Capability
    monkeypatch.setattr(M, "ROOT", tmp_path)
    (tmp_path / "here.txt").write_text("x")

    ok = Capability("a", "", (Need("here.txt", "repo"),))
    derivable = Capability("b", "", (Need("here.txt", "repo"),
                                     Need("gone.txt", "derive", "run something")))
    blocked = Capability("c", "", (Need("gone.txt", "obtain", "ask someone"),))
    assert M.status(ok) == "ok"
    assert M.status(derivable) == "derivable"
    assert M.status(blocked) == "needs-data"


def test_check_reports_one_row_per_need():
    for cap in CAPABILITIES:
        assert len(check(cap)) == len(cap.needs)
