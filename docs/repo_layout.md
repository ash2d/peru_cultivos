# Why the repository is shaped like this

A record of the choices a reader would otherwise have to reverse-engineer, made when the
project was prepared for collaborators. If you are trying to *do* something, you want
[`howto/`](howto/) instead.

---

## Workspaces are named, not exported

Which dataset a command touches used to be three environment variables — `CC_PROC`, `CC_FEAT`,
`CC_RUNS` — that had to agree. Forgetting one `export` did not fail: it silently read or wrote
the wrong store, and every recipe in the docs began with an export line for exactly that reason.

`workspaces.yaml` names the triple; `-w national` selects it. It still sets those same three
variables, so nothing downstream changed and `paths.py` still resolves at **call** time — the
rule that keeps one workspace from overwriting another's tables. Every `-w` run prints what it
resolved, because a silent mis-target is the failure this replaces.

## Commands are grouped by task, and the old names still work

`data` / `labels` / `splits` / `satellite` / `train` / `evaluate` / `analysis` / `labelling` /
`advanced` / `archive`.

The historical groups — `features`, `perennial`, `allperu` — grouped commands by *which strand
of the research built them*, which is the right axis for someone who worked on the project and
the wrong one for someone arriving at it (`allperu` alone held 27 commands, of which about four
are the pipeline). They are hidden from `--help` and still resolve, so old scripts and shell
history keep working.

## `cc evaluate` reports four splits, not one

CV, LODO, LOYO and LODYO in one table, each beside its majority-class floor. They existed as
four separate commands, and reading only the first is how `centroid_lat`, an entire
architecture, and a `--climate both` recommendation all got adopted before a held-out department
overturned them. An evaluation that is easy to forget will be skipped, so
the protocol is the default and a single split is the special case. See
[`LESSONS.md`](LESSONS.md).

It leads with **mean ± sd over held-out units**, not the pooled score. Pooling weights the
answer toward whichever department was largest; the mean is what to expect from a *new* one.

## Label spaces are files

`config/labels/*.yaml`, one per space, with the reasoning beside the mapping. They were a dict
literal inside `labelling/train_prep.py`, which made adding one — the thing most likely to be
needed when new labels arrive — a Python edit inside a module that must never be imported
alongside torch. A test adds a brand-new label set and trains on it with nothing under `src/`
edited, so the promise is enforced rather than claimed.

## Closed routes are moved, not deleted

`src/crop_classifier/archive/` holds code whose *estimand* failed a pre-registered gate. The
code works; the question did not survive contact with the data. The README there says what each
one tried and which gate killed it. Deleting it would mean the next person spends a month
rediscovering it.

Not everything closed is in there. `perennial/panel.py`, `diagnostics.py`, `trajectories.py`
and `harmonization.py` belong to failed estimands too, but live code depends on them —
`features/assemble.py` calls `harmonization.oli_to_etm`, and `perennial/report_figures.py`
builds committed figures out of the panel. Tangling the live path to tidy the dead one is a bad
trade, so they stay where they are.

## `data/demo/` is committed on purpose

`data/` is 33 GB, gitignored, and partly licensed. Without a committed sample, a fresh clone
could run nothing at all. The demo is 1,302 real parcels over six departments — enough that
`cc advanced lodo` runs, so a newcomer can see cross-validation and leave-one-department-out
disagree before they have any data of their own.

It is *stratified*, not representative. No share, area or prevalence from it means anything.

## The tests guard the documentation too

* every markdown link resolves, and every doc path named in a code comment exists;
* every `uv run cc …` line printed anywhere in the docs resolves against the real Typer app —
  137 of them — including the options they name;
* every committed figure is cited by something;
* `CLAUDE.md` stays orientation-sized;
* no file under `src/` contains an absolute path to one machine;
* the README quickstart runs in CI on a clean checkout, and its output still contains the floor
  and the select-on-LODO warning.

---

## Deliberately not done

**`allperu/tenure_did.py` was not split up.** It is 1,092 lines and it would read better as
three files. It also produced the project's only actual estimate, and the numbers of record
depend on it. Splitting it is a real risk against a cosmetic gain; if it is ever done, do it
with the DiD outputs regenerated and diffed.

**`labelling/build_html.py` still carries its HTML template inline.** 820 lines, and the
template would be better as a file beside it. Lower priority than anything above.
