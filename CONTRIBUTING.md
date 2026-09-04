# Contributing

## Environment

**`uv` only.** Never bare `pip` or `python`.

```bash
uv sync
uv run pytest -q
uv run ruff check .
```

Add dependencies with `uv add <pkg>` — never hand-edit `pyproject.toml`'s dependency list.

## Before you push

Both must be clean:

```bash
uv run pytest -q
uv run ruff check .
```

## Never commit

* Anything under `data/` — **except** `data/demo/`, which is synthetic and deliberately tracked.
  The raw archive is licensed and not redistributable ([`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md)).
* Anything under `runs/` or `logs/`.
* Credentials of any kind. `.gitignore` covers `.env`, `*-key.json` and `credentials.json`,
  but check.
* A machine-specific absolute path in `src/` — there is a test that greps for this, because a
  grep is the only thing that notices. Configuration goes in `workspaces.yaml`.

## If a result changes, stop

The numbers in [`docs/RESULTS.md`](docs/RESULTS.md) are the record. A refactor that moves one is
not a refactor.

If you find that a change alters a metric: **stop, and say so** — in the pull request, with the
before and after. Do not update `RESULTS.md` to match. Either the change is wrong, or the number
was, and which one it is deserves a decision rather than a silent edit.

## Evaluation is not optional

Any claim about a model is reported over **CV / LODO / LOYO / LODYO**, with the majority-class
floor beside every macro-F1. `cc evaluate` does all of that in one command specifically so it
cannot be skipped by accident.

A cross-validation number on its own is not a result here. That is not a style preference — it
is the finding that overturned a feature, an architecture and a covariate recommendation in this
project, each time only when a whole department was held out. See
[`docs/LESSONS.md`](docs/LESSONS.md).

## Adding things

**A new label set** is a YAML file in `src/crop_classifier/config/labels/` — no Python.
[`docs/howto/06_reference.md`](docs/howto/06_reference.md).

**A new workspace** is an entry in `workspaces.yaml`.

**A closed route** — code whose estimand failed a pre-registered gate — goes in
`src/crop_classifier/archive/`, with a row in the README there saying what it tried and which
gate killed it. It does not get deleted: "we tried this and measured why it does not work" is a
result, and deleting it means the next person spends a month rediscovering it.

## Docs

`tests/test_docs.py` enforces that the documentation set is deliberate: every markdown link
resolves, every doc-path mentioned in a code comment exists, every committed figure is cited by
something, and `CLAUDE.md` stays orientation-sized. Adding a doc means adding it to
`EXPECTED_DOCS` — on purpose, not by accident.

## Two things that will waste your afternoon

**Never import torch and lightgbm in one process on macOS.** Each bundles its own libomp;
co-loading them segfaults with exit 139 and no traceback. The model registry is lazy for this
reason, and the test suite trains LightGBM in a subprocess. Loop over models in the **shell**.

**`quality_ok` is a nullable boolean** where NA means "not measured". Compare with `== True` or
`.isna()`, never with plain truthiness.
