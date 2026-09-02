# 1. Set up

About 15 minutes. No data and no Earth Engine account needed for this page.

---

## Install

The project uses [uv](https://docs.astral.sh/uv/), not pip. Python 3.11.

```bash
git clone <this repo>
cd Coding
uv sync            # installs everything, exactly as pinned
uv run pytest -q   # ~1 min, should be all green
```

Start every command with `uv run`. A bare `python` or `pip` uses whatever interpreter your
shell happens to have, and this project pins its versions for a reason.

## Check it works

```bash
uv run cc -w demo train --model lightgbm --run-name demo
uv run cc -w demo evaluate runs/demo/demo
```

That trains and scores a model on `data/demo/` — 1,302 real parcels over six departments,
committed so a fresh clone runs something. Shares computed from it mean nothing (it is a
balanced sample), but the pipeline is the real one.

## Workspaces: which dataset a command touches

Every command takes `-w NAME`. A workspace is a named set of three folders, defined in
`workspaces.yaml` in the repository root.

| workspace | what it is |
|---|---|
| `demo` | the small committed sample; runs with no raw data |
| `national` | 14 departments, Landsat imagery — the main one |
| `national_s2` | the same parcels, Sentinel-2 imagery — the best classifier |
| `piura`, `perennial` | the earlier single-department work |
| `tenure_did` | the tenure difference-in-differences sample |

```bash
uv run cc workspaces     # where each one lands on your machine, and what exists
```

Every command that writes anything prints the workspace it resolved, on the first line. Read
that line — using the wrong workspace is the easiest mistake here.

## Earth Engine (only for new imagery)

You need this only to pull imagery for parcels that have none. Training, evaluating and
reproducing the published results do not touch it.

```bash
uv run earthengine authenticate
```

Then put your own Google Cloud project id in `workspaces.yaml` under `gee_project:`. Earth
Engine bills a project, so the id in this repository is not usable by anyone else. Free
research access: <https://earthengine.google.com/>.

Check it:

```bash
uv run python -c "from crop_classifier.features.landsat_gee import init_ee; init_ee(); print('ok')"
```

## Two things that will bite you

- **Never train a LightGBM model and a torch model in one process on macOS.** They each ship
  their own copy of libomp and loading both crashes with no error message. Run one model per
  command; loop in the shell, not in Python.
- **You do not have the raw archive** unless someone gave it to you. Most things still run —
  [`../DATA_ACCESS.md`](../DATA_ACCESS.md) says what.

Next: [`02_reproduce_results.md`](02_reproduce_results.md).
