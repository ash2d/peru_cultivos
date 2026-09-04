# 1. Set up

About 15 minutes, once. You do not need any data or a Google account for this page.

---

## Install

This project uses [uv](https://docs.astral.sh/uv/) to manage Python, not pip. Python 3.11.

```bash
git clone <this repo>
cd Coding
uv sync            # installs everything at the exact versions used here
uv run pytest -q   # about 1 minute, should be all green
```

Start every command with `uv run`. Plain `python` or `pip` will use whatever version of Python
your computer happens to have, and this project depends on specific versions.

Check that it works. This trains and tests a model on `data/demo/`, a small set of 1,302 real
parcels included so that a fresh copy can run something straight away:

```bash
uv run cc -w demo train --model lightgbm --run-name demo
uv run cc -w demo evaluate runs/demo/demo
uv run cc data verify        # and what your computer can do with the real data
```

## Workspaces: choosing which dataset a command uses

Most commands take `-w NAME`. A workspace is a named set of three folders, listed in
`workspaces.yaml` in the main folder of the repository.

| workspace | what it holds |
|---|---|
| `national` | 14 departments, Landsat images. The main one |
| `national_s2` | the same parcels with Sentinel-2 images. The best model |
| `demo` | the small included sample. Runs with no other data |
| `piura`, `perennial`, `tenure_did` | earlier work on one department, and the land-title study |

```bash
uv run cc workspaces     # where each one is on your computer, and what exists
```

Every command that writes a file prints which workspace it used, on the first line. Read that
line. Using the wrong workspace is the easiest mistake to make here.

## Google Earth Engine, only if you need new images

You need this only to download satellite images for parcels that do not have any yet, which is
pages [`03`](03_score_parcels.md) and [`04`](04_label_and_train.md). Nothing else uses it.

```bash
uv run earthengine authenticate
```

Then put your own Google Cloud project id in `workspaces.yaml`, under `gee_project:`. Earth
Engine charges a project, so the id already in this repository will not work for anyone else.
Research access is free: <https://earthengine.google.com/>.

Check it:

```bash
uv run python -c "from crop_classifier.features.landsat_gee import init_ee; init_ee(); print('ok')"
```

## Two things that will catch you out

- **Do not train a LightGBM model and a neural model in the same command on a Mac.** They each
  bring their own copy of a shared library, and loading both crashes with no error message. Run
  one model per command.
- **You probably do not have the raw government files.** Most things still work without them.
  [`../DATA_ACCESS.md`](../DATA_ACCESS.md) says which.

Next: pick a task from [`README.md`](README.md).
