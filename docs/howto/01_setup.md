# 1. Setup

About 15 minutes, plus however long Google takes to approve an Earth Engine account.

---

## Install

The project uses **[uv](https://docs.astral.sh/uv/)**, not pip. Python 3.11.

```bash
git clone <this repo>
cd Coding
uv sync                      # installs everything from uv.lock, exactly
uv run pytest -q             # ~1 min; should be all green
```

⚠️ **Always `uv run …`.** A bare `python` or `pip` uses whatever interpreter your shell
happens to have, and this project pins its dependencies for a reason — `lightgbm` and `torch`
each bundle their own copy of libomp and the combination is version-sensitive.

## Check it works, with no data

```bash
uv run cc -w demo train --model lightgbm --run-name demo
uv run cc -w demo evaluate runs/demo/demo
```

That trains and scores a model on `data/demo/` — a small **synthetic** workspace committed to
the repository so a fresh clone runs something. Numbers from it are meaningless; it is a
plumbing check. See [`data/demo/README.md`](../../data/demo/README.md).

---

## Point the project at your data

One file: **`workspaces.yaml`** in the repository root.

```yaml
data_root: ./data                     # where data/raw and data/processed live
runs_root: ./runs                     # where trained models are written
gee_project: your-gcp-project-id      # your own; see below
cenagro_source_dir: null              # only for regenerating the census extract
```

A **workspace** is a named set of three directories, and every command takes `-w NAME`:

| workspace | what it is |
|---|---|
| `demo` | the synthetic sample above |
| `piura` | the original single-department build |
| `perennial` | 3-class labels over the Piura pixels |
| `national` | ⭐ 14 departments, Landsat — the main one |
| `national_s2` | the same parcels, Sentinel-2 |
| `tenure_did` | the difference-in-differences sample |

Then:

```bash
uv run cc workspaces
```

which prints where each one resolves on your machine, whether it exists, and which
workspaces deliberately **share** a feature store (`piura` and `perennial` do — same parcels,
same pixels, only the label column differs).

⚠️ Getting the workspace wrong used to be this project's easiest mistake, because it was three
environment variables that had to agree. Every command that writes anything now prints the
workspace it resolved, on stderr. **Read that line.**

You do not have the raw data unless someone gave it to you — see
[`../DATA_ACCESS.md`](../DATA_ACCESS.md).

---

## Earth Engine

Needed only for extracting new imagery, not for training on features that already exist.

```bash
uv run earthengine authenticate
```

Then put **your own** Google Cloud project id in `workspaces.yaml`. Earth Engine bills a
project; the one in this repository's history is not yours and there is no shared default that
could be. Free research access: <https://earthengine.google.com/>.

Verify:

```bash
uv run python -c "from crop_classifier.features.landsat_gee import init_ee; init_ee(); print('ok')"
```

---

## Two things that will bite you

**Never import torch and lightgbm in one process on macOS.** Each bundles its own libomp and
co-loading them segfaults (exit 139) with no traceback. The model registry loads lazily for
this reason, and the CLI runs one model arm per process. If you write a loop over models, loop
in the **shell**, not in Python.

**`quality_ok` is a nullable boolean**, where NA means "not measured". Compare with `== True`
or `.isna()`, never with plain truthiness.

Next: [`02_get_satellite_data.md`](02_get_satellite_data.md).
