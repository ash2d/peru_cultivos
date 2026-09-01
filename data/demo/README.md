# ⚠️ Synthetic demo workspace — no number here is a result

Everything in this directory is **generated**, by `tools/make_demo_workspace.py`, from a fixed
seed. None of it is Peruvian land, crops, or imagery.

It exists for one reason: the raw data this project is built on (PETT/COFOPRI land-titling
records and the 2012 CENAGRO census) was obtained under a research agreement and cannot be
redistributed, so a fresh clone has no inputs at all. Without this, nothing in the repository
runs until someone has ~20 GB of licensed files. See [`docs/DATA_ACCESS.md`](../../docs/DATA_ACCESS.md).

What it *is* faithful to is the **schema and the discipline**: the same tables, columns and
dtypes; regions assigned whole to a split and whole to a fold; the buffer dead-zone columns
present; the acquisition-metadata features correlated with label year exactly as the real store
has them, so `--drop-features meta` has something real to drop.

```bash
uv run cc -w demo train --model lightgbm --run-name demo
uv run cc -w demo evaluate runs/demo/demo
```

The parcel table carries a `synthetic` column set to `True`, and the departments are named
`DEMOLANDIA` and `EJEMPLIA`, so a table derived from this cannot quietly be mistaken for one
derived from the real store.

Regenerate with `uv run python tools/make_demo_workspace.py`.

**Real results are in [`docs/RESULTS.md`](../../docs/RESULTS.md).**
