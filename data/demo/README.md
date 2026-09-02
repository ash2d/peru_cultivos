# The demo workspace

**A real sample — 1,302 parcels across six departments**, with their real declared labels, their
real extracted Landsat features and the real spatial split assignment inherited from the national
build.

It exists because `data/` is 33 GB and gitignored, so a fresh clone could otherwise run nothing
at all. Redistributing *this sample* was confirmed as permitted. The **full** archive is licensed
and is not here — [`docs/DATA_ACCESS.md`](../../docs/DATA_ACCESS.md).

```bash
uv run cc -w demo train --model lightgbm --drop-features meta,location --run-name demo
uv run cc -w demo advanced lodo --drop-features meta,location --tag demo --min-parcels 100
uv run cc -w demo evaluate runs/demo/demo --tag demo
```

Under a minute end to end, no Earth Engine account needed.

## What it is for

It reproduces the project's central finding at small scale:

```
split     mean±sd over units   pooled   floor   skill              units         n
CV           0.5381 ± 0.0705   0.5533   0.171   0.443         5 CV folds     1,091
LODO         0.4940 ± 0.0516   0.5015   0.167   0.393      6 departments     1,302
```

Cross-validation says 0.538. Holding out a **whole department** says 0.494. That gap is the thing
this repository is mostly about, and you can see it here before you have any data of your own.
[`docs/howto/03_train_and_evaluate.md`](../../docs/howto/03_train_and_evaluate.md).

The six departments were chosen to span the country's range of class balance — TUMBES is 75 %
perennial nationally, LAMBAYEQUE 6 %. Six similar departments would have made
leave-one-department-out look easy, which is the opposite of the lesson.

## It is a stratified sample, not a representative one

Classes are balanced within each department so that every fold can train. **No share, area or
prevalence computed from this workspace means anything** — the same caveat the real national
sample carries, for the same reason.

Accuracy figures from it are indicative of the pipeline, not of Peru. The numbers of record are
in [`docs/RESULTS.md`](../../docs/RESULTS.md).

## What is in here

| file | what |
|---|---|
| `modeling_parcels.parquet` | parcels: geometry, label, year, department, split, fold, buffer columns |
| `features/features_lightgbm.parquet` | the extracted Landsat summary features |
| `label_map.json` | class name → id |
| `lodo_*_demo.*` | a completed leave-one-department-out run, committed so `cc evaluate` has something to read on a fresh clone |

Regenerate with `uv run python tools/make_demo_workspace.py` — which needs the national workspace
on disk. Using the demo does not.
