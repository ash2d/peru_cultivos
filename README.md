# Peru crop classifier

Classifying agricultural land use in Peru from Landsat and Sentinel-2 imagery, using the
country's legacy land-titling records as labels. The research question is whether farmland has
shifted from domestic annual crops to export perennial crops, and whether securing legal land
title makes that shift more likely.

**Where it stands:** the single-year 3-class classifier works (locked-test macro-F1 **0.681** in
Piura, **0.628** CV nationally). Reading *change* out of it does not — four estimands failed for
measured reasons. One survived and returned a bounded null. A Sentinel-2 endpoint-labelling
campaign is the live work. See [`docs/STATUS.md`](docs/STATUS.md).

## Setup

```bash
uv sync                              # Python 3.11, deps from uv.lock
uv run earthengine authenticate      # required for any GEE extraction
uv run pytest -q                     # 411 tests
uv run python -m crop_classifier.cli --help
```

Never use bare `pip`/`python` — always `uv run`. `data/`, `runs/` and `logs/` are gitignored and
local-only.

## Documentation

| file | what it is |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | orientation for a coding agent — environment, data traps, gotchas |
| [`docs/STATUS.md`](docs/STATUS.md) | ⭐ what is done, what is closed, **what to do next** |
| [`docs/RESULTS.md`](docs/RESULTS.md) | every strand, its verdict, the numbers of record |
| [`docs/LESSONS.md`](docs/LESSONS.md) | method findings that generalise beyond this project |
| [`docs/DATA.md`](docs/DATA.md) | raw datasets, linkage chains, the four silent traps |
| [`docs/PIPELINE.md`](docs/PIPELINE.md) | module reference, CLI table, cookbook |
| [`docs/REPORT.md`](docs/REPORT.md) | the readable narrative, with figures |
| [`docs/s2_labelling/`](docs/s2_labelling/) | the live labelling campaign + frozen codebook |

**Start with [`docs/REPORT.md`](docs/REPORT.md)** if you want to understand the project, or
[`docs/STATUS.md`](docs/STATUS.md) if you want to work on it.
