# 2. Reproduce the published results

Two commands, about 90 seconds. No licensed data, no Earth Engine account, no long job: the
tables each check needs are committed to this repository.

---

```bash
uv sync
uv run cc reproduce
```

Every published number is printed beside the one your machine just computed:

```
check           quantity                                          published   measured
demo            demo quickstart, CV macro-F1                        +0.5381    +0.5381  ok
national-lodo   national Landsat, LODO macro-F1 (14 departments)    +0.4789    +0.4789  ok
s2-model        Sentinel-2 3-class (t3w, --climate temp), CV        +0.7620    +0.7585  ok
perennial-shift perennial share of parcels, 1999 → 2012 (pp)          9.900      9.900  ok
                perennial share of cadastral area (pp)               12.500     12.500  ok
                titled − untitled gap in that change (pp)            -2.100     -2.100  ok
tenure-did      DiD headline effect on perennial probability        -0.0011    -0.0011  ok
```

`ok` on every row means your clone reproduces the paper.

## What each check is

| check | what it re-derives | time |
|---|---|---|
| `demo` | trains a model on the committed sample and compares its CV score | ~10 s |
| `national-lodo` | reads back the national Landsat evaluation record | instant |
| `s2-model` | refits the Sentinel-2 classifier and compares its CV score | ~1 min |
| `perennial-shift` | the headline: perennial change 1999 → 2012, and the tenure split | ~30 s |
| `tenure-did` | the difference-in-differences estimate | ~10 s |

```bash
uv run cc reproduce --list          # the checks and where their numbers are published
uv run cc reproduce s2-model        # run one
uv run cc reproduce s2-model -v     # and print its full tables
```

## Reading the output

- Two checks **retrain a model**, so they land within a few thousandths rather than exactly.
  LightGBM is not bit-identical across machines. The other checks are exact.
- A row marked `missing` names the file it wanted. Run `uv run cc data verify` to see what
  your clone has, file by file.
- A row that is `ok` on the number but differs in the fourth decimal is not a problem. A row
  that fails is: it means the code and the published number have drifted apart, and the
  section named in `--list` is where to look.

## What a clone can and cannot do

```bash
uv run cc data verify
```

It prints every capability as `ok` (everything is here), `derivable` (a command rebuilds it,
and the output names the command) or `needs-data` (the licensed archive or a new imagery
extraction). On a fresh clone the demo, the national model, the Sentinel-2 labels, the
census comparison and the tenure estimate are all `ok`.

Details of what is committed and what is not: [`../DATA_ACCESS.md`](../DATA_ACCESS.md).

Next: [`03_train_and_evaluate.md`](03_train_and_evaluate.md).
