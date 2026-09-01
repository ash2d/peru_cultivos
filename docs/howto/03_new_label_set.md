# 3. Make a new label set

**No Python.** A label set is a YAML file; adding one is copying a file and editing it.

There are two kinds, because there are two kinds of label in this project.

---

## A. Photo-interpreted labels (Sentinel-2 campaign) — `config/labels/`

An annotator records one of five classes for a parcel:

```
PERENNIAL   ANNUAL   OTHER   WOODY_NON_CROP   NON_AGRICULTURE
```

(plus `UNSURE`, which is an abstain and never a class). A **label set** says how those collapse
into what a model is asked to predict.

```bash
uv run cc labels list
```

To add one, copy the nearest file in `src/crop_classifier/config/labels/`:

```yaml
# src/crop_classifier/config/labels/orchard2.yaml
name: orchard2                     # must match the filename
about: >-
  Perennial against everything else that is farmed, dropping non-agriculture entirely.
collapse:
  ANNUAL: FARMED_NOT_PERENNIAL     # merge into a new class
  OTHER: FARMED_NOT_PERENNIAL
  WOODY_NON_CROP: PERENNIAL
  NON_AGRICULTURE: null            # `null` DROPS the class from training
rules_compatible: true             # can the hand-written `rules` baseline be scored here?
```

Anything not named in `collapse` keeps its own name. Then:

```bash
uv run cc labels list                                          # it should appear
uv run cc -w national_s2 labelling train prep --target orchard2
uv run cc -w national_s2 labelling train fit  --target orchard2 --model lightgbm
uv run cc -w national_s2 evaluate runs/s2_labels/ws_orchard2/lightgbm
```

A key in `collapse:` that is not one of the five recorded classes is an **error**, naming the
valid ones — not a silently ignored line that leaves your arm training in a different label
space than you think.

---

## B. Declared-crop labels (the PETT registry) — `config/perennial*.yaml`

These map free-text Spanish crop names (`MANGO`, `ARROZ`, `PASTO NATURAL`, ~thousands of them,
dirty) onto classes. Same idea, more vocabulary.

Variants use `extends:` so a derived set is a **diff, not a copy**:

```yaml
extends: perennial.yaml
add:
  perennial: [PALTO, PITAHAYA]     # APPEND to the inherited list, skipping duplicates
drop: [some_group]                 # delete an inherited key entirely
```

`add:` is the important one. A derived lexicon is only safe if it is **additive** — if it
changed an existing token's group, a measured "change in the land" would be partly a change of
definition. When the derived file was a full copy (they were 559 lines each), nothing enforced
that and nothing showed it; a reader had to diff the whole file to find out.

```bash
uv run cc -w perennial labels build --config src/crop_classifier/config/perennial.yaml
```

---

## ⚠️ Read the unmapped tail. Every time.

Both builders print the tokens that fell through to the catch-all, **most frequent first, top
ten**. Read them.

An unmapped-token catch-all is never uniformly distributed, and a percentage budget cannot see
that. Mapping the 2012 census vocabulary onto these classes left 4.09 % of tokens falling to a
blanket `ANNUAL` — of which **80 % was the single token `VERGEL FRUTICOLA`**, "fruit orchard",
which is a perennial. The project's headline number moved from **+2.4 pp to +12.5 pp** when it
was fixed. The budget check passed either way.

The tail was already being written to `unassigned_tokens.csv`. It was written and not read,
which for this purpose is the same as not written.

---

## ⚠️ Then read your result at both ends of the bracket

`t4` and `t3w` differ in exactly one decision — where `WOODY_NON_CROP` goes — and that decision
moves `PERENNIAL` F1 by **0.190**, which is larger than most effects anyone measures here.

So: **anything that decides something, re-run at both ends.** `--climate both` was adopted on
12 of 14 departments at p = 0.004; re-asked at the other end of the same bracket it is 7 of 14
at p = 0.345, and the rainfall column is worth −0.001. Temperature held at both ends. Rainfall
did not replicate, and only the bracket showed it.

## ⚠️ And macro-F1 is not comparable between label sets

Collapsing four classes to two raised macro-F1 from 0.672 to 0.715 — and raised the
always-guess-the-largest-class floor from **0.171 to 0.467**. Normalised, the two-class arm was
the *worst* in the study, and it bought +0.004 on the class anyone cared about.

`cc evaluate` prints the floor and a normalised `skill` beside every score for this reason.
Compare label sets on `skill`, or on the F1 of the one class the research question is about.

Next: [`04_train_and_evaluate.md`](04_train_and_evaluate.md).
