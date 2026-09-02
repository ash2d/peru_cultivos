# 7. Make a new label set

A label set says which classes a model is asked to predict. Adding one is copying a YAML file
and editing it — no Python.

There are two kinds, because there are two kinds of label here.

---

## A. Photo-interpreted labels (Sentinel-2)

An annotator records one of five classes for a parcel:

```
PERENNIAL   ANNUAL   OTHER   WOODY_NON_CROP   NON_AGRICULTURE
```

(plus `UNSURE`, an abstain, which is never a class). A label set says how those collapse into
what the model predicts. See what exists with `uv run cc labels list`.

To add one, copy the nearest file in `src/crop_classifier/config/labels/`:

```yaml
# src/crop_classifier/config/labels/orchard2.yaml
name: orchard2                     # must match the filename
about: >-
  Perennial against everything else that is farmed, dropping non-agriculture.
collapse:
  ANNUAL: FARMED_NOT_PERENNIAL     # merge into a new class
  OTHER: FARMED_NOT_PERENNIAL
  WOODY_NON_CROP: PERENNIAL
  NON_AGRICULTURE: null            # null drops the class from training
rules_compatible: true
```

Anything not named in `collapse` keeps its own name. Then:

```bash
uv run cc labels list
uv run cc -w national_s2 labelling train prep --target orchard2
uv run cc -w national_s2 labelling train fit  --target orchard2 --model lightgbm
uv run cc -w national_s2 evaluate runs/s2_labels/ws_orchard2/lightgbm
```

A key in `collapse:` that is not one of the five recorded classes is an error naming the valid
ones — not a silently ignored line that leaves you training in a label space you did not mean.

---

## B. Declared-crop labels (the PETT registry)

These map free-text Spanish crop names (`MANGO`, `ARROZ`, `PASTO NATURAL`, thousands of them,
dirty) onto classes. Same idea, much more vocabulary, in `config/perennial*.yaml`.

Variants use `extends:`, so a derived set is a difference rather than a copy:

```yaml
extends: perennial.yaml
add:
  perennial: [PALTO, PITAHAYA]     # append to the inherited list
drop: [some_group]                 # remove an inherited key
```

`add:` matters: a derived lexicon is only safe if it is additive. If it changed an existing
crop's group, a measured "change in the land" would partly be a change of definition. When
these files were full copies, nothing enforced that and nothing showed it.

```bash
uv run cc -w perennial labels build --config src/crop_classifier/config/perennial.yaml
```

---

## Read the unmapped tail, every time

Both builders print the crop names that fell through to the catch-all class, most frequent
first. Read the top ten.

A catch-all is never evenly spread, and a percentage budget cannot see that. Mapping the 2012
census vocabulary left 4.09 % of names falling into a blanket `ANNUAL` — and 80 % of that was
one name, `VERGEL FRUTICOLA`, "fruit orchard", which is perennial. The project's headline moved
from +2.4 pp to +12.5 pp when it was fixed, and the percentage check had passed either way.

## Check your result at both ends of the bracket

`t4` and `t3w` differ in one decision — where `WOODY_NON_CROP` goes — and it moves
`PERENNIAL` F1 by **0.190**, more than most effects anyone measures here. So anything that
decides something should be re-run at both ends.

`--climate both` was adopted on 12 of 14 departments at p = 0.004; at the other end of the same
bracket it is 7 of 14 at p = 0.345, and the rainfall column is worth −0.001. Temperature held
at both ends. Only the bracket showed the difference.

## macro-F1 is not comparable between label sets

Collapsing four classes to two raised macro-F1 from 0.672 to 0.715 — and raised the
always-guess-the-largest-class floor from 0.171 to 0.467. Normalised, the two-class version was
the worst in the study, and it gained +0.004 on the class anyone cared about.

`cc evaluate` prints the floor and a normalised `skill` beside every score. Compare label sets
on `skill`, or on the F1 of the one class your question is about.

Next: [`08_perennial_change_by_tenure.md`](08_perennial_change_by_tenure.md).
