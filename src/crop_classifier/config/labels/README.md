# Label sets

One file per label space. **Adding a label set is editing this directory — never Python.**

A label set says how the five real classes an annotator can record

    PERENNIAL   ANNUAL   OTHER   WOODY_NON_CROP   NON_AGRICULTURE

(plus `UNSURE`, an abstain, which is never a class) are collapsed into the classes a model is
asked to predict. That is a **modelling decision**, so it lives in a file that can be read and
diffed rather than in a dict inside a training call.

```yaml
name: t3w
about: >-
  What it is and why anyone would run it.
collapse:
  NON_AGRICULTURE: OTHER        # rename / merge into another class
  WOODY_NON_CROP: PERENNIAL
  SOME_CLASS: null              # `null` DROPS the class from training entirely
rules_compatible: true          # can the hand-written `rules` baseline be scored here?
```

Anything not named in `collapse` keeps its own name. An empty `collapse:` is the full
five-class space.

## Add one

1. Copy the nearest file and give it a new `name:` matching the filename.
2. Edit `collapse:`.
3. `uv run cc labels list` — it should appear, with its class count.
4. `uv run cc labelling train prep --target <name>` then `... fit --target <name>`.
5. `uv run cc evaluate <run> --tag <tag>`.

## ⚠️ Read the verdict across the bracket, not at one point

`t4` and `t3w` differ only in what happens to `WOODY_NON_CROP`, and that one decision moves
`PERENNIAL` F1 by **0.190** — larger than most effects anyone measures here. A feature adopted
at one end of that bracket and never re-asked at the other is a feature that has not been
tested: `--climate both` was adopted on 12 of 14 departments at p = 0.004, and re-run at the
other end it is 7 of 14 at p = 0.345.

**Re-run anything that decides something at both ends of the bracket.**

## ⚠️ And macro-F1 is not comparable across these files

Collapsing to two classes raises macro-F1 *and* raises the always-guess-the-largest-class floor
further. `cc evaluate` prints the floor and the normalised `skill` beside every score for this
reason; compare label spaces on `skill`, or on the F1 of the class you actually care about.
`docs/RESULTS.md` §8.2c.
