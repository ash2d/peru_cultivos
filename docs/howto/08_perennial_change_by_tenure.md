# 8. Measure perennial change, and its relation to land tenure

The question the project exists for: has Peruvian farmland shifted from domestic annual crops
to export perennials, and does secure legal title make that shift more likely?

- **Shift: yes, large.** +9.9 pp of parcels and +12.5 pp of cadastral area between about 1999
  and 2012, nationally, with no satellite and no classifier in the measurement.
- **Title: no, and slightly the other way.** Titled parcels shifted *less* (−2.1 pp ± 0.6),
  and by area the gap disappears (+0.3 pp). The causal design returns a bounded null:
  −0.0011 [−0.0126, +0.0104].

Three routes below. They use different instruments and answer different questions.

---

## Route A — two declarations, no classifier

The strongest result here, because there is no model in it: the same land declared twice, by
two official instruments (PETT ~1999, CENAGRO 2012).

```bash
uv run cc -w national analysis perennial-shift            # ~30 s, draws the figure too
uv run cc -w national analysis perennial-shift --figure   # redraw only
```

This runs on a clone as it stands; `uv run cc reproduce perennial-shift` checks the three
headline numbers. Rebuilding the name crosswalk underneath it needs the licensed files
([`../DATA_ACCESS.md`](../DATA_ACCESS.md)):

```bash
uv run cc -w national data cenagro-link      # by farmer name, 14 departments, ~6 min
```

Three things to get right when reading the output:

- **Read the like-for-like row, not the first one.** The census cannot record fallow — it asks
  which crop is grown, so a fallow parcel leaves the frame. `PASTURE_FALLOW` falling
  30.8 % → 6.3 % is that difference between instruments, not land change.
- **Post-stratification is not optional.** The name link needs a name on both sides and a
  district match, which favours larger, better-documented parcels: the linked panel is 16.6 %
  perennial against the population's 9.9 %. Reweighting on department × declared class raises
  the change from +8.4 to +9.9 pp, so composition was damping the effect. It cannot fix
  selection *within* a cell.
- **Any share or area figure needs `sample_weight`.** The national sample doubles the
  perennial share by design.

## Route B — the causal design

A two-period difference-in-differences on two dated tenure observations already on disk: the
declaration-time status (~1997–2006) and the cadastre's status and transaction date (≈2011–12).
1,780,580 parcels have both; 8.6 % move from unregistered to registered. The classifier
supplies the outcome at both dates, drift included — the design works because the drift is
common to both arms and cancels.

```bash
uv run cc -w tenure_did analysis did \
    --preds data/processed/all_peru_did/panel_predictions_nolat_aug_yleak10.parquet \
    --tag mycheck                     # placebo first, then the headline
```

`--tag` names the configuration; a run refuses to overwrite an existing one, so pick a new
word. `uv run cc reproduce tenure-did` does the same check without writing anything.

The commands that designed it, in this order:

```bash
uv run cc -w tenure_did analysis did-feasibility    # run feasibility first, always
uv run cc -w tenure_did analysis did-sample
uv run cc -w tenure_did analysis did-register       # refuses to overwrite
```

**This is complete. Do not re-run it for a bigger sample** — 6,559 treated parcels is every one
that exists after the pre-period restriction. The order is enforced by the code: the placebo is
estimated first and the registration file cannot be overwritten.

`did-feasibility` compares the sample size the variance requires against the sample size the
archive can supply. It takes minutes and it is the highest-value command on this page. Size for
the test that will decide, and use an equivalence test rather than a significance test — a wide
enough interval passes any significance gate.

## Route C — the cross-section

```bash
uv run cc -w national advanced tenure-error --run <run> --tenure <tenure.parquet>
uv run cc -w national allperu tenure-xsec --preds <panel_predictions.parquet> --tag <tag>
```

Descriptive, never reported as an estimate. The baseline registered/unregistered gap among
at-risk parcels *is* the classifier's differential false-positive rate, not agronomy, and its
sign differs by department. Always read the per-department table: the Piura 24.9 / 10.9 figure
does not replicate elsewhere.

---

## Routes that are closed

| route | why |
|---|---|
| Per-parcel yearly trajectories over 25 years | predictions flicker 0.43–0.98 against a 0.15 criterion, and it is not a coverage problem. A ~0.55–0.59 classifier cannot support a per-parcel series |
| The 5-year window pivot | the control group drifts 7× further than the signal, the other way, on every arm |
| Recalibration, quantile alignment, density matching | measured: they moved the target by 0.0001, −0.1023 and about a third of the artefact |
| Admitting Landsat 8/9 (OLI), tried twice | the sensor difference depends on the cover type (0.025 NDVI between classes); no global correction removes it, and fitting one locally is worse than none |
| Collapsing to 2 classes | macro-F1 rises and its floor rises further; normalised it is the lowest-skill arm, for +0.004 on the class that matters |

They still run, under `cc archive`, and `src/crop_classifier/archive/README.md` records which
check killed each. They are kept because "we tried this and measured why it fails" is a result.

## Three instruments are not one series

The headline figure has three points in time. The census legs are one instrument; the 2025
points are 214 and 364 photo-interpreted parcels from a different frame with a different
restriction. Each imagery reading starts from its own baseline rather than continuing the
census line — joining them would invent a trend neither instrument measured. The
`WOODY_NON_CROP` decision alone moves the imagery endpoint by 26 pp, against a census effect of
about 10 pp, which is why that arm is not quoted on tenure.

Full numbers and caveats: [`../RESULTS.md`](../RESULTS.md) §7, §8.5, §8.6.
