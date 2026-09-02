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

## The table underneath all of it

If what you want is the data rather than an estimate — every parcel, what was declared on it,
what the 2012 census recorded, and what 2019+ satellite imagery shows — this is one command
and about ten seconds:

```bash
uv run cc -w national analysis parcel-table                       # -> data/processed/cenagro/parcel_table.parquet
uv run cc -w national analysis parcel-table --out parcels.csv     # or a CSV to open in Excel
```

One row per parcel, one column per observation of it:

| column | what it is |
|---|---|
| `COD_PREDIO`, `dept`, `area_ha` | the parcel |
| `tenure`, `frac_inscrito`, `reg_year` | registered or not at the declaration, and when |
| `pett_year`, `pett_class` | the crop the farmer declared, and the year they declared it |
| `cen_class`, `cen_sown_ha`, `cen_any_export` | the 2012 census reading |
| `s2_label`, `s2_class`, `s2_class_woody_perennial` | a person's reading of 2019+ imagery |
| `s2_pred_label`, `s2_pred_class`, `s2_pred_proba` | the Sentinel-2 classifier's reading |
| `s2_pred_source`, `weight`, `n_observations` | how to read the three columns above |

All four class columns use the same three words — `PERENNIAL`, `ANNUAL`, `PASTURE_FALLOW` —
so you can crosstab any two of them directly.

**It is 95,941 parcels, and the imagery columns are filled on 157 of them.** That is not a
bug and not fixable by re-running anything. The table's universe is the parcels with *both* a
declaration and a 2012 census record; the labelling campaign drew its 865 parcels from the
much larger declaration population, so the overlap is incidental. `n_observations` says how
many of the four each row actually has — filter on it rather than assuming.

Four things to know before you compute anything from it:

- **Weight the imagery columns.** The campaign deliberately over-sampled perennial parcels,
  so an unweighted share of `s2_class` is about three times the real one. Use `weight`. It is
  empty for the 120 pilot parcels, which were drawn under a different design.
- **`s2_pred_source` tells you whether a prediction is honest.** The classifier was trained on
  these parcels, so the default fills the column only from predictions made on parcels held
  out of training (`out_of_fold`, `locked_test`). If you pass `--preds` with your own
  `cc predict` output it is marked `applied`, and applying the model to parcels it trained on
  produces scores far better than the 0.774 it earned on held-out data.
- **`WOODY_NON_CROP` appears twice on purpose.** `s2_class` leaves tree cover that is not a
  crop unmapped; `s2_class_woody_perennial` counts it as perennial. That single choice moves
  the imagery result by 26 points against a census effect of about 10, so report both.
- **Crop *names* are not in it** — the class only. The declared crop text needs the 417 MB
  national parcel table and the census crop text needs the licensed archive
  ([`../DATA_ACCESS.md`](../DATA_ACCESS.md)); neither ships with the repo.

To fill the 2025 columns for parcels of your own, score them first
([`04_predict_new_parcels.md`](04_predict_new_parcels.md) §C, needs Earth Engine) and pass the
result:

```bash
uv run cc -w national analysis parcel-table --preds preds2025.parquet
```

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
