# 5. Measure perennial change, and its relation to land tenure

The question the project exists for: **has Peruvian farmland shifted from domestic annual crops
to export perennials, and does secure legal title make that shift more likely?**

Short answers, from `docs/RESULTS.md`:

* **Shift: yes, large.** +9.9 pp of parcels and +12.5 pp of cadastral area between ~1999 and
  2012, nationally, with no satellite and no classifier anywhere in the measurement.
* **Title: no, and slightly the other way.** Titled parcels shifted **less** (−2.1 pp ± 0.6),
  and **by area the gap vanishes** (+0.3 pp). A causal design returned a bounded null:
  −0.0011 [−0.0126, +0.0104].

There are three routes to it. They use different instruments and answer different questions, so
run whichever matches what you are asking.

---

## Route A ⭐ — two declarations, no classifier

The strongest thing in the project, because there is no model in it: the same land, declared
twice, by two different official instruments (PETT ~1999, CENAGRO 2012).

```bash
uv run cc -w national data cenagro-link      # crosswalk by farmer name, 14 depts, ~6 min
uv run cc -w national analysis perennial-shift               # ~3 min, draws the figure too
uv run cc -w national analysis perennial-shift --figure      # redraw only
```

Needs `data/raw/Cenagro_IV/` — see [`../DATA_ACCESS.md`](../DATA_ACCESS.md) §4.

Writes `data/processed/cenagro/national_*.csv` and `national_panel.parquet`, and
`docs/figures/perennial_over_time_by_tenure.png`.

### ⚠️ Read the like-for-like row, not the first one

The census **cannot record fallow** — question 024 asks which crop is grown, so a fallow parcel
contributes no row and leaves the frame entirely. `PASTURE_FALLOW` going 30.8 % → 6.3 % is that
instrument difference, not land change. The defensible comparison is conditional on a crop being
recorded on both sides.

### ⚠️ Post-stratification is not optional

The name link is not a random sample. It needs a name on both sides plus a district agreement,
and the parcels that satisfy that are the larger, valley-floor, better-documented ones — the
linked panel is 16.6 % perennial against the population's 9.9 %. Reweighting on
department × declared class restores the national cell counts, and the change **rises** from
+8.4 to +9.9 pp, so composition was damping the effect, not manufacturing it. What reweighting
cannot fix is selection *within* a cell.

### ⚠️ Any area or share figure must use `sample_weight`

The national sample doubles the perennial share **by design** (sqrt allocation). An unweighted
share is wrong, and it is wrong in the direction that flatters the headline.

---

## Route B — the causal design

A two-period difference-in-differences, using two *dated* tenure observations that were already
on disk: the declaration-time `ESTADO en RRPP` (~1997–2006) and the cadastre's `estado` +
`fech_tran` (≈2011–12). 1,780,580 parcels have both; 8.6 % move NO INSCRITO → REGISTERED.

The classifier supplies the outcome at both dates, drift and all — the point of the design is
that the drift is **common to both arms and differences out**.

```bash
uv run cc -w tenure_did analysis did-feasibility    # ⭐ feasibility FIRST, always
uv run cc -w tenure_did analysis did-sample
uv run cc -w tenure_did analysis did-register       # refuses to overwrite
uv run cc -w tenure_did analysis did                # placebo first, then the headline
```

**This is complete. Do not re-run it for a bigger sample — the population is exhausted.** 6,559
treated parcels is every one that exists after the pre-period restriction.

The order is enforced by the code, not by discipline: the placebo is estimated *first*, and the
registration file refuses to be overwritten.

### ⭐ Check feasibility before funding an extraction

`did-feasibility` compares the sample size the variance requires against the sample size the
archive can supply. It takes minutes and it is the highest-value command in this list. Size for
the gate that will actually decide, and use an **equivalence** test rather than a significance
test — the earlier gate design rewarded imprecision, because a wide enough interval passes
anything.

---

## Route C — the cross-section

```bash
uv run cc -w national advanced tenure-error --run <run> --tenure <tenure.parquet>
uv run cc -w national allperu tenure-xsec --preds <panel_predictions.parquet> --tag <tag>
```

⚠️ **Descriptive, and never reported as an estimate.** The baseline INSCRITO/NO INSCRITO gap
among at-risk parcels *is* the classifier's differential false-positive rate, not agronomy, and
the contrast's sign is **department-specific**. Always read the per-department table, never the
national average alone — the Piura 24.9 / 10.9 figure does not replicate elsewhere.

---

## ⛔ Routes that are closed. Do not reopen them.

| route | why |
|---|---|
| Per-parcel annual trajectories over 25 years | flicker 0.43–0.98 against a 0.15 criterion; **not** a coverage problem — there are no thin years nationally. A ~0.55–0.59 classifier cannot support a per-parcel series |
| The 5-year window pivot | the control pool drifts **7× further than the signal, the other way**, on every arm |
| Recalibration / quantile alignment / density matching | measured: moved the target metric by 0.0001, −0.1023, and about a third of the artefact |
| Admitting Landsat 8/9 (OLI), twice | the sensor difference is **cover-type dependent** (0.025 NDVI between classes). No global linear map removes it, and refitting locally is worse than no correction |
| Collapsing to 2 classes | macro-F1 rises, the floor rises further; normalised it is the lowest-skill arm in the study, for +0.004 on the class you care about |

They are still runnable, under `cc archive`, with `src/crop_classifier/archive/README.md`
recording what each tried and which gate killed it. They are kept because "we tried this and
measured why it does not work" is a result.

---

## ⚠️ Three instruments are not one series

The headline figure has three points in time. The census legs are one instrument; the 2025
points are 214 / 364 photo-interpreted parcels from a *different frame with a different
restriction*. Each imagery reading is drawn from **its own baseline**, not continued off the
census line. Connecting them would invent a trend that neither instrument measured.

And the `WOODY_NON_CROP` decision moves the imagery endpoint by **26 pp**, against a census
effect of ~10 pp — which is why that arm is not quoted on tenure at all.

Full numbers, intervals and caveats: [`../RESULTS.md`](../RESULTS.md) §7, §8.5, §8.6.
