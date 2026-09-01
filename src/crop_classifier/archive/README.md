# Closed routes

Each module here answers a question the project **stopped asking**, because a gate that was
written down in advance failed. The code works; the estimand did not survive contact with the
data. Kept for reproduction, and so the same idea is not re-proposed as if it were new.

| module | what it tried to estimate | why it closed | numbers |
|---|---|---|---|
| `windows.py` | Perennial share in a 5-year window around each parcel's title date, pivoting the panel from per-parcel to per-window | The **control pool drifts 7× further than the signal, and in the opposite direction**, on every arm. Probability aggregation fixed the flicker; it did not fix the control. | `RESULTS.md` §5 |
| `window_sample.py` | The dept × tenure × label stratified sample that fed the above | Closed with the estimand it served | §5 |
| `estimate.py` | A national area estimate of perennial cover from the classifier | Rests on the panel, which failed its gate; an area figure from a ~0.55–0.59 classifier over 25 years is not defensible | §9 |
| `external.py` | Validation against external land-cover products | MapBiomas Peru **never assigns a perennial class** (codes 36/46/47/48, 15) anywhere in Piura, so it cannot confirm or deny the thing being measured. Structural, not a fair loss. | §9 |
| `oli_overlap.py` | Admitting Landsat 8/9 (OLI) to the panel via a paired same-day overlap sample | The sensor difference is **cover-type dependent** (0.025 NDVI between classes). No global linear map can remove it. | §6.4 |
| `oli_refit.py` | The same, refitting Roy's coefficients locally instead of using the published ones | Refitting is *worse* than no correction. ⛔ **Do not propose this again** — reopening OLI needs a cleaner paired sample, not better fitting. | §6.4 |

## Running them

```bash
uv run cc archive --help
```

Options and behaviour are unchanged from when they lived under `allperu`; only the command
prefix moved (`cc allperu windows …` → `cc archive windows …`).

## The rule these came from

Three of the six closed for the *same* reason, discovered three times: a number that looks like
signal is produced by something the model was supposed to be blind to. `frac_l7` manufactured
change, statics manufactured stability, `centroid_lat` manufactured accuracy that does not leave
the training departments. **Report CV / LODO / LOYO / LODYO, never CV alone** —
`docs/LESSONS.md`.
