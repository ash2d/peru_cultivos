# 6. Label more parcels, then train on them

Follow this page to make the classifier better. The model of record was trained on **865**
parcels that people looked at by eye; more labelled parcels is the change most likely to help.

No Python. Every step is one command you copy, plus one step where people look at pictures in
a web browser.

```
pick parcels  →  make pictures  →  people label them  →  put the CSVs back  →  train
```

---

## Before you start

You need the repository installed ([`01_setup.md`](01_setup.md)), a Google Earth Engine
account (free for research), and an internet connection — the aerial pictures come from Esri's
public map service.

**Pick a round name** — `round2`, `piura2027`, your initials — and pass `--round <name>` to
every command below. It keeps the new work in its own folders:

```
labels_s2            the campaign of record, never written to
labels_s2_round2     your round: sample, pictures, labels
features_s2_round2   its satellite features
```

Without `--round`, a new draw would overwrite the sample the existing 865 labels are keyed to.
The code refuses to do that, and the flag is how you avoid meeting the error.

---

## Step 1 — choose which parcels to label

```bash
uv run cc -w national_s2 labelling campaign universe --round round2
uv run cc -w national_s2 labelling campaign pool     --round round2
uv run cc -w national_s2 labelling campaign probe    --round round2 --workers 8
uv run cc -w national_s2 labelling campaign draw     --round round2 --total 100 --pilot-n 0
uv run cc -w national_s2 labelling campaign split    --round round2
```

| step | what it does | how long |
|---|---|---|
| `universe` | every parcel that could be labelled: has a boundary, at least 0.15 ha | seconds |
| `pool` | a shortlist, about 2.5× the number you want | seconds |
| `probe` | asks Esri, parcel by parcel, whether the aerial imagery is sharp and recent enough | ~1 h for a few thousand |
| `draw` | the sample: balanced across departments and classes, max 2 parcels per 5 km cell | seconds |
| `split` | freezes which parcels are for training and which are held out | seconds |

`probe` is slow for a good reason: Esri returns a valid grey image above the zoom it actually
serves, rather than an error, so every parcel has to be checked before anyone is asked to look
at it.

**Do not change the sample after `draw`.** Dropping parcels later — because they look hard, or
the imagery disappoints — changes what the sample represents. That is why the imagery check
comes first.

`--total` sets how many parcels to draw; the default is the campaign of record's 1,000. A
100-parcel round is `draw --total 100`, and the per-department floor and the double-labelled
overlap scale with it. `--pilot-n 0` skips the separate pilot batch, which a second round does
not need.

Before funding a round, `uv run cc labelling budget` shows the learning curve on the labels
that already exist. On the current 865, a further 100 labels move macro-F1 by roughly +0.02;
the clearly worthwhile use of 100 labellings is the overlap shard that measures how much the
existing labels disagree with themselves ([`../STATUS.md`](../STATUS.md)).

---

## Step 2 — make the pictures

```bash
uv run cc -w national_s2 labelling campaign chips    --round round2 --workers 3
uv run cc -w national_s2 labelling campaign extract  --round round2      # Earth Engine, hours
uv run cc -w national_s2 labelling campaign assemble --round round2
uv run cc -w national_s2 labelling campaign html     --round round2 --lang es
```

- `chips` downloads the aerial image around each parcel and draws its boundary on it.
- `extract` pulls the Sentinel-2 pixels. This is the long step and it can fail quietly — read
  the warnings in [`05_get_satellite_data.md`](05_get_satellite_data.md) first, and check it
  by counting rows, never by "the command finished".
- `assemble` turns the pixels into the greenness curve labellers see and the numbers the model
  trains on.
- `html` writes the labelling pages, `--lang es` or `--lang en`.

You now have `labels_s2_round2/html_es/` holding `shard01_A.html`, `shard01_B.html`, … and
`item_key.csv`. Each shard is a self-contained file with the images inside it: email it, put
it on a USB stick, open it by double-clicking. No server needed.

`item_key.csv` says which parcel each `item_id` is. Do not delete or edit it — without it the
returned labels cannot be joined to anything.

---

## Step 3 — people label the parcels

Send each person one or more shard files. In the page they type their name, then for each
parcel look at the wide view, the close-up and the greenness curve and press `1`–`6`. At the
end they press **Descargar CSV / Download CSV** and send you the file. Progress is saved in
the browser, so they can stop and come back.

The six options are in [`../s2_labelling/codebook.md`](../s2_labelling/codebook.md), and the
same text sits behind the *Manual* button in every page, so nobody is working from a different
version of the rules.

**`UNSURE` is one of the six and it is a real answer.** Someone made to guess produces a label
that looks exactly like a confident one. Say so when you send the files out.

---

## Step 4 — put the CSVs back

Drop every returned CSV into one folder, exactly as received — do not rename them, merge them
in Excel, or fix anything by hand.

```bash
uv run cc -w national_s2 labelling campaign ingest --round round2 \
    --csv-dir data/processed/all_peru/labels_s2_round2/returned
```

This checks the files, joins them to parcels, measures how often two labellers who saw the
same parcel agreed, and writes `labelled_parcels.parquet`.

It stops with an error — rather than quietly accepting — on a label outside the six, an
`item_id` that is not in `item_key.csv`, or a missing column. The fix is in the CSV, not in
the code.

Parcels are flagged unusable, not deleted, when the answer was `UNSURE`, when the labeller
said the boundary did not match what they saw, or when the parcel has too few Sentinel-2
pixels to summarise.

---

## Step 5 — train on old and new labels together

```bash
uv run cc -w national_s2 labelling campaign combine --round round2
```

That writes `labels_s2_round2_all`: the 865 of record plus your new ones, both originals left
untouched. Then train the way the project's own model was trained:

```bash
uv run cc -w national_s2 labelling train prep --round round2_all --target t3w --climate temp
uv run cc -w national_s2 labelling train fit  --round round2_all --target t3w \
    --model lightgbm --climate temp
uv run cc -w national_s2 labelling train lodo --round round2_all --target t3w --climate temp
```

- `--target t3w` is the label set: 3 classes, woody canopy read as perennial. Others:
  [`07_new_label_set.md`](07_new_label_set.md).
- `--climate temp` adds mean temperature, the one covariate that helped both cross-validation
  and unseen departments.
- `--model lightgbm` is the model carried forward. LTAE wins cross-validation and loses every
  unseen department. `combine` does not rebuild the LTAE tensor, so a merged round trains
  LightGBM only.

The run lands in `runs/labels_s2_round2_all/ws_t3w__clim_temp/lightgbm` — a round trains into
its own directory, named after the round and the arm. That path is what you pass to
`cc predict` later.

If `prep` stops with *"N labelled parcels have no row in …"*, the round's features were not
built: run `campaign assemble` for the round, then `campaign combine`, then `prep` again. The
check exists because a parcel with no features trains as an empty example rather than as an
error.

**Read the `lodo` number, not the cross-validation number.** Cross-validation holds out
parcels inside departments the model already trained on, so it cannot separate real signal
from the model memorising where things are. A change that improves CV and not LODO has not
improved anything: [`03_train_and_evaluate.md`](03_train_and_evaluate.md).

**Do not score the locked test again.** It was spent once, on 2026-09-01, and your combined
round contains those same parcels. Choose on CV and LODO, and leave `--eval-test` alone.

---

## If something goes wrong

| what you see | what it means |
|---|---|
| `label_sample.parquet already exists` | you forgot `--round`, or that round already has a sample. Use a new name |
| `N item_ids in the CSVs are not in item_key` | those CSVs came from another round's pages. Match them to the round that produced them |
| `labels outside the label scheme` | a CSV was edited by hand. Use the file the page produced |
| `N labelled parcels have no climate value` | the parcels are outside the climate table. Rebuild with `uv run cc -w national data climate normals`, or train without `--climate` |
| `ws_… not built` | run `prep` with the same `--target` and `--climate` you are about to fit with |
| Earth Engine looks finished but nothing appeared | it did not finish. Count the output rows |

`uv run cc data verify` says what your machine can currently do, at any point.

---

## The whole loop, end to end

Labelling 100 parcels, retraining on every label, then scoring a fresh set of parcels:

| | |
|---|---|
| 1. draw and label 100 parcels | steps 1–4 above, with `draw --total 100 --pilot-n 0` |
| 2. combine with the 865 of record and train | step 5 above |
| 3. read the LODO score, not the CV score | [`03_train_and_evaluate.md`](03_train_and_evaluate.md) |
| 4. score a new set of parcels with the model you just trained | [`04_predict_new_parcels.md`](04_predict_new_parcels.md), section C |

Next: [`07_new_label_set.md`](07_new_label_set.md).
