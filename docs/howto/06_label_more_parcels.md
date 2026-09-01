# 6. Label more parcels, then train on them

**This is the page to follow if you want the classifier to get better.** The model of record
was trained on **865** parcels that people looked at by eye. More parcels is the single change
most likely to help.

You do not need to write any Python. Every step below is one command you copy, plus one step
where people look at pictures in a web browser.

The whole thing is one loop:

```
pick parcels  →  make pictures  →  people label them  →  put the CSVs back  →  train
```

---

## Before you start

**You need three things.**

| | |
|---|---|
| the repository, installed | `uv sync` — see [`01_setup.md`](01_setup.md) |
| a Google Earth Engine account | free for research; [`02_get_satellite_data.md`](02_get_satellite_data.md) |
| an internet connection | the aerial images come from Esri's public map service |

**Everything gets a round name.** Pick one word — `round2`, `piura2027`, your initials — and
pass `--round <name>` to **every** command on this page. That name keeps the new work in its
own folder:

```
data/processed/all_peru/labels_s2            ← the campaign of record. Never written to.
data/processed/all_peru/labels_s2_round2     ← your new round: sample, pictures, labels
data/processed/all_peru/features_s2_round2   ← its satellite features
```

⛔ **Without `--round`, a new draw would overwrite the sample the existing 865 labels are
keyed to, and they would stop being readable.** The code refuses to do it, but the flag is
how you avoid ever meeting that error.

---

## Step 1 — choose which parcels to label

```bash
uv run cc -w national_s2 labelling campaign universe --round round2
uv run cc -w national_s2 labelling campaign pool     --round round2
uv run cc -w national_s2 labelling campaign probe    --round round2 --workers 8
uv run cc -w national_s2 labelling campaign draw     --round round2
uv run cc -w national_s2 labelling campaign split    --round round2
```

What each one does:

| step | what it does | how long |
|---|---|---|
| `universe` | every parcel that *could* be labelled: has a boundary, is at least 0.15 ha | seconds |
| `pool` | a candidate shortlist, ~2.5× the number you want | seconds |
| `probe` | asks Esri, per parcel, **is there aerial imagery here, sharp enough and recent enough** | ~1 h for a few thousand |
| `draw` | the actual sample — balanced across departments and crop classes, capped at 2 parcels per 5 km cell | seconds |
| `split` | freezes which parcels are training and which are held-out test | seconds |

`probe` is the slow one and it is slow for a good reason: **Esri returns a valid grey image
above the zoom level it actually serves, not an error**, so imagery has to be checked parcel
by parcel before anyone is asked to look at it.

⚠️ **Do not change the sample after `draw`.** Dropping parcels afterwards — because they look
hard, or because the imagery is disappointing — changes what the sample represents. That is
why the imagery check happens *before* the draw and not after.

**Want a different number of parcels?** The default is 1,000. Ask what a campaign is worth
before you fund one:

```bash
uv run cc labelling budget          # the learning curve on the labels that already exist
```

---

## Step 2 — make the pictures

```bash
uv run cc -w national_s2 labelling campaign chips   --round round2 --workers 3
uv run cc -w national_s2 labelling campaign extract --round round2      # Earth Engine
uv run cc -w national_s2 labelling campaign assemble --round round2
uv run cc -w national_s2 labelling campaign html    --round round2 --lang es
```

* `chips` downloads the aerial image around each parcel and draws its boundary on it.
* `extract` pulls the Sentinel-2 pixels from Earth Engine — this is the long one, hours, and
  it can **fail silently**: read the Earth Engine warnings in
  [`02_get_satellite_data.md`](02_get_satellite_data.md) before starting it, and **check the
  output by counting rows, never by "the command finished"**.
* `assemble` turns those pixels into the greenness curve each labeller sees, and into the
  numbers the model will train on.
* `html` writes the labelling pages.

You now have, in `data/processed/all_peru/labels_s2_round2/html_es/`:

```
shard01_A.html   shard01_B.html   shard02_A.html   …   item_key.csv
```

Each `shard*.html` is a **self-contained file with the images inside it**. Email it, put it on
a USB stick, open it by double-clicking. No server, no internet needed to use it.

`item_key.csv` is the file that says which parcel each `item_id` refers to. **Do not delete
it and do not edit it** — without it the returned labels cannot be joined to anything.

Use `--lang es` for Spanish (what the real campaign used) or `--lang en` for English.

---

## Step 3 — people label the parcels

Send each person one or more shard files. In the page they:

1. type their name at the top (the name goes into the file they send back);
2. for each parcel, look at the wide view, the close-up and the greenness curve, and press
   `1`–`6`;
3. press **Descargar CSV / Download CSV** at the end and send you the file.

The six options are in [`../s2_labelling/codebook.md`](../s2_labelling/codebook.md), and the
same text is behind the *Manual* button inside every page, so nobody is reading a different
version of the rules than the one on the record.

⭐ **`UNSURE` is one of the six and it is a real answer.** A labeller who is made to guess
produces a label indistinguishable from a confident one. Tell people to use it.

Progress is saved in the browser as they go, so they can close the page and come back.

---

## Step 4 — put the CSVs back

Make a folder and drop every returned CSV into it, exactly as received. Do not rename them,
do not merge them in Excel, do not fix anything by hand:

```
data/processed/all_peru/labels_s2_round2/returned/
    shard01_Maria.csv
    shard02_Jose.csv
    …
```

Then:

```bash
uv run cc -w national_s2 labelling campaign ingest --round round2 \
    --csv-dir data/processed/all_peru/labels_s2_round2/returned
```

This checks the files, joins them to the parcels, computes how often two labellers who saw the
same parcel agreed, and writes `labelled_parcels.parquet`.

It **fails loudly** rather than quietly on: a label that is not one of the six, an `item_id`
that is not in `item_key.csv`, a missing column. If it complains, the fix is in the CSV, not
in the code.

Parcels are automatically flagged as unusable — not deleted — when they are `UNSURE`, when the
labeller reported that the boundary did not match what they saw, or when the parcel has too
few Sentinel-2 pixels to summarise.

---

## Step 5 — train on old labels + new labels together

```bash
uv run cc -w national_s2 labelling campaign combine --round round2
```

That writes `labels_s2_round2_all` — the 865 of record **plus** your new ones — leaving both
originals untouched. Then train exactly as the project's own model was trained:

```bash
uv run cc -w national_s2 labelling train prep --round round2_all --target t3w --climate temp
uv run cc -w national_s2 labelling train fit  --round round2_all --target t3w \
    --model lightgbm --climate temp
```

and read the result:

```bash
uv run cc -w national_s2 labelling train lodo --round round2_all --target t3w --climate temp
```

* `--target t3w` is the label set (3 classes, woody canopy read as perennial). Others:
  [`03_new_label_set.md`](03_new_label_set.md).
* `--climate temp` adds mean temperature, the one covariate that helped both cross-validation
  and unseen departments.
* `--model lightgbm` is the model carried forward. LTAE wins cross-validation and **loses**
  every unseen department, which is the whole lesson of this project.

### ⭐ Read the LODO number, not the CV number

Cross-validation holds out parcels *inside departments the model already trained on*. It
cannot tell real signal from the model memorising where things are. `lodo` holds out a whole
department. **A change that improves CV and not LODO has not improved anything.** Full
explanation: [`04_train_and_evaluate.md`](04_train_and_evaluate.md).

### ⚠️ Do not score the locked test again

The held-out test was spent once, on 2026-09-01. The combined round contains those same
parcels. Choose your model on CV and LODO and leave `--eval-test` alone; a number from a test
set that has already been read confirms nothing.

---

## If something goes wrong

| what you see | what it means |
|---|---|
| `label_sample.parquet already exists` | you forgot `--round`, or you are re-drawing a round that already has a sample. Use a new round name |
| `N item_ids in the CSVs are not in item_key` | those CSVs came from a **different round's** pages. Match the CSVs to the round that produced them |
| `labels outside the label scheme` | a CSV was edited by hand. Use the file the page produced |
| `N labelled parcels have no climate value` | the new parcels are outside the climate table. Rebuild it: `uv run cc -w national data climate normals`, or train without `--climate` |
| `ws_… not built` | run the `prep` line above with the *same* `--target` and `--climate` you are about to fit with |
| Earth Engine seems finished but nothing appeared | it did not finish. **Count the output rows.** [`02_get_satellite_data.md`](02_get_satellite_data.md) |

Check what your machine can currently do at any point:

```bash
uv run cc data verify
```
