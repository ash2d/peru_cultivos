# 4. Label more images, then train on them

The current model learned from **865** parcels that people looked at by eye. Adding more
labelled parcels is the change most likely to make it better.

You do not need to write any code. Every step is one command you copy, plus one step where
people look at pictures in a web browser.

```
pick parcels  ->  make pictures  ->  people label them  ->  put the CSV files back  ->  train
```

You need this repository installed ([`01_setup.md`](01_setup.md)), a Google Earth Engine
account, and an internet connection. The aerial pictures come from Esri's public map service.

**Choose a name for this round of labelling** first: `round2`, `piura2027`, your initials.
Pass `--round <name>` to every command below. This keeps the new work in its own folders and
leaves the existing 865 labels untouched. Without it, a new selection of parcels would
overwrite the list the old labels are attached to. The code refuses to do that, and the flag is
how you avoid meeting the error.

---

## Step 1: choose which parcels to label

```bash
uv run cc -w national_s2 labelling campaign universe --round round2
uv run cc -w national_s2 labelling campaign pool     --round round2
uv run cc -w national_s2 labelling campaign probe    --round round2 --workers 8
uv run cc -w national_s2 labelling campaign draw     --round round2 --total 100 --pilot-n 0
uv run cc -w national_s2 labelling campaign split    --round round2
```

| step | what it does | how long |
|---|---|---|
| `universe` | lists every parcel that could be labelled: it has a boundary and is at least 0.15 hectares | seconds |
| `pool` | makes a shortlist, about 2.5 times the number you want | seconds |
| `probe` | checks with Esri, parcel by parcel, whether the aerial picture is sharp and recent enough | about 1 hour for a few thousand |
| `draw` | picks the final sample: spread across departments and crop types, at most 2 parcels per 5 km square | seconds |
| `split` | fixes which parcels are for training and which are held back | seconds |

`probe` is slow for a good reason. If you ask Esri to zoom in further than it has pictures for,
it returns a plain grey image rather than an error, so every parcel has to be checked before
anyone is asked to look at it.

**Do not change the sample after `draw`.** If you drop parcels later because they look hard, or
because the picture disappoints, the sample no longer represents anything. That is why the
picture check comes first.

`--total` sets how many parcels to pick. `--pilot-n 0` skips the separate trial batch, which a
second round does not need. Before you commit to a round, `uv run cc labelling budget` shows
how much accuracy past labels bought. On the current 865, another 100 labels are worth roughly
0.02 more macro-F1.

## Step 2: make the pictures

```bash
uv run cc -w national_s2 labelling campaign chips    --round round2 --workers 3
uv run cc -w national_s2 labelling campaign extract  --round round2      # Earth Engine, hours
uv run cc -w national_s2 labelling campaign assemble --round round2
uv run cc -w national_s2 labelling campaign html     --round round2 --lang es
```

- `chips` downloads the aerial picture around each parcel and draws the parcel's outline on it.
- `extract` downloads the Sentinel-2 measurements. This is the long step and it can fail
  without saying so. Check it by counting rows, not by seeing the command end
  ([`06_reference.md`](06_reference.md)).
- `assemble` turns those measurements into the greenness curve labellers see and the numbers
  the model trains on.
- `html` writes the labelling pages. Use `--lang es` for Spanish or `--lang en` for English.

You now have `labels_s2_round2/html_es/` containing `shard01_A.html`, `shard01_B.html` and so
on, plus `item_key.csv`. Each of those HTML files holds its own pictures inside it, so you can
email it, put it on a USB stick, or open it by double-clicking. No server is needed.

`item_key.csv` records which parcel each item in the pages refers to. Do not delete or edit it.
Without it the returned labels cannot be matched to parcels.

## Step 3: people label the parcels

Send each person one or more of the HTML files. In the page they type their name, then for each
parcel they look at the wide view, the close-up and the greenness curve, and press `1` to `6`.
At the end they press **Descargar CSV / Download CSV** and send you the file. Their progress is
saved in the browser, so they can stop and come back.

The six options are explained in [`../s2_labelling/codebook.md`](../s2_labelling/codebook.md).
The same text is behind the *Manual* button in every page, so nobody works from a different
version of the rules.

**`UNSURE` is one of the six and it is a real answer.** Someone forced to guess produces a
label that looks exactly like a confident one. Say so when you send the files out.

## Step 4: put the CSV files back

Put every returned CSV in one folder, exactly as you received it. Do not rename the files,
merge them in Excel, or correct anything by hand.

```bash
uv run cc -w national_s2 labelling campaign ingest --round round2 \
    --csv-dir data/processed/all_peru/labels_s2_round2/returned
```

This checks the files, matches them to parcels, measures how often two people who saw the same
parcel agreed, and writes `labelled_parcels.parquet`.

It stops with an error, rather than quietly accepting, if a label is not one of the six, if an
item is not in `item_key.csv`, or if a column is missing. The fix is in the CSV, not in the
code. Parcels are marked unusable rather than deleted when the answer was `UNSURE`, when the
labeller said the outline did not match what they saw, or when the parcel has too few
Sentinel-2 pixels to measure.

## Step 5: train on the old and new labels together

```bash
uv run cc -w national_s2 labelling campaign combine --round round2
```

That creates `labels_s2_round2_all`: the original 865 plus your new ones, with both originals
left untouched. Then train:

```bash
uv run cc -w national_s2 labelling train prep --round round2_all --target t4 --climate temp
uv run cc -w national_s2 labelling train fit  --round round2_all --target t4 --climate temp \
    --model lightgbm
uv run cc -w national_s2 labelling train lodo --round round2_all --target t4 --climate temp
```

- `--target t4` is the set of classes: `PERENNIAL`, `ANNUAL`, `WOODY_NON_CROP`, `OTHER`. This is
  what the parcel table uses. `t3w` instead counts non-crop trees as `PERENNIAL` and is what the
  published results used. Other sets, and how to add your own, are in
  [`06_reference.md`](06_reference.md).
- `--climate temp` adds each parcel's average temperature. It is the only extra input in this
  project that improved both cross-validation and performance on new departments.
- `--model lightgbm` is the model to use. The neural model (LTAE) wins cross-validation and
  loses on every unseen department. `combine` does not rebuild the input the neural model needs,
  so a combined round trains LightGBM only.

The result goes to `runs/labels_s2_round2_all/ws_t4__clim_temp/lightgbm`. That is the folder you
pass to `cc predict-s2 --model` on [page 3](03_score_parcels.md).

**Read the `lodo` score, not the cross-validation score.** Cross-validation hides parcels inside
departments the model has already trained on, so a good score there can mean the model memorised
where things grow. `lodo` hides a whole department. A change that improves cross-validation but
not `lodo` has not improved anything. Three separate inputs and one whole model type were caught
this way ([`06_reference.md`](06_reference.md)).

**Do not score the final test set again.** It was used once, on 2026-09-01, and your combined
round contains those same parcels. Choose your model on cross-validation and `lodo`, and leave
`--eval-test` alone.

## If something goes wrong

| what you see | what it means |
|---|---|
| `label_sample.parquet already exists` | you forgot `--round`, or that round already has a sample. Use a new name |
| `N item_ids in the CSVs are not in item_key` | those CSVs came from a different round's pages |
| `labels outside the label scheme` | a CSV was edited by hand. Use the file the page produced |
| `N labelled parcels have no row in ...` | the round's numbers were not built. Run `campaign assemble`, then `combine`, then `prep` again |
| `N labelled parcels have no climate value` | rebuild with `cc -w national data climate normals`, or train without `--climate` |
| `ws_... not built` | run `prep` with the same `--target` and `--climate` you are about to train with |
| Earth Engine seemed to finish but nothing appeared | it did not finish. Count the rows it produced |

`uv run cc data verify` tells you what your computer can currently do, at any point.
