# How to

Four things people come here to do. Each page gives the commands in order, and the few things
that can go wrong without an error message.

| I want to | page | do I need anything |
|---|---|---|
| **Get one table of every parcel and its crop type**, from all four records: what the farmer declared, the 2012 census, what a person saw in 2025 images, and what the model predicts for 2025 | [`02_parcel_table.md`](02_parcel_table.md) | no |
| **Run the model on parcels nobody has labelled.** A sample from all of Peru, or your own field boundaries | [`03_score_parcels.md`](03_score_parcels.md) | a Google Earth Engine account |
| **Label more images and train a better model** | [`04_label_and_train.md`](04_label_and_train.md) | Earth Engine, and people to do the labelling |
| **Get summary numbers:** share of land in tree crops, share of parcels with a title, and how both changed | [`05_summary_stats.md`](05_summary_stats.md) | no |

Start with [`01_setup.md`](01_setup.md). It takes about 15 minutes and you only do it once.

[`06_reference.md`](06_reference.md) has everything else: checking the published numbers,
training and testing a model, downloading images, changing the list of crop classes, and the
research questions that are finished or abandoned.

```bash
uv run cc --help          # every command
uv run cc data verify     # what your computer can currently do
```
