# Labelling codebook — S2 endpoint campaign

> **Frozen 2026-08-13, before any parcel was labelled. Revised the same day** — a sixth
> value (`NON_AGRICULTURE`) was added, `OTHER` was narrowed to make room for it, the
> confidence control was removed, and the two image panels changed (see
> [`plan.md`](plan.md) §5). **Still frozen before any
> parcel was labelled**, so nothing on record was collected under the old text.
>
> The same text is embedded in every labelling HTML behind the *Codebook* button, so what a
> labeller reads and what is on the record here cannot drift apart.
>
> The codebook is the single biggest determinant of κ. Read it once end to end before
> starting, and re-read the decision rules whenever a parcel makes you hesitate.

## What you are doing

**Type your name in the box at the top before you start.** The letter in the filename
(`shard01_A.html`) is only a suggestion of who should take that file; what goes into the
CSV is whatever you type. Your progress is saved against that name, so two people can use
the same computer without clobbering each other.

For each parcel you see three things:

1. **context** (left) — the whole parcel outlined in **yellow**, its neighbours in thin
   **cyan**, **plus a margin of the land around it**, on Esri World Imagery at the finest
   zoom the service serves there. The same crop reads differently in different geographies,
   so this panel is there to show you what the parcel *sits in* — a river, a town edge,
   forest, a block of other field types;
2. **zoom** (right) — the same centre, **200 m across**, so canopy texture is legible even
   when the parcel is far too large for the left panel to show it;
3. **NDVI trace** — 24 months of Sentinel-2 greenness, one point per clear observation,
   with a shaded band around it and a dashed yellow line at the date the aerial imagery was
   taken.

**Each panel carries its own scale bar and they are not the same scale.** The context panel
is 400 m across for most parcels and up to 2.7 km for the largest; the zoom panel is always
200 m.

You pick **exactly one of six values** for every parcel.

## The six values

| key | value | covers |
|---|---|---|
| `1` | **PERENNIAL** | woody or multi-year crop expected to hold the parcel >3 years: mango, lime, avocado, olive, coffee, cacao, banana/plantain, oil palm |
| `2` | **ANNUAL** | sown and harvested within a cycle: rice, maize, cotton, potato, beans, wheat |
| `3` | **OTHER** | **farmable land that is not currently a crop**: pasture, fallow, ploughed or prepared bare ground, weeds and scrub on ground that could be sown |
| `4` | **WOODY_NON_CROP** | trees that are not a crop: windbreaks, riparian strips, invaded/abandoned parcels |
| `5` | **UNSURE** | you genuinely cannot tell |
| `6` | **NON_AGRICULTURE** | **land out of agricultural use altogether**: buildings, yards, settlement, greenhouses, industrial sheds, roads and tracks, canals and reservoirs, open water, active riverbed sand and gravel, quarries, bare rock |

⚠️ **`UNSURE` stays on key `5` and `NON_AGRICULTURE` takes `6`.** `NON_AGRICULTURE` was
added after the rest of the list existed; renumbering the abstain to make the table read
tidily would change what an already-briefed labeller's fingers do.

**UNSURE is not a failure and it is not scored against you — use it freely.** An honest
UNSURE is worth more than a guess: a guess is indistinguishable from a real label
downstream, whereas an UNSURE parcel is simply set aside. It is the right answer for cloud,
deep shadow, a parcel that is half one thing and half another, or imagery too coarse to
tell whether those blobs are crowns.

**PERENNIAL** — *visual:* regular crown pattern or row structure, canopy texture, green in
both seasons of the trace.
⚠️ **Sugarcane is ANNUAL here** (matching `config/perennial_allperu.yaml`'s
`cana_policy: annual`, and MapBiomas).

**ANNUAL** — *visual:* uniform texture, no crowns, sharp field boundaries; one or two NDVI
peaks with returns to bare between them.

**OTHER** — *visual:* no crop geometry at all, or crop geometry with nothing growing across
both seasons of the trace.

**WOODY_NON_CROP** — *visual:* tree cover **without** rows or a planting grid, often
following a watercourse or a field edge rather than filling the parcel.
⚠️ **Never fold this into PERENNIAL.** Declared woody non-crop is 2.79 % of the national
pool, and that is a *lower* bound — it counts only what was declared, not twenty years of
invasion. It is bigger than the effect this project is trying to measure.

**NON_AGRICULTURE** — *visual:* roofs, pavement, vehicle tracks, water, or unvegetated
sand/rock, with no field geometry at all.

## ⭐ The one rule that separates OTHER from NON_AGRICULTURE

These two classes are the pair most likely to cost κ, because until 2026-08-13 they were a
single class. `OTHER` used to be defined as *"everything else: pasture, fallow, prepared
bare ground, scrub, natural vegetation, water, built-up, road, riverbed"* — the last four of
which are now `NON_AGRICULTURE`. Apply this test literally:

> **Could this ground be sown next season exactly as it stands?**
>
> * **Yes** → **OTHER**
> * **No** — something would have to be demolished, dug up or drained first, or it is
>   permanently water, rock or pavement → **NON_AGRICULTURE**

Worked cases, in both directions:

| what you see | class | why |
|---|---|---|
| dry bare fallow, ploughed soil | `OTHER` | farmland between crops — sow it tomorrow |
| grazed pasture, however rough | `OTHER` | in agricultural use |
| scrub on flat farmable ground | `OTHER` | clear the weeds and it is a field |
| riverbed sand and gravel | `NON_AGRICULTURE` | active channel, not farmable ground |
| quarry floor, bare rock, steep scree | `NON_AGRICULTURE` | no soil to sow |
| houses, yards, sheds, greenhouses | `NON_AGRICULTURE` | would have to be demolished |
| road, track, canal, reservoir | `NON_AGRICULTURE` | permanent infrastructure |

**Why this matters and is not pedantry:** a fallow field can convert to a perennial and a
road cannot. Pooling them puts a structurally impossible outcome into the same class as the
one this project exists to measure.

## Decision rules for the ambiguous cases

These are where κ is won. Every parcel gets one of the six, so each rule ends somewhere.

**Work down this ladder and stop at the first line that fits >50 % of the parcel.** The
order is what keeps two labellers from splitting the same parcel between `4` and `6`, or
between `3` and `6`:

1. woody / multi-year **crop** → **1 PERENNIAL**
2. sown-and-harvested **crop**, growing or between cycles → **2 ANNUAL**
3. tree or shrub cover that is **not** a crop → **4 WOODY_NON_CROP**
   — *before 6*: riparian trees along a river are woody non-crop; the water and gravel
   beside them are non-agriculture
4. surface that could not be sown as it stands → **6 NON_AGRICULTURE**
5. anything else — farmable ground, not currently cropped → **3 OTHER**
6. you cannot tell → **5 UNSURE**

Then the specific cases:

* **Mixed parcel** → the class covering **>50 %** of the parcel. If genuinely even,
  `UNSURE`. A farmhouse or shed inside a cropped field does **not** make the parcel
  `NON_AGRICULTURE` — only a parcel that is *mostly* built is.
* **Agroforestry with a closed tree canopy** → `PERENNIAL`.
* **Young plantings** → `PERENNIAL` if a regular planting grid is legible **or** the trace
  shows low-amplitude green persisting through both dry seasons. Otherwise `UNSURE`.
  **Never `ANNUAL` by default** — an immature orchard read as annual is a false negative on
  exactly the transition this project is about.
* **Unreadable** (cloud, deep shadow, partial coverage) → `UNSURE`.
* **The visible field boundary disagrees with the yellow outline** → label **what is inside
  the outline**, and tick **boundary mismatch** (`b`).

## Reading the NDVI trace

The trace is the single most useful thing on the page, and it is what makes ANNUAL vs OTHER
callable at all — a single aerial photograph cannot tell a harvested field from a fallow one.

* **fallow / bare** — flat and low across both seasons;
* **annual** — one or two clear peaks, returning near bare in between;
* **mature perennial** — persistently green with a *small* seasonal swing;
* **young orchard** — low-amplitude green that never quite goes to bare.

The y-axis is fixed at −0.1 to 1.0 on every parcel, so traces are directly comparable.

### The shaded band is spread, not uncertainty

The line is the parcel's **median** greenness on each date. The band around it is the
**middle half of the parcel's own pixels** on that date (25th to 75th percentile).

⚠️ **It is not an error bar.** A wide band does not mean the measurement is poor; it means
the parcel is not doing the same thing everywhere.

* **narrow** — the parcel is uniform: one field of one thing, or a closed canopy;
* **persistently wide** — the parcel is internally varied: tree crowns against bare
  inter-row (a good sign for `PERENNIAL`), or genuinely half one thing and half another,
  which is a prompt to check the >50 % rule or press `5`.

## UNSURE is the only abstain

If you cannot call a parcel, press `5`.

**There is no confidence control any more.** It used to sit beside the label as a 1–3 grade
and it has been removed: a low-confidence guess and a real label are indistinguishable once
they are in the training set, so a half-abstain helped nobody and split the signal with the
abstain that actually works. Either call the parcel or press `5`. G2 fails if more than
25 % of the sample is `UNSURE`, so an honest abstain is read directly by a gate.

## Reading the two images

The left panel is **context** — it tells you the parcel's **shape, its neighbours, and what
it sits in**. Perennial land usually looks different from the annual fields next to it, and
that contrast is often the easiest signal on the page; the surrounding landscape (river,
town edge, forest, irrigation block) is what makes the same crop legible in Piura and in
Huancavelica. The right panel tells you the **texture**: regular crowns on a grid means a
planted orchard; irregular blobs mean woody non-crop; smooth uniform tone means an annual
field or bare ground.

⚠️ Most imagery is 1.2 m, so the zoom panel **enlarges rather than resolves**. If it is
still not clear at 200 m across, the information is not there — that is an `UNSURE`, not a
harder look.

## Controls

| key | action |
|---|---|
| `1` `2` `3` `4` `5` `6` | assign the class **and advance** |
| `b` | toggle boundary mismatch |
| `←` `→` | move without labelling |

Your work autosaves to the browser after every action, keyed to the shard *and your name*,
so closing the tab loses nothing. Press **Download CSV** when you finish a shard (or partway — it can be
downloaded as often as you like) and send the file back.

⚠️ **Do not look anything up.** The parcel's 1998 crop declaration exists and is
deliberately not shown to you. If you were anchored on it you would manufacture agreement
between the declaration and what you see, which is the exact failure this campaign is
designed to avoid measuring.

## What you will not be told, and why

The declared PETT class, the train/test assignment and the CV fold are **not present
anywhere in the HTML file**, not merely hidden — `tests/test_build_html.py` asserts on the
raw file text. Department is shown, because it is obvious from the imagery anyway.
