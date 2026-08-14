# Documentation cleanup plan

> **For an agent working alone.** Do the tasks in order. Everything you need is in this file
> plus the repo. Do not ask questions; where a judgement call is needed the rule is stated.
>
> **Goal:** 23 markdown files / 13,336 lines → **10 files / ~2,500 lines**, with nothing
> useful lost and the next actual task obvious within 60 seconds of opening the repo.

---

## 0. Why this is needed

The project ran five research strands over a month. Four failed, one returned a number, and a
sixth is live. Every strand left behind a plan doc *and* a results doc, and every failure was
appended to `CLAUDE.md` rather than folded into it. The result:

* **`CLAUDE.md` is 1,041 lines / 81 KB** and is ~70 % results narrative. It is loaded into
  every agent's context automatically. This is the single biggest problem.
* **`docs/all_peru/RESULTS.md` is 2,227 lines**, `docs/perennial/RESULTS.md` 1,549.
* **Six plan docs describe routes that are closed** (window pivot, temporal OOD, endpoint
  labels, perennial panel, all-Peru panel, the original root `plan.md`).
* **Three overlapping summaries** exist (`SUMMARY.md`, `SUMMARY_FULL.md`,
  `SUMMARY_FULL_SHORT.md`).
* An agent cannot tell from any single file **what to do next**.

The information itself is good. It is the *shape* that is wrong.

---

## 1. Do this first — you cannot undo anything yet

**The repo has no commits.** `git log` returns
`fatal: your current branch 'main' does not have any commits yet`. Every file is untracked. If
you delete a doc now, it is gone.

```bash
# 1a. runs/ is 782 MB and is NOT gitignored. Exclude it, but keep the two small
#     selection records that docs reference by path.
```

Append to `.gitignore`:

```
# Model run outputs (782 MB). Selection records are force-added below.
runs/
!runs/all_peru/selected_model.json
!runs/all_peru/selected_model_20260809.json
```

Then:

```bash
git add -A
git add -f runs/all_peru/selected_model.json runs/all_peru/selected_model_20260809.json
git commit -m "Baseline: full project state before documentation cleanup"
git tag docs-cleanup-baseline
```

**Verify before continuing:** `git show docs-cleanup-baseline --stat | head` lists the docs.
Everything deleted later is recoverable with
`git show docs-cleanup-baseline:docs/all_peru/RESULTS.md`. Do not start section 3 until this
commit exists.

---

## 2. The target: 10 files

| file | lines (budget) | what it is | built from |
|---|---|---|---|
| `README.md` | 40 | what this project is, how to run it, doc map | new (currently empty) |
| `CLAUDE.md` | **220 max** | agent orientation only — no results narrative | rewrite of the 1,041-line file |
| `docs/STATUS.md` | 130 | ⭐ what is done, what is next, concrete task list | new |
| `docs/DATA.md` | 260 | every raw dataset, the linkage chains, the silent traps | `DATASETS.md` + `all_peru/DATA_AUDIT.md` + 4 others |
| `docs/PIPELINE.md` | 320 | code guide: modules, CLI, workspaces, gotchas | trim existing 670 |
| `docs/RESULTS.md` | 550 | ⭐ every strand, its verdict, the numbers that survive | `perennial/RESULTS.md` + `all_peru/RESULTS.md` |
| `docs/LESSONS.md` | 130 | methodological findings that generalise | scattered ⭐/⚠️ blocks |
| `docs/REPORT.md` | 484 | human-facing narrative (pairs with the .tex/.pdf) | `SUMMARY_FULL_SHORT.md`, moved unchanged |
| `docs/s2_labelling/plan.md` | 180 | ⭐ the **live** campaign: remaining steps + gates | `s2_labelling_plan.md` + `s2_labelling_RESULTS.md` |
| `docs/s2_labelling/codebook.md` | 212 | frozen labelling codebook | moved **verbatim** |

Folders `docs/all_peru/` and `docs/perennial/` disappear. `docs/figures/` stays as is.

---

## 3. Write the new files

Do these in the order given — later files link to earlier ones.

### 3.1 `docs/RESULTS.md` (new, ~550 lines)

The hardest one; do it first while you have the sources open. Read in full:
`docs/perennial/RESULTS.md`, `docs/all_peru/RESULTS.md`, `docs/SUMMARY_FULL.md`.

**Structure — one section per strand, in chronological order:**

```
# Results

## How to read this
(3 lines: one section per strand; each opens with a verdict line; numbers are
 measured unless marked. Closed means do not reopen — reasons are in the section.)

## 0. Scoreboard
(a table: strand | question | verdict | where the code is | ~10 rows)

## 1. 12-class crop classifier — CLOSED, superseded
## 2. 3-class land-state classifier (Piura) — WORKS
## 3. The Piura 28-year panel — ⛔ FAILED its gate
## 4. All of Peru — WORKS single-year, ⛔ panel FAILED
## 5. The window pivot — ⛔ FAILED (T1, T2, T3)
## 6. Temporal OOD fixes — ⛔ FAILED; two routes closed with reasons
## 7. The tenure DiD — ⚖️ COMPLETE: a bounded null (the project's only estimate)
## 8. S2 endpoint labelling — LIVE (see docs/s2_labelling/plan.md)
```

**Budget each strand section at 40–70 lines.** Rules for what to keep:

* **Keep:** the verdict, the headline numbers with their uncertainty, the *reason* a thing
  failed, and anything a future agent could waste days rediscovering.
* **Keep every "do not reopen" reason**, especially: the OLI route (closed twice — the sensor
  difference is cover-type dependent, so no global linear map can fix it); the tenure DiD
  population ceiling (6,559 treated parcels is all of Peru); the perennial panel estimand.
* **Cut:** blow-by-blow chronology, superseded intermediate numbers, "what was NOT run and
  why" sections, repeated restatements of the same finding across strands, and anything that
  reads as a diary entry rather than a fact.
* **Cut the "Reproducing" shell blocks** from both source files — they move to
  `docs/PIPELINE.md` §Cookbook, deduplicated.
* When a number is superseded (e.g. the DiD pilot's −0.038 → +0.034), state **only the final
  number** and one line saying the earlier one reversed and why. Do not preserve both
  arguments.

**Two things must survive verbatim in substance** because they are the project's most
valuable results:

1. `centroid_lat` = **+0.047 CV / +0.047 LOYO / −0.060 LODO** — and that only LODYO sees both
   axes.
2. The DiD result: headline **−0.0011 [−0.0126, +0.0104]**, placebo **−0.0029**, corrected at
   M = 7 **+0.0195 [−0.0330, +0.0720]** ⇒ NOT-SEPARABLE, plus the caveat that tenure is last
   observed ~2011 while the outcome runs to 2023 (attenuates toward zero).

### 3.2 `docs/LESSONS.md` (new, ~130 lines)

Pull out the findings that are **method, not Peru**. These are currently buried and are the
most reusable thing in the repo. One short entry each: claim, evidence in one line, what to do
differently. Cover at least:

* An OOD evaluation is blind along the axis it holds fixed (CV / LODO / LOYO / LODYO).
* Gain ≠ contribution — measured twice (`frac_l7`, the metadata ablation).
* Time-invariant features manufacture *stability*; flicker was monotone in static count,
  exactly as pre-registered.
* A pre-trend gate should ask "could this overturn my conclusion?", not "does it beat a
  chosen number" — v1 rewarded imprecision, v2 was unpassable at any sample size.
* Check feasibility (required n from variance vs available n from the archive) **before**
  funding any extraction.
* Verify a finished job by **counting its output**, never by "the process ended".
* Silent failures seen here: GEE hangs with no exception; GEE throttles arriving both as
  errors and as 900 s hangs; Esri returning a valid grey image above its zoom; a `.dbf` under
  the wrong basename returning zero columns; content-addressed caches silently skipping new
  columns.
* An unused correction is an untested correction (`harmonization.py`, written months before
  it was first run, and harmful when finally run).
* An abstain must be its own label, not a low confidence.
* Verify a sensor step against a placebo cut on raw bands, never an index.

### 3.3 `docs/DATA.md` (new, ~260 lines)

Merge, in this order:

1. `docs/DATASETS.md` (raw file inventory) — drop the Piura-superseded framing, lead with the
   national folders.
2. `CLAUDE.md` §3 (data inventory) and §4 (the two linkage chains) — these are good; keep them
   nearly as they are.
3. `docs/all_peru/DATA_AUDIT.md` — keep §1 (byte-identical check, one line) and **§4 in full**
   (the four silent traps: zero-padded keys, wrong-basename `.dbf`, 3D geometry, departments
   split across workbooks). Keep §5's population/sample/class-mix table.
4. `docs/perennial/l7_coverage.md` + `docs/perennial/panel_budget.md` → compress to **one
   table each, ~8 lines**. Keep the sibling `.csv` files: move them to `docs/figures/`.
5. `docs/LANDSAT_MISSION_AUDIT.md` + `docs/LANDSAT_MISSIONS_REVIEW.md` → these are two audits
   of one question with the same answer. Compress both to **≤15 lines**: L5/L7 pooling is
   sound (measured step 0.005–0.012 reflectance), OLI is excluded by user decision, and the
   OLI harmonisation route is closed (link to `RESULTS.md` §6).
6. From `CLAUDE.md` §6, keep the caveats that are *about the data*: unreliable dates, spatial
   autocorrelation (~86 % of neighbours share a crop ⇒ splits must be blocked), intercropping,
   ~1 crop-year per polygon, AREA units, bridge choice.

### 3.4 `docs/PIPELINE.md` (trim 670 → ~320 lines)

Keep §2 (pipeline at a glance), §3 (module reference), §4 (artifacts inventory), §6 (gotchas).

* **Delete §1 "Where the project stands"** (264 lines) — that is now `docs/STATUS.md`.
* **Delete §7 "Known limitations & open decisions"** — the live ones move to `STATUS.md`, the
  settled ones to `RESULTS.md`.
* **Add a CLI table.** `src/crop_classifier/cli.py` is 709 lines with ~45 typer commands and
  they are documented nowhere as a set. Generate the list with
  `uv run python -m crop_classifier.cli --help` and each subgroup's `--help`, then write a
  table: command | what it does | when you would run it. Mark closed-route commands
  (`allperu oli*`, `allperu windows`, `allperu estimate`, `allperu external`,
  `perennial trajectories`/`area`) as **CLOSED — built, unrun, blocked**.
* **Add a §Cookbook** holding the deduplicated shell blocks from the two RESULTS files and
  `SUMMARY_FULL.md` §9. Keep the two warnings that matter: never import torch and lightgbm in
  one process on macOS (libomp segfault), and never run two `nbconvert --inplace` on one
  notebook.
* Keep the workspace-switch table (`CC_PROC` / `CC_FEAT` / `CC_RUNS`) — it is easy to get
  wrong and appears in several places.

### 3.5 `docs/s2_labelling/plan.md` (new, ~180 lines)

```bash
mkdir -p docs/s2_labelling
git mv docs/all_peru/s2_labelling_codebook.md docs/s2_labelling/codebook.md   # verbatim, do not edit
```

The codebook is **frozen and embedded in the labelling HTML**. Do not change a word of its
body; only fix the internal link to `s2_labelling_RESULTS.md` so it points at the new plan.

Then write `docs/s2_labelling/plan.md` from `s2_labelling_plan.md` + `s2_labelling_RESULTS.md`:

* **§1 State (10 lines).** Built and done: universe 614,876 parcels; 4,519 Esri probes, G0
  passes at 87.6 %; 2,224 chips; 117,768 S2 parcel-dates; 9 HTML shards, 1,112 parcels; split
  frozen in `config/split_s2labels.yaml`.
* **§2 What is left — the only live work in the project.** Steps 3, 8, 9, 10 of the old §11,
  as a numbered checklist with the gate on each:
  1. Pilot: 120 parcels, both labellers → `allperu s2-labels ingest` → **G1: κ_called ≥ 0.75**.
  2. Label 992 parcels (~37 h human) → **G2: UNSURE share**, **G3: ≥150 per real class**.
  3. Assemble features → train → CV / LODO / locked test → **G4**.
  4. Report the weighted declared→observed transition matrix with CIs.
* **§3 Decisions already frozen (30 lines).** Six label values; the priority ladder
  (`WOODY_NON_CROP` before `NON_AGRICULTURE`); the "could this ground be sown next season
  exactly as it stands?" separating test; no confidence control; labeller types their own
  name; blindness asserted on the raw HTML string.
* **§4 Known limits (15 lines).** One round only, no second pass. 1,000 labels buys a national
  model and one national test number — **not** per-(dept × class) accuracy. No sierra/selva
  labels ever (8 departments have no bridge). Training prior is a ~5× `PERENNIAL` oversample —
  prior-correct the model. Piura is one of the three worst-covered departments for recent
  high-res imagery (0.61 eligible).
* **§5 Build record (20 lines).** Only what a future agent needs: the harmonisation check took
  three attempts and what finally worked (fit the season out per parcel, subtract a placebo
  cut one year earlier, read raw bands not an index); the ribbon needed per-pixel percentiles
  that did not exist, forcing a full re-extraction with a hand-invalidated cache. Cut the rest
  of the 701-line build record.

### 3.6 `docs/STATUS.md` (new, ~130 lines)

The file an agent reads to know what to do. Nothing here may duplicate `RESULTS.md` — link
instead.

```
# Status

Last updated: <date>

## The question
(3 lines: has land shifted annual → perennial, and does secure title cause it?)

## What works
(5 bullets max: the single-year 3-class classifier, Piura 0.681 locked test / national
 0.628 CV; the national label build; the DiD design.)

## What is closed, and must not be reopened
(a table: route | why closed | one-line reason | RESULTS.md section)

## ⭐ What to do next
(the S2 labelling checklist from docs/s2_labelling/plan.md §2, restated in 8 lines,
 with the pilot marked as the immediate next action)

## Standing constraints on any new work
- Never select a model on CV alone — report CV / LODO / LOYO / LODYO.
- The national locked test is UNSPENT. Keep it that way until an estimand passes its gate.
- The Piura locked test is SPENT TWICE (2026-08-05). It is not a clean estimate any more.
- Run a feasibility check (required n vs available n) before funding any GEE extraction.
- Verify a finished extraction by counting its rows, not by the process exiting.
```

### 3.7 `CLAUDE.md` (rewrite, **220 lines max**)

This is the file that costs the most, because it is auto-loaded. It must be orientation, not
history. Target structure:

```
# Peru crop classifier — agent orientation

## Read these, in this order          (5 lines: STATUS → RESULTS → PIPELINE → DATA)
## The one result to know before modelling   (centroid_lat, ~6 lines)
## What this project is                (10 lines)
## Environment                         (uv, python 3.11, GEE auth, workspace env vars — 20 lines)
## Data, in one paragraph each         (~40 lines, pointing at DATA.md for detail)
## The four silent data traps          (~12 lines — keep these here, they cost hours)
## Code layout                         (~20 lines: src/ tree, one line per package)
## Notebooks                           (~15 lines, one line each)
## Gotchas for an agent                (~30 lines — keep current §9 nearly as is)
## Where results live                  (~10 lines: pointers only, zero numbers except the one above)
```

**Delete outright from `CLAUDE.md`:** every dated ⛔/⭐/⚖️ status block in §8 (they are
`RESULTS.md` and `STATUS.md` now), the per-strand narratives, and all repeated findings. The
current §8 alone is ~600 lines and reduces to a ~10-line pointer section.

**Fix the broken link:** `CLAUDE.md` references
`docs/all_peru/AGENT_PROMPT_tenure_did.md`, **which does not exist**. Remove the reference
(the DiD is complete and closed).

### 3.8 `README.md` (new, ~40 lines)

Currently 0 bytes. Write: one paragraph on what the project is; the `uv` setup and GEE auth in
four commands; a doc map table (the 10 files, one line each); a link to `docs/REPORT.md` for
the readable narrative.

### 3.9 `docs/REPORT.md`

```bash
git mv docs/SUMMARY_FULL_SHORT.md docs/REPORT.md
```

Keep the content — it is current (2026-08-12), well written, and pairs with
`docs/project_report.tex` / `SUMMARY_FULL_SHORT.pdf`. Only edit its link block at the top to
point at the new file names. **Rename the PDF to match** (`docs/REPORT.pdf`) and update
`project_report.tex` if it references the old name.

---

## 4. Delete these 19 files

Only after the baseline commit exists **and** the replacement content is written and checked.
23 project `.md` files today − 19 deleted here − 2 moved in §3.5/§3.9 = 2 kept in place
(`CLAUDE.md`, `docs/PIPELINE.md`, both rewritten) + 5 new = **10**.

```bash
git rm plan.md \
       docs/DATASETS.md \
       docs/SUMMARY.md \
       docs/SUMMARY_FULL.md \
       docs/LANDSAT_MISSION_AUDIT.md \
       docs/LANDSAT_MISSIONS_REVIEW.md \
       docs/perennial/plan.md \
       docs/perennial/RESULTS.md \
       docs/perennial/l7_coverage.md \
       docs/perennial/panel_budget.md \
       docs/all_peru/plan.md \
       docs/all_peru/RESULTS.md \
       docs/all_peru/DATA_AUDIT.md \
       docs/all_peru/window_plan.md \
       docs/all_peru/temporal_ood_plan.md \
       docs/all_peru/tenure_did_plan.md \
       docs/all_peru/endpoint_labels_plan.md \
       docs/all_peru/s2_labelling_plan.md \
       docs/all_peru/s2_labelling_RESULTS.md
```

Move the surviving data files out first:

```bash
git mv docs/perennial/l7_coverage.csv docs/figures/l7_coverage.csv
git mv docs/perennial/panel_budget.csv docs/figures/panel_budget.csv
git mv docs/all_peru/did_panel_budget.csv docs/figures/did_panel_budget.csv
```

Then `docs/all_peru/` and `docs/perennial/` should be empty — confirm with `ls -A` and remove.

**Why each deletion is safe:**

| file | reason |
|---|---|
| `plan.md` (root) | 2026-07-20 design doc; the pipeline it proposed was built. Superseded by `PIPELINE.md`. |
| `DATASETS.md` | merged into `docs/DATA.md` §1. |
| `SUMMARY.md` | 12-class one-pager, stale since 2026-07-30, says so in its own header. |
| `SUMMARY_FULL.md` | 1,136-line narrative; folded into `RESULTS.md` + `LESSONS.md`. `REPORT.md` is the readable version. |
| both Landsat audits | same question, same answer; compressed into `DATA.md`. |
| `perennial/plan.md`, `all_peru/plan.md` | fully executed; outcomes recorded. |
| both `RESULTS.md` | folded into `docs/RESULTS.md`. |
| `l7_coverage.md`, `panel_budget.md` | one table each survives in `DATA.md`; the `.csv`s are kept. |
| `window_plan.md`, `temporal_ood_plan.md` | routes closed; verdicts and reasons in `RESULTS.md` §5–6. |
| `tenure_did_plan.md` | study complete; result and design in `RESULTS.md` §7. |
| `endpoint_labels_plan.md` | superseded by the S2 plan, by its own header. |
| `s2_labelling_*` | rewritten into `docs/s2_labelling/`. |

---

## 5. Fix every link

After deleting, no link may dangle. There are ~90 cross-doc links today.

```bash
python3 - <<'EOF'
import re, os, glob
files = ['README.md','CLAUDE.md'] + glob.glob('docs/**/*.md', recursive=True)
bad = []
for f in files:
    d = os.path.dirname(f)
    for m in re.findall(r'\]\(([^)]+\.md)[^)]*\)', open(f).read()):
        if m.startswith('http'):
            continue
        t = os.path.normpath(os.path.join(d, m))
        if not os.path.exists(t):
            bad.append((f, m))
for b in bad:
    print(b)
print('broken:', len(bad))
EOF
```

**This must print `broken: 0`.** It prints `broken: 1` today (the missing
`AGENT_PROMPT_tenure_did.md`).

Also check non-`.md` references still resolve — `docs/figures/*.png|svg|csv` are cited
throughout, and three `.csv` files move in §4.

---

## 6. Tests — read this before changing anything

**Run the suite first:** `uv run pytest -q` → **409 passed** in ~40 s today.

**No test should be removed.** Every one of the 22 test files maps to a live module, and the
modules that look dead are not: `windows.py` is imported by `tenure_did.py`, which produced the
project's only result; `harmonization.py` is imported by `features/assemble.py`. Deleting tests
while keeping the code is the worst of both — the code rots silently and the next agent trusts
it. Do these four things instead:

1. **Fix the one warning.** `tests/test_label_sample.py::TestDraw::test_region_cap_holds_globally`
   uses a class-scoped fixture defined as an instance method (`PytestRemovedIn10Warning`).
   Convert it to a `@classmethod` setting attributes on `cls`.
2. **Mark closed-route tests** so they read as archive, not as live contract. Add to
   `pyproject.toml`:
   ```toml
   [tool.pytest.ini_options]
   markers = ["closed_route: guards code for an estimand that was abandoned; kept so it does not rot"]
   ```
   Apply `@pytest.mark.closed_route` at module level in `tests/test_trajectories.py`,
   `tests/test_area_estimation.py`, `tests/test_window_pivot.py`, and the OLI tests inside
   `tests/test_temporal_ood.py`. Add one line to each file's docstring naming the
   `docs/RESULTS.md` section that closed it.
3. **Add a docs test.** New `tests/test_docs.py` with two checks: no broken internal `.md`
   link (the §5 script as an assertion), and `CLAUDE.md` is under 250 lines. This is what stops
   the problem coming back.
4. **Re-run** `uv run pytest -q`. Expect **411 passed, 0 warnings**.

---

## 7. Housekeeping (do it, it is cheap)

* **Two notebooks share the number 05** — `05_crop_label_cleaning.ipynb` and
  `05_perennial_trends.ipynb`. Rename the second to `06_perennial_trends.ipynb` and update the
  references in `CLAUDE.md` and `docs/REPORT.md`.
* **`logs/` (1.6 MB, 43 files)** is already gitignored. Leave it.
* **`.pytest_cache/README.md`** is generated noise — confirm `.pytest_cache/` is gitignored;
  add it if not. Do not count it as a project doc.
* **`.ruff_cache/`** likewise.
* **Run `uv run ruff check .`** at the end and fix anything the doc edits touched.

---

## 8. Finish

1. `uv run pytest -q` → 411 passed.
2. The §5 link script → `broken: 0`.
3. `find . -name '*.md' -not -path './.git/*' -not -path './.venv/*' -not -path './.pytest_cache/*' | wc -l` → **10**.
4. `wc -l CLAUDE.md` → **≤ 220**.
5. Total doc lines → `~2,500` (from 13,336).
6. **Delete this plan**: `git rm docs/DOC_CLEANUP_PLAN.md`.
7. Commit:
   ```bash
   git commit -m "Consolidate 23 docs into 10; CLAUDE.md 1041 -> ~220 lines"
   ```

**The acceptance test for the whole job:** open `README.md` cold, and inside a minute you can
say what the project is, what works, what is closed, and that the next action is the 120-parcel
S2 labelling pilot with a κ ≥ 0.75 gate. If you cannot, `STATUS.md` is not doing its job.

---

## 9. Rules while you work

* **Never delete a number that was measured.** Delete the paragraph explaining how it felt to
  measure it. If a fact is unique to a file you are deleting, it belongs in the replacement.
* **Preserve every "do not reopen" reason.** They are the most expensive knowledge here — each
  one represents days of measurement. The OLI closure and the DiD population ceiling in
  particular.
* **Prefer a table to prose** for anything enumerable (datasets, runs, commands, gates).
* **One fact, one home.** If it appears in two of the new files, cut it from one and link.
* **Dates:** keep them on results (`measured 2026-08-11`), drop them from narration.
* Drop the emoji density to at most one marker per section heading (⭐ live/important,
  ⛔ closed, ⚠️ trap). They are currently load-bearing *and* everywhere, which cancels out.
