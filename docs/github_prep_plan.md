# Plan: make this repo publishable and usable by collaborators

**Audience:** a coding agent working in this repo.
**Goal:** a collaborator with no prior context can clone, install, get satellite data, build a
label set, train, evaluate, and reproduce the tenure/perennial analysis — without editing Python.

**Decision already made: modify this repo in place. Do not rewrite from scratch.**
The 21k lines in `src/` encode four silent data traps (`CLAUDE.md` §"four silent data traps"),
a CV/LODO/LOYO/LODYO protocol that overturned three separate wrong conclusions, and ~30 test
files. A clean rewrite loses all of it and re-earns the bugs. What is broken is the *interface*:
environment-variable workspaces, 45 CLI commands of which 7 are dead ends, personal absolute
paths, label spaces hardcoded in Python, and docs addressed to an agent that already knows the
project. Every phase below is refactoring and documentation. **No modelling result may change.**

---

## Ground rules for the whole job

1. **No numbers move.** After every phase run `uv run pytest -q` and `uv run ruff check .`.
   If a metric in `docs/RESULTS.md` would change, stop and report instead.
2. **Work in phases, commit per phase**, with the phase name in the message. Phases 1-4 are
   independent; 5-8 depend on 1-3.
3. **Do not delete analysis code**, even for closed routes. Move it (Phase 3). "We tried this
   and measured why it fails" is a result and the docs cite it.
4. **Preserve the CV/LODO/LOYO/LODYO discipline everywhere.** Any new convenience command that
   reports CV alone is a regression — see `CLAUDE.md` §"the one result to know before modelling".
5. Always `uv run`. Never bare `pip`/`python`.

---

## Phase 1 — Kill the environment variables (the single biggest usability blocker)

**Problem.** `CC_PROC`, `CC_FEAT`, `CC_RUNS` decide which dataset you touch. Get one wrong and you
silently read or write the wrong store — there is no error. Every cookbook recipe in
`docs/PIPELINE.md` §7 starts with an `export` line, and a collaborator who forgets it corrupts
the wrong workspace. This is the thing that most makes the project feel unrunnable.

**Do this.**

1. Create `workspaces.yaml` at the repo root — the one file a collaborator edits to point at
   their data:

   ```yaml
   # Where the raw data lives on this machine. Edit these two lines and nothing else.
   data_root: ./data
   gee_project: peru-crop-classifier      # your own Google Cloud project id

   workspaces:
     piura:            {proc: processed,                    feat: processed/features,             runs: runs/piura}
     national:         {proc: processed/all_peru,           feat: processed/all_peru/features,    runs: runs/all_peru}
     national_s2:      {proc: processed/all_peru,           feat: processed/all_peru/features_s2, runs: runs/s2_labels}
     tenure_did:       {proc: processed/all_peru_did,       feat: processed/all_peru_did/features, runs: runs/did}
   ```

2. Rewrite `src/crop_classifier/paths.py` to resolve from this file. Keep the existing
   **call-time, never import-time** rule — the module docstring explains why, preserve that
   warning verbatim. Keep `CC_PROC`/`CC_FEAT`/`CC_RUNS` working as an override so nothing in
   flight breaks, but they stop being the documented path.

3. Add a global `--workspace / -w` option on the Typer app that sets the active workspace for
   the whole invocation. Default: `national`. Every subcommand inherits it.

4. **Print the resolved workspace on every command that writes anything**, e.g.
   `workspace=national  proc=data/processed/all_peru  runs=runs/all_peru`. Silent
   mis-targeting is the failure mode; make it non-silent.

5. Add `cc workspaces` — lists each workspace, its resolved paths, and whether the expected
   inputs exist on this machine (✓/✗ per artefact). This is the first command a new
   collaborator should run.

6. Update the `docs/PIPELINE.md` §7 cookbook: `export CC_PROC=... CC_FEAT=... CC_RUNS=...`
   becomes `-w national`.

**Accept when:** no recipe in any doc requires an `export`, and `cc -w national workspaces`
prints a correct existence report.

---

## Phase 2 — Remove every machine-specific path and secret assumption

1. `src/crop_classifier/allperu/cenagro_extract.py:38` hardcodes
   `/Users/ash/Library/CloudStorage/OneDrive-SharedLibraries-UDEP/...`. Move it to
   `workspaces.yaml` as `cenagro_source_dir:` (nullable). If unset and the command is invoked,
   fail with a message that names the file to edit and what the source share is.
2. `src/crop_classifier/features/landsat_gee.py:38` hardcodes `GEE_PROJECT =
   "peru-crop-classifier"`. Read `gee_project` from `workspaces.yaml`, falling back to the
   `GEE_PROJECT` env var. Do the same in `features/s2_gee.py` if it carries its own copy.
3. `grep -rn "Users/\|OneDrive" src tests notebooks docs` must return nothing but prose.
4. Add `.env.example` if any credential path is still expected, and confirm `.gitignore` covers
   it (it already covers `*-key.json`, `credentials.json`, `.env`).

**Accept when:** the grep is clean and a fresh clone on another machine needs exactly one file
edited (`workspaces.yaml`).

---

## Phase 3 — Split the CLI into "what a collaborator runs" and "what the paper needed"

**Problem.** `uv run python -m crop_classifier.cli allperu --help` lists ~30 commands. Seven are
closed routes, several more are one-off audits. A newcomer cannot tell which four commands are
the pipeline.

**Do this.**

1. **Add a console script** so the entry point is `cc`, not
   `uv run python -m crop_classifier.cli`. Add to `pyproject.toml`:
   `[project.scripts] cc = "crop_classifier.cli:app"`. Every doc becomes `uv run cc ...`.

2. **Restructure the command tree to task-shaped top-level verbs**, keeping the existing
   functions — this is renaming and regrouping, not rewriting:

   ```
   cc workspaces                 # what's configured, what exists on disk
   cc data      build            # PETT chain A -> linked polygons + labels
   cc data      check            # the four data traps, run against real files
   cc labels    build            # label table from config/labels/*.yaml
   cc labels    list             # available label sets and their class counts
   cc satellite extract          # GEE: Landsat or Sentinel-2 (--sensor)
   cc satellite assemble         # pixel store -> model features
   cc train                      # --labels <set> --model lightgbm|ltae|rules
   cc evaluate                   # runs CV + LODO + LOYO + LODYO together, one table
   cc predict                    # apply a trained model to a parcel set
   cc analysis  tenure-shift     # PETT -> CENAGRO 2012 perennial change by tenure
   cc analysis  transitions      # declared -> observed transition matrix
   cc analysis  did              # the two-period tenure DiD
   cc labelling ...              # the human photo-interpretation campaign
   cc archive   ...              # closed routes, unchanged, kept for reproduction
   ```

3. **Move the closed routes out of the main tree.** Create
   `src/crop_classifier/archive/` and move `allperu/windows.py`, `window_sample.py`,
   `estimate.py`, `external.py`, `oli_overlap.py`, `oli_refit.py` (~1.9k lines) plus the
   panel/flicker modules the gate closed (`perennial/panel.py`, `diagnostics.py`,
   `trajectories.py`, `gapfill.py`, `harmonization.py`). Keep their tests. Add
   `src/crop_classifier/archive/README.md` listing, per module, the estimand, the gate it
   failed, and the `RESULTS.md` section. Wire them under `cc archive`.

4. **`cc evaluate` must report all four splits in one table by default**, with the
   majority-class floor printed beside every macro-F1 (`CLAUDE.md` §"Two traps"). Today a user
   must know to run `lodo`, `loyo` and `lodyo` as separate commands and know that CV alone is
   misleading. Bake the protocol into the tool so it cannot be skipped by accident.

5. Keep old command paths working as hidden aliases for one release so nothing in `docs/` or a
   shell history breaks mid-migration; mark them deprecated in `--help`.

**Accept when:** `uv run cc --help` fits on one screen, every listed command is one a
collaborator would actually run, and `cc evaluate` alone produces the CV/LODO/LOYO/LODYO table.

---

## Phase 4 — Make a new label set a YAML edit, never a code edit

**Problem.** The 3-class policy is already config-driven (`config/perennial*.yaml` with the
`extends:`/`add:`/`drop:` loader — good, keep it). But the S2 label spaces are hardcoded as a
Python dict, `TARGETS` in `src/crop_classifier/labelling/train_prep.py:59`, and the codebook is
frozen in `docs/s2_labelling/codebook.md`. Adding a label set today means editing Python in two
places and knowing which. This is the requirement "adapt if new labels come later".

**Do this.**

1. Create `src/crop_classifier/config/labels/` and give every label set one file:
   `piura_12class.yaml`, `perennial_3class.yaml`, `s2_t5.yaml`, `s2_t4.yaml`, `s2_t3w.yaml`,
   `s2_t2.yaml`. Use the existing `extends:` loader so a variant is a **diff, not a copy** —
   that discipline is already written down in `config_loader.py` and it is correct.
2. Each file declares: `classes`, `group_priority`, the token→class lists, the collapse map
   from the parent space, and `max_unassigned_frac`.
3. Replace `train_prep.TARGETS` with a loader over that directory. `cc labels list` enumerates
   it. `--labels s2_t3w` replaces `--target t3w` everywhere.
4. **Keep and make loud the unmapped-token audit.** `CLAUDE.md` records that a 4.09 % catch-all
   was 80 % one token and moved the headline from +2.4 pp to +12.5 pp. On every label build,
   print the unmapped tail **sorted by frequency, top ten**, and fail if it exceeds
   `max_unassigned_frac`. This must be impossible to miss, not a CSV a collaborator has to know
   to open.
5. Write `docs/howto/new_label_set.md`: copy the nearest YAML, add `extends:`, add tokens under
   `add:`, run `cc labels build --labels mine`, read the printed tail, run
   `cc train --labels mine` then `cc evaluate --labels mine`. No Python.
6. Add a test that round-trips a synthetic new label set end to end, so the "no code change"
   promise is enforced rather than claimed.

**Accept when:** a new 4-class set can be added and trained with zero edits under `src/`, and
the test proves it.

---

## Phase 5 — Solve "collaborators have no data" (otherwise none of the above is runnable)

`data/` is 33 GB, gitignored, and partly under a UDEP OneDrive licence. Right now a collaborator
who clones this repo can run **nothing**. Fix in three layers:

1. **`docs/DATA_ACCESS.md`** — for each raw input: what it is, size, who owns it, how to request
   it, and which analyses die without it. Sources are already characterised in `docs/DATA.md`;
   this is the *acquisition* view, aimed at someone who has none of it.
2. **A committed demo workspace.** Add `data/demo/` (tracked, a few MB): ~500 real parcels from
   one department with geometry, declared crop, year, tenure, and their already-extracted
   feature rows. Add workspace `demo` to `workspaces.yaml`. Then
   `uv run cc -w demo labels build && uv run cc -w demo train && uv run cc -w demo evaluate`
   works on a fresh clone with no GEE account and no 33 GB. Check with the repo owner before
   committing any real parcel data; if licensing blocks it, generate a synthetic store with the
   same schema and label it clearly as synthetic (metrics from it are meaningless — say so in
   the output).
3. **A CI job** that runs the demo path end to end, so the quickstart cannot silently rot.

**Accept when:** a clone with no `data/raw/` and no GEE credentials completes the demo
quickstart, and CI enforces it.

---

## Phase 6 — Rewrite the docs around tasks, not around the project's history

Current docs are excellent as a *record* and poor as an *entry point*: 5,000 lines addressed to
someone who already knows the project. Keep the record; add the entry.

**New top-level `README.md`** (~150 lines, the only thing you can assume is read):
question → one figure → what works and what doesn't in five bullets → install → the demo
quickstart → a table linking the five how-tos → where the detailed docs are → citation, licence,
contact.

**New `docs/howto/` — five task-shaped guides, each start-to-finish, copy-pasteable, with
expected runtime and expected output at every step:**

| file | covers |
|---|---|
| `01_setup.md` | `uv sync`, GEE auth + your own GCP project, editing `workspaces.yaml`, `cc workspaces` to verify |
| `02_get_satellite_data.md` | `cc satellite extract` for Landsat and S2: the two-stage design, chunk sizing, cost, resumability, and **⚠️ the three ways GEE fails silently — verify a finished job by counting its output, never by "the process ended"** |
| `03_new_label_set.md` | Phase 4's YAML workflow, incl. the unmapped-token tail |
| `04_train_and_evaluate.md` | training on the PETT labels vs the S2 photo-interpreted labels; **why CV alone is not a verdict**; reading the CV/LODO/LOYO/LODYO table; the majority-class floor; the libomp/one-arm-per-process rule |
| `05_perennial_change_by_tenure.md` | the headline analysis: `cc analysis tenure-shift`, `transitions`, `did`; what each estimand is, what it assumes, and what the answer currently is (a bounded null) |

**Restructure existing docs:**

* `docs/RESULTS.md` (1,763 lines) — keep as the numbers of record, add a summary table at the
  top so the headline results are visible without reading it all.
* `docs/LESSONS.md` — keep; this is the most transferable thing in the repo.
* `docs/PIPELINE.md` — strip the cookbook (it becomes the how-tos) and keep it as the module
  reference.
* `docs/STATUS.md` — keep, it is the "what to do next" page.
* `CLAUDE.md` — keep as agent orientation, retarget the paths to the new CLI.
* `docs/DATA.md` — keep. The four traps stay prominent.

**Accept when:** someone who has never seen the project can go from clone to a trained,
correctly-evaluated model using only `README.md` + `docs/howto/`.

---

## Phase 7 — Repo hygiene

1. **Untracked build junk in `reports/`** — `peru_report.{aux,fdb_latexmk,fls,log,out,synctex.gz,toc}`
   must never be committed. Add `reports/*.aux` etc. to `.gitignore`. Decide whether
   `peru_report.pdf`/`.tex` supersede `REPORT.md`/`REPORT.pdf`/`project_report.tex` and keep
   **one** current narrative; the other three are duplicates.
2. **`reports/figures/` duplicates `docs/figures/`** (9 files, several byte-identical). Keep one
   directory, `docs/figures/`, and have the report reference it.
3. `.pytest_cache/` and `.ruff_cache/` are present — confirm they are ignored and untracked.
4. **Add `LICENSE`** (ask the owner; MIT or Apache-2.0 for code) and **`CITATION.cff`**.
   Note the data licence separately in `DATA_ACCESS.md` — the code licence does not cover PETT
   or CENAGRO.
5. **Add `CONTRIBUTING.md`**: `uv` only, run `pytest` + `ruff` before pushing, never commit
   under `data/` or `runs/`, and the rule that a result changing means stop and report.
6. Git history is 46 MB packed. Fine — leave it.
7. **Move the notebooks to `notebooks/exploratory/`** with a README saying they are historical
   and superseded by `src/`, except `04_inspect_parcel_basemaps.ipynb` which is a live tool.

---

## Phase 8 — Make the code itself easier to read

Do this **last** and **conservatively** — it is the phase most likely to change a number.

1. `src/crop_classifier/cli.py` is 953 lines with heavy logic inside command bodies (see
   `allperu s2-train`, ~120 lines of branching). Move that logic into the modules it belongs to;
   commands become thin argument parsing. Behaviour identical.
2. `allperu/tenure_did.py` is 1,092 lines. Split into `did/sample.py`, `did/estimate.py`,
   `did/pretrend.py`. No logic changes.
3. `labelling/build_html.py` is 820 lines with an HTML template inline — extract the template to
   a `.html` file beside it.
4. Add module-level docstrings that say **what a reader wants to do with this file**, not only
   what it does. Many already have excellent *why* docstrings — keep every warning verbatim,
   especially the call-time/import-time rule in `paths.py` and the `extends:` rationale in
   `config_loader.py`.
5. Add type hints at public function boundaries where missing.
6. **Do not** reformat wholesale, rename public functions gratuitously, or "simplify" anything
   whose docstring explains why it is shaped that way. Those shapes are usually a bug that was
   already paid for.

---

## Suggested order and rough effort

| phase | effort | unblocks |
|---|---|---|
| 1 workspaces | M | everything |
| 2 machine paths | S | a second machine |
| 3 CLI + archive | M | comprehensibility |
| 4 label YAML | M | "new labels later" |
| 5 demo data + CI | M | a collaborator running anything |
| 6 docs | L | understanding |
| 7 hygiene | S | publishing |
| 8 code shape | M | maintenance |

Phases 1-3 alone make the repo runnable by someone else. Phase 5 is the one people forget and it
is the one that decides whether a collaborator ever gets past `uv sync`.

## Final acceptance checklist

- [ ] Fresh clone, no data, no GEE: demo quickstart in `README.md` runs to a trained model.
- [ ] `grep -rn "Users/\|OneDrive" src tests` returns nothing.
- [ ] No documented recipe requires an `export`.
- [ ] `uv run cc --help` fits one screen; no closed route is listed outside `cc archive`.
- [ ] A new label set is addable with zero edits under `src/`, proven by a test.
- [ ] `cc evaluate` reports CV **and** LODO **and** LOYO **and** LODYO with the majority-class
      floor, by default.
- [ ] `uv run pytest -q` and `uv run ruff check .` clean.
- [ ] Every number in `docs/RESULTS.md` unchanged.
- [ ] `LICENSE`, `CITATION.cff`, `CONTRIBUTING.md`, `docs/DATA_ACCESS.md` present.
