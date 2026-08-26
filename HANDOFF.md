# Bridge Condition Forecasting Model — Handoff

**Prepared:** 25 August 2026
**Author:** Steven Ma
**Repository:** `https://github.com/stevenmaswim/bridge-condition-model`
**State at handoff:** branch `main`, commit `3079a47`
**Owning team:** _(to be filled in)_ · **Supervisor:** _(to be filled in)_

> Read §1 and §9 first. §1 is what the thing is; §9 is what still has to happen before anyone
> else can rely on it.

---

## 1. What this is, in plain terms

A forecasting tool that estimates what a Texas bridge's NBI condition ratings (deck,
superstructure, substructure, culvert) will be a chosen number of years in the future.

It exists to support **maintenance and capital budget planning** — screening which bridges are
trending toward poor condition so engineers can look at them sooner. It is decision *support*.
It does not make funding decisions, and it does not replace inspection.

It learns from roughly 1.7 million historical inspection records covering ~85,000 Texas bridges
from 1992 to 2025.

**The core insight the project is built on:** the strongest predictor of a bridge's future
condition is its own current condition plus how many years ahead you are looking. Physical
attributes matter, but far less. The first version of this project ignored that and predicted
condition from attributes alone; the current version does not.

---

## 2. Status and confidence

| | |
|---|---|
| Code | Complete and working. 57/57 unit tests pass (verified 25 Aug 2026 on commit `3079a47`). |
| Evaluation | Leak-free, split by bridge. Numbers are honest and defensible. |
| Engineering review | Done — calibration, benchmark vs. Markov, ranking tests, two negative experiments. |
| Bridge-engineer (SME) review | **Not done.** Packet is written and waiting in `sme_review/`. |
| Production use | **Not approved.** Scope decision never made (see §10). |
| Deployment | None. It runs from a laptop; there is no server, schedule, or pipeline. |

**Be careful with this claim:** the accuracy figures are real and were measured on bridges the
model never trained on. But nobody with a bridge-engineering background has yet confirmed the
deterioration behavior is physically sensible. Until that happens, treat every output as a
screening suggestion, not a finding.

---

## 3. What is in the project

### Code (all in the GitHub repo)

| Path | What it does |
|---|---|
| `main.py` | End-to-end training pipeline. Trains both models, writes outputs. |
| `config.yaml` | Every setting: data source, features, model parameters, thresholds. Start here. |
| `src/data_loader.py` | Load raw data (CSV or Snowflake), rename columns, clean. |
| `src/features.py` | Feature engineering (age at inspection, deck area, traffic density), encoding. |
| `src/enrichment.py` | Joins static physical attributes from the AssetWise extract onto the panel. |
| `src/deterioration.py` | **The primary model.** Inspection events → forward pairs → train → forecast → watch-list. |
| `src/model.py` | Attributes-only fallback model (XGBoost + linear baseline). |
| `src/forecast.py` | Single-bridge forecasting using both models. |
| `src/predict.py` | Batch predictions and grouped summaries. |
| `src/baselines.py` | Reference baselines (persistence, age curve) and shared metric definitions. |
| `src/snowflake_loader.py` | Live Snowflake connection. Credentials from env vars only. |
| `src/report_template.py` | HTML/CSS for the branded forecast report. |
| `src/txdot_logo.py` | Official TxDOT logo vector, extracted from the corporate .potx. |
| `predict_future.py` | CLI: forecast one bridge to a target year. |
| `build_watchlist.py` | CLI: build a ranked budget watch-list. |
| `forecast_ui.py` | CLI: build the interactive HTML report. |
| `discover_snowflake.py` | Browse Snowflake databases/schemas/tables to find the right source. |
| `test_snowflake_connection.py` | **Run before training from Snowflake.** Verifies the source is a panel, not a current-values table. |
| `tests/` | 57 unit tests covering cleaning, encoding, date parsing, pair building, the split, and the serve path. |
| `sql/bridge_data_extract.sql` | The AssetWise/SNBI extract query, with every field's status annotated. |

### Documents (currently **not** in the repo — see §9)

| File | Audience |
|---|---|
| `Bridge_Model_Status_Report_v2.md` | Supervisor. What changed, what the numbers are, decision menu. |
| `Data_Parameters_and_Testing_Report.md` / `.pdf` | Technical. Line-by-line data accounting, every parameter, full testing methodology, plain-English glossary. |
| `Bridge_Condition_Technical_Report.pdf` | Technical deep-dive. |
| `sme_review/SME_Review_Packet.md` + `SME_Review.pdf` | Bridge engineer. Five judgment questions, no stats background needed. |
| `sme_review/*.csv` | The validation exhibits the reports and the HTML page are built from. |
| `README.md` | Developer quick-start. **This one is in the repo.** |
| `HANDOFF.md` | This document. Currently untracked — see §9. |

### Data and models (not in the repo, by design)

- `data/raw/REAL_ML_LEARNING_DATA.csv` — the inspection panel, 1,712,921 rows × 65 columns.
- `data/new_SQL_Querry_for_BRG_P_M.csv` — the enrichment extract, 63,242 bridges × 41 columns, headerless.
- `models/deterioration/*.pkl` — four trained deterioration bundles (the ones actually served).
- `models/*.pkl` — eight attributes-only models (4 targets × 2 model types).
- `data/outputs/` — predictions, metrics, feature importance, watch-lists.

---

## 4. How to run it

### Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run everything with the venv interpreter: `./venv/Scripts/python.exe <script>`.

Verified working on Python 3.14.5, pandas 2.3.3, numpy 2.5.0, scikit-learn 1.9.0, xgboost 3.3.0,
snowflake-connector-python 4.7.1, python-dotenv 1.2.2. `requirements.txt` covers everything
including the Snowflake path — no extra installs are needed.

### Credentials

Copy `.env.example` to `.env` and fill in:

```
SNOWFLAKE_ACCOUNT=txdot-dm_bal_c
SNOWFLAKE_USER=your.name@txdot.gov
```

There is **no password**. TxDOT uses Microsoft SSO — `config.yaml` sets
`authenticator: externalbrowser`, which opens a browser window to log in. Do **not** append
`.privatelink` to the account: this account has no PrivateLink endpoint, and the suffix produces a
confusing host-resolution error with nothing pointing at the cause.

`.env` is git-ignored and must stay that way; the repository is public.

### Verify the data source before training

```powershell
./venv/Scripts/python.exe test_snowflake_connection.py --district 12
```

This is not optional when pointing at a new table. It checks that the source is an **inspection
panel** (many dated inspections per bridge), not a current-values table (one row per bridge). A
current-values table trains without any error and produces a completely useless model, because
there are no inspection-to-inspection transitions to learn from. This check makes that failure
loud — and it has already caught exactly that mistake once (see §5).

### Train

```powershell
./venv/Scripts/python.exe main.py              # trains both models
./venv/Scripts/python.exe main.py --dry-run    # synthetic smoke test, writes to _dry_run/
```

Retrain whenever the source data changes. The attributes-only model derives its categorical codes
at training time, so reusing an old model against refreshed data can silently desync the codes.
The deterioration model persists its encoders and is not affected.

Expect the full run to be memory-hungry: building forward pairs self-joins each bridge's
inspections before sampling down to the 1.2M cap.

### Use it

```powershell
# One bridge, forecast to a year
./venv/Scripts/python.exe predict_future.py --nbi 120200152401017 --year 2040

# Ranked budget watch-list for a district
./venv/Scripts/python.exe build_watchlist.py --district 12

# Interactive branded HTML report
./venv/Scripts/python.exe forecast_ui.py --district 12
```

The HTML report is a single self-contained offline file. It opens in any browser, needs no
server, and can be emailed.

---

## 5. Where the data comes from

Three distinct sources. Being precise about this matters — it is the first thing anyone will ask.

**A. The training panel** — 1,712,921 rows × 65 columns, 1992–2025. This is a *time-series
panel*: ~85,000 bridges each appearing ~20 times, not 1.7M distinct bridges. Misreading this was
the original defect (§7). It can be read from a local CSV or live from Snowflake.

**B. The enrichment extract** — 63,242 bridges × 41 columns, one current row per bridge, produced
by `sql/bridge_data_extract.sql`. Supplies static physical attributes the historical panel lacks.
Static attributes legitimately back-fill a bridge's whole history. **The file is headerless**, so
column names are assigned positionally from `EXTRACT_COLUMNS` in `src/enrichment.py` — that list
must stay in exact sync with the SELECT order in the SQL. Export it *with* its header row and set
`enrichment.has_header: true` to remove this fragility.

**C. The full SNBI export** — ~143 columns. Not used directly; it is where the field selection
came from.

### The live Snowflake source, and the trap in it

```
Rows:   PRD_ADS_BRIDGE_INSP_HIST.BAL_ADS_BRIDGE_INSP_HIST.HIST_BRG_INSP_DATA
Filter: WHERE bridge number IN (SELECT ... FROM CORE_SNBI_DATA JOIN FEATURES
                                ON B.F.01 = 'H01' AND B.F.02 = 'C')
```

Rows come from the **inspection history** table. `CORE_SNBI_DATA` is used only to decide *which
bridges qualify* — never to supply rows. This distinction is the single most important thing to
understand before touching this query.

**Why it matters, learned the hard way (22 Aug 2026):** an earlier version selected *from*
`CORE_SNBI_DATA`. That table returns exactly one row per bridge — 6,865 rows for 6,865 bridges in
district 12 — so it is a current-values table, not history. The deterioration model learns from
inspection-to-inspection transitions, so training on it would have produced a model with nothing
to learn from, and **nothing would have raised an error**. `test_snowflake_connection.py` step 4
is what caught it. Run it before repointing this query at any new table.

The same commit fixed a second silent failure: `parse_inspection_date` assumed the packed integer
`MMDDYYYY` form, and `NaT`-ed every row of a source returning real dates. Because a `NaT`
inspection date is dropped rather than raised, that surfaced only as an empty training set. It now
dispatches per row across native datetimes, packed integers, and ISO strings, with a regression
test covering all four shapes plus mixed input.

**The highway restriction** is the semi-join. `FEATURES` carries one row per feature a structure
interacts with; `B.F.01 = 'H01'` is a highway feature and `B.F.02 = 'C'` is the feature the bridge
*carries*. Together they exclude pedestrian, rail and other structures the agency does not plan
budget against. Measured effect in district 12: 8,736 bridges → 6,865, about **21% dropped**.
`IN (...)` rather than a `JOIN` is deliberate — a join would multiply history rows by the number
of qualifying features per bridge; a semi-join filters without changing cardinality.

The query supplies `year_reconstructed`, `num_spans_main`, `num_spans_approach`, `structure_type`,
`deck_type` and `wearing_surface` directly from the panel, so those no longer depend on the
enrichment extract's match rate. Three attributes still come only from the extract:
`load_posting_status`, `scour_vulnerability`, and `adt_truck`. Load posting is the **second most
important feature** for three of the four components, so its coverage is worth improving — but the
saved models were trained with it arriving from the extract, and category vocabularies may differ
between sources. **If you move it to the panel, retrain in the same change** — a persisted encoder
will silently bucket unseen codes as missing.

---

## 6. The two models

### Deterioration model (primary) — `src/deterioration.py`

Predicts a future rating from the current rating + years ahead + age + physical attributes.
Trained on real inspection-to-inspection transitions.

Each saved bundle holds **three** models:

1. **Point forecast** — the most-likely future rating.
2. **Conservative forecast** — the 25th-percentile "plan for this" rating, for budgeting. It
   deliberately errs toward worse condition, catching ~75% of true 2-point decliners versus ~56%
   for the point forecast, at about a 7% false-alarm cost.
3. **Risk model** — a calibrated `P(rating ≤ 5)`. This is the best ranker for budget priorities
   and gives a probability you can risk-weight against.

A **hybrid rule** is applied automatically at serve time: for horizons under ~3 years it carries
the last rating forward, because ratings barely move and nothing beats that; beyond it, the model
takes over. This is the honest answer to "why not just use the model everywhere."

Because years-ahead is itself an input, one model serves every horizon — a 20-year forecast is a
single evaluation, not twenty one-year steps compounded.

### Attributes-only model (fallback) — `src/model.py`

Predicts a rating from attributes alone, for bridges with no usable inspection history. Weaker,
and only used when the primary model has nothing to work from.

### Train/test split — the setting that makes the numbers real

`GroupShuffleSplit` on `bridge_id`, `test_size = 0.30`, `random_state = 42`
(`config.yaml`, `src/deterioration.py`).

70/30 **by bridge, not by row**. All ~20 inspections of a bridge go entirely to train or entirely
to test. Roughly 60,000 training bridges and 25,000 test bridges. The deterioration model trains
on 575k+ real transitions, capped at 1.2M sampled forward pairs per target.

If you change one setting in this project, do not let it be this one.

---

## 7. Results, and where each number comes from

Measured on held-out bridges, leak-free.

| Component | 10-yr within ±1 | 20-yr within ±1 | 20-yr MAE (model vs. carry-forward) |
|---|---|---|---|
| Deck | 92.7% | 92.4% | 0.44 vs 0.71 |
| Superstructure | 92.7% | 92.0% | 0.46 vs 0.71 |
| Substructure | 90.6% | 90.9% | 0.47 vs 0.75 |
| Culvert | 95.9% | 95.5% | 0.44 vs 0.54 |

On the ~58,000 **on-system** (state-maintained) bridges TxDOT is responsible for, deck accuracy
holds at ~94.8% within ±1 at 10 years and 94.2% at 20 years. The model is more accurate on
on-system bridges because they are better maintained and better documented. We do **not** retrain
on on-system data only — tested, and the difference is +0.2 points, so the all-data model is kept
and the outputs are filtered instead.

It beats the industry-standard material-stratified Markov deterioration-curve method by ~13–17%
at the 20-year horizon, which is what justifies the added complexity.

**One exception, stated plainly:** for culverts at 10 years, carry-forward is marginally better
than the model (MAE 0.371 vs 0.377). Culvert ratings move so little over a decade that "no change"
is hard to beat. The advantage returns by 20 years, and the hybrid rule already defers to
carry-forward at short horizons.

**Top drivers (deck), from `sme_review/feature_importance.csv`:** current rating 38.7%, load
posting status 8.0%, years ahead 4.4%, span continuity 3.5%, wearing surface 3.5%. Note that
**traffic is not a driver** (ADT is 1.0%, 27th of 31) and **age is not top-five** (2.8%, 9th).
Span material is 7th for deck but 2nd for superstructure. If anyone assumes age and traffic
dominate, this table is the correction.

> These figures were measured before the highway-only filter was introduced, which drops about 21%
> of bridges from the population. They should be regenerated from a training run on the current
> source before being quoted externally.

### Figures that cannot currently be reproduced

These come from the status and testing reports and are **not** regenerable from the exhibit CSVs
shipped alongside the code. Regenerate them from a training run before citing them externally:
risk-model AUC 0.93 · top-1% precision 86% vs 81% vs 45% naive · plan-for catch rate ~75% vs ~56%.
`forecast_ui.py` already flags these as pending regeneration in the report appendix.

---

## 8. Known limits — where not to use this

- **It cannot foresee sudden failures.** Scour, impact, and similar events are events, not trends.
  Inspection remains the safety net. Nothing here changes that.
- **It is weakest on bridges already rated 0–4**, where it tends to over-predict recovery. Those
  are fewer than ~600 of ~68,000 test cases. Do not use it to justify deferring work on a
  structure that is already poor — those should be inspected directly.
- **It treats rating increases (repairs, rehab) as noise** rather than predicting them.
- **It predicts a rating trend, not inspection findings.**
- **Long horizons rest on fewer examples.** 20-year accuracy is measured on the bridges that
  actually have 20-year histories.
- **Three attributes still depend on the enrichment extract** (`load_posting_status`,
  `scour_vulnerability`, `adt_truck`), whose match rate is quoted as ~74% in the reports and ~79%
  in a `config.yaml` comment. Reconcile those two numbers before either goes in front of anyone.
  Coverage also skews toward newer bridges, so older on-system bridges fall back to fewer features.
- **Climate-zone grouping is not active.** The field is NULL pending a county/district → zone
  lookup; summaries fall back to district only.
- **R² is modest (~0.5)** and that is expected, not a defect: ratings barely move and inspectors
  routinely disagree by a point. "Within ±1" is the meaningful measure for a screening tool.

---

## 9. Open items

### Still blocking a clean handoff

**1. The documentation is not in the repository.** `.gitignore` currently excludes
`Bridge_Model_Status_Report_v2.md`, `Data_Parameters_and_Testing_Report.md`, `*.pdf`,
`sme_review/`, and `forecast_ui.html`. Anyone who clones the repo gets **code only** — no reports,
no validation exhibits, no SME packet, no sample output. This is the biggest remaining risk: the
analysis that makes the code trustworthy does not travel with the code. Decide deliberately what
should be tracked and what should live on a shared drive, and write that decision down. If the
exclusions were meant to keep TxDOT data out of a public repo, that is a sound reason — but the
documents then need a named, findable home elsewhere. **This file (`HANDOFF.md`) is untracked and
falls under the same decision.**

**2. The repository is public, under a personal GitHub account.** For an office handoff it needs
to move to a TxDOT-owned organization with team access. The code was written to be safe in public
— credentials are env-var only, `.env` is git-ignored and confirmed untracked, data and models are
excluded — but ownership by an individual is not a durable arrangement for a team asset.

### Fixed on 26 August 2026 (commit `3079a47`, merged to `main` in `517bd2d`)

- **`.env.example` gave a Snowflake account that cannot connect** (`txdot-dm_bal_c.privatelink`).
  Corrected, with a comment explaining why the suffix must not be added.
- **`requirements.txt` omitted `snowflake-connector-python[pandas]` and `python-dotenv`**, both
  required because `config.yaml` ships with `data.source: snowflake`. A fresh clone following the
  documented setup crashed on first run. Both added.
- **Work was stranded on a feature branch.** PR #1 merged an older snapshot of
  `feat/txdot-branded-forecast-report`, and `main` sat 13 commits behind it afterwards — still
  carrying the `.privatelink` account, the missing Snowflake packages, and a `config.yaml` aimed
  at the pre-highway-filter query. Anyone cloning `main` got that broken setup.

  Merged in `517bd2d` (a merge commit, not a fast-forward: `main` carried PR #1's merge commit,
  which was never on the branch). Local `main` had one unique commit, `edd8f4c`, a rebase
  duplicate holding an older copy of the coordinate decoder — confirmed equivalent with
  `git cherry` and confirmed superseded by diffing the two versions before it was discarded.
  `main` and the branch are now identical; a cold clone of `main` was verified to receive the
  working setup.

### Smaller items, worth fixing but not blocking

- `src/deterioration.py` — `fit_encoders` calls `.astype(str)` before `.fillna("NA")`, so the
  `fillna` never fires; missing categoricals become the literal string `"nan"` and get their own
  code. Behavior is consistent between training and serving so results are unaffected, but the
  docstring describes something the code does not do.
- `build_watchlist.py` — when no district is given, the output filename is built from the string
  `"ALL districts"`, producing a filename with a space in it. Cosmetic, but awkward to script
  against.
- `src/enrichment.py` — emits a pandas `FutureWarning` about `replace` downcasting. Harmless
  today; will break on a future pandas.
- Two leftover stashes (`filter-branch: rewrite`, `intro-panel-wip`) and a dangling pre-rewrite
  commit history. Housekeeping.
- The enrichment extract is exported headerless, making column names positional. Re-export with
  the header row and flip `enrichment.has_header` to `true`.
- The validation exhibits in `sme_review/` predate the highway-only filter and should be
  regenerated (see the note in §7).

---

## 10. Decisions the office needs to make

These were raised for supervisor decision and never resolved. They are listed in the order that
unblocks the most work.

1. **SME review.** Have a bridge engineer sanity-check the deterioration curves and a sample of
   forecasts. The packet is written and needs about an hour of an engineer's time. Everything else
   is gated behind this.
2. **Scope.** Personal tool, shared team tool, or an input to budget and priority decisions? This
   sets the hardening bar — a personal tool needs nothing more; a budget input needs ownership,
   retraining cadence, and change control.
3. **Data completeness.** Approve broadening the AssetWise join to lift coverage of the three
   extract-only attributes, particularly for pre-1970 on-system bridges — and retrain when it
   changes.
4. **Retraining cadence.** Manual rerun on each AssetWise export, or a schedule. Today it is
   whenever someone remembers.
5. **Where it runs.** Today it runs from one laptop against a personal GitHub repo. If it is going
   to be used by more than one person, it needs a home.

---

## 11. If you are picking this up cold

The fastest path to understanding it:

1. Read `Data_Parameters_and_Testing_Report.md` §10 — a plain-English glossary of every term used
   in this project, written for someone with no statistics background.
2. Read `config.yaml` top to bottom. Nearly every setting carries a comment explaining why it has
   the value it does. It is the most honest map of the project's thinking.
3. Run `main.py --dry-run`. It uses synthetic data, writes to `_dry_run/` folders, and cannot
   touch real models or outputs.
4. Run the test suite: `./venv/Scripts/python.exe -m pytest tests/ -q`. All 57 should pass in
   about 25 seconds.
5. Open a generated `forecast_ui.html` in a browser. Its methodology appendix explains the model
   to a non-technical reader better than any of the source files.

The code is heavily commented, and the comments explain *why* rather than *what* — including the
mistakes that were corrected and the reason each correction was necessary. Read them; several
record decisions that are not recoverable from the code alone. The commit messages are worth
reading for the same reason.

---

## 12. Contacts

| Role | Name | Notes |
|---|---|---|
| Built by | Steven Ma | |
| Supervisor | _(fill in)_ | |
| Owning team | _(fill in)_ | |
| Data source owner (AssetWise / Snowflake) | _(fill in)_ | For access and extract questions |
| SME reviewer | _(unassigned)_ | Needed to unblock everything in §10 |
