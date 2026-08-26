# Start here

First-day setup for the bridge condition forecasting model. Follow it in order — each step ends
with something you should actually see on screen. If you don't see it, stop there rather than
continuing; several failures in this codebase are silent, and a later step will look fine while
producing nonsense.

For background on what the model is and how it works, read `HANDOFF.md` after step 2. This file
is only about getting it running.

Budget about an hour, most of it waiting on installs and access.

---

## Before you start — things only Steven can give you

- [ ] **GitHub access** to `stevenmaswim/bridge-condition-model` (public to read; you need write
      access to push)
- [ ] **Snowflake SSO**, role `Public`, warehouse `WH_SMALL_GEN_BI`, on **both**
      `PRD_ADS_BRIDGE_INSP_HIST` and `PRD_BRIDGE_INSP` — the query reads across the two
- [ ] **The shared-drive folder** (path: _ask Steven — it is not in the repo_) containing:
  - `data/new_SQL_Querry_for_BRG_P_M.csv` — **the important one**, see step 5
  - `models/` — the trained models, so you don't have to retrain to try things
- [ ] **AssetWise access**, eventually, for regenerating that extract

Steps 1–3 work without any of this. Do them while you wait.

---

## Step 1 — Install (10 min)

```powershell
git clone https://github.com/stevenmaswim/bridge-condition-model.git
cd bridge-condition-model
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Python 3.14.5 is what it's verified on. `requirements.txt` covers everything including the
Snowflake path — no extra installs.

**Run everything with `./venv/Scripts/python.exe <script>`**, not a bare `python`. Mixing
interpreters is the most common way to get confusing import errors here.

> ✅ **You should see:** `pip` finishing without errors, and `./venv/Scripts/python.exe -c "import xgboost, snowflake.connector; print('ok')"` printing `ok`.

---

## Step 2 — Prove the code works, before any data (5 min)

```powershell
./venv/Scripts/python.exe -m pytest tests/ -q
./venv/Scripts/python.exe main.py --dry-run
```

The dry run trains on synthetic data. It needs no database, no files, and writes only into
`_dry_run/` subfolders, so it cannot overwrite anything real.

> ✅ **You should see:** `62 passed`, then `Pipeline complete!` with `Models trained: 8` and
> `Predictions: 200 rows`.

If both pass, your environment is correct and every later problem is data or access, not setup.
That's worth knowing before you start debugging credentials.

Good moment to read `HANDOFF.md` §1–§2 while you wait on access.

---

## Step 3 — Credentials (5 min)

```powershell
copy .env.example .env
```

Edit `.env`:

```
SNOWFLAKE_ACCOUNT=txdot-dm_bal_c
SNOWFLAKE_USER=your.name@txdot.gov
```

**There is no password.** TxDOT uses Microsoft SSO; a browser window opens instead. Two things
that will cost you an afternoon if you get them wrong:

- Do **not** append `.privatelink` to the account. This account has no PrivateLink endpoint and
  the suffix gives a host-resolution error that points at nothing.
- `.env` is git-ignored and must stay that way. **The repository is public.**

Optional but recommended — otherwise you get two browser prompts per run:

```powershell
pip install "snowflake-connector-python[secure-local-storage]"
```

---

## Step 4 — Verify the data source (5 min, and never skip it)

```powershell
./venv/Scripts/python.exe test_snowflake_connection.py
```

Five checks: credentials → query → columns → **panel shape** → **value ranges**. The last two
exist because both of their failure modes are silent and both have already happened on this
project. Details in step 6.

> ✅ **You should see:** `PANEL CONFIRMED` with roughly **12 inspections per bridge**, then
> `All checked columns are in the units the model expects`, and finally
> `You're ready to set data.source: snowflake`.

If panel shape reports ~**1.0 rows per bridge**, the query is pointed at a current-values table.
Stop. Do not train on it — see step 6.

---

## Step 5 — Add the data files

Copy from the shared-drive folder into your clone:

```
data/new_SQL_Querry_for_BRG_P_M.csv     <- enrichment extract
models/                                  <- trained models
```

`data/` and `models/` are git-ignored, so these never get committed.

**Why the extract matters:** it supplies span material, deck type, load posting and six other
attributes. Without it the pipeline prints **one line** — `extract file not found ... using panel
features only` — and carries on training a materially different model. It does not error. Check
for that line.

---

## Step 6 — Build something real

```powershell
# One bridge
./venv/Scripts/python.exe predict_future.py --nbi 120200152401017 --year 2040

# A district's interactive HTML report
./venv/Scripts/python.exe forecast_ui.py --district 12

# All 25 districts + an index page, zipped and ready to send
./venv/Scripts/python.exe build_district_reports.py --zip
```

The last one takes about 3 minutes and one SSO prompt. Add `--on-system-only` for the
state-maintained subset — the scope that matters for budget questions. The two scopes write to
different folders and archive names deliberately, so they can't overwrite each other.

---

## The three traps

All three fail **silently**. An absence of errors here does not mean success.

**1. `CORE_SNBI_DATA` is a current-values table, not inspection history.** One row per bridge.
The model learns from inspection-to-inspection transitions, so selecting rows from it trains on
zero transitions and raises nothing at all — you get a model that runs and is worthless. It is
used only as a semi-join to decide *which bridges qualify*. Step 4's panel check exists for this;
run it before repointing any query at any table.

**2. A missing enrichment extract degrades to panel-only features.** One line of output, no error,
materially different model. See step 5.

**3. Inspection dates arrive in three shapes** — packed `MMDDYYYY` integer, native datetime, ISO
string. `parse_inspection_date` dispatches per row. A bad parse becomes `NaT`, and `NaT` rows are
*dropped*, not raised — so a wrong assumption shows up only as a mysteriously empty training set.

---

## What to read next

| | |
|---|---|
| `HANDOFF.md` | The real document. What it is, results and where each number comes from, known limits, open items. |
| `README.md` | Commands, the report packs, the two model types. |
| `config.yaml` | Every setting, commented. The Snowflake query lives here. |
| `sme_review/SME_Review_Packet.md` | What we're asking a bridge engineer to check. |

## Known-open, so you're not surprised

- **Possible target leakage** via `load_posting_status` — bridges get posted *because* they
  deteriorated, and it's back-filled across each bridge's whole history. It's the #2 feature for
  deck and substructure. The ablation that would settle it has not been run. `HANDOFF.md` §9.
- **No temporal holdout.** The split is by bridge, not by period, so the accuracy figures answer
  "generalizes to an unseen bridge," not "generalizes to a future year."
- **Validation exhibits in `sme_review/` predate the highway-only filter**, which drops ~21% of
  bridges. Regenerate before quoting externally.
- **Not yet reviewed by a bridge engineer.**
