# Bridge Condition Forecasting Model

Forecasts TxDOT bridge condition ratings (deck, superstructure, substructure, culvert) from
historical NBI inspection data. Built to support **maintenance/capital budget planning** — flagging
which bridges are trending toward poor condition — as a decision-support aid for engineers, not an
automatic funding decision.

## Two models

1. **Deterioration model** (primary, `src/deterioration.py`) — predicts a bridge's **future** rating
   from its current rating, how many years ahead, its age, and its physical attributes. Trained on
   real inspection-to-inspection transitions. Each bundle holds three models: the point forecast, a
   **conservative (25th-pctile) "plan-for-this"** forecast for budgeting, and a calibrated **risk
   model** giving `P(rating ≤ 5)` — a probability you can risk-weight a budget on, which ranks
   priorities better than the point forecast (AUC 0.93; top-1% precision 86% vs. 81%). A **hybrid
   rule** carries the last rating forward for the near term (≤ ~3 yr) and uses the model long-term.
2. **Attributes-only model** (fallback, `src/model.py`) — predicts a rating from attributes alone,
   for bridges with no usable inspection history.

Evaluation is **leak-free** (split by `bridge_id`, so no bridge is in both train and test). On
held-out bridges the deterioration model lands within ±1 rating for ~90–96% of bridges at 10–20 yr,
beats "assume no change," and beats a material-stratified Markov deterioration-curve model by
~13–17% at 20 yr.

> **New to this project?** Read [START_HERE.md](START_HERE.md) first — an ordered first-day
> setup with a checkpoint after every step. Then [HANDOFF.md](HANDOFF.md) for what the model is,
> what the numbers mean, and what is still open.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Run everything with the venv interpreter, e.g. `./venv/Scripts/python.exe main.py`.

## Data files (required)

- `data/raw/REAL_ML_LEARNING_DATA.csv` — the inspection panel (raw AssetWise/SNBI export; path set by
  `config.yaml: data.raw_file`). This is a **time-series panel** (~85k bridges × ~20 annual snapshots),
  not one row per bridge.
- `data/new_SQL_Querry_for_BRG_P_M.csv` — the current-values extract of static physical attributes
  (span material, deck type, spans, scour, …), joined on by `src/enrichment.py`. Produced by
  `sql/bridge_data_extract.sql`. Optional — the pipeline degrades to core features if it is absent.

## Train

```powershell
python main.py                 # trains BOTH models on real data, saves to models/ and models/deterioration/
python main.py --dry-run       # synthetic data smoke test (skips the deterioration model)
python main.py --config=other.yaml
```
Outputs land in `data/outputs/`, models in `models/` (attributes-only) and `models/deterioration/`.

> **Retrain when the data changes.** The attributes-only model's categorical codes are derived from
> the data at train time; if the raw or extract file is updated, retrain (`python main.py`) rather
> than reusing an old model, or the codes can desync. The deterioration model persists its encoders,
> so it is not affected.

## Forecast tools (after training)

```powershell
# Forecast one bridge to a future year (most-likely + conservative "plan-for", plus the fallback)
python predict_future.py --nbi 120200152401017 --year 2040

# Build a budget watch-list: bridges rated 5-7 forecast to reach poor condition, worst-first
python build_watchlist.py --district 12

# Generate an interactive HTML page (search codes, horizon slider, deterioration-curve charts)
python forecast_ui.py --codes 120200152401017,121700017714043     # or --file codes.txt / --district 12
```

## District report packs (what you send someone)

`build_district_reports.py` builds one self-contained HTML report per district plus an index
page linking them, and optionally zips the lot. It pulls the source panel **once** and reuses it
for all 25 districts — with Snowflake's `externalbrowser` auth, a naive loop would prompt you to
log in 25 times.

```powershell
# Overview -- every highway bridge, state-maintained and locally owned
./venv/Scripts/python.exe build_district_reports.py --zip
#   -> reports/  +  TxDOT_Bridge_Condition_Forecast_<date>.zip

# Budget scope -- state-maintained (on-system) bridges only, what TxDOT funds
./venv/Scripts/python.exe build_district_reports.py --on-system-only --zip
#   -> reports_on_system/  +  TxDOT_Bridge_Condition_Forecast_on-system_<date>.zip

# A few districts, e.g. to check something quickly
./venv/Scripts/python.exe build_district_reports.py --districts 12,15,18
```

The two scopes write to **different folders and different archive names on purpose** — the output
folder is cleared on entry, so sharing a path would mean the second build silently destroyed the
first. Roughly 3 minutes and one SSO prompt for a full statewide run; about 52 MB of HTML that
deflates to under 4 MB.

Which to send: the overview for "here is the inventory," the on-system build for anything that
feeds a funding decision — `config.yaml`'s own watch-list defaults to `on_system_only: true`
for that reason.

**Before trusting a run against a new source**, check it:

```powershell
./venv/Scripts/python.exe test_snowflake_connection.py          # one district, seconds
```

Five steps: credentials, query, columns, **panel shape**, **value ranges**. Steps 4 and 5 exist
because both failures are silent — a current-values table trains a model with no transitions to
learn from and raises nothing, and a source whose coordinates are packed integers or whose load
ratings are ×100 just falls past every learned split with the feature quietly contributing
nothing. Both have already happened here.

## Project structure

- `src/data_loader.py` — load raw data, rename columns, clean (numeric coercion, keep id + inspection date)
- `src/features.py` — feature engineering (inspection-year age, sanity-bounded year_built), enrichment
  join + encoding, feature-column selection
- `src/enrichment.py` — join the static SNBI physical attributes onto the panel (cached read)
- `src/model.py` — attributes-only XGBoost + linear baseline, group split, native-NaN handling
- `src/deterioration.py` — deterioration model: inspection events, forward pairs, conservative model,
  hybrid forecast, watch-list
- `src/forecast.py` — single-bridge forecasting orchestration (both models)
- `src/predict.py` — batch predictions + grouped summaries for the attributes-only model
- `src/baselines.py` — reference baselines (persistence, age-curve) and shared metric definitions
- `main.py` — end-to-end training pipeline · `config.yaml` — all settings
- `sme_review/` — SME review packet (visual PDF + exhibits) · `Bridge_Model_Status_Report_v2.md` — supervisor report

## Notes / current limits

- Grouping by **climate zone** is not yet active (the field is NULL in the extract, pending a
  county/district → zone lookup); summaries fall back to district only.
- The model cannot foresee sudden failures (scour, impact) — those are events, not trends; keep
  inspection as the safety net. It is weakest on already-poor bridges (ratings 0–4).
- The newest physical fields cover ~74% of bridges and skew toward newer ones; older on-system
  bridges fall back to core features.
