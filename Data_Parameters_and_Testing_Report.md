# Data, Parameters & Testing — Defensible Reference

**Purpose.** A complete, line-by-line accounting of (1) exactly what data the model uses, (2) which
fields were kept versus filtered out of the ~143-field export and *why*, (3) the parameters and their
values, and (4) how the data was tested. Written for a technical meeting so every claim is verifiable
against the actual files.

---

## 1. What data am I using? (three files, be precise)

There are **three** files in play. Being exact about this is the first thing to get right in the room.

| # | File | Rows / Cols | Role |
|---|---|---|---|
| A | `TxDOT_Bridges_SNBI_…csv` | ~current bridges / **143 cols** | The **full SNBI export** — the universe of available fields (this is the "~120"). |
| B | `REAL_ML_LEARNING_DATA.csv` | **1,712,921** rows / **65 cols** | The **training data**: a historical inspection *panel* (1992–2025). Both models train on this. |
| C | `new_SQL_Querry_for_BRG_P_M.csv` | 63,242 bridges / **41 cols** | The **enrichment extract**: current-value physical attributes joined onto B by bridge ID. |

**Key point for the meeting:** the model does **not** train on the 143-column export directly. It
trains on the 65-column historical panel (B) and joins 11 static attributes from the current-value
extract (C). The 143-column file is where the field selection *came from*.

### Data lineage (how 143 becomes the model's inputs)

```
143 fields  (full SNBI export, file A)
   │  historical export kept a working subset
   ▼
 65 columns (training panel, file B)  ──────────────►  41 DROPPED (see §3)
   │                                                    4 TARGETS + 1 ID + 1 inspection date
   ▼                                                   18 features KEPT
 + 11 static physical features (from extract C)
   ▼
~31 model features  (+ derived: bridge_age, deck_area, traffic_density,
                      + deterioration-only: current rating r0, horizon, age_t0)
```

---

## 2. The prediction targets (what the model predicts)

Four NBI **condition ratings**, each 0–9 (9 = excellent, 0 = failed; ≤5 is "poor"):

| Target column (file B) | Model name | Fill % | Distinct values |
|---|---|---|---|
| `B.C.01: DECK CONDITION RATING` | deck_cond_rating | 99.8 | 0–9 + "N" |
| `B.C.02: SUPER CONDITION RATING` | superstructure_cond_rating | 99.8 | 0–9 + "N" |
| `B.C.03: SUB CONDITION RATING` | substructure_cond_rating | 99.8 | 0–9 + "N" |
| `B.C.04: CULVERT CONDITION RATING` | culvert_cond_rating | 99.8 | 0–9 + "N" |

`"N"` means the member does not exist on that structure (e.g. a culvert has no deck); it is coerced to
missing so those members are not predicted. A fifth rating, **channel condition** (`B.C.09`), is present
but currently unused — a candidate future target.

---

## 3. Line-by-line: every column in the 65-column training panel

Disposition counts: **4 targets · 1 ID · 1 inspection date · 13 numeric features · 5 categorical
features · 41 dropped.** Fill % / cardinality measured on a 250k-row sample.

### 3a. Kept — features (18) + targets/keys (6)

| Raw column | Used as | Fill % | Card. |
|---|---|---|---|
| `B.ID.01: BRG NUM` | ID (lookup key) | 100 | — |
| `B.IE.02: INSPEC BEGIN DATE (RT)` | inspection date (bookkeeping) | 99.9 | — |
| `B.C.01/02/03/04 …` | 4 targets | 99.8 | — |
| `B.L.04: HWY AGENCY DIST` → txdot_district | categorical feature | 100 | 26 |
| `B.CL.01: OWNER` → owner | categorical feature | 99.9 | 22 |
| `B.CL.02: MAINT RESP` → maintenance_resp | categorical feature | 99.9 | 21 |
| `B.H.01: FUNC CLASSIFICATION (1)` → functional_class | categorical feature | 99.9 | 7 |
| `B.LR.01: DES LOAD` → design_load | categorical feature | 99.8 | 12 |
| `B.W.01: YEAR BUILT` → year_built | numeric feature | 99.9 | 132 |
| `B.H.09: AADT` → adt | numeric feature | 99.9 | 22,000 |
| `B.H.11: YR OF AADT` → adt_year | numeric feature | 99.9 | 55 |
| `B.G.02: TOTAL BRIDGE LENGTH` → structure_length | numeric feature | 99.9 | 3,099 |
| `B.G.03: MAX SPAN LENGTH` → max_span_length | numeric feature | 99.9 | 710 |
| `B.G.05: BRG WIDTH OUT-TO-OUT` → deck_width | numeric feature | 100 | 1,881 |
| `B.G.06: BRIDGE WIDTH CURB-TO-CURB` → roadway_width | numeric feature | 100 | 1,399 |
| `B.G.09: APPROACH RDWY WIDTH` → approach_roadway_width | numeric feature | 99.8 | 355 |
| `B.G.11: SKEW` → skew_angle | numeric feature | 99.9 | 89 |
| `B.L.05: LATITUDE` → latitude | numeric feature | 78.9 | 93,363 |
| `B.L.06: LONGITUDE` → longitude | numeric feature | 78.9 | 93,414 |
| `B.LR.05: INVNTRY LOAD RTG FACTOR` → inventory_load_rating_factor | numeric feature | 99.9 | 78 |
| `B.LR.06: OPR LOAD RTG FACTOR` → operating_load_rating_factor | numeric feature | 99.9 | 97 |

### 3b. Dropped (41) — grouped by reason

**Near-empty (<5% fill) — no usable signal:**
`B.IE.05 INSPEC INTERVAL (NSTM/UW)` (1.3–1.5%), `B.IE.02 INSPEC BEGIN DATE (NSTM/UW)` (1.3–1.5%),
`B.L.07/08/09 BORDER BRIDGE …` (0.1%), `B.N.06 SUBSTR NAV PROTECTION` (2%),
`B.N.03 MOVABLE BRG NAV CLRNCE` (0.5%). *(These only populate for specialized inspections / border /
movable bridges.)*

**High-cardinality free text — noise when encoded as integers:**
`FACILITY CARRIED` (36,653 distinct), `FEATURE CROSSED` (30,160), `B.L.11 BRG LOCATION` (87,263),
`B.RT.02 ROUTE NUMBER` (4,850), `B.L.03 PLACE CODE` (2,177), `B.H.16 HWY MAX USABLE SURFACE WIDTH`
(1,383), `88 SPEC FLAGS` (668).

**Redundant with a kept feature:**
`B.L.02 COUNTY CODE` (redundant with district), `22 1 MAINT SECT`, `B.G.07/08 CURB/SIDEWALK WIDTH`
(covered by deck/roadway width), `B.H.08 LANES ON HWY` (covered by width + traffic).

**Administrative / low predictive value for deterioration:**
`B.CL.05 TOLL`, `B.G.10 BRG MEDIAN`, `B.H.12/13 HWY VERT/USABLE CLRNCE`, `B.N.02/05 NAV CLEARANCES`,
`B.RT.05 SERVICE TYPE`, `B.H.03 NHS` / `B.H.05 STRAHNET DESIGNATION`, `41 2 LOAD 1000lb`,
`B.IE.05 INSPEC INTERVAL (RT)`, `B.IR.01/03 NSTM/UW INSP REQD`, `B.LR.04 LOAD RTG METHOD` (60% fill).

**Superseded / handled elsewhere:**
`DATE TIMESTAMP` (annual submission date — superseded by the real inspection date `B.IE.02`),
`B.PS.01 LOAD POSTING` (dropped here, **re-added via the enrichment extract** as a top-3 feature).

**Present but not yet used — candidate future features/targets:**
`B.C.09 CHANNEL CONDITION RATING` (a 5th condition rating), `B.AP.01 APPROACH RDWY ALIGNMENT`
(appraisal), `B.N.01 NAVIGABLE WATERWAY` (water-exposure flag), `ON/OFF` (on/off-system).

> **Defensible one-liner:** *"Of 65 columns, 24 carry the signal we model — four ratings, an ID, the
> inspection date, and 18 predictive attributes. The other 41 are either near-empty, near-unique free
> text, redundant with a kept field, purely administrative, or handled elsewhere. Nothing predictive was
> discarded without a reason, and the borderline candidates (channel rating, waterway, on/off-system)
> are documented as next additions."*

---

## 4. Enrichment extract (file C): 41 columns → 11 physical features added

The current-value extract exists to add the **strong physical predictors** the historical panel lacks.
Of its 41 columns, **11 static attributes** are joined onto every bridge (static, so the current value
back-fills history). The rest are IDs or duplicates of panel fields.

**Used (11):** `structure_kind` (span material), `structure_type` (span type), `span_continuity`,
`deck_type`, `wearing_surface`, `deck_protection`, `num_spans_main`, `num_beam_lines`, `adt_truck`
(truck traffic), `scour_vulnerability`, `load_posting_status`.

**Not used:** the extract also carries the ratings/geometry/traffic already present in the panel (avoids
duplication), plus `climate_zone`, `year_reconstructed`, `num_spans_approach`, `membrane_type` — which
are **NULL / not yet populated** (documented data gaps).

---

## 5. The final model feature set

**~31 features** reach the model:

- **13 numeric** (panel): year_built, adt, adt_year, structure_length, max_span_length, deck_width,
  roadway_width, approach_roadway_width, skew_angle, latitude, longitude, inventory_ & operating_load_rating_factor.
- **5 categorical** (panel): txdot_district, owner, maintenance_resp, functional_class, design_load.
- **11 static physical** (enrichment): the list in §4.
- **3 engineered** (`features.py`): bridge_age (= inspection year − year_built, bounded), deck_area,
  traffic_density.
- **Deterioration model only, +3:** current rating `r0`, `horizon` (years ahead), `age_t0`.

Feature-importance (deck) confirms sensible drivers: current rating dominates (~39%), then load posting
(~8%), horizon (~4%), then material, wearing surface, load rating, and age (~3% each).

---

## 6. Parameters — values and why

All in `config.yaml`. Production (deterioration) values:

| Parameter | Value | What it does | Why this value |
|---|---|---|---|
| n_estimators | 300 | boosting trees | enough for sticky ratings, not over-grown |
| max_depth | 6 | interaction depth | captures age×material without memorizing |
| learning_rate | 0.1 | shrinkage per tree | standard, stable with 300 trees |
| subsample | 0.8 | rows per tree | regularization on 1M+ rows |
| colsample_bytree | 0.8 | features per tree | decorrelates trees |
| min_child_weight | 3 | min evidence/leaf | avoids leaves on rare-rating rows |
| reg_alpha / reg_lambda | 0.1 / 1.0 | L1 / L2 penalty | mild regularization on the tail |
| random_state | 42 | seed | reproducible split + models |
| **test_size** | **0.30** | held-out fraction | ~360k-pair test, stable metrics |
| **split.method** | **group** | split by bridge | **the key correctness setting (no leakage)** |
| conservative_quantile | 0.25 | "plan-for" quantile | best catch/false-alarm trade for budgeting |
| risk_threshold | 5.0 | "poor" cutoff | NBI ≤5 is the poor/act line |
| hybrid_threshold_years | 3.0 | carry-forward vs model | below it, carry-forward is unbeatable |
| max_horizon / pair_cap | 25 / 1.2M | training bounds | keeps horizon distribution + runtime sane |

The attributes-only model additionally uses `RandomizedSearchCV` (20 iterations, 3-fold CV, scored on
negative MAE) over depth/learning-rate/regularization; the deterioration model uses the fixed values
above (point accuracy is at its ceiling, so tuning yields little — see §7).

---

## 7. How was the data tested? (validation methodology)

Every headline number is measured on **bridges the model never trained on.** The test design is the
most important thing to be able to defend:

1. **Leak-free split (the core test).** `GroupShuffleSplit` on `bridge_id` — no bridge appears in both
   train and test. This is what makes the metrics honest; a plain random split inflated the original
   numbers by letting the model memorize individual bridges.
2. **By-horizon evaluation.** Accuracy is reported separately at ~2 / 5 / 10 / 20 years, because it
   legitimately falls the further ahead we forecast.
3. **Reference baselines.** Every result is shown against *carry-forward* (assume no change) and an
   *age-curve*, plus a *persistence* bar — so "good" is relative to what a simple rule achieves.
4. **Calibration & bias check.** Confirmed the model is unbiased (mean error within ±0.01 at every
   horizon) and well-calibrated for common ratings; documented its weakness on rare poor bridges.
5. **Benchmark vs. the standard method.** Beats a fair, material-stratified Markov deterioration-curve
   model by ~13–17% at 20 years.
6. **Ranking test for budgeting (precision@K).** Of the top 1% the risk model flags, 86% truly become
   poor (vs 45% naive) — the deployment-relevant metric.
7. **Adversarial / negative experiments.** Tested two plausible accuracy boosters (recent-decline rate,
   other components' ratings); both moved MAE ~0, confirming the point-accuracy ceiling rather than
   assuming it.
8. **Unit tests.** 56 automated tests cover cleaning, encoding, date parsing, event/pair building, the
   split, and the serve path.

### Headline results (leak-free)

| Component | 10-yr within ±1 | 20-yr within ±1 | 20-yr MAE (model vs carry-forward) |
|---|---|---|---|
| Deck | 92.7% | 92.4% | 0.44 vs 0.71 |
| Superstructure | 92.7% | 92.0% | 0.46 vs 0.71 |
| Substructure | 90.6% | 90.9% | 0.47 vs 0.75 |
| Culvert | 95.9% | 95.5% | 0.44 vs 0.54 |

---

## 8. Likely supervisor questions — prepared answers

- **"You started with 143 fields — why only ~31?"** → Because 41 of the 65 working columns are
  near-empty, near-unique free text, redundant, or administrative; nothing predictive was dropped
  silently. The strong physical predictors were *added back* via the enrichment extract. (§3–§5)
- **"How do I know the accuracy is real and not overfit?"** → It's measured on bridges held out by
  a by-bridge split, broken out by horizon, and it beats both the naive baseline and the industry
  Markov method. (§7)
- **"Can it be more accurate?"** → Point accuracy is at the inspector label-noise ceiling (two feature
  experiments confirmed no lift). Further gains require *data* — reconstruction year, element-level
  condition, and filling the older-bridge coverage gap — not more tuning. (§6–§7)
- **"What should we actually use it for?"** → Conservative-forecast + risk-ranked screening of which
  fair-to-good bridges are trending toward poor, with an engineer confirming any bridge before funding.
- **"What's missing / next?"** → Reconstruction year, channel-rating and on/off-system as features,
  climate zone, and broader enrichment coverage for pre-1970 on-system bridges.

---

## 9. On-system bridges (the agency's population)

The agency is responsible for **on-system** (state-maintained) bridges; off-system bridges are locally
owned. The data contains both, and the tool is scoped accordingly.

| System | Bridges | Share |
|---|---|---|
| On-system (state-maintained) | 58,449 | 68.3% |
| Off-system (locally owned) | 27,127 | 31.7% |
| Reclassified over time (mixed) | 1,482 | 1.7% |

**The model is more accurate on on-system bridges** (better maintained, better documented). On held-out
on-system bridges:

| Component | On-system within ±1 | MAE | R² |
|---|---|---|---|
| Deck | 95.1% | 0.35 | 0.52 |
| Superstructure | 94.7% | 0.37 | 0.57 |
| Substructure | 94.4% | 0.37 | 0.52 |
| Culvert | 95.9% | 0.37 | 0.39 |

Deck holds at **94.8% within ±1 at 10 years, 94.2% at 20 years** for on-system specifically.

**Do we retrain on on-system only? No.** Trained on all data vs. on-system-only, evaluated on the *same*
on-system test bridges, the difference is negligible (+0.2 pts). Off-system data adds deterioration
signal without biasing on-system predictions, so we keep the all-data model and **filter the outputs.**

> **Now the default.** `build_watchlist.py` defaults to on-system only (`--all-system` overrides);
> statewide it flags ~1,322 on-system deck bridges. **Meeting headline:** *on the ~58,000 on-system
> bridges TxDOT is responsible for, the model forecasts deck condition within one rating point for ~95%
> of bridges, out to 10–20 years.*

---

## 10. Plain-English guide — every term explained

No math background assumed. If a word above was unfamiliar, it is defined here.

**The basic idea**
- **Machine-learning model.** A computer program that learns patterns from many past examples instead of
  being given fixed rules. Like a seasoned inspector who has seen thousands of bridges and can eyeball how
  one will age — except it is a program that found those patterns in the data.
- **Training / training data.** The past examples we show it — here, ~1.7M historical inspections. The
  textbook and worked problems a student studies before an exam.
- **Feature (input / predictor).** One piece of information about a bridge used to make the estimate: age,
  material, traffic, current rating. The clues a detective works from.
- **Target (label).** The thing we predict — the future condition rating. The answer in the back of the book.
- **Prediction / forecast.** The program's best estimate of the target for a bridge you ask about.

**The engine: how it learns**
- **Decision tree.** A flowchart of yes/no questions ending in an estimate ("Older than 40? → Steel? →
  estimate this rating"). A choose-your-own-adventure that lands on an answer.
- **XGBoost / gradient boosting.** Our method: it builds hundreds of small decision trees, each trained to
  fix the mistakes of the ones before it, then adds them up. A team of specialists who each catch what the
  last one missed. "Boosting" = each round boosts accuracy by focusing on prior errors.
- **Parameters vs. hyperparameters.** Parameters are what it learns on its own (the questions in the trees).
  Hyperparameters are the dials we set beforehand (how many trees, how deep, how fast). Hyperparameters are
  the oven settings; parameters are how the cake turns out. Ours are in §6.
- **Encoding.** Computers do math, not words, so "Concrete"/"Steel" become numbers — like jersey numbers.
- **Missing values.** When a field is blank, the program handles it sensibly instead of guessing a wrong number.

**Making sure it is honest**
- **Train/test split.** Teach on one set of bridges, test on a different set it never saw — study, then take
  an exam with new problems.
- **Group split (by bridge).** Because each bridge appears ~20 times, all of a bridge's records go to study
  OR exam, never both — otherwise it just recognizes the bridge on the exam. The key correctness step.
- **Data leakage.** When it accidentally peeks at the answers (or test bridges) during training, looking
  smarter than it is. Fixing this is why the honest numbers are a little lower — and trustworthy.
- **Overfitting.** Memorizing the examples instead of learning general patterns — aces practice, fails on new
  cases. A student who memorizes the answer key. Guarded against by the §6 settings.
- **Baseline.** A dead-simple rule we must beat to earn our keep: "assume the rating never changes."

**How we measure accuracy**
- **MAE (mean absolute error).** Average size of the miss, in rating points. 0.4 = typically off by < half a point.
- **RMSE.** Like MAE but punishes big misses more; much larger than MAE means a few forecasts were badly off.
- **R² (R-squared).** How much of the real variation the model explains, 0–1. 0 = no better than guessing the
  average; 1 = perfect. Modest (~0.5) here because ratings barely move and inspectors disagree ~±1 (a ceiling).
- **Within ±1.** The intuitive one — how often the forecast is within one rating point. ~95% = almost always right.
- **Calibration.** Do the risk percentages mean what they say? If it says "70% chance of poor," do ~70% of
  those bridges end up poor? Ours does — the percentages are trustworthy.
- **AUC.** 0.5–1.0 score for how well the risk model separates bad from not-bad. 0.5 = coin flip, 1.0 = perfect.
  Ours is 0.93.
- **Precision@K.** Of the top few bridges flagged, how many truly go bad. "Top 1% precision 86%" = of the 1%
  it's most worried about, 86% reach poor condition. The budget-relevant measure.

**The specific pieces**
- **Regression vs. classification.** Regression predicts a number (the rating); classification predicts a
  category / yes-no (will it be poor?). We use both.
- **Quantile / "conservative" forecast.** Instead of the single most-likely rating, a deliberately cautious
  estimate. Our 25th-percentile "plan-for" number = plan for a bit worse than average — safer for budgeting.
- **Risk probability.** The percent chance a bridge reaches poor condition by the target year — lets you rank
  and risk-weight, not just read one rating.
- **Horizon.** How many years ahead we forecast.
- **Deterioration model.** Our main model — predicts how condition declines over time from today.
- **Markov model / deterioration curves.** The older, standard industry method using simple year-to-year
  transition probabilities. We beat it, which justifies the more advanced approach.

---

*All figures verified against the actual files on 2026-07-31. Column dispositions come directly from
`config.yaml` and the data profiles; parameters from `config.yaml`; metrics from held-out (leak-free)
evaluation.*
