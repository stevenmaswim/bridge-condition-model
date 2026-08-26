# Bridge Condition Model — Status Report (v2)
**TxDOT — Prepared for supervisor review**

## Summary

The model now **forecasts future condition** (deck, superstructure, substructure, culvert) rather
than just estimating a rating from a bridge's attributes. On held-out bridges the model never
trained on, it predicts condition **10–20 years out** within ±1 rating for **~91–96%** of bridges,
and cuts average error **~30–40% versus assuming no change** at the 20-year horizon.

This v2 corrects a data issue that made v1's headline look better than it truly was, and reframes
the model around how the data actually behaves. The numbers below are **honest and defensible** —
evaluated so that no bridge appears in both training and testing.

## What changed since v1 (and why numbers moved)

1. **Fixed a hidden data leak.** The 1.7M-row dataset is not 1.7M bridges — it is ~85,000 bridges
   each inspected ~20 times (annual snapshots, 1992–2025). v1 split rows randomly, so the *same
   bridge* appeared in both training and testing; the model partly memorized bridges, inflating the
   score. v2 splits **by bridge**, so accuracy reflects genuinely new bridges.
2. **Reframed to a deterioration model.** Bridges are re-inspected on a ~2-year cycle, and condition
   changes only 16% of the time between inspections. The strongest predictor of a bridge's future
   condition is **its own current condition plus how many years ahead** — which v1 ignored. v2 uses it.
3. **Added physical predictors** from AssetWise SNBI (span material, deck type, span counts, scour),
   now joined onto ~70–74% of bridges.
4. **Corrected engineering bugs** (bridge age was computed from today's date, not the inspection
   date; missing values were mishandled; a numeric field was mistyped as a category).

## Model 1 — Deterioration model (primary)

Predicts a bridge's future rating from its current rating + years ahead + age + physical attributes.
Evaluated by forecast horizon on held-out bridges. "Carry-forward" = assume the rating never changes.

| Target | Horizon | Carry-forward (MAE) | **Model (MAE)** | Model within ±1 |
|---|---|---|---|---|
| Deck | 10 yr | 0.50 | **0.41** | 92.7% |
| Deck | 20 yr | 0.71 | **0.44** | 92.4% |
| Superstructure | 20 yr | 0.71 | **0.46** | 92.0% |
| Substructure | 20 yr | 0.75 | **0.47** | 90.9% |
| Culvert | 20 yr | 0.54 | **0.44** | 95.5% |

**Key finding — use the right tool for the horizon:**
- **Near-term (≤ ~3–5 yr):** nothing beats "carry the last rating forward" (~98% within ±1). Ratings
  barely move, so the deployed tool uses carry-forward here.
- **Long-term (10–20 yr — the capital-planning range):** the deterioration model clearly wins, and
  is what should inform budget/priority decisions.

The forecast tool (`predict_future.py`) applies this **hybrid rule automatically.**

## Model 2 — Attributes-only model (fallback)

For bridges with no usable inspection history. Retrained on the honest by-bridge split with the
bug fixes and new features. Honest held-out performance:

| Target | within ±1 | R² |
|---|---|---|
| Deck | 89.8% | 0.41 |
| Superstructure | 89.2% | 0.51 |
| Substructure | 87.1% | 0.48 |
| Culvert | 93.8% | 0.30 |

*(v1 reported 91–97% within ±1 / R² 0.59–0.70, but on the leaky split. These honest numbers are the
right basis for decisions.)*

## Validation and decision-ready tooling (added 2026-07-28)

An engineering review put the model through several checks and turned the findings into tools:

- **Calibration:** forecasts are **unbiased** (not over-optimistic) and well-calibrated for the
  common ratings (5–8) where nearly all bridges sit.
- **Benchmark vs. standard practice:** the model **beats the industry-standard deterioration-curve
  (Markov) method by ~13–17%** at the 20-year horizon — the added complexity is justified.
- **Conservative forecast for budgeting:** every bridge now gets a *most-likely* rating **and** a
  conservative *plan-for-this* rating. Budgeting off the conservative number catches ~75% of true
  2-point decliners (vs. ~56% for the point forecast) at a ~7% false-alarm cost.
- **Budget watch-list:** `build_watchlist.py` produces a ranked, per-district list of bridges heading
  toward poor condition (e.g., 261 in one district), with the review's usage rules built in — rank by
  *forecast condition*, exclude already-poor bridges (inspect those) and new bridges (normal aging).
- **Confirmed limits:** it cannot foresee sudden failures (those are events, not trends — keep
  inspection as the safety net); it is weakest on already-poor bridges; and the newest physical fields
  skew toward newer bridges, so older on-system bridges need a data backfill.

## Honest caveats

- Long-horizon accuracy is measured on the bridges that actually have long histories; very-long
  horizons (20 yr+) have fewer examples.
- New physical features cover ~70–74% of bridges; the rest fall back to core features. Broadening
  the AssetWise join would add a little accuracy.
- Ratings occasionally jump *up* (repairs/reconstruction); the model treats these as noise rather
  than predicting specific repair events.
- Not yet validated by a bridge engineer (see decision menu).

## Supervisor decision menu

1. **SME review** — have a bridge engineer sanity-check the deterioration curves and a few forecasts.
2. **Scope** — personal tool, shared team tool, or input to budget/priority decisions? Sets the
   hardening bar.
3. **Data completeness** — approve broadening the AssetWise join (reconstruction year, approach
   spans) to lift coverage above ~74%.
4. **Retraining cadence** — manual rerun on each AssetWise export, or a schedule.

*Source: honest by-bridge evaluation on 1,712,921 inspection records / ~85,000 bridges; deterioration
model trained on 575k+ real inspection-to-inspection transitions.*
