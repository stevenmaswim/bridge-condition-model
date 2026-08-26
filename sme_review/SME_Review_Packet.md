# Bridge Condition Model — SME Review Packet
**Purpose:** Ask a bridge engineer to sanity-check the model's behavior before it informs any decisions.
No coding or statistics background needed — the questions below are all engineering-judgment calls.

## What the model does (plain language)

For a given bridge, the model estimates what its condition ratings (deck, superstructure,
substructure, culvert) will be a chosen number of years in the future. It learns from ~85,000 Texas
bridges and their inspection histories (1992–2025). Its main inputs are:

- the bridge's **most recent condition rating**,
- **how many years ahead** we are forecasting,
- the bridge's **age**, **material and structure type**, **deck type**, **traffic**, and **district**.

For the near term (about 3 years or less) the tool simply carries the last rating forward, because
ratings rarely change that fast. Beyond that, it uses the learned deterioration model.

## Exhibits in this folder

1. **sample_forecasts_backtest.csv** — the most important one. Real bridges where we took a *past*
   inspection, had the model forecast ~10 years ahead, and compared its prediction to the rating the
   bridge **actually** received later. Columns: current rating, years ahead, predicted, actual, error.
   *This is a true backtest — the model never saw these bridges in training.*
2. **deterioration_by_material.csv** — average predicted vs. actual decline over ~10 years, grouped by
   span material. Lets you check whether the model declines faster/slower for the right material types.
3. **feature_importance.csv** — which inputs the model leans on most, per rating. A quick check that it
   relies on sensible factors (prior condition, load posting, age, material) rather than noise.
4. **accuracy_by_starting_rating.csv** — honest accuracy broken out by the bridge's starting rating.
   Shows the model is strongest for common ratings (6–8: ~88–96% within ±1) and weaker for rare low
   ratings (0–4), where it tends to over-predict recovery. Fewer than ~600 of ~68,000 test cases start
   below 5.

## Questions we'd like your judgment on

1. **Realism of the trajectories:** In the backtest sample, are the predicted 10-year ratings
   reasonable given the starting rating and bridge type? Any that look clearly wrong?
2. **Deterioration by material:** Does the predicted decline by material (exhibit 2) match your
   experience (e.g., timber vs. prestressed concrete vs. steel)?
3. **Drivers:** In the feature importance (exhibit 3), are the top factors ones you'd expect to drive
   condition? Anything that looks like it shouldn't matter but does, or vice versa?
4. **Repairs/rehab:** The model treats rating *increases* (repairs) as noise rather than predicting
   them. Is that acceptable for planning use, or a gap we should address?
5. **Use boundaries:** For what horizon and purpose would you trust this (e.g., 10-year screening for
   prioritization) and where would you not (e.g., condemning/programming a specific structure)?

## Known limitations (already understood on our side)

- Accuracy is honest but not perfect: ~90–96% of forecasts land within ±1 rating at 10–20 years.
- New physical fields (material/deck type/scour) currently cover ~74% of bridges; the rest use core
  fields only.
- Long horizons (20 yr+) rest on fewer historical examples.
- The model predicts a rating trend, not specific inspection findings or failure events.
- **Weakest on already-poor bridges (rating 0–4):** it tends to over-predict recovery for these rare
  cases, so it should not be relied on to forecast structures already in poor condition.

## What we need back

A short thumbs-up / thumbs-down with any specific bridges or patterns that look off, plus your view on
question 5 (appropriate use). That's enough for us to decide whether to pilot it or refine further.
