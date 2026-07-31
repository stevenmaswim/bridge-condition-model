"""Reference baselines for the bridge condition model.

A model's metrics only mean something next to a baseline: "R2 = 0.65" is impressive
or embarrassing depending on how a trivial rule does on the same data. Two baselines
matter for condition ratings:

  * persistence  -- predict each inspection's rating = the SAME bridge's PREVIOUS
                    inspection rating. Condition ratings change slowly, so this is a
                    very strong, very honest bar. If the ML model cannot beat "assume
                    it's unchanged since last time," the model is not earning its keep.
  * age curve    -- predict rating from bridge age alone (mean rating per age). Captures
                    the single dominant trend (older -> worse) with no other features.

These are deliberately simple and make no train/test split of their own except where a
fit is required (age curve); persistence needs no fitting.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred, clip=(0, 9)):
    """MAE / RMSE / R2 / within-N, matching the model's metric definitions and its
    deployed clip-to-[0,9] behavior so baseline and model numbers are comparable."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), clip[0], clip[1])
    abs_err = np.abs(y_true - y_pred)
    return {
        "n": len(y_true),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan"),
        "within_1": (abs_err <= 1.0).mean() * 100,
    }


def persistence_baseline(panel, id_col, date_col, target):
    """Predict each inspection's rating from the same bridge's previous inspection.

    Evaluated over every inspection that HAS a prior inspection for that bridge (the set
    of real year-to-year transitions in the panel). Returns (metrics, detail_df)."""
    cols = [id_col, date_col, target]
    d = panel[cols].copy()
    d[target] = pd.to_numeric(d[target], errors="coerce")
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[id_col, date_col, target])
    d = d.sort_values([id_col, date_col])
    d["prev_rating"] = d.groupby(id_col)[target].shift(1)
    d = d.dropna(subset=["prev_rating"])
    if d.empty:
        return None, d
    return compute_metrics(d[target], d["prev_rating"]), d


def age_curve_baseline(panel, age_col, target, test_mask=None, random_state=42, test_frac=0.3):
    """Predict rating from age alone: fit mean rating per integer age on the train rows,
    predict that mean on the test rows. If test_mask is not supplied, a random row split
    is used purely to give the baseline an honest held-out score. Returns (metrics, detail)."""
    d = panel[[age_col, target]].copy()
    d[age_col] = pd.to_numeric(d[age_col], errors="coerce")
    d[target] = pd.to_numeric(d[target], errors="coerce")
    d = d.dropna(subset=[age_col, target])
    if len(d) < 10:
        return None, d
    d["age_bin"] = d[age_col].round().astype(int)

    if test_mask is None:
        rng = np.random.default_rng(random_state)
        is_test = rng.random(len(d)) < test_frac
    else:
        is_test = test_mask.loc[d.index].to_numpy()

    train, test = d[~is_test], d[is_test]
    age_means = train.groupby("age_bin")[target].mean()
    global_mean = train[target].mean()
    pred = test["age_bin"].map(age_means).fillna(global_mean)
    return compute_metrics(test[target], pred), test
