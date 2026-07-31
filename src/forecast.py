import numpy as np
import pandas as pd

from src.data_loader import load_config, load_raw_data, rename_raw_columns, clean_data
from src.features import prepare_model_data
from src.model import compute_linear_impute_values, load_model
from src.deterioration import forecast_bridge, load_deterioration_model


def find_bridge_row(df, id_col, nbi_code):
    """Looks up a single bridge by its NBI/bridge code. String-compares after stripping
    whitespace so formatting quirks (e.g. accidental padding) don't cause a false miss."""

    matches = df[df[id_col].astype(str).str.strip() == str(nbi_code).strip()]
    if matches.empty:
        return None
    return matches.iloc[0]


def build_future_row(row, target_year):
    """Projects a bridge's age-dependent engineered features forward to target_year, as if
    that year were "now" -- everything else (district, materials, traffic, etc.) is held
    constant. This is a deterioration-curve read of the model, not a true per-bridge
    time series -- there's no historical data to project from, only the model's learned
    age-vs-condition relationship."""

    row = row.copy()

    year_built = row.get("year_built")
    if year_built is None or pd.isna(year_built):
        return row, False  # can't project without a year_built on record

    # clip to [0, 130] to match the training-time cap (features.engineer_features); an unbounded
    # far-future age would be out-of-domain and just saturate the trees.
    row["bridge_age"] = min(max(target_year - year_built, 0), 130)

    if "years_since_reconstruction" in row.index:
        year_reconstructed = row.get("year_reconstructed")
        if year_reconstructed is not None and not pd.isna(year_reconstructed) and year_reconstructed > 0:
            row["years_since_reconstruction"] = min(max(target_year - year_reconstructed, 0), 130)
        else:
            row["years_since_reconstruction"] = row["bridge_age"]

    return row, True


def build_model_input(future_row, feature_cols, model_type, linear_impute_values=None):
    """Builds a one-row model-ready DataFrame from a bridge's (possibly object-dtype) Series.

    A row pulled from a DataFrame via .iloc[] gets a single dtype for the whole Series --
    "object" here, since the source df mixes int/float/category-code columns across its
    columns. Casting to float BEFORE fillna/transpose is required, or XGBoost/sklearn will
    reject every column as non-numeric even though the underlying values are genuinely
    numeric."""

    x_series = future_row[feature_cols].astype(float)
    if model_type == "linear_regression":
        x_series = x_series.fillna(linear_impute_values or {})
    # else (xgboost): keep NaN -- native missing-value handling, matches training
    return x_series.to_frame().T


def _load_deterioration_models(config):
    models = {}
    for target in config["targets"]:
        try:
            models[target] = load_deterioration_model(target, config)
        except FileNotFoundError:
            continue
    return models


def _attributes_only_forecast(df_feat, feature_cols, config, id_col, nbi_code, target_year):
    """Existing attributes-only projection, factored out so predict_future_all can reuse the
    single data load. Returns {target: {model_type: pred}} or None."""
    row = find_bridge_row(df_feat, id_col, nbi_code)
    if row is None:
        return None
    future_row, ok = build_future_row(row, target_year)
    if not ok:
        return None
    linear_impute_values = compute_linear_impute_values(df_feat, feature_cols)
    results = {}
    for target in config["targets"]:
        results[target] = {}
        for model_type in config["model"].get("compare_types", ["xgboost", "linear_regression"]):
            try:
                model = load_model(target, config, model_type=model_type)
            except FileNotFoundError:
                continue
            X = build_model_input(future_row, feature_cols, model_type, linear_impute_values)
            results[target][model_type] = round(float(np.clip(model.predict(X)[0], 0, 9)), 1)
    return results


def predict_future_all(nbi_code, target_year, config_path="config.yaml"):
    """Forecast a bridge with BOTH models from a single data load:
      * deterioration (hybrid) -- primary, uses the bridge's own inspection history;
      * attributes-only        -- fallback for bridges with no usable history.
    """
    config = load_config(config_path)
    id_col = config.get("data", {}).get("id_col")
    if not id_col:
        raise ValueError("config.yaml has no data.id_col configured -- cannot look up a bridge by ID")

    df_raw = rename_raw_columns(load_raw_data(config), config)
    df_clean = clean_data(df_raw, config)

    det_models = _load_deterioration_models(config)
    det = forecast_bridge(df_clean.copy(), config, det_models, nbi_code, target_year) if det_models else None

    df_feat, feature_cols = prepare_model_data(df_clean.copy(), config)
    attr = _attributes_only_forecast(df_feat, feature_cols, config, id_col, nbi_code, target_year)

    _print_forecast(nbi_code, target_year, det, attr)
    return {"deterioration": det, "attributes_only": attr}


def _print_forecast(nbi_code, target_year, det, attr):
    print(f"\n=== Forecast for bridge {nbi_code} in {target_year} ===")
    if det:
        print("\nDeterioration model (recommended -- uses this bridge's inspection history):")
        print(f"  {'target':<28} {'likely':>6}  {'plan-for':>8}  {'risk':>6}   detail")
        for target, r in det.items():
            note = {"deterioration": "model", "carry_forward": "near-term: last rating held",
                    "current": "target year <= last inspection"}.get(r["method"], r["method"])
            cons = r.get("prediction_conservative", r["prediction"])
            risk = r.get("risk_poor_pct")
            risk_s = "-" if risk is None else f"{risk:.0f}%"
            print(f"  {target:<28} {r['prediction']:>6}  {cons:>8}  {risk_s:>6}   "
                  f"(from {r['current_rating']:.0f} at {r['last_inspection_year']}, "
                  f"+{r['horizon_years']:.0f}y; {note})")
        print("  'plan-for' = conservative forecast; 'risk' = chance of reaching poor condition (<=5).")
    else:
        print("\n(no deterioration models found -- train them first via the pipeline)")
    if attr:
        print("\nAttributes-only model (fallback for bridges without history):")
        for target, preds in attr.items():
            if preds:
                print(f"  {target:<28} " + ", ".join(f"{mt}={v}" for mt, v in preds.items()))
    if not det and not attr:
        print(f"  No bridge found with id '{nbi_code}', or no usable data to project.")
