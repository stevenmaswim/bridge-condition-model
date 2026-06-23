import os
import pandas as pd
import numpy as np


def predict_all_targets(df, feature_cols, trained_models):
    predictions = df.copy()
    for target, model in trained_models.items():
        X = df[feature_cols].fillna(-999)
        pred_col = f"{target}_predicted"
        predictions[pred_col] = model.predict(X)
        predictions[pred_col] = predictions[pred_col].clip(0, 9).round(1)
    return predictions


def summarize_by_group(predictions, config, trained_models):
    group_cols = [
        config["grouping"]["district_col"],
        config["grouping"]["climate_zone_col"],
    ]
    available_groups = [c for c in group_cols if c in predictions.columns]
    if not available_groups:
        print("Warning: no grouping columns found, skipping grouped summary")
        return pd.DataFrame()

    pred_cols = [f"{t}_predicted" for t in trained_models.keys()]
    actual_cols = [t for t in trained_models.keys() if t in predictions.columns]
    agg_cols = pred_cols + actual_cols

    summary = predictions.groupby(available_groups)[agg_cols].agg(["mean", "median", "count"]).reset_index()
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    return summary


def save_predictions(predictions, summary, metrics_df, fi_df, config):
    out_dir = config["data"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    pred_path = config["output"]["predictions_file"]
    predictions.to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")

    metrics_path = config["output"]["metrics_file"]
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")

    fi_path = config["output"]["feature_importance_file"]
    fi_df.to_csv(fi_path, index=False)
    print(f"Saved feature importance to {fi_path}")

    if not summary.empty:
        summary_path = os.path.join(out_dir, "grouped_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"Saved grouped summary to {summary_path}")
