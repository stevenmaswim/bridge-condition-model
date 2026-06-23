import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def build_model(config):
    params = config["model"]["params"].copy()
    return XGBRegressor(**params)


def train_single_target(df, feature_cols, target, config):
    subset = df.dropna(subset=[target])
    if len(subset) < 10:
        print(f"  Skipping {target}: only {len(subset)} valid rows")
        return None, None

    X = subset[feature_cols].fillna(-999)
    y = subset[target]

    test_size = config["model"]["test_size"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=config["model"]["params"]["random_state"]
    )

    model = build_model(config)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    metrics = {
        "target": target,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }
    print(f"  {target}: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, R2={metrics['r2']:.3f}")
    return model, metrics


def train_all_targets(df, feature_cols, config):
    targets = config["targets"]
    trained = {}
    all_metrics = []

    for target in targets:
        if target not in df.columns:
            print(f"  Target column {target} not in data, skipping")
            continue
        print(f"Training model for: {target}")
        model, metrics = train_single_target(df, feature_cols, target, config)
        if model is not None:
            trained[target] = model
            all_metrics.append(metrics)

    return trained, pd.DataFrame(all_metrics)


def save_models(trained_models, config):
    model_dir = config["output"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    for target, model in trained_models.items():
        path = os.path.join(model_dir, f"{target}_xgb.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
    print(f"Saved {len(trained_models)} models to {model_dir}/")


def load_model(target, config):
    path = os.path.join(config["output"]["model_dir"], f"{target}_xgb.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def get_feature_importance(trained_models, feature_cols):
    rows = []
    for target, model in trained_models.items():
        importances = model.feature_importances_
        for feat, imp in zip(feature_cols, importances):
            rows.append({"target": target, "feature": feat, "importance": imp})
    fi_df = pd.DataFrame(rows)
    return fi_df.sort_values(["target", "importance"], ascending=[True, False])
