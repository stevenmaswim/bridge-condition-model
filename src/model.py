import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


def build_model(config, model_type="xgboost"):
    if model_type == "xgboost":
        params = config["model"]["params"].copy()
        return XGBRegressor(**params)
    elif model_type == "linear_regression":
        return LinearRegression()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def tune_xgboost(X_train, y_train, config):
    tuning_cfg = config["model"]["tuning"]
    base = XGBRegressor(random_state=config["model"]["params"]["random_state"])
    param_grid = tuning_cfg["param_grid"]
    common = dict(
        cv=tuning_cfg.get("cv", 3),
        scoring=tuning_cfg.get("scoring", "neg_mean_absolute_error"),
        n_jobs=-1,
    )
    if tuning_cfg.get("method", "random") == "grid":
        search = GridSearchCV(base, param_grid, **common)
    else:
        search = RandomizedSearchCV(
            base, param_grid,
            n_iter=tuning_cfg.get("n_iter", 20),
            random_state=tuning_cfg.get("random_state", 42),
            **common,
        )
    search.fit(X_train, y_train)
    print(f"    [tuning] best params: {search.best_params_} (best CV {common['scoring']}={search.best_score_:.4f})")
    return search.best_estimator_, search.best_params_


def compute_linear_impute_values(df, feature_cols):
    """Per-column medians for Linear Regression's imputation, since -999 (the XGBoost sentinel) would badly distort OLS coefficients."""
    medians = df[feature_cols].median()
    medians = medians.fillna(0)
    return medians.to_dict()


def split_train_test(X_raw, y, subset, config):
    """Split into train/test sets.

    Defaults to a GROUP-aware split keyed on the bridge id (config data.id_col): the raw
    data is a panel with ~20 inspections per bridge, so a plain random split scatters
    inspection rows of the SAME bridge across both train and test. The model then partly
    memorizes individual bridges, which inflates held-out metrics and does not reflect
    accuracy on a genuinely new bridge. GroupShuffleSplit guarantees every bridge lands
    entirely in train OR test, never both.

    Falls back to the legacy random split when no usable group column is present (e.g. the
    --dry-run synthetic data has no bridge id) or when config.model.split.method == "random"
    (kept so we can reproduce the old, leaky number for an honest before/after comparison)."""
    test_size = config["model"]["test_size"]
    random_state = config["model"]["params"]["random_state"]
    split_cfg = config.get("model", {}).get("split", {})
    method = split_cfg.get("method", "group")
    id_col = config.get("data", {}).get("id_col")

    if method == "group" and id_col and id_col in subset.columns and subset[id_col].nunique() > 1:
        groups = subset[id_col]
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X_raw, y, groups))
        return (X_raw.iloc[train_idx], X_raw.iloc[test_idx],
                y.iloc[train_idx], y.iloc[test_idx])

    if method == "group":
        print("    [split] group split requested but no usable id column found -- falling back to random split")
    return train_test_split(X_raw, y, test_size=test_size, random_state=random_state)


def train_single_target(df, feature_cols, target, config, model_type="xgboost", linear_impute_values=None):
    subset = df.dropna(subset=[target])
    if len(subset) < 10:
        print(f"  Skipping {target} ({model_type}): only {len(subset)} valid rows")
        return None, None

    X_raw = subset[feature_cols]
    y = subset[target]

    X_train_raw, X_test_raw, y_train, y_test = split_train_test(X_raw, y, subset, config)

    if model_type == "xgboost":
        # Keep NaN: XGBoost has native sparsity-aware missing-value handling that learns the best
        # direction for missing at each split. The old -999 sentinel defeated that by making
        # "missing" look like a real, extreme numeric value.
        X_train, X_test = X_train_raw, X_test_raw
    elif model_type == "linear_regression":
        impute_values = linear_impute_values or {}
        X_train, X_test = X_train_raw.fillna(impute_values), X_test_raw.fillna(impute_values)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    tuning_cfg = config.get("model", {}).get("tuning", {})
    best_params = None
    if model_type == "xgboost" and tuning_cfg.get("enabled", False):
        model, best_params = tune_xgboost(X_train, y_train, config)
    else:
        model = build_model(config, model_type)
        if model_type == "xgboost":
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        else:
            model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, 9)  # match predict_all_targets()'s clipping so metrics reflect real deployed behavior --
                                     # Linear Regression can extrapolate to nonsense values on unusual rows; unclipped,
                                     # a handful of those can dominate RMSE/R2 while leaving MAE looking deceptively normal
    abs_err = np.abs(y_test.to_numpy() - y_pred)
    metrics = {
        "target": target,
        "model_type": model_type,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "within_0.5": (abs_err <= 0.5).mean() * 100,
        "within_1": (abs_err <= 1.0).mean() * 100,
        "within_2": (abs_err <= 2.0).mean() * 100,
        "best_params": str(best_params) if best_params else "",
    }
    print(
        f"  {target} [{model_type}]: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, "
        f"R2={metrics['r2']:.3f}, within 1 rating={metrics['within_1']:.1f}% (held-out test, n={metrics['n_test']})"
    )
    return model, metrics


def train_all_targets(df, feature_cols, config, linear_impute_values=None):
    targets = config["targets"]
    model_types = config["model"].get("compare_types", ["xgboost", "linear_regression"])
    trained = {}
    all_metrics = []

    for target in targets:
        if target not in df.columns:
            print(f"  Target column {target} not in data, skipping")
            continue
        trained[target] = {}
        for model_type in model_types:
            print(f"Training {model_type} for: {target}")
            model, metrics = train_single_target(
                df, feature_cols, target, config, model_type=model_type,
                linear_impute_values=linear_impute_values,
            )
            if model is not None:
                trained[target][model_type] = model
                all_metrics.append(metrics)
        if not trained[target]:
            del trained[target]

    return trained, pd.DataFrame(all_metrics)


def save_models(trained_models, config):
    model_dir = config["output"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    count = 0
    for target, models_by_type in trained_models.items():
        for model_type, model in models_by_type.items():
            path = os.path.join(model_dir, f"{target}_{model_type}.pkl")
            with open(path, "wb") as f:
                pickle.dump(model, f)
            count += 1
    print(f"Saved {count} models to {model_dir}/")


def load_model(target, config, model_type="xgboost"):
    path = os.path.join(config["output"]["model_dir"], f"{target}_{model_type}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def get_feature_importance(trained_models, feature_cols):
    rows = []
    for target, models_by_type in trained_models.items():
        for model_type, model in models_by_type.items():
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                importance_type = "gain_importance"
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_)
                importance_type = "abs_coefficient"
            else:
                print(f"  Warning: {model_type} exposes neither feature_importances_ nor coef_, skipping for {target}")
                continue
            for feat, imp in zip(feature_cols, importances):
                rows.append({
                    "target": target,
                    "model_type": model_type,
                    "feature": feat,
                    "importance": imp,
                    "importance_type": importance_type,
                })
    fi_df = pd.DataFrame(rows)
    if fi_df.empty:
        return fi_df
    if "linear_regression" in fi_df["model_type"].values:
        print(
            "  Note: linear_regression 'importance' is abs(coef_), which is NOT scale-normalized "
            "like XGBoost's feature_importances_ (which sum to 1). Compare feature rankings within "
            "a model_type only -- do not compare raw magnitudes across model types."
        )
    return fi_df.sort_values(["target", "model_type", "importance"], ascending=[True, True, False])
