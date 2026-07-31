import numpy as np
import pandas as pd

from src.data_loader import load_config
from src.model import (
    build_model,
    compute_linear_impute_values,
    get_feature_importance,
    load_model,
    save_models,
    train_all_targets,
    train_single_target,
)


def make_training_df():
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "feat_a": rng.uniform(0, 100, n),
        "feat_b": rng.uniform(0, 50, n),
        "feat_c": rng.integers(0, 5, n).astype(float),
        "deck_cond_rating": np.clip(rng.normal(6, 1.5, n), 0, 9).round(0),
    })


def test_build_model_returns_xgb():
    config = load_config()
    model = build_model(config)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_build_model_linear_regression():
    config = load_config()
    model = build_model(config, model_type="linear_regression")
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_build_model_invalid_type_raises():
    config = load_config()
    try:
        build_model(config, model_type="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_train_single_target_returns_metrics():
    config = load_config()
    df = make_training_df()
    feature_cols = ["feat_a", "feat_b", "feat_c"]
    model, metrics = train_single_target(df, feature_cols, "deck_cond_rating", config)
    assert model is not None
    assert metrics["model_type"] == "xgboost"
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0


def test_train_single_target_linear_regression():
    config = load_config()
    df = make_training_df()
    feature_cols = ["feat_a", "feat_b", "feat_c"]
    impute_values = compute_linear_impute_values(df, feature_cols)
    model, metrics = train_single_target(
        df, feature_cols, "deck_cond_rating", config,
        model_type="linear_regression", linear_impute_values=impute_values,
    )
    assert model is not None
    assert metrics["model_type"] == "linear_regression"
    assert np.isfinite(metrics["mae"])
    assert np.isfinite(metrics["rmse"])
    assert np.isfinite(metrics["r2"])
    # sanity bound: a correctly-imputed linear model on this synthetic data shouldn't blow up MAE
    assert metrics["mae"] < 5


def test_train_skips_small_dataset():
    config = load_config()
    df = make_training_df().head(5)
    feature_cols = ["feat_a", "feat_b", "feat_c"]
    model, metrics = train_single_target(df, feature_cols, "deck_cond_rating", config)
    assert model is None
    assert metrics is None


def test_compute_linear_impute_values_handles_all_nan_column():
    df = make_training_df()
    df["feat_all_nan"] = np.nan
    impute_values = compute_linear_impute_values(df, ["feat_a", "feat_all_nan"])
    assert impute_values["feat_all_nan"] == 0
    assert not np.isnan(impute_values["feat_a"])


def test_train_all_targets_produces_both_model_types():
    config = load_config()
    df = make_training_df()
    df = df.rename(columns={"feat_a": "structure_length", "feat_b": "deck_width", "feat_c": "skew_angle"})
    feature_cols = ["structure_length", "deck_width", "skew_angle"]
    # only deck_cond_rating exists in this synthetic df; other 3 targets get skipped, which is fine
    trained_models, metrics_df = train_all_targets(df, feature_cols, config)
    assert set(trained_models["deck_cond_rating"].keys()) == {"xgboost", "linear_regression"}
    assert metrics_df[metrics_df["target"] == "deck_cond_rating"]["model_type"].nunique() == 2


def test_tuning_disabled_uses_fixed_params():
    config = load_config()
    config["model"]["tuning"]["enabled"] = False
    df = make_training_df()
    feature_cols = ["feat_a", "feat_b", "feat_c"]
    model, metrics = train_single_target(df, feature_cols, "deck_cond_rating", config, model_type="xgboost")
    assert model.get_params()["n_estimators"] == config["model"]["params"]["n_estimators"]
    assert model.get_params()["max_depth"] == config["model"]["params"]["max_depth"]
    assert metrics["best_params"] == ""


def test_tuning_enabled_runs_search_and_returns_valid_model():
    config = load_config()
    config["model"]["tuning"]["enabled"] = True
    config["model"]["tuning"]["n_iter"] = 2
    config["model"]["tuning"]["cv"] = 2
    config["model"]["tuning"]["param_grid"] = {
        "n_estimators": [50, 100],
        "max_depth": [3, 4],
    }
    df = make_training_df()
    feature_cols = ["feat_a", "feat_b", "feat_c"]
    model, metrics = train_single_target(df, feature_cols, "deck_cond_rating", config, model_type="xgboost")
    assert model is not None
    assert hasattr(model, "predict")
    assert metrics["best_params"] != ""


def test_get_feature_importance_handles_linear_and_xgb():
    config = load_config()
    df = make_training_df()
    feature_cols = ["feat_a", "feat_b", "feat_c"]
    xgb_model, _ = train_single_target(df, feature_cols, "deck_cond_rating", config, model_type="xgboost")
    impute_values = compute_linear_impute_values(df, feature_cols)
    lr_model, _ = train_single_target(
        df, feature_cols, "deck_cond_rating", config,
        model_type="linear_regression", linear_impute_values=impute_values,
    )
    trained_models = {"deck_cond_rating": {"xgboost": xgb_model, "linear_regression": lr_model}}
    fi_df = get_feature_importance(trained_models, feature_cols)
    assert set(fi_df["model_type"].unique()) == {"xgboost", "linear_regression"}
    assert set(fi_df["importance_type"].unique()) == {"gain_importance", "abs_coefficient"}


def test_save_and_load_model_roundtrip_per_model_type(tmp_path):
    config = load_config()
    config["output"]["model_dir"] = str(tmp_path)
    df = make_training_df()
    feature_cols = ["feat_a", "feat_b", "feat_c"]
    trained_models, _ = train_all_targets(df, feature_cols, config)
    save_models(trained_models, config)

    xgb_loaded = load_model("deck_cond_rating", config, model_type="xgboost")
    lr_loaded = load_model("deck_cond_rating", config, model_type="linear_regression")
    assert hasattr(xgb_loaded, "predict")
    assert hasattr(lr_loaded, "predict")
