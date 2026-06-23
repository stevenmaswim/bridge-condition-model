import pandas as pd
import numpy as np
from src.data_loader import load_config
from src.model import build_model, train_single_target


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


def test_train_single_target_returns_metrics():
    config = load_config()
    df = make_training_df()
    feature_cols = ["feat_a", "feat_b", "feat_c"]
    model, metrics = train_single_target(df, feature_cols, "deck_cond_rating", config)
    assert model is not None
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0


def test_train_skips_small_dataset():
    config = load_config()
    df = make_training_df().head(5)
    feature_cols = ["feat_a", "feat_b", "feat_c"]
    model, metrics = train_single_target(df, feature_cols, "deck_cond_rating", config)
    assert model is None
    assert metrics is None
