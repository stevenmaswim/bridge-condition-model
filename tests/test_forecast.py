import numpy as np
import pandas as pd

from src.forecast import build_future_row, build_model_input, find_bridge_row


def make_bridge_row(**overrides):
    base = {
        "bridge_id": "010600013603019",
        "year_built": 1980,
        "year_reconstructed": 0,
        "bridge_age": 46,
        "years_since_reconstruction": 46,
        "deck_width": 30.0,
    }
    base.update(overrides)
    return pd.Series(base)


def test_build_future_row_projects_age_forward():
    row = make_bridge_row()
    future_row, ok = build_future_row(row, 2050)
    assert ok is True
    assert future_row["bridge_age"] == 70  # 2050 - 1980


def test_build_future_row_uses_reconstruction_year_when_present():
    row = make_bridge_row(year_reconstructed=2010)
    future_row, ok = build_future_row(row, 2030)
    assert ok is True
    assert future_row["years_since_reconstruction"] == 20  # 2030 - 2010
    assert future_row["bridge_age"] == 50  # 2030 - 1980, unaffected by reconstruction


def test_build_future_row_falls_back_to_bridge_age_when_never_reconstructed():
    row = make_bridge_row(year_reconstructed=0)
    future_row, ok = build_future_row(row, 2030)
    assert future_row["years_since_reconstruction"] == future_row["bridge_age"]


def test_build_future_row_fails_gracefully_without_year_built():
    row = make_bridge_row(year_built=np.nan)
    future_row, ok = build_future_row(row, 2030)
    assert ok is False


def test_build_future_row_clips_negative_age_to_zero():
    row = make_bridge_row(year_built=2020)
    future_row, ok = build_future_row(row, 2015)  # target year before it was even built
    assert ok is True
    assert future_row["bridge_age"] == 0


def test_find_bridge_row_finds_exact_match():
    df = pd.DataFrame({"bridge_id": ["001", "002", "003"], "year_built": [1980, 1990, 2000]})
    row = find_bridge_row(df, "bridge_id", "002")
    assert row is not None
    assert row["year_built"] == 1990


def test_find_bridge_row_returns_none_when_not_found():
    df = pd.DataFrame({"bridge_id": ["001", "002", "003"], "year_built": [1980, 1990, 2000]})
    row = find_bridge_row(df, "bridge_id", "999999999")
    assert row is None


def test_find_bridge_row_tolerates_whitespace():
    df = pd.DataFrame({"bridge_id": [" 001 ", "002"], "year_built": [1980, 1990]})
    row = find_bridge_row(df, "bridge_id", "001")
    assert row is not None


def test_build_model_input_handles_object_dtype_row_from_mixed_df():
    # Mixed column dtypes (str id, int, float, int8-like category code) forces pandas to
    # upcast a single-row .iloc[] extraction to dtype "object" -- the exact bug XGBoost
    # rejected with "DataFrame.dtypes for data must be int, float, bool or category".
    df = pd.DataFrame({
        "bridge_id": ["001", "002"],
        "year_built": [1980, 1990],
        "deck_width": [30.5, 28.0],
        "owner": pd.Series([0, 1], dtype="int8"),
    })
    row = df.iloc[0]
    assert row.dtype == object  # confirms the bug scenario is actually reproduced

    feature_cols = ["year_built", "deck_width", "owner"]
    X = build_model_input(row, feature_cols, model_type="xgboost")
    assert X.shape == (1, 3)
    for col in feature_cols:
        assert pd.api.types.is_numeric_dtype(X[col])


def test_build_model_input_linear_regression_uses_impute_values():
    df = pd.DataFrame({
        "year_built": [1980],
        "deck_width": [np.nan],
    })
    row = df.iloc[0]
    X = build_model_input(row, ["year_built", "deck_width"], model_type="linear_regression",
                           linear_impute_values={"deck_width": 25.0})
    assert X["deck_width"].iloc[0] == 25.0
