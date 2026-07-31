import pandas as pd
import numpy as np
from src.data_loader import load_config
from src.features import engineer_features, encode_categoricals, get_feature_columns


def make_sample_df():
    return pd.DataFrame({
        "year_built": [1990, 2005, 1970],
        "year_reconstructed": [0, 2015, 2000],
        "structure_length": [100, 200, 300],
        "deck_width": [10, 20, 15],
        "adt": [5000, 20000, 1000],
        "owner": ["State", "County", "City"],
        "txdot_district": ["Austin", "Dallas", "Houston"],
        "climate_zone": ["Humid Subtropical", "Semi-Arid", "Arid"],
        "deck_cond_rating": [7, 5, 3],
        "superstructure_cond_rating": [6, 5, 4],
        "substructure_cond_rating": [7, 6, 3],
        "culvert_cond_rating": [np.nan, 5, np.nan],
    })


def test_engineer_features_creates_bridge_age():
    config = load_config()
    df = make_sample_df()
    result = engineer_features(df, config)
    assert "bridge_age" in result.columns
    assert (result["bridge_age"] >= 0).all()


def test_engineer_features_creates_deck_area():
    config = load_config()
    df = make_sample_df()
    result = engineer_features(df, config)
    assert "deck_area" in result.columns
    assert result["deck_area"].iloc[0] == 100 * 10


def test_encode_categoricals_produces_numeric():
    config = load_config()
    df = make_sample_df()
    result = encode_categoricals(df, config)
    for col in ["owner", "txdot_district", "climate_zone"]:
        assert pd.api.types.is_numeric_dtype(result[col])


def test_get_feature_columns_excludes_targets():
    config = load_config()
    df = make_sample_df()
    df = engineer_features(df, config)
    df = encode_categoricals(df, config)
    feature_cols = get_feature_columns(df, config)
    for target in config["targets"]:
        assert target not in feature_cols


def test_get_feature_columns_excludes_inspection_date():
    config = load_config()
    df = make_sample_df()
    df["inspection_date"] = [5152010, 5152010, 5152010]  # numeric MMDDYYYY -- must NOT be a feature
    df = engineer_features(df, config)
    df = encode_categoricals(df, config)
    assert config["data"]["inspection_date_col"] not in get_feature_columns(df, config)


def test_engineer_features_uses_inspection_year_for_age():
    # age must be measured from the INSPECTION year, not today's date
    config = load_config()
    df = make_sample_df()
    df["inspection_date"] = [5152010, 5152010, 5152010]  # -> 2010
    result = engineer_features(df, config)
    assert result["bridge_age"].iloc[0] == 2010 - 1990  # 20
    assert result["bridge_age"].iloc[2] == 2010 - 1970  # 40


def test_engineer_features_bounds_garbage_year_built():
    config = load_config()
    df = make_sample_df()
    df["year_built"] = [110, 9650, 1970]  # garbage, garbage, valid
    df["inspection_date"] = [5152010, 5152010, 5152010]
    result = engineer_features(df, config)
    assert pd.isna(result["year_built"].iloc[0])       # 110  -> out of [1900, insp_year] -> NaN
    assert pd.isna(result["year_built"].iloc[1])       # 9650 -> NaN
    assert pd.isna(result["bridge_age"].iloc[0])       # NaN year_built -> NaN age
    assert result["bridge_age"].iloc[2] == 2010 - 1970
