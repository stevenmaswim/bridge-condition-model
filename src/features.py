import pandas as pd
import numpy as np
from datetime import datetime


def engineer_features(df, config):
    """Engineers the new column based upon the existing columns in the dataframe, in this case it is age, deck size, traffic density, and years since reconstruction."""
    if "year_built" in df.columns:
        current_year = datetime.now().year
        df["bridge_age"] = current_year - df["year_built"]
        df["bridge_age"] = df["bridge_age"].clip(lower=0)

    if "year_reconstructed" in df.columns and "year_built" in df.columns:
        df["years_since_reconstruction"] = np.where(
            df["year_reconstructed"] > 0,
            current_year - df["year_reconstructed"],
            df.get("bridge_age", 0),
        )

    if "deck_area" not in df.columns:
        if "structure_length" in df.columns and "deck_width" in df.columns:
            df["deck_area"] = df["structure_length"] * df["deck_width"]

    if "adt" in df.columns and "deck_area" in df.columns:
        df["traffic_density"] = df["adt"] / df["deck_area"].replace(0, np.nan)

    return df


def encode_categoricals(df, config):
    """Scans through the dataframe and only includes features specified in the configuration file, and encodes categorical features as numeric codes."""
    categorical_feats = config["features"]["categorical"]
    for col in categorical_feats:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("Unknown")
            df[col] = df[col].astype("category").cat.codes
    return df


def get_feature_columns(df, config):
    """Returns a list of feature columns based on the configuration file, excluding target and grouping columns."""
    targets = config["targets"]
    group_cols = [
        config["grouping"]["district_col"],
        config["grouping"]["climate_zone_col"],
    ]
    exclude = set(targets + group_cols)
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype != "object"]
    return feature_cols


def prepare_model_data(df, config):
    """Prepares the dataframe for modeling by engineering features, encoding categorical variables, and selecting feature columns based on the configuration."""
    df = engineer_features(df, config)
    df = encode_categoricals(df, config)
    feature_cols = get_feature_columns(df, config)
    print(f"Prepared {len(feature_cols)} features for modeling")
    return df, feature_cols
