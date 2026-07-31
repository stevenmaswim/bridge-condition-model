import pandas as pd
import numpy as np
from datetime import datetime

from src.enrichment import attach_static_features


def _inspection_year(df, config):
    """Per-row inspection year parsed from the MMDDYYYY inspection-date column, or None if absent
    (e.g. --dry-run synthetic data)."""
    insp_col = config.get("data", {}).get("inspection_date_col")
    if insp_col and insp_col in df.columns:
        s = df[insp_col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(8)
        return pd.to_datetime(s, format="%m%d%Y", errors="coerce").dt.year
    return None


def engineer_features(df, config):
    """Engineers age/size/traffic features.

    bridge_age uses the INSPECTION year (not today's date) so a bridge's age aligns with the rating
    recorded at that historical inspection -- the previous code used datetime.now(), which gave every
    one of a bridge's ~20 inspection rows the same (wrong) age. year_built is sanity-bounded to
    [1900, inspection_year]; garbage values (the raw data holds years like 110 and 9650) become NaN."""
    insp_year = _inspection_year(df, config)
    ref_year = insp_year if insp_year is not None else datetime.now().year

    if "year_built" in df.columns:
        yb = pd.to_numeric(df["year_built"], errors="coerce")
        upper = insp_year if insp_year is not None else datetime.now().year
        yb = yb.where((yb >= 1900) & (yb <= upper))
        df["year_built"] = yb
        df["bridge_age"] = (ref_year - yb).clip(lower=0, upper=130)

    if "year_reconstructed" in df.columns and "year_built" in df.columns:
        df["years_since_reconstruction"] = np.where(
            df["year_reconstructed"] > 0,
            ref_year - df["year_reconstructed"],
            df.get("bridge_age", 0),
        )

    if "deck_area" not in df.columns:
        if "structure_length" in df.columns and "deck_width" in df.columns:
            df["deck_area"] = df["structure_length"] * df["deck_width"]

    if "adt" in df.columns and "deck_area" in df.columns:
        df["traffic_density"] = df["adt"] / df["deck_area"].replace(0, np.nan)

    return df


def attach_and_encode_enrichment(df, config):
    """Join the static physical attributes from the AssetWise extract (src/enrichment.py) onto the
    panel and encode any enrichment-only categoricals. Config-listed categoricals are left for
    encode_categoricals. No-ops safely when there's no id column (dry-run) or no extract file."""
    id_col = config.get("data", {}).get("id_col")
    if not id_col or id_col not in df.columns:
        return df
    df, feats = attach_static_features(df, config)
    config_cats = set(config.get("features", {}).get("categorical", []))
    for c in feats:
        if c in config_cats:
            continue  # handled by encode_categoricals
        # encode any non-numeric (string/object/category) column to integer codes. Checking
        # is_numeric_dtype rather than == "object" is required on pandas >= 3.0, where text loads
        # as StringDtype (not object) and would otherwise slip through as a bogus numeric feature.
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).fillna("Unknown").astype("category").cat.codes
    return df


def encode_categoricals(df, config):
    """Encodes the configured categorical features as integer codes."""
    categorical_feats = config["features"]["categorical"]
    for col in categorical_feats:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("Unknown")
            df[col] = df[col].astype("category").cat.codes
    return df


def get_feature_columns(df, config):
    """Returns feature columns, excluding targets, grouping, id, and inspection-date columns."""
    targets = config["targets"]
    group_cols = [
        config["grouping"]["district_col"],
        config["grouping"]["climate_zone_col"],
    ]
    exclude = set(targets + group_cols)
    id_col = config.get("data", {}).get("id_col")
    if id_col:
        exclude.add(id_col)
    # inspection_date is stored as an integer MMDDYYYY, so it loads as numeric and would otherwise
    # be picked up as a model feature. It's a bookkeeping column for the deterioration model only.
    insp_col = config.get("data", {}).get("inspection_date_col")
    if insp_col:
        exclude.add(insp_col)
    # is_numeric_dtype (not != "object") so pandas>=3.0 StringDtype text columns are excluded too.
    feature_cols = [c for c in df.columns
                    if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return feature_cols


def prepare_model_data(df, config):
    """Engineers features, joins enrichment attributes, encodes categoricals, and selects features."""
    df = engineer_features(df, config)
    df = attach_and_encode_enrichment(df, config)
    df = encode_categoricals(df, config)
    feature_cols = get_feature_columns(df, config)
    print(f"Prepared {len(feature_cols)} features for modeling")
    return df, feature_cols
