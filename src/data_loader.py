import os
import numpy as np
import pandas as pd
import yaml


def load_config(config_path="config.yaml"):
    """Loads the configuration from a YAML file."""
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(config):
    """Loads the raw data into pandas. Reads a local CSV/xlsx by default, or live from Snowflake
    when config.data.source == 'snowflake'."""
    id_col = config.get("data", {}).get("id_col")
    rename_map = config.get("data", {}).get("column_rename_map", {})
    # raw header(s) that map to the id column -- must load as string to keep leading zeros.
    id_raw_cols = [raw for raw, mapped in rename_map.items() if id_col and mapped == id_col]

    # --- live source: Snowflake ---
    if config.get("data", {}).get("source", "csv") == "snowflake":
        from src.snowflake_loader import load_from_snowflake
        df = load_from_snowflake(config)
        for col in id_raw_cols + ([id_col] if id_col else []):
            if col in df.columns:
                df[col] = df[col].astype(str)   # protect leading zeros in the bridge code
        return df

    # --- default source: local file ---
    filepath = os.path.join(config["data"]["raw_dir"], config["data"]["raw_file"])
    dtype_hint = {raw: str for raw in id_raw_cols}
    if filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath, engine="openpyxl", dtype=dtype_hint or None)
    else:
        df = pd.read_csv(filepath, dtype=dtype_hint or None, low_memory=False)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {filepath}")
    return df


def rename_raw_columns(df, config):
    """Renames verbose export column headers (e.g. AssetWise/SNBI labels) to the
    model's snake_case names, per config.yaml's data.column_rename_map. Columns not
    listed in the map are left untouched -- safe to call even when the raw file
    already uses the model's own column names (e.g. the --dry-run synthetic path)."""

    rename_map = config.get("data", {}).get("column_rename_map", {})
    existing_map = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing_map:
        df = df.rename(columns=existing_map)
        print(f"Renamed {len(existing_map)} columns per column_rename_map")
    return df


# Columns the legacy inspection-history export stores in a packed integer form, and the plausible
# range of the real decoded value. Anything already inside its range is left alone, so this is a
# no-op on sources that already hand us decimals (e.g. the SNBI CSV export).
_DEGREES_COLS = ("latitude", "longitude")
_HUNDREDTHS_COLS = ("inventory_load_rating_factor", "operating_load_rating_factor")


def _decode_packed_degrees(series):
    """DDMMSSss integer -> decimal degrees. 29153429 -> 29 deg 15' 34.29" -> 29.2595.

    The legacy HIST_BRG_INSP_DATA table stores NBI items 16/17 this way, while the SNBI export
    stores plain decimals. Values already in the plausible decimal range (|v| <= 200) pass
    through untouched, so this is safe to run on either source."""
    v = pd.to_numeric(series, errors="coerce")
    packed = v.abs() > 200
    if not packed.any():
        return v
    a = v.abs()
    decoded = (a // 1_000_000) + ((a % 1_000_000) // 10_000) / 60.0 + ((a % 10_000) / 100.0) / 3600.0
    return v.where(~packed, decoded * np.sign(v).replace(0, 1))


def normalize_legacy_encodings(df, config):
    """Undo the legacy export's packed encodings so every source presents the same units.

    Why this exists: the models were fit on decimal degrees and load-rating factors around
    1.0-2.5. The live Snowflake history table returns 29153429 for latitude and 227 for an
    inventory rating of 2.27. Feeding those to a fitted tree ensemble raises nothing -- the
    values simply fall to one side of every learned split, so the feature quietly stops
    contributing. Inventory load rating is the 3rd-ranked feature for substructure, so that is
    not a rounding problem.

    Every rule below is guarded on magnitude, so a source that is already in the right units is
    unchanged.
    """
    for col in _DEGREES_COLS:
        if col in df.columns:
            df[col] = _decode_packed_degrees(df[col])
    # TxDOT bridges are all in the western hemisphere; the packed form carries no sign.
    if "longitude" in df.columns:
        lon = pd.to_numeric(df["longitude"], errors="coerce")
        df["longitude"] = lon.where(lon <= 0, -lon)
    for col in _HUNDREDTHS_COLS:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            df[col] = v.where(v.abs() <= 10, v / 100.0)
    return df


def clean_data(df, config):
    """Cleans data in accordance with the configuration file because the configuration determines which data is important"""
    
    targets = config["targets"]
    numeric_feats = config["features"]["numeric"]
    categorical_feats = config["features"]["categorical"]
    id_col = config.get("data", {}).get("id_col")
    # inspection date and the on/off-system flag are not model features but must survive cleaning:
    # the inspection date rebuilds real inspection events, the system flag scopes the watch-list.
    insp_col = config.get("data", {}).get("inspection_date_col")
    system_col = config.get("data", {}).get("system_col")
    all_cols = targets + numeric_feats + categorical_feats
    for extra in (insp_col, system_col, id_col):
        if extra and extra not in all_cols:
            all_cols = [extra] + all_cols

    available = [c for c in all_cols if c in df.columns]
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        print(f"Warning: columns not found in data, skipping: {missing}")

    df = df[available].copy()

    for col in targets + numeric_feats:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[t for t in targets if t in df.columns], how="all")
    df = normalize_legacy_encodings(df, config)

    print(f"After cleaning: {len(df)} rows, {len(df.columns)} columns")
    return df


def save_processed(df, config):
    """Saves the processed data to a CSV file in the specified directory."""

    os.makedirs(config["data"]["processed_dir"], exist_ok=True)
    out_path = os.path.join(config["data"]["processed_dir"], "cleaned.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved processed data to {out_path}")
    return out_path
