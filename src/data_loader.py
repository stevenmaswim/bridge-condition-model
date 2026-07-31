import os
import pandas as pd
import yaml


def load_config(config_path="config.yaml"):
    """Loads the configuration from a YAML file."""
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(config):
    """Loads the raw data from the specified file path in the configuration file into pandas."""

    filepath = os.path.join(config["data"]["raw_dir"], config["data"]["raw_file"])

    # Force the ID column to load as a string, whatever its raw (pre-rename) header is --
    # a numeric-looking bridge code (e.g. "010600013603019") would otherwise get parsed as
    # an int and silently lose its leading zero.
    id_col = config.get("data", {}).get("id_col")
    rename_map = config.get("data", {}).get("column_rename_map", {})
    dtype_hint = {raw: str for raw, mapped in rename_map.items() if id_col and mapped == id_col}

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

    print(f"After cleaning: {len(df)} rows, {len(df.columns)} columns")
    return df


def save_processed(df, config):
    """Saves the processed data to a CSV file in the specified directory."""

    os.makedirs(config["data"]["processed_dir"], exist_ok=True)
    out_path = os.path.join(config["data"]["processed_dir"], "cleaned.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved processed data to {out_path}")
    return out_path
