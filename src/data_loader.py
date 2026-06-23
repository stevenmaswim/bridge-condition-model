import os
import pandas as pd
import yaml


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(config):
    filepath = os.path.join(config["data"]["raw_dir"], config["data"]["raw_file"])
    if filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath, engine="openpyxl")
    else:
        df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {filepath}")
    return df


def clean_data(df, config):
    targets = config["targets"]
    numeric_feats = config["features"]["numeric"]
    categorical_feats = config["features"]["categorical"]
    all_cols = targets + numeric_feats + categorical_feats

    available = [c for c in all_cols if c in df.columns]
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        print(f"Warning: columns not found in data, skipping: {missing}")

    df = df[available].copy()
    df = df.dropna(subset=[t for t in targets if t in df.columns], how="all")

    for col in numeric_feats:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"After cleaning: {len(df)} rows, {len(df.columns)} columns")
    return df


def save_processed(df, config):
    os.makedirs(config["data"]["processed_dir"], exist_ok=True)
    out_path = os.path.join(config["data"]["processed_dir"], "cleaned.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved processed data to {out_path}")
    return out_path
