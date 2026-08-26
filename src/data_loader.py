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
_HUNDREDTHS_COLS = ("inventory_load_rating_factor", "operating_load_rating_factor")

# Generous bounding box around Texas. Used only to decide whether a decoded coordinate is
# believable -- anything outside becomes NaN rather than being passed on as a number. A tree
# ensemble handles NaN natively and correctly; it cannot tell that 0.49 is a failed decode.
_GEO_BOUNDS = {"latitude": (24.0, 38.0), "longitude": (-108.0, -92.0)}


def _decode_packed_degrees(series):
    """Packed sexagesimal integer -> decimal degrees, tolerant of the widths in the wild.

    The legacy history table stores NBI items 16/17 as a packed integer, but not at a single
    width: 29153429 is DDMMSSss (29 deg 15' 34.29") while 291534 is DDMMSS with the hundredths
    dropped. Decoding everything as DDMMSSss turns the six-digit form into 0 deg 29' 15.34",
    i.e. 0.49 -- a number that looks fine to a fitted model and is off by thirty degrees.

    Widths are therefore dispatched by magnitude, and values already in decimal degrees pass
    through. Zero means "not recorded" in this export, not the Gulf of Guinea, so it is dropped.
    """
    v = pd.to_numeric(series, errors="coerce")
    a = v.abs()
    sign = np.where(v < 0, -1.0, 1.0)
    mag = pd.Series(np.nan, index=v.index, dtype="float64")

    decimal = a.gt(0) & a.le(200)                                    # 29.2595
    mag[decimal] = a[decimal]

    ddmmssss = a.ge(1_000_000) & a.lt(1_000_000_000)                 # 29153429
    d = a[ddmmssss]
    mag[ddmmssss] = d // 1_000_000 + (d % 1_000_000) // 10_000 / 60.0 + (d % 10_000) / 100.0 / 3600.0

    ddmmss = a.ge(10_000) & a.lt(1_000_000)                          # 291534
    d = a[ddmmss]
    mag[ddmmss] = d // 10_000 + (d % 10_000) // 100 / 60.0 + (d % 100) / 3600.0

    return mag * sign


def normalize_legacy_encodings(df, config=None, verbose=True):
    """Undo the legacy export's packed encodings so every source presents the same units.

    Why this exists: the models were fit on decimal degrees and load-rating factors around
    1.0-2.5. The live Snowflake history table returns 29153429 for latitude and 227 for an
    inventory rating of 2.27. Feeding those to a fitted tree ensemble raises nothing -- the
    values simply fall to one side of every learned split, so the feature quietly stops
    contributing. Inventory load rating is the 3rd-ranked feature for substructure, so that is
    not a rounding problem.

    Every rule is guarded on magnitude, so a source already in the right units is unchanged.
    """
    for col, (lo, hi) in _GEO_BOUNDS.items():
        if col not in df.columns:
            continue
        before = pd.to_numeric(df[col], errors="coerce")
        out = _decode_packed_degrees(before)
        if col == "longitude":
            # the packed form carries no sign and every TxDOT bridge is west of the meridian
            out = out.where(out <= 0, -out)
        # A coordinate we cannot land inside the state is a failed decode or an unrecorded
        # value. NaN is the honest answer; a wrong number is not.
        implausible = out.notna() & ~out.between(lo, hi)
        out = out.mask(implausible)
        if verbose:
            lost = int(implausible.sum())
            unrecorded = int((before.notna() & before.eq(0)).sum())
            if lost or unrecorded:
                print(f"  [units] {col}: {lost:,} implausible -> NaN, "
                      f"{unrecorded:,} recorded as 0 (not captured)")
            # Report the digit widths that failed. A packed width this decoder does not handle
            # would show up as one dominant length -- that is a decoder gap to fix, not missing
            # data to accept. Without this the two are indistinguishable from the outside.
            if lost:
                widths = (before[implausible].abs().dropna().astype("int64").astype(str)
                          .str.len().value_counts().sort_index())
                shape = ", ".join(f"{w}-digit x{c:,}" for w, c in widths.items() if c)
                print(f"           failed widths: {shape}")
        df[col] = out

    for col in _HUNDREDTHS_COLS:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            df[col] = v.where(v.abs() <= 10, v / 100.0)

    # District codes arrive zero-padded from some sources and bare from others -- the live
    # history table contains BOTH '1' and '01' for district 1. Two consequences, both quiet:
    # grouping splits one district into two, and because txdot_district is a model feature the
    # encoder (fitted on the bare form) treats '01' as a category it has never seen. Collapse
    # numeric codes to their bare form; leave anything non-numeric alone.
    dist_col = ((config or {}).get("grouping", {}) or {}).get("district_col", "txdot_district")
    if dist_col in df.columns:
        v = df[dist_col].astype(str).str.strip()
        numeric = v.str.fullmatch(r"\d+").fillna(False)
        out = v.mask(numeric, v.str.lstrip("0").replace("", "0"))
        # astype(str) turns a missing district into the literal "None", which then survives
        # dropna() and produces a "district None" report. Put the real nulls back.
        df[dist_col] = out.where(df[dist_col].notna())
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
        # This is a SCHEMA message -- these columns are absent from the source query, which is a
        # different thing from a bridge having a blank value. Saying only "missing" invites the
        # reading that most bridges lack them. Several are expected: bridge_age is engineered at
        # runtime and never appears in a source, and the enrichment extract supplies its own set
        # further down the pipeline, after this point.
        enr = config.get("enrichment", {}) or {}
        from_extract = set(enr.get("static_numeric", []) or []) | set(enr.get("static_categorical", []) or [])
        derived = {"bridge_age", "deck_area"}
        later = sorted(c for c in missing if c in from_extract)
        engineered = sorted(c for c in missing if c in derived)
        absent = sorted(c for c in missing if c not in from_extract and c not in derived)
        if absent:
            print(f"Note: not provided by this source: {absent}")
        if later:
            print(f"      (supplied later by the enrichment extract: {later})")
        if engineered:
            print(f"      (derived during feature engineering: {engineered})")

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
