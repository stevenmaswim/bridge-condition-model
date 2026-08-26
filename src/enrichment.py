"""Join static per-bridge physical attributes (from the AssetWise SNBI extract) onto the
inspection panel.

The extract produced by sql/bridge_data_extract.sql is one CURRENT row per bridge. Its physical
attributes (span material/type, deck type, span counts, scour) are static -- they do not change
between inspections barring reconstruction -- so the latest value legitimately back-fills every
historical inspection row of that bridge, keyed on bridge_id (= as_code).

Handles both a headered export (preferred; column names come from the SQL aliases) and the legacy
headerless export (names assigned positionally from EXTRACT_COLUMNS, which must match the SELECT
order in sql/bridge_data_extract.sql).
"""
import os
from functools import lru_cache

import numpy as np
import pandas as pd

# Canonical output column order of sql/bridge_data_extract.sql. KEEP IN SYNC with that SELECT list.
EXTRACT_COLUMNS = [
    "as_id", "as_code", "deck_cond_rating", "superstructure_cond_rating", "substructure_cond_rating",
    "culvert_cond_rating", "txdot_district", "climate_zone", "structure_length", "max_span_length",
    "deck_width", "roadway_width", "adt", "adt_year", "adt_truck", "year_built", "year_reconstructed",
    "skew_angle", "num_spans_main", "num_beam_lines", "num_spans_approach", "owner", "maintenance_resp",
    "structure_kind", "structure_type", "span_continuity", "deck_type", "wearing_surface", "membrane_type",
    "deck_protection", "design_load", "approach_roadway_width", "functional_class", "facility_carried",
    "features_intersected", "channel_cond_rating", "scour_vulnerability", "load_posting_status",
    "county_code", "inspection_date", "inspection_interval",
]

# Static physical attributes worth joining. Defaults; overridable via config.enrichment.
DEFAULT_STATIC_NUMERIC = ["num_spans_main", "num_beam_lines", "adt_truck"]
DEFAULT_STATIC_CATEGORICAL = [
    "structure_kind", "structure_type", "span_continuity", "deck_type",
    "wearing_surface", "deck_protection", "scour_vulnerability", "load_posting_status",
]


@lru_cache(maxsize=4)
def _read_extract(path, has_header):
    """Read the extract as strings, normalizing NULL/empty to NaN. Assigns positional names when
    the file has no header row. Cached per (path, has_header): the 63k-row extract is otherwise
    re-parsed on every enrichment call (5x in a full pipeline run). Callers must not mutate the
    returned frame in place (load_static_features only reads/copies from it)."""
    if has_header:
        df = pd.read_csv(path, dtype=str, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
    else:
        df = pd.read_csv(path, header=None, names=EXTRACT_COLUMNS, dtype=str, low_memory=False)
    return df.replace({"NULL": np.nan, "": np.nan})


@lru_cache(maxsize=4)
def _build_lookup(path, has_header, id_col, numeric, categorical):
    """Build the per-bridge lookup once per (file, config) combination.

    _read_extract already caches the file read, but the de-duplication and rename below ran on
    every call -- 63,242 rows re-processed and re-announced once per district, 25 times over in
    a full build. Arguments are plain hashables so lru_cache can key on them.
    """
    df = _read_extract(path, has_header=has_header)
    if "as_code" not in df.columns:
        print("  [enrichment] no 'as_code' column found; skipping enrichment")
        return None
    feats = [c for c in list(numeric) + list(categorical) if c in df.columns]
    lookup = df[["as_code"] + feats].drop_duplicates("as_code").rename(columns={"as_code": id_col})
    print(f"  [enrichment] loaded {len(lookup):,} bridges x {len(feats)} static features "
          f"({', '.join(feats)})")
    return lookup


def load_static_features(config):
    """Return a per-bridge lookup DataFrame keyed on the model's id column (bridge_id), carrying the
    configured static physical features. Returns None if enrichment is disabled or the file is
    missing, so the pipeline degrades gracefully to panel-only features."""
    cfg = config.get("enrichment", {}) or {}
    path = cfg.get("extract_file")
    if not path or not os.path.exists(path):
        if path:
            print(f"  [enrichment] extract file not found ({path}); using panel features only")
        return None

    numeric = cfg.get("static_numeric", DEFAULT_STATIC_NUMERIC)
    categorical = cfg.get("static_categorical", DEFAULT_STATIC_CATEGORICAL)
    id_col = config.get("data", {}).get("id_col", "bridge_id")
    lookup = _build_lookup(path, bool(cfg.get("has_header", False)), id_col,
                           tuple(numeric), tuple(categorical))
    if lookup is None:
        return None
    lookup = lookup.copy()   # callers get their own frame; the cached one stays pristine
    for c in numeric:
        if c in lookup.columns:
            lookup[c] = pd.to_numeric(lookup[c], errors="coerce")
    return lookup


def attach_static_features(df, config):
    """Left-join the static physical features onto a panel/event DataFrame by bridge_id.
    Returns (df, feature_names). Unmatched bridges get NaN (handled natively by XGBoost)."""
    lookup = load_static_features(config)
    if lookup is None:
        return df, []
    id_col = config.get("data", {}).get("id_col", "bridge_id")
    feats = [c for c in lookup.columns if c != id_col]
    # Don't re-add columns the panel already carries. The live Snowflake query already supplies
    # num_spans_main/structure_type/deck_type/etc.; merging them again would collide and pandas would
    # suffix both copies (_x/_y), breaking every downstream lookup by name. Keep the values on df.
    overlap = [c for c in feats if c in df.columns]
    if overlap:
        lookup = lookup.drop(columns=overlap)
        feats = [c for c in feats if c not in overlap]
    if not feats:
        return df, []
    merged = df.merge(lookup, on=id_col, how="left")
    matched = merged[feats[0]].notna().mean() * 100 if feats else 0.0
    print(f"  [enrichment] matched {matched:.1f}% of rows to the extract")
    return merged, feats
