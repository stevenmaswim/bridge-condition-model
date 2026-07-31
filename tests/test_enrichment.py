import os

import numpy as np
import pandas as pd
import pytest

from src.enrichment import (
    EXTRACT_COLUMNS,
    load_static_features,
    attach_static_features,
)


def _write_headerless_extract(path, rows):
    """Write a CSV with no header row, one field per EXTRACT_COLUMNS position."""
    df = pd.DataFrame(rows, columns=EXTRACT_COLUMNS)
    df.to_csv(path, index=False, header=False)


def _make_row(as_code, **overrides):
    row = {c: "NULL" for c in EXTRACT_COLUMNS}
    row["as_code"] = as_code
    row.update(overrides)
    return row


def _config(path):
    return {
        "data": {"id_col": "bridge_id"},
        "enrichment": {"extract_file": path, "has_header": False},
    }


def test_extract_columns_count():
    # Must match the 41-column SELECT in sql/bridge_data_extract.sql
    assert len(EXTRACT_COLUMNS) == 41
    assert EXTRACT_COLUMNS[1] == "as_code"


def test_load_static_features_headerless(tmp_path):
    path = os.path.join(tmp_path, "extract.csv")
    _write_headerless_extract(path, [
        _make_row("0001", num_spans_main="4", deck_type="C01", scour_vulnerability="A"),
        _make_row("0002", num_spans_main="2", deck_type="0", scour_vulnerability="B"),
    ])
    lookup = load_static_features(_config(path))
    assert lookup is not None
    assert "bridge_id" in lookup.columns
    assert set(lookup["bridge_id"]) == {"0001", "0002"}
    # numeric static features are coerced
    assert pd.api.types.is_numeric_dtype(lookup["num_spans_main"])
    assert lookup.set_index("bridge_id").loc["0001", "num_spans_main"] == 4


def test_null_becomes_nan(tmp_path):
    path = os.path.join(tmp_path, "extract.csv")
    _write_headerless_extract(path, [_make_row("0001", deck_type="NULL")])
    lookup = load_static_features(_config(path))
    assert pd.isna(lookup.set_index("bridge_id").loc["0001", "deck_type"])


def test_missing_file_returns_none():
    cfg = _config("does/not/exist.csv")
    assert load_static_features(cfg) is None


def test_load_static_features_with_header(tmp_path):
    # headered export: names come from the header row, not positional EXTRACT_COLUMNS
    path = os.path.join(tmp_path, "extract_headered.csv")
    pd.DataFrame([_make_row("0001", num_spans_main="4", deck_type="C01")],
                 columns=EXTRACT_COLUMNS).to_csv(path, index=False, header=True)
    cfg = {"data": {"id_col": "bridge_id"},
           "enrichment": {"extract_file": path, "has_header": True}}
    lookup = load_static_features(cfg)
    assert lookup is not None
    assert set(lookup["bridge_id"]) == {"0001"}
    assert lookup.set_index("bridge_id").loc["0001", "num_spans_main"] == 4


def test_attach_static_features_unmatched_is_nan(tmp_path):
    path = os.path.join(tmp_path, "extract.csv")
    _write_headerless_extract(path, [_make_row("0001", num_spans_main="4")])
    panel = pd.DataFrame({"bridge_id": ["0001", "0002"], "x": [1, 2]})
    merged, feats = attach_static_features(panel, _config(path))
    assert "num_spans_main" in feats
    m = merged.set_index("bridge_id")
    assert m.loc["0001", "num_spans_main"] == 4
    assert pd.isna(m.loc["0002", "num_spans_main"])  # bridge not in extract


def test_attach_no_config_is_noop():
    panel = pd.DataFrame({"bridge_id": ["0001"], "x": [1]})
    merged, feats = attach_static_features(panel, {"data": {"id_col": "bridge_id"}})
    assert feats == []
    assert merged.equals(panel)
