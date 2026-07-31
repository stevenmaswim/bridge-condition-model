import numpy as np
import pandas as pd

from src.data_loader import clean_data, rename_raw_columns


def _config():
    return {
        "data": {"id_col": "bridge_id", "inspection_date_col": "inspection_date",
                 "column_rename_map": {"B.C.01: DECK": "deck_cond_rating"}},
        "targets": ["deck_cond_rating", "culvert_cond_rating"],
        "features": {"numeric": ["adt"], "categorical": ["owner"]},
    }


def test_rename_raw_columns_maps_only_listed():
    df = pd.DataFrame({"B.C.01: DECK": [7], "other": [1]})
    out = rename_raw_columns(df, _config())
    assert "deck_cond_rating" in out.columns
    assert "other" in out.columns  # unlisted columns untouched


def test_clean_data_keeps_id_and_inspection_date_and_drops_unlisted():
    cfg = _config()
    df = pd.DataFrame({"bridge_id": ["A"], "inspection_date": [5152010], "deck_cond_rating": [7],
                       "culvert_cond_rating": [np.nan], "adt": ["5000"], "owner": ["State"], "extra": [1]})
    out = clean_data(df, cfg)
    assert "bridge_id" in out.columns and "inspection_date" in out.columns
    assert "extra" not in out.columns                 # non-configured column dropped
    assert pd.api.types.is_numeric_dtype(out["adt"])  # numeric feature coerced


def test_clean_data_coerces_N_and_drops_all_nan_target_rows():
    cfg = _config()
    df = pd.DataFrame({
        "bridge_id": ["A", "B"], "inspection_date": [5152010, 5152010],
        "deck_cond_rating": ["N", "7"],       # A: N -> NaN
        "culvert_cond_rating": ["N", "N"],    # both N -> NaN
        "adt": ["1", "2"], "owner": ["State", "City"],
    })
    out = clean_data(df, cfg)
    assert len(out) == 1                       # row A had all-NaN targets -> dropped
    assert out.iloc[0]["deck_cond_rating"] == 7
