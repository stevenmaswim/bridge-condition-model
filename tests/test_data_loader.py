import numpy as np
import pandas as pd

from src.data_loader import clean_data, rename_raw_columns, normalize_legacy_encodings


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


def test_normalize_decodes_packed_legacy_encodings():
    """The live history table packs coordinates as DDMMSSss and load ratings x100, while the
    CSV export uses plain decimals. Feeding packed values to a fitted tree ensemble raises
    nothing -- they fall off the end of every learned split and the feature silently stops
    contributing -- so the decode has to happen before the model ever sees them."""
    live = pd.DataFrame({
        "latitude": [29153429.0], "longitude": [95210096.0],
        "inventory_load_rating_factor": [227.0], "operating_load_rating_factor": [236.0]})
    out = normalize_legacy_encodings(live, verbose=False)
    assert round(out["latitude"].iloc[0], 4) == 29.2595
    assert round(out["longitude"].iloc[0], 4) == -95.3503     # western hemisphere, sign restored
    assert out["inventory_load_rating_factor"].iloc[0] == 2.27
    assert out["operating_load_rating_factor"].iloc[0] == 2.36


def test_normalize_leaves_already_decimal_values_alone():
    """Guarded on magnitude, so running it on a source that is already in the right units --
    which is what the training CSV is -- must change nothing."""
    csv = pd.DataFrame({
        "latitude": [31.32], "longitude": [-97.17],
        "inventory_load_rating_factor": [1.21], "operating_load_rating_factor": [1.65]})
    out = normalize_legacy_encodings(csv.copy(), verbose=False)
    pd.testing.assert_frame_equal(out, csv)


def test_normalize_handles_both_packed_widths_and_drops_the_undecodable():
    """The export does not use one packed width. 29153429 is DDMMSSss; 291534 is the same
    coordinate with the hundredths dropped. Decoding the short form as if it were the long one
    yields 0 deg 29' 15.34" = 0.49 -- a plausible-looking float thirty degrees from Texas, which
    is exactly the kind of value a fitted model cannot flag. Anything that will not decode into
    the state becomes NaN, which the tree ensemble handles natively."""
    df = pd.DataFrame({
        "latitude":  [29153429.0, 291534.0, 29.2595, 0.0, None, 12345678.0],
        "longitude": [95210096.0, 952100.0, -95.35,  0.0, None, 999.0]})
    out = normalize_legacy_encodings(df, verbose=False)

    # the two widths differ only by the dropped hundredths of an arcsecond -- about 9 m
    assert abs(out["latitude"].iloc[0] - out["latitude"].iloc[1]) < 1e-3
    assert all(round(out["latitude"].iloc[i], 2) == 29.26 for i in (0, 1, 2))
    assert round(out["longitude"].iloc[0], 2) == round(out["longitude"].iloc[1], 2) == -95.35
    # 0 means "not recorded", and neither the out-of-state nor the unparseable value survives
    assert out.loc[3:5, "latitude"].isna().all()
    assert out.loc[3:5, "longitude"].isna().all()


def test_normalize_tolerates_missing_columns_and_nulls():
    out = normalize_legacy_encodings(pd.DataFrame({"latitude": [None, 29153429.0]}), verbose=False)
    assert pd.isna(out["latitude"].iloc[0])
    assert round(out["latitude"].iloc[1], 4) == 29.2595
