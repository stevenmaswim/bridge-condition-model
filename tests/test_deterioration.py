import numpy as np
import pandas as pd
import pytest

from src.deterioration import (
    parse_inspection_date,
    build_inspection_events,
    build_forward_pairs,
    fit_encoders,
    apply_encoders,
    forecast_bridge,
    build_watchlist,
)


def _config():
    return {
        "data": {"id_col": "bridge_id", "inspection_date_col": "inspection_date"},
        "targets": ["deck_cond_rating"],
        "model": {"test_size": 0.3, "params": {"random_state": 42}},
        "deterioration": {"hybrid_threshold_years": 3.0, "max_horizon": 25, "pair_cap": 1000},
    }


def test_parse_inspection_date_mmddyyyy():
    s = pd.Series([3211991, 12281994, np.nan])  # 7-digit and 8-digit forms
    out = parse_inspection_date(s)
    assert out.iloc[0] == pd.Timestamp("1991-03-21")
    assert out.iloc[1] == pd.Timestamp("1994-12-28")
    assert pd.isna(out.iloc[2])


def test_build_inspection_events_collapses_carried_forward():
    # bridge A has a duplicated (id, date) row (carried-forward annual snapshot)
    df = pd.DataFrame({
        "bridge_id": ["A", "A", "A", "B"],
        "inspection_date": [3211991, 3211991, 5152000, 6011995],
        "deck_cond_rating": [7, 7, 6, 8],
    })
    ev = build_inspection_events(df, _config())
    assert len(ev) == 3  # A collapses to 2 events, B has 1
    assert ev[ev["bridge_id"] == "A"].shape[0] == 2


def test_build_forward_pairs_horizon_and_ratings():
    df = pd.DataFrame({
        "bridge_id": ["A", "A", "B"],
        "inspection_date": [3211991, 5152000, 6011995],
        "deck_cond_rating": [7, 6, 8],
    })
    ev = build_inspection_events(df, _config())
    pairs = build_forward_pairs(ev, _config(), "deck_cond_rating")
    # only bridge A (2 events) yields a forward pair
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["r0"] == 7 and row["r1"] == 6
    assert 9.0 < row["horizon"] < 9.4  # 1991-03 -> 2000-05 is ~9.15y


def test_encoders_roundtrip_and_unseen():
    train = pd.DataFrame({"c": ["x", "y", "x"]})
    enc = fit_encoders(train, ["c"])
    out = apply_encoders(pd.DataFrame({"c": ["x", "y", "z"]}), enc)
    assert out["c"].iloc[0] == enc["c"]["x"]
    assert out["c"].iloc[1] == enc["c"]["y"]
    assert pd.isna(out["c"].iloc[2])  # unseen -> NaN


class _StubModel:
    """Returns a fixed prediction so we can test the hybrid routing without a trained model."""
    def predict(self, X):
        return np.full(len(X), 4.2)


def _bridge_df():
    return pd.DataFrame({
        "bridge_id": ["A", "A"],
        "inspection_date": [5152010, 5152020],
        "deck_cond_rating": [8, 7],
        "year_built": [1990, 1990],
    })


def _models():
    return {"deck_cond_rating": {
        "model": _StubModel(), "encoders": {}, "feature_cols": ["r0", "horizon", "age_t0"],
    }}


def test_forecast_hybrid_long_horizon_uses_model():
    res = forecast_bridge(_bridge_df(), _config(), _models(), "A", 2040)
    r = res["deck_cond_rating"]
    assert r["method"] == "deterioration"
    assert r["prediction"] == 4.2
    assert r["last_inspection_year"] == 2020


def test_forecast_hybrid_short_horizon_carries_forward():
    res = forecast_bridge(_bridge_df(), _config(), _models(), "A", 2022)  # +2y <= threshold
    r = res["deck_cond_rating"]
    assert r["method"] == "carry_forward"
    assert r["prediction"] == 7  # last rating held


def test_forecast_past_year_returns_current():
    res = forecast_bridge(_bridge_df(), _config(), _models(), "A", 2018)  # before last inspection
    assert res["deck_cond_rating"]["method"] == "current"


def test_forecast_unknown_bridge_returns_none():
    assert forecast_bridge(_bridge_df(), _config(), _models(), "ZZZ", 2040) is None


class _StubConservative:
    def predict(self, X):
        return np.full(len(X), 4.0)  # a fixed pessimistic forecast


class _StubRisk:
    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 0.3), np.full(n, 0.7)])  # P(poor) = 0.7 -> 70%


def _models_with_conservative():
    return {"deck_cond_rating": {
        "model": _StubModel(), "model_conservative": _StubConservative(), "model_risk": _StubRisk(),
        "encoders": {}, "feature_cols": ["r0", "horizon", "age_t0"],
    }}


def _wl_config():
    c = _config()
    c["grouping"] = {"district_col": "txdot_district", "climate_zone_col": "climate_zone"}
    return c


def test_forecast_includes_conservative_and_risk():
    res = forecast_bridge(_bridge_df(), _config(), _models_with_conservative(), "A", 2040)
    r = res["deck_cond_rating"]
    assert r["prediction"] == 4.2
    assert r["prediction_conservative"] == 4.0  # from the conservative model
    assert r["risk_poor_pct"] == 70             # from the risk model (P=0.7)


def test_build_watchlist_filters_to_sweet_spot_and_flags_poor():
    # A: rated 6 (in range), conservative forecast 4.0 <= threshold -> INCLUDE
    # B: rated 8 (too good) -> exclude;  C: rated 3 (already poor) -> exclude
    df = pd.DataFrame({
        "bridge_id": ["A", "B", "C"],
        "inspection_date": [5152020, 5152020, 5152020],
        "deck_cond_rating": [6, 8, 3],
        "year_built": [1980, 1980, 1980],
        "txdot_district": ["12", "12", "12"],
    })
    wl = build_watchlist(df, _wl_config(), _models_with_conservative(),
                         target="deck_cond_rating", horizon=10)
    assert list(wl["bridge_id"]) == ["A"]
    assert wl.iloc[0]["forecast_conservative"] == 4.0
    assert wl.iloc[0]["current_rating"] == 6
    assert wl.iloc[0]["risk_poor_pct"] == 70  # risk model surfaced in the watch-list


def test_build_watchlist_missing_target_raises():
    with pytest.raises(ValueError):
        build_watchlist(_bridge_df(), _wl_config(), _models_with_conservative(),
                        target="culvert_cond_rating")  # not present in the models dict


def test_build_watchlist_district_no_match_is_empty():
    df = pd.DataFrame({
        "bridge_id": ["A"], "inspection_date": [5152020], "deck_cond_rating": [6],
        "year_built": [1980], "txdot_district": ["12"],
    })
    wl = build_watchlist(df, _wl_config(), _models_with_conservative(),
                         target="deck_cond_rating", district="99")
    assert wl.empty

def test_parse_inspection_date_accepts_native_dates_and_iso_strings():
    """Regression: the parser used to assume integer MMDDYYYY and silently NaT'd every row of a
    source that returns real dates. Because a NaT date is dropped rather than raised, that
    failure was invisible until the model had no training pairs -- CORE_SNBI_DATA hit it with
    0 of 6,865 rows parsed. Both shapes must work, including mixed."""
    packed = parse_inspection_date(pd.Series([3211991, 10211991]))
    assert list(packed.dt.strftime("%Y-%m-%d")) == ["1991-03-21", "1991-10-21"]

    iso = parse_inspection_date(pd.Series(["2024-10-21", "2022-11-22"]))
    assert list(iso.dt.strftime("%Y-%m-%d")) == ["2024-10-21", "2022-11-22"]

    native = parse_inspection_date(pd.to_datetime(pd.Series(["2024-10-21"])))
    assert native.iloc[0].strftime("%Y-%m-%d") == "2024-10-21"

    mixed = parse_inspection_date(pd.Series([3211991, "2024-10-21", "", None, "junk"]))
    assert mixed.notna().sum() == 2
    assert mixed.iloc[0].strftime("%Y-%m-%d") == "1991-03-21"
    assert mixed.iloc[1].strftime("%Y-%m-%d") == "2024-10-21"
