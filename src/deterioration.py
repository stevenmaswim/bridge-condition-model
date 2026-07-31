"""Deterioration model: predict a bridge's FUTURE condition rating from its current rating, how
many years ahead, its age, and its physical attributes.

Why this exists (validated on real data, 2026-07-28): the panel is annual NBI snapshots, but
bridges are re-inspected on a ~2-year cycle, so ~half the rows are carried-forward duplicates. A
naive "carry forward last rating" rule is unbeatable at short horizons but degrades badly past ~10
years -- exactly the capital-planning range. A model trained on real inspection-to-inspection
transitions cuts long-horizon error ~30-40% and is the honest basis for forecasting.

Pipeline:
  clean panel -> build_inspection_events (collapse carried-forward rows to real inspections)
              -> build_forward_pairs (t0 -> t1 with horizon, r0, r1)
              -> attach static physical features (src/enrichment.py)
              -> train per target, evaluate BY HORIZON vs carry-forward + age-curve baselines.

Kept alongside the attributes-only model (src/model.py), which handles bridges with no history.
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor, XGBClassifier

from src.baselines import compute_metrics
from src.enrichment import attach_static_features

# Feature sets (module defaults; override via config.deterioration.{numeric,categorical}).
# High-cardinality free-text panel fields (facility_carried, features_intersected) are deliberately
# excluded -- as raw codes they are noise. approach_roadway_width is treated as numeric.
DET_NUMERIC = [
    "structure_length", "max_span_length", "deck_width", "roadway_width", "adt", "adt_year",
    "skew_angle", "latitude", "longitude", "inventory_load_rating_factor",
    "operating_load_rating_factor", "approach_roadway_width",
]
DET_CATEGORICAL = ["owner", "maintenance_resp", "design_load", "txdot_district", "functional_class"]

# Horizon buckets for honest per-horizon reporting.
HORIZON_BINS = [0, 3.5, 7.5, 12.5, 25]
HORIZON_LABELS = ["~2y", "~5y", "~10y", "~20y"]


def parse_inspection_date(series):
    """SNBI B.IE.02 is an integer MMDDYYYY (e.g. 3211991 -> 1991-03-21), NOT a native date.
    Zero-pad to 8 chars and parse explicitly; a naive to_datetime treats it as nanoseconds."""
    s = series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(8)
    return pd.to_datetime(s, format="%m%d%Y", errors="coerce")


def build_inspection_events(df, config):
    """Collapse carried-forward annual rows into one row per real inspection (bridge, inspection
    date), carrying the ratings and features recorded at that inspection."""
    id_col = config["data"]["id_col"]
    insp_col = config["data"]["inspection_date_col"]
    ev = df.copy()
    ev[insp_col] = parse_inspection_date(ev[insp_col])
    ev = ev.dropna(subset=[id_col, insp_col])
    ev = ev.drop_duplicates(subset=[id_col, insp_col]).sort_values([id_col, insp_col])
    return ev.reset_index(drop=True)


def _feature_lists(config, columns):
    det = config.get("deterioration", {}) or {}
    numeric = [c for c in det.get("numeric", DET_NUMERIC) if c in columns]
    categorical = [c for c in det.get("categorical", DET_CATEGORICAL) if c in columns]
    return numeric, categorical


def build_forward_pairs(events, config, target, max_horizon=None, cap=None, random_state=42):
    """Build forward inspection pairs (t0 -> t1) for one target. Returns a DataFrame with r0
    (current rating), r1 (future rating), horizon (years), age_t0, plus t0 features and the
    joined static physical features."""
    det = config.get("deterioration", {}) or {}
    max_horizon = max_horizon if max_horizon is not None else det.get("max_horizon", 25)
    cap = cap if cap is not None else det.get("pair_cap", 1_200_000)
    id_col = config["data"]["id_col"]
    insp_col = config["data"]["inspection_date_col"]
    yb = "year_built"
    numeric, categorical = _feature_lists(config, events.columns)
    t0_feats = numeric + categorical + ([yb] if yb in events.columns else [])

    ed = events.copy()
    ed[target] = pd.to_numeric(ed[target], errors="coerce")
    ed = ed.dropna(subset=[target])

    left = ed[[id_col, insp_col, target] + t0_feats].rename(columns={insp_col: "d0", target: "r0"})
    right = ed[[id_col, insp_col, target]].rename(columns={insp_col: "d1", target: "r1"})
    pairs = left.merge(right, on=id_col)
    pairs = pairs[pairs["d1"] > pairs["d0"]].copy()
    pairs["horizon"] = (pairs["d1"] - pairs["d0"]).dt.days / 365.25
    pairs = pairs[(pairs["horizon"] > 0) & (pairs["horizon"] <= max_horizon)]
    if cap and len(pairs) > cap:
        pairs = pairs.sample(cap, random_state=random_state)
    if yb in pairs.columns:
        pairs["age_t0"] = (pairs["d0"].dt.year - pd.to_numeric(pairs[yb], errors="coerce")).clip(0, 130)
    else:
        pairs["age_t0"] = np.nan

    pairs, _ = attach_static_features(pairs, config)
    return pairs


def fit_encoders(df, cols):
    """Map each categorical value to an integer code; unseen values at predict time -> NaN
    (handled natively by XGBoost). Persisted with the model for forecast-time consistency."""
    enc = {}
    for c in cols:
        vals = df[c].astype(str).fillna("NA").unique()
        enc[c] = {v: i for i, v in enumerate(vals)}
    return enc


def apply_encoders(df, encoders):
    df = df.copy()
    for c, mapping in encoders.items():
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("NA").map(mapping).astype("float")
    return df


def predict_with_bundle(bundle, frame):
    """Encode, align to the bundle's feature_cols, and predict for a frame that already has r0,
    horizon, and age_t0 set. Returns (most_likely, conservative, risk_pct):
      * most_likely  -- point forecast of the future rating,
      * conservative -- lower-quantile "plan-for" forecast, clamped <= most_likely,
      * risk_pct     -- P(future rating <= risk_threshold) as a 0-100 percentage (NaN if no risk
                        model in the bundle).
    Single source of truth for the serve-time prediction path
    (forecast_bridge / build_watchlist / forecast_ui)."""
    enc = apply_encoders(frame, bundle["encoders"])
    X = enc.reindex(columns=bundle["feature_cols"])
    for c in bundle["feature_cols"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    likely = np.clip(bundle["model"].predict(X), 0, 9)
    mc = bundle.get("model_conservative")
    conservative = np.clip(mc.predict(X), 0, 9) if mc is not None else likely.copy()
    conservative = np.minimum(conservative, likely)
    mr = bundle.get("model_risk")
    risk = mr.predict_proba(X)[:, 1] * 100 if mr is not None else np.full(len(X), np.nan)
    return likely, conservative, risk


def _model_feature_columns(pairs, config, static_feats):
    numeric, categorical = _feature_lists(config, pairs.columns)
    static_num = [c for c in static_feats if pd.api.types.is_numeric_dtype(pairs[c])]
    static_cat = [c for c in static_feats if c not in static_num]
    numeric_cols = numeric + static_num + ["r0", "horizon", "age_t0"]
    categorical_cols = categorical + static_cat
    return numeric_cols, categorical_cols


def train_deterioration_target(pairs, config, target, params=None):
    """Group-split by bridge, train XGBoost to predict r1, and evaluate BY HORIZON against
    carry-forward and age-curve baselines. Returns (model, encoders, feature_cols, metrics_df)."""
    id_col = config["data"]["id_col"]
    # static_feats = the enrichment columns = everything not a bookkeeping/derived column and not
    # already claimed as a panel feature. Derive the panel set from the CONFIG-resolved lists (not
    # the module constants) so a config override can't place a column in both panel and static ->
    # duplicate feature column.
    panel_numeric, panel_categorical = _feature_lists(config, pairs.columns)
    reserved = {id_col, "d0", "d1", "r0", "r1", "horizon", "age_t0", "year_built"}
    reserved |= set(panel_numeric) | set(panel_categorical)
    static_feats = [c for c in pairs.columns if c not in reserved]
    numeric_cols, categorical_cols = _model_feature_columns(pairs, config, static_feats)
    feature_cols = numeric_cols + categorical_cols

    encoders = fit_encoders(pairs, categorical_cols)
    enc = apply_encoders(pairs, encoders)

    gss = GroupShuffleSplit(n_splits=1, test_size=config["model"]["test_size"],
                            random_state=config["model"]["params"]["random_state"])
    tr_idx, te_idx = next(gss.split(enc, groups=enc[id_col]))
    train, test = enc.iloc[tr_idx], enc.iloc[te_idx]

    p = dict(params or config["model"]["params"])
    p.pop("random_state", None)
    model = XGBRegressor(random_state=config["model"]["params"]["random_state"], n_jobs=-1, **p)
    model.fit(train[feature_cols], train["r1"])

    # Conservative "plan-for-this" model: a lower-quantile forecast. Point forecasts under-flag the
    # rare fast decliners (those drops are events, not trends, so no feature fixes them); a low
    # quantile deliberately errs toward worse condition so budget planning catches more of them.
    q = (config.get("deterioration", {}) or {}).get("conservative_quantile", 0.25)
    model_conservative = XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=q,
        random_state=config["model"]["params"]["random_state"], n_jobs=-1, **p)
    model_conservative.fit(train[feature_cols], train["r1"])

    # Risk model: a calibrated P(future rating <= poor_threshold) classifier. Directly models the
    # budget-relevant event and ranks priorities better than the point forecast (validated: AUC 0.93,
    # top-1% precision 86% vs 81% for the regression ranking, 45% naive).
    poor = (config.get("deterioration", {}) or {}).get("risk_threshold", 5.0)
    model_risk = None
    y_poor = (train["r1"] <= poor).astype(int)
    if y_poor.nunique() > 1:  # need both classes present to train
        model_risk = XGBClassifier(
            eval_metric="logloss", random_state=config["model"]["params"]["random_state"],
            n_jobs=-1, **p)
        model_risk.fit(train[feature_cols], y_poor)

    test = test.copy()
    test["pred"] = np.clip(model.predict(test[feature_cols]), 0, 9)
    test["carry"] = test["r0"]
    age_means = train.assign(ab=(train["age_t0"] + train["horizon"]).round()).groupby("ab")["r1"].mean()
    gmean = train["r1"].mean()
    test["age_pred"] = (test["age_t0"] + test["horizon"]).round().map(age_means).fillna(gmean)
    test["hb"] = pd.cut(test["horizon"], HORIZON_BINS, labels=HORIZON_LABELS)

    rows = []
    for hb, g in test.groupby("hb", observed=True):
        for name, col in [("deterioration", "pred"), ("carry_forward", "carry"), ("age_curve", "age_pred")]:
            m = compute_metrics(g["r1"], g[col])
            rows.append({"target": target, "horizon": str(hb), "method": name, "n": len(g),
                         "mae": round(m["mae"], 3), "within_1": round(m["within_1"], 1),
                         "r2": round(m["r2"], 3)})
    metrics_df = pd.DataFrame(rows)
    return model, model_conservative, model_risk, encoders, feature_cols, metrics_df


def train_all_deterioration(df, config):
    """Train a deterioration model per target from a cleaned panel. Returns
    (models, metrics_df) where models[target] = {model, encoders, feature_cols}."""
    events = build_inspection_events(df, config)
    print(f"[deterioration] {len(events):,} real inspection events "
          f"from {events[config['data']['id_col']].nunique():,} bridges")
    models, all_metrics = {}, []
    for target in config["targets"]:
        if target not in events.columns:
            continue
        pairs = build_forward_pairs(events, config, target)
        if len(pairs) < 100:
            print(f"[deterioration] {target}: only {len(pairs)} pairs, skipping")
            continue
        model, model_cons, model_risk, encoders, feats, metrics = train_deterioration_target(
            pairs, config, target)
        models[target] = {"model": model, "model_conservative": model_cons, "model_risk": model_risk,
                          "encoders": encoders, "feature_cols": feats}
        all_metrics.append(metrics)
        ten = metrics[(metrics["horizon"] == "~10y")].set_index("method")
        if "deterioration" in ten.index and "carry_forward" in ten.index:
            print(f"[deterioration] {target} @10y: model MAE={ten.loc['deterioration','mae']} "
                  f"within1={ten.loc['deterioration','within_1']}%  "
                  f"vs carry-forward MAE={ten.loc['carry_forward','mae']}")
    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    return models, metrics_df


def forecast_bridge(df_clean, config, models, nbi_code, target_year, hybrid_threshold=None):
    """Forecast one bridge's condition ratings in target_year using the HYBRID rule:
    carry the last inspection rating forward for near-term horizons, use the deterioration model
    for long-term. Returns {target: {method, prediction, current_rating, last_inspection_year,
    horizon_years}} or None if the bridge isn't found. df_clean is a cleaned panel (post clean_data)."""
    if hybrid_threshold is None:
        hybrid_threshold = (config.get("deterioration", {}) or {}).get("hybrid_threshold_years", 3.0)
    id_col = config["data"]["id_col"]
    insp_col = config["data"]["inspection_date_col"]

    events = build_inspection_events(df_clean, config)
    b = events[events[id_col].astype(str).str.strip() == str(nbi_code).strip()]
    if b.empty:
        return None
    latest = b.sort_values(insp_col).iloc[-1]
    last_year = int(latest[insp_col].year)
    horizon = target_year - last_year

    one = latest.to_frame().T
    one, _ = attach_static_features(one, config)
    yb = pd.to_numeric(pd.Series([latest.get("year_built")]), errors="coerce").iloc[0]
    age_t0 = float(np.clip(last_year - yb, 0, 130)) if pd.notna(yb) else np.nan

    results = {}
    for target, bundle in models.items():
        r0 = pd.to_numeric(pd.Series([latest.get(target)]), errors="coerce").iloc[0]
        if pd.isna(r0):
            continue  # this member type doesn't exist on this bridge (e.g. no deck on a culvert)
        conservative = None
        risk = float("nan")
        if horizon <= 0:
            method, pred = "current", float(r0)
        elif horizon <= hybrid_threshold:
            method, pred = "carry_forward", float(r0)
        else:
            row = one.copy()
            row["r0"], row["horizon"], row["age_t0"] = float(r0), float(horizon), age_t0
            likely_arr, cons_arr, risk_arr = predict_with_bundle(bundle, row)
            pred = float(likely_arr[0])
            conservative = float(cons_arr[0])
            risk = float(risk_arr[0])
            method = "deterioration"
        if conservative is None:
            conservative = float(r0)  # near-term / no quantile model: worse-case == last rating
        results[target] = {
            "method": method, "prediction": round(pred, 1),
            "prediction_conservative": round(conservative, 1),
            "risk_poor_pct": None if pd.isna(risk) else round(risk, 0),
            "current_rating": float(r0),
            "last_inspection_year": last_year, "horizon_years": round(float(horizon), 1),
        }
    return results


def build_watchlist(df_clean, config, models, target="deck_cond_rating", horizon=10,
                    district=None, current_min=5, current_max=7, poor_threshold=5.0,
                    on_system_only=True):
    """Budget watch-list: forecast every bridge's `target` rating `horizon` years from its latest
    inspection and return the bridges worth funding attention, ranked worst-first.

    Usage rules baked in (from the SME review): rank by FORECAST condition, not by size of decline
    (decline-ranking just surfaces healthy new bridges). Keep bridges currently rated
    [current_min, current_max] whose CONSERVATIVE forecast reaches <= poor_threshold; exclude
    already-poor bridges (inspect those directly) and new/high-rated bridges (normal aging)."""
    id_col = config["data"]["id_col"]
    insp_col = config["data"]["inspection_date_col"]
    dist_col = config["grouping"]["district_col"]
    if target not in models:
        raise ValueError(f"no deterioration model loaded for {target}")
    bundle = models[target]

    events = build_inspection_events(df_clean, config)
    latest = events.sort_values(insp_col).groupby(id_col, as_index=False).tail(1).copy()
    latest[target] = pd.to_numeric(latest[target], errors="coerce")
    latest = latest.dropna(subset=[target])
    system_col = config.get("data", {}).get("system_col")
    if on_system_only and system_col and system_col in latest.columns:
        latest = latest[latest[system_col].astype(str).str.upper().str.startswith("ON")]
    if district is not None and dist_col in latest.columns:
        latest = latest[latest[dist_col].astype(str).str.strip() == str(district).strip()]
    if latest.empty:
        return pd.DataFrame()

    latest["district"] = latest[dist_col] if dist_col in latest.columns else None
    latest["last_inspection_year"] = latest[insp_col].dt.year
    latest["r0"] = latest[target]
    latest["horizon"] = float(horizon)
    latest["age_t0"] = (latest["last_inspection_year"]
                        - pd.to_numeric(latest["year_built"], errors="coerce")).clip(0, 130)
    latest, _ = attach_static_features(latest, config)

    likely, conservative, risk = predict_with_bundle(bundle, latest)
    latest["forecast_most_likely"] = np.round(likely, 1)
    latest["forecast_conservative"] = np.round(conservative, 1)
    latest["risk_poor_pct"] = np.round(risk, 0)
    latest["predicted_decline"] = (latest["r0"] - latest["forecast_most_likely"]).round(1)

    wl = latest[(latest["r0"] >= current_min) & (latest["r0"] <= current_max)
                & (latest["forecast_conservative"] <= poor_threshold)].copy()
    # rank by the risk model (best budget ranker: P@1% 86% vs 81% for the forecast); fall back to
    # the conservative forecast where the risk model is unavailable.
    wl = wl.sort_values(["risk_poor_pct", "forecast_conservative"], ascending=[False, True],
                        na_position="last")
    cols = [id_col, "district", "year_built", "last_inspection_year", "r0", "risk_poor_pct",
            "forecast_most_likely", "forecast_conservative", "predicted_decline"]
    return wl[cols].rename(columns={"r0": "current_rating"})


def save_deterioration_models(models, config):
    out_dir = os.path.join(config["output"]["model_dir"], "deterioration")
    os.makedirs(out_dir, exist_ok=True)
    for target, bundle in models.items():
        with open(os.path.join(out_dir, f"{target}_deterioration.pkl"), "wb") as f:
            pickle.dump(bundle, f)
    print(f"[deterioration] saved {len(models)} models to {out_dir}/")


def load_deterioration_model(target, config):
    path = os.path.join(config["output"]["model_dir"], "deterioration", f"{target}_deterioration.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)
