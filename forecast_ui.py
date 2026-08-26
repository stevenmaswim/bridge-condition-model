"""Generate a self-contained TxDOT-branded HTML forecast report for a set of bridges.

Runs the real deterioration model once over the requested bridges, precomputes each bridge's
forecast trajectory (most-likely + conservative + risk) at several horizons for every condition
rating it has, and embeds it in a single offline HTML file with:

  * an executive KPI band built from the validation exhibits on disk,
  * a bridge explorer (search, horizon slider, deterioration curve) with a live side rail that
    explains how the number on screen was actually produced,
  * a methodology & validation appendix.

Presentation lives in src/report_template.py; this module is data assembly + CLI.

Examples:
  python forecast_ui.py --codes 120200152401017,121700017714043
  python forecast_ui.py --file my_bridges.txt          # one NBI code per line
  python forecast_ui.py --district 12                  # every bridge in a district
"""
import argparse
import datetime as dt
import html
import json
import os
import re

import numpy as np
import pandas as pd

from src.data_loader import load_config, load_raw_data, rename_raw_columns, clean_data
from src.deterioration import build_inspection_events, predict_with_bundle, load_deterioration_model
from src.enrichment import attach_static_features
from src.report_template import BRAND_CSS, HTML_TEMPLATE, LOGO_SVG

HORIZONS = [0, 2, 5, 10, 15, 20]

# r0 / horizon / age_t0 are supplied by the forecast call itself, not by the bridge record, so
# they are reported separately in the side rail and excluded from the attribute-coverage count.
RUNTIME_FEATURES = ("r0", "horizon", "age_t0")

# Plain-English feature names. Mirrors prettyFeat() in report_template.py -- the appendix table
# and the side rail must not disagree about what a feature is called. "r0"/"age_t0" are the
# training-code names and mean nothing to a reader.
PRETTY_FEATURE = {
    "r0": "Rating at last inspection", "horizon": "Years ahead",
    "age_t0": "Age at last inspection", "load_posting_status": "Load posting status",
    "span_continuity": "Span continuity", "wearing_surface": "Wearing surface",
    "structure_kind": "Structure material", "structure_type": "Structure type",
    "deck_type": "Deck type", "deck_protection": "Deck protection",
    "inventory_load_rating_factor": "Inventory load rating",
    "operating_load_rating_factor": "Operating load rating",
    "txdot_district": "District", "maintenance_resp": "Maintenance responsibility",
    "owner": "Owner", "num_beam_lines": "Beam lines", "num_spans_main": "Main spans",
    "adt": "Average daily traffic", "adt_truck": "Truck traffic", "adt_year": "Traffic count year",
    "deck_width": "Deck width", "roadway_width": "Roadway width",
    "max_span_length": "Max span length", "structure_length": "Structure length",
    "approach_roadway_width": "Approach width", "functional_class": "Functional class",
    "design_load": "Design load", "scour_vulnerability": "Scour vulnerability",
    "skew_angle": "Skew angle", "latitude": "Latitude", "longitude": "Longitude",
}

NICE_TARGET = {
    "deck_cond_rating": "Deck",
    "superstructure_cond_rating": "Superstructure",
    "substructure_cond_rating": "Substructure",
    "culvert_cond_rating": "Culvert",
}


# =============================================================================================
# Forecasting
# =============================================================================================
def _forecast_target(base, bundle, target, hybrid):
    """Return {row_index: {'current', 'likely':{h}, 'cons':{h}, 'risk':{h}}} for one target.
    risk = P(rating <= poor threshold) as a percent; None near-term."""
    r0 = pd.to_numeric(base[target], errors="coerce").to_numpy()
    likely = {h: np.full(len(base), np.nan) for h in HORIZONS}
    cons = {h: np.full(len(base), np.nan) for h in HORIZONS}
    risk = {h: np.full(len(base), np.nan) for h in HORIZONS}
    for h in HORIZONS:
        if h == 0 or h <= hybrid:  # near-term: carry the last rating forward (matches the hybrid rule)
            likely[h] = r0.copy()
            cons[h] = r0.copy()
            continue
        frame = base.copy()
        frame["r0"] = r0
        frame["horizon"] = float(h)
        likely[h], cons[h], risk[h] = predict_with_bundle(bundle, frame)  # conservative clamped <= likely
    out = {}
    for i in range(len(base)):
        if np.isnan(r0[i]):
            continue
        out[i] = {
            "current": round(float(r0[i]), 1),
            "likely": {h: round(float(likely[h][i]), 1) for h in HORIZONS},
            "cons": {h: round(float(cons[h][i]), 1) for h in HORIZONS},
            "risk": {h: (None if np.isnan(risk[h][i]) else round(float(risk[h][i]), 0)) for h in HORIZONS},
        }
    return out


def _display_value(v):
    """Render a feature value for the side rail: whole numbers stay whole, floats get one
    decimal, everything else passes through as text."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() else f"{float(v):.1f}"
    text = str(v).strip()
    return text or None


def load_panel(config, codes=None, district=None):
    """One pull from the configured source -> the de-duplicated inspection-event panel.

    Split out from build_data so a multi-district run pulls once and reuses the frame. Each
    call opens its own Snowflake connection, and with externalbrowser auth that means an
    interactive SSO round trip -- doing it 25 times is not a build, it is a hostage situation.
    """
    id_col = config["data"]["id_col"]
    dist_col = config["grouping"]["district_col"]
    if config.get("data", {}).get("source") == "snowflake":
        if codes:
            config["_snowflake_filter"] = {"column": id_col, "values": [str(c).strip() for c in codes]}
        elif district is not None:
            config["_snowflake_filter"] = {"column": dist_col, "values": [str(district).strip()]}
    df = clean_data(rename_raw_columns(load_raw_data(config), config), config)
    return build_inspection_events(df, config)


def build_data(config, codes=None, district=None, driver_features=(), events=None):
    """Run the model over the requested bridges. Returns (bridges, targets, meta).

    Pass `events` to reuse a panel already loaded by load_panel()."""
    id_col = config["data"]["id_col"]
    insp_col = config["data"]["inspection_date_col"]
    dist_col = config["grouping"]["district_col"]
    det_cfg = config.get("deterioration", {}) or {}
    hybrid = det_cfg.get("hybrid_threshold_years", 3.0)

    if events is None:
        events = load_panel(config, codes=codes, district=district)
    latest = events.sort_values(insp_col).groupby(id_col, as_index=False).tail(1).copy()

    if codes:
        wanted = {str(c).strip() for c in codes}
        latest = latest[latest[id_col].astype(str).str.strip().isin(wanted)]
    elif district is not None and dist_col in latest.columns:
        latest = latest[latest[dist_col].astype(str).str.strip() == str(district).strip()]
    latest = latest.reset_index(drop=True)
    if latest.empty:
        return [], [], {}

    latest["age_t0"] = (latest[insp_col].dt.year
                        - pd.to_numeric(latest["year_built"], errors="coerce")).clip(0, 130)
    base, _ = attach_static_features(latest, config)

    per_target = {}
    targets = []
    feature_cols = []
    for target in config["targets"]:
        try:
            bundle = load_deterioration_model(target, config)
        except FileNotFoundError:
            continue
        targets.append(target)
        per_target[target] = _forecast_target(base, bundle, target, hybrid)
        if not feature_cols:  # identical across targets -- built from the same feature lists
            feature_cols = list(bundle.get("feature_cols", []))

    # Attribute coverage. predict_with_bundle reindexes onto feature_cols, so an attribute the
    # record does not carry silently becomes an all-NaN column and the forecast is still produced
    # -- from less information. Counting it here is what lets the side rail say so.
    attr_cols = [c for c in feature_cols if c not in RUNTIME_FEATURES]
    present_mask = pd.DataFrame(
        {c: (base[c].notna() if c in base.columns else pd.Series(False, index=base.index))
         for c in attr_cols},
        index=base.index) if attr_cols else pd.DataFrame(index=base.index)

    value_cols = [c for c in driver_features if c in base.columns and c not in RUNTIME_FEATURES]

    bridges = []
    for i, row in latest.iterrows():
        entry = {
            "id": str(row[id_col]),
            "district": (str(row[dist_col]) if dist_col in latest.columns and pd.notna(row[dist_col])
                         else ""),
            "year_built": (int(row["year_built"]) if pd.notna(row.get("year_built")) else None),
            "last_year": int(row[insp_col].year),
            "last_inspection": row[insp_col].strftime("%Y-%m-%d"),
            "age_t0": (int(row["age_t0"]) if pd.notna(row.get("age_t0")) else None),
            "targets": {},
        }
        if attr_cols:
            missing = [k for k, c in enumerate(attr_cols) if not bool(present_mask.at[i, c])]
            entry["feat_total"] = len(attr_cols)
            entry["feat_present"] = len(attr_cols) - len(missing)
            entry["feat_missing"] = missing[:6]      # indices into meta.attr_names
        else:
            entry["feat_total"] = entry["feat_present"] = 0
            entry["feat_missing"] = []
        # positional, aligned to meta.val_keys -- a dict here repeats every key on every bridge
        entry["vals"] = [_display_value(base.at[i, c]) for c in value_cols]
        for target in targets:
            if i in per_target[target]:
                entry["targets"][target] = per_target[target][i]
        if entry["targets"]:
            bridges.append(entry)

    # On/off-system counts, classified per bridge from its latest record. Reported because
    # "how many on-system bridges is this" is the first question asked of any scoped pull, and
    # deriving it afterwards from the page is impossible -- the payload does not carry the flag.
    system_col = config.get("data", {}).get("system_col")
    n_on = n_off = None
    if system_col and system_col in latest.columns:
        sysv = latest[system_col].astype(str).str.strip().str.upper()
        n_on = int(sysv.eq("ON").sum())
        n_off = int(len(sysv) - n_on)

    meta = {
        "n_on_system": n_on,
        "n_off_system": n_off,
        "val_keys": value_cols,      # names for the positional bridge["vals"] array
        "attr_names": attr_cols,     # names for the bridge["feat_missing"] indices
        "source_label": _source_label(config),
        "generated": dt.datetime.now().strftime("%d %b %Y, %H:%M"),
        "model_trained": _model_trained_on(targets, config),
        "hybrid": hybrid,
        "risk_threshold": det_cfg.get("risk_threshold", 5.0),
        "conservative_quantile": det_cfg.get("conservative_quantile", 0.25),
        "n_bridges": len(bridges),
    }
    return bridges, targets, meta


def _source_label(config):
    data_cfg = config.get("data", {}) or {}
    if data_cfg.get("source") == "snowflake":
        sf = config.get("snowflake", {}) or {}
        table = sf.get("table") or f"{sf.get('database', '')}.{sf.get('schema', '')}"
        return f"Snowflake live · {table}"
    return f"CSV · {os.path.join(data_cfg.get('raw_dir', ''), data_cfg.get('raw_file', ''))}"


def _model_trained_on(targets, config):
    """Timestamp of the deterioration model files actually being served."""
    model_dir = os.path.join(config.get("output", {}).get("model_dir", "models"), "deterioration")
    stamps = []
    for target in targets:
        path = os.path.join(model_dir, f"{target}_deterioration.pkl")
        if os.path.exists(path):
            stamps.append(os.path.getmtime(path))
    if not stamps:
        return None
    return dt.datetime.fromtimestamp(max(stamps)).strftime("%d %b %Y")


# =============================================================================================
# Validation exhibits -- read from disk at build time, each independently optional.
# data/ and sme_review/ are both gitignored, so a fresh clone has none of this; the page must
# still build, just without the KPI band and appendix.
# =============================================================================================
def _read_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def load_validation_exhibits(review_dir="sme_review", top_drivers=8):
    """Load the on-disk validation evidence. Returns {} when nothing is available."""
    ex = {}

    bench = _read_csv(os.path.join(review_dir, "benchmark_vs_markov.csv"))
    if bench is not None and not bench.empty:
        ex["benchmark"] = bench.to_dict("records")

    acc = _read_csv(os.path.join(review_dir, "accuracy_by_starting_rating.csv"))
    if acc is not None and not acc.empty:
        ex["accuracy"] = acc.to_dict("records")
        n = acc["n"].sum()
        if n:
            # Report the whole test set, not an unweighted average of ten uneven buckets.
            ex["overall"] = {
                "n": int(n),
                "mae": float((acc["MAE"] * acc["n"]).sum() / n),
                "within1": float((acc["within_1_pct"] * acc["n"]).sum() / n),
            }

    fi = _read_csv(os.path.join(review_dir, "feature_importance.csv"))
    if fi is not None and not fi.empty:
        drivers = {}
        for target, grp in fi.groupby("target"):
            top = grp.sort_values("importance", ascending=False).head(top_drivers)
            drivers[str(target)] = [{"feature": str(r.feature), "importance": float(r.importance)}
                                    for r in top.itertuples()]
        ex["drivers"] = drivers

    mat = _read_csv(os.path.join(review_dir, "deterioration_by_material.csv"))
    if mat is not None and not mat.empty:
        ex["material"] = mat.head(8).to_dict("records")

    return ex


def driver_feature_names(exhibits):
    """Every feature named by any component's driver list -- the values worth shipping per bridge."""
    names = set()
    for rows in (exhibits.get("drivers") or {}).values():
        names.update(r["feature"] for r in rows)
    return sorted(names)


# =============================================================================================
# Static HTML fragments (KPI band + methodology appendix), rendered once at build time
# =============================================================================================
def _kpi(value, unit, label, source):
    unit_html = f'<span class="u">{html.escape(unit)}</span>' if unit else ""
    return (f'<div class="kpi"><div class="val">{html.escape(value)}{unit_html}</div>'
            f'<div class="lbl">{html.escape(label)}</div>'
            f'<div class="src">{html.escape(source)}</div></div>')


def render_kpis(exhibits, n_bridges):
    tiles = []
    overall = exhibits.get("overall")
    if overall:
        tiles.append(_kpi(f"{overall['mae']:.2f}", "", "Mean absolute error",
                          "Rating points, across all starting ratings"))
        tiles.append(_kpi(f"{overall['within1']:.1f}", "%", "Within ±1 rating",
                          f"Of {overall['n']:,} held-out test forecasts"))

    bench = exhibits.get("benchmark")
    if bench:
        gains = [(1 - r["ml_MAE"] / r["carry_forward_MAE"]) for r in bench
                 if r.get("horizon_yr") == 20 and r.get("carry_forward_MAE")]
        if gains:
            tiles.append(_kpi(f"{max(gains) * 100:.0f}", "%", "Less error at 20 years",
                              "Vs. assuming condition does not change"))

    tiles.append(_kpi(f"{n_bridges:,}", "", "Bridges in this report",
                      "Each forecast from its own inspection history"))
    tiles.append(_kpi("1.2", "M", "Training transitions",
                      "Sampled inspection-to-inspection pairs per component"))
    return "\n".join(tiles)


# --- small SVG helpers -----------------------------------------------------------------------
S1, S2, S3 = "#0056A9", "#D90D0D", "#27713E"   # accent1, accent2, snapped accent3
SLATE = "#333F48"                              # accent6
GRID, INK, MUTED = "#CFD4DA", "#1A1A1A", "#5A6470"

# Texture is the TxDOT templates' own secondary encoding (their sample charts stripe series 2 and
# hatch series 3). Here it is mandatory, not decorative: the brand red/green pair sits in the 6-8
# CVD floor band, which is only legal alongside a non-colour channel.
_DEFS = f"""<defs>
  <pattern id="patDash" width="8" height="4" patternUnits="userSpaceOnUse">
    <rect width="8" height="4" fill="{S2}"/><rect x="0" y="1.2" width="4.4" height="1.6" fill="#FFFFFF"/>
  </pattern>
  <pattern id="patDots" width="6" height="6" patternUnits="userSpaceOnUse">
    <rect width="6" height="6" fill="{S3}"/>
    <circle cx="1.5" cy="1.5" r="0.95" fill="#FFFFFF"/><circle cx="4.5" cy="4.5" r="0.95" fill="#FFFFFF"/>
  </pattern>
</defs>"""


def _nice_axis(vmax, ticks=4):
    """A shared axis top on a round step -- 0.647 becomes 0.8 in steps of 0.2, not five
    arbitrary decimals."""
    raw = vmax / ticks
    for step in (0.05, 0.1, 0.125, 0.2, 0.25, 0.5, 1.0, 2.0):
        if raw <= step:
            return step * ticks, step
    return vmax, vmax / ticks


def _benchmark_panel(rows, horizon, axis_max, step, width=470, height=280):
    """Grouped bars: model vs Markov vs carry-forward MAE, one panel per horizon.
    Lower is better, so the shortest bar wins -- stated on the axis, not left to inference.

    axis_max is supplied by the caller and shared across horizons: these are small multiples of
    the same measure, so giving each panel its own scale would make a 20-year error look like a
    10-year one."""
    rows = [r for r in rows if r.get("horizon_yr") == horizon]
    if not rows:
        return ""
    pl, pr, pt, pb = 46, 12, 26, 54
    plot_w, plot_h = width - pl - pr, height - pt - pb
    top = axis_max
    y = lambda v: pt + plot_h * (1 - v / top)
    group_w = plot_w / len(rows)
    bar_w = (group_w - 20) / 3 - 2          # 2px surface gap between adjacent bars

    s = [f'<svg viewBox="0 0 {width} {height}" role="img" '
         f'aria-label="Mean absolute error at {horizon} years by component and method">{_DEFS}',
         f'<text x="{pl}" y="14" font-size="13" font-weight="700" fill="{INK}">'
         f'{horizon}-year forecast</text>']
    n_ticks = int(round(top / step))
    for i in range(n_ticks + 1):
        v = step * i
        s.append(f'<line x1="{pl}" y1="{y(v):.1f}" x2="{width - pr}" y2="{y(v):.1f}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{pl - 7}" y="{y(v) + 4:.1f}" font-size="11" fill="{MUTED}" '
                 f'text-anchor="end">{v:.2f}</text>')
    for gi, r in enumerate(rows):
        gx = pl + group_w * gi + 10
        for bi, (val, fill) in enumerate([(r["ml_MAE"], S1),
                                          (r["markov_MAE"], "url(#patDash)"),
                                          (r["carry_forward_MAE"], "url(#patDots)")]):
            bx = gx + bi * (bar_w + 2)
            s.append(f'<rect x="{bx:.1f}" y="{y(val):.1f}" width="{bar_w:.1f}" '
                     f'height="{plot_h - (y(val) - pt):.1f}" fill="{fill}"/>')
        # direct-label the model bar only -- a number on every bar is unreadable
        # paint-order halo: the model's bar is the shortest, so its label sits beside a taller
        # neighbour and would otherwise read on top of the pattern fill
        s.append(f'<text x="{gx + bar_w / 2:.1f}" y="{y(r["ml_MAE"]) - 5:.1f}" font-size="11" '
                 f'font-weight="700" fill="{S1}" text-anchor="middle" '
                 f'stroke="#FFFFFF" stroke-width="3" paint-order="stroke">{r["ml_MAE"]:.2f}</text>')
        s.append(f'<text x="{gx + group_w / 2 - 10:.1f}" y="{height - 32}" font-size="11.5" '
                 f'fill="{INK}" text-anchor="middle">{html.escape(str(r["component"]))}</text>')
    s.append(f'<text x="{pl + plot_w / 2:.1f}" y="{height - 8}" font-size="11.5" fill="{MUTED}" '
             f'text-anchor="middle">mean absolute error in rating points &mdash; lower is better</text>')
    s.append("</svg>")
    return "".join(s)


def _accuracy_chart(rows, width=980, height=300):
    """One series: share of forecasts within +/-1 rating, by the rating the bridge started from.
    All bars take slot 1 -- colouring them by value would re-encode bar height and burn the
    identity channel. The weak left tail is called out with a rule and a label instead."""
    if not rows:
        return ""
    pl, pr, pt, pb = 46, 16, 22, 66
    plot_w, plot_h = width - pl - pr, height - pt - pb
    y = lambda v: pt + plot_h * (1 - v / 100)
    step = plot_w / len(rows)
    bar_w = step - 16

    s = [f'<svg viewBox="0 0 {width} {height}" role="img" '
         f'aria-label="Share of forecasts within one rating point, by starting rating">']
    for i in range(5):
        v = 100 * i / 4
        s.append(f'<line x1="{pl}" y1="{y(v):.1f}" x2="{width - pr}" y2="{y(v):.1f}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{pl - 7}" y="{y(v) + 4:.1f}" font-size="11" fill="{MUTED}" '
                 f'text-anchor="end">{v:.0f}%</text>')
    for i, r in enumerate(rows):
        bx = pl + step * i + 8
        val = float(r["within_1_pct"])
        s.append(f'<rect x="{bx:.1f}" y="{y(val):.1f}" width="{bar_w:.1f}" '
                 f'height="{plot_h - (y(val) - pt):.1f}" fill="{S1}"/>')
        s.append(f'<text x="{bx + bar_w / 2:.1f}" y="{y(val) - 6:.1f}" font-size="11.5" '
                 f'font-weight="700" fill="{INK}" text-anchor="middle">{val:.0f}%</text>')
        s.append(f'<text x="{bx + bar_w / 2:.1f}" y="{height - 44}" font-size="12.5" '
                 f'font-weight="700" fill="{INK}" text-anchor="middle">{int(r["starting_rating"])}</text>')
        s.append(f'<text x="{bx + bar_w / 2:.1f}" y="{height - 30}" font-size="10.5" '
                 f'fill="{MUTED}" text-anchor="middle">n={int(r["n"]):,}</text>')
    # boundary between the poor-condition tail and the bulk of the inventory
    bx5 = pl + step * 5
    s.append(f'<line x1="{bx5:.1f}" y1="{pt}" x2="{bx5:.1f}" y2="{pt + plot_h}" '
             f'stroke="{MUTED}" stroke-width="1"/>')
    s.append(f'<text x="{bx5 - 8:.1f}" y="{pt + 13}" font-size="11" fill="{MUTED}" '
             f'text-anchor="end">already poor &mdash; weakest</text>')
    s.append(f'<text x="{pl + plot_w / 2:.1f}" y="{height - 8}" font-size="11.5" fill="{MUTED}" '
             f'text-anchor="middle">rating the bridge started from</text>')
    s.append("</svg>")
    return "".join(s)


def _drivers_table(drivers, targets, top=6):
    """Top drivers per component, side by side. A table rather than a fourth chart: the side rail
    already plots these interactively, and the comparison across components is what matters."""
    cols = [t for t in targets if t in drivers] or list(drivers)
    if not cols:
        return ""
    head = "".join(f'<th>{html.escape(NICE_TARGET.get(c, c))}</th>' for c in cols)
    body = []
    for rank in range(top):
        cells = []
        for c in cols:
            rows = drivers[c]
            if rank < len(rows):
                feat = rows[rank]["feature"]
                pretty = PRETTY_FEATURE.get(feat, feat.replace("_", " "))
                cells.append(f'<td>{html.escape(pretty)}'
                             f'<span class="muted"> &middot; {rows[rank]["importance"] * 100:.1f}%</span></td>')
            else:
                cells.append("<td></td>")
        body.append(f'<tr><td class="name">{rank + 1}</td>{"".join(cells)}</tr>')
    return (f'<div class="scroll"><table><thead><tr><th>#</th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def _material_table(rows):
    if not rows:
        return ""
    body = []
    for r in rows:
        gap = abs(float(r["pred_decline"]) - float(r["actual_decline"]))
        body.append(
            f'<tr><td class="name">{html.escape(str(r["structure_kind"]))}</td>'
            f'<td class="num">{int(r["n"]):,}</td>'
            f'<td class="num">{float(r["mean_from"]):.2f}</td>'
            f'<td class="num">{float(r["pred_decline"]):.2f}</td>'
            f'<td class="num">{float(r["actual_decline"]):.2f}</td>'
            f'<td class="num">{gap:.2f}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Structure code</th>'
            '<th class="num">Bridges</th><th class="num">Mean start</th>'
            '<th class="num">Predicted decline</th><th class="num">Actual decline</th>'
            '<th class="num">Gap</th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def render_method(exhibits, targets, meta):
    """The methodology & validation appendix."""
    out = []

    out.append(
        '<h3>How a forecast is produced</h3>'
        '<p class="note">Every bridge in the inventory is inspected on a roughly two-year cycle, and '
        'the raw export repeats the last result annually in between. Those repeats are collapsed so '
        'only real inspection events remain, then every inspection is paired with every later '
        'inspection of the same bridge. Each pair becomes one training example: <i>this bridge was '
        'rated r0, and n years later it was rated r1</i>. Up to 1.2 million such pairs are sampled '
        'per component, spanning gaps of up to 25 years.</p>'
        '<p class="note">Because the number of years is itself an input, one model serves every '
        'horizon &mdash; a 20-year forecast is a single evaluation, not twenty one-year steps '
        'compounded. Train and test are split <b>by bridge</b>, so no bridge appears on both sides '
        'and the accuracy below is measured on structures the model has never seen.</p>')

    bench = exhibits.get("benchmark")
    if bench:
        out.append('<h3 class="sub">Measured against the alternatives</h3>')
        out.append('<p class="note">Two reference methods, both currently used in practice: assuming '
                   'condition does not change (&ldquo;carry-forward&rdquo;), and a '
                   'material-stratified Markov deterioration curve. Lower error is better; both '
                   'panels share one scale, so the growth from 10 to 20 years is directly '
                   'comparable.</p>')
        worst = max(max(r["carry_forward_MAE"], r["markov_MAE"], r["ml_MAE"]) for r in bench)
        axis_max, step = _nice_axis(worst * 1.05)
        out.append('<div class="two">'
                   f'<div>{_benchmark_panel(bench, 10, axis_max, step)}</div>'
                   f'<div>{_benchmark_panel(bench, 20, axis_max, step)}</div></div>')
        out.append('<div class="legend">'
                   f'<span><i class="bx" style="background:{S1}"></i>This model</span>'
                   '<span><i class="bx" style="background:repeating-linear-gradient(180deg,'
                   f'{S2} 0 2px,#fff 2px 3px)"></i>Markov curve</span>'
                   '<span><i class="bx" style="background:radial-gradient(#fff 22%,transparent 23%) 0 0/4px 4px,'
                   f'{S3}"></i>Carry-forward</span></div>')
        culvert10 = next((r for r in bench
                          if str(r.get("component")).lower() == "culvert" and r.get("horizon_yr") == 10), None)
        if culvert10 and culvert10["carry_forward_MAE"] < culvert10["ml_MAE"]:
            out.append(
                '<div class="callout warn"><b>One exception, stated plainly.</b> For culverts at 10 '
                f'years, carry-forward is marginally better than the model ({culvert10["carry_forward_MAE"]:.3f} '
                f'vs {culvert10["ml_MAE"]:.3f}). Culvert ratings move so little over a decade that '
                '&ldquo;no change&rdquo; is hard to beat. The advantage returns by 20 years '
                f'({culvert10["ml_MAE"]:.3f} → '
                f'{next(r["ml_MAE"] for r in bench if str(r.get("component")).lower() == "culvert" and r.get("horizon_yr") == 20):.3f} '
                'against a carry-forward that degrades faster), and the built-in hybrid rule already '
                'defers to carry-forward at short horizons.</div>')

    acc = exhibits.get("accuracy")
    if acc:
        out.append('<h3 class="sub">Accuracy by the condition the bridge started in</h3>')
        out.append('<p class="note">Inspectors themselves routinely disagree by a rating point, so '
                   '&ldquo;within &plusmn;1&rdquo; is the practical bar for a screening tool.</p>')
        out.append(f'<div class="chartwrap">{_accuracy_chart(acc)}</div>')
        out.append(
            '<div class="callout warn"><b>Where it is weak.</b> Accuracy falls off sharply for '
            'bridges already rated 0&ndash;4, where the model tends to over-predict recovery. Those '
            'structures are a small share of the inventory and are already inspection priorities, but '
            'the model should not be used to justify deferring work on them.</div>')

    material = exhibits.get("material")
    if material:
        out.append('<h3 class="sub">Predicted vs. actual decline, by structure type</h3>')
        out.append('<p class="note">A check that the model is not simply accurate on average while '
                   'wrong about which structures decline fastest. Predicted and actual decline are '
                   'compared within each material class over the held-out period. Codes are raw SNBI '
                   'structure-kind values, shown as recorded.</p>')
        out.append(_material_table(material))

    drivers = exhibits.get("drivers")
    if drivers:
        out.append('<h3 class="sub">What the model leans on</h3>')
        out.append('<p class="note">Model-wide importance across the full training set, per '
                   'component. The current rating dominates every component, which is expected and '
                   'is exactly why bridges with no usable inspection history fall back to the '
                   'separate attributes-only model.</p>')
        out.append(_drivers_table(drivers, targets))

    out.append(
        '<div class="callout prov"><b>Provenance.</b><ul>'
        f'<li>Bridge records read from <b>{html.escape(meta.get("source_label", "—"))}</b>.</li>'
        f'<li>Deterioration models trained <b>{html.escape(meta.get("model_trained") or "—")}</b>; '
        'retrain when the source data is refreshed.</li>'
        f'<li>Hybrid threshold <b>{meta.get("hybrid")} years</b> &middot; plan-for quantile '
        f'<b>{meta.get("conservative_quantile")}</b> &middot; poor-condition threshold '
        f'<b>{meta.get("risk_threshold")}</b>.</li>'
        f'<li>Page generated <b>{html.escape(meta.get("generated", ""))}</b>.</li>'
        '</ul></div>')

    out.append(
        '<h3 class="sub">Figures from the internal status report &mdash; pending regeneration</h3>'
        '<p class="note">These were measured during model development and are quoted from the '
        'internal status and testing reports. They are <b>not</b> reproducible from the exhibit files '
        'shipped alongside this page; regenerate them from a training run before citing them '
        'externally.</p>'
        '<div class="scroll"><table><thead><tr><th>Figure</th><th>Reported value</th>'
        '<th>What it measures</th></tr></thead><tbody>'
        '<tr><td class="name">Risk model AUC</td><td>0.93</td>'
        '<td>Ranking quality for &ldquo;will reach poor condition&rdquo;</td></tr>'
        '<tr><td class="name">Top-1% precision</td><td>86% vs 81% vs 45%</td>'
        '<td>Risk ranking vs. point-forecast ranking vs. naive selection</td></tr>'
        '<tr><td class="name">Plan-for catch rate</td><td>~75% vs ~56%</td>'
        '<td>Share of true 2-point decliners flagged, at ~7% false alarms</td></tr>'
        '<tr><td class="name">Within &plusmn;1 by horizon</td><td>~90&ndash;96%</td>'
        '<td>Per-component accuracy at 10 and 20 years</td></tr>'
        '</tbody></table></div>')

    return "\n".join(out)


# =============================================================================================
# Assembly
# =============================================================================================
def write_html(bridges, targets, meta, exhibits, out_path):
    html_str = (HTML_TEMPLATE
                .replace("%%CSS%%", BRAND_CSS)
                .replace("%%LOGO%%", LOGO_SVG)
                .replace("%%KPIS%%", render_kpis(exhibits, len(bridges)))
                .replace("%%METHOD%%", render_method(exhibits, targets, meta))
                .replace("%%N%%", f"{len(bridges):,}")
                .replace("%%SOURCE%%", html.escape(meta.get("source_label", "—")))
                .replace("%%GENERATED%%", html.escape(meta.get("generated", "")))
                .replace("%%DATA%%", json.dumps(bridges))
                .replace("%%TARGETS%%", json.dumps(targets))
                .replace("%%META%%", json.dumps(meta))
                .replace("%%EXHIBITS%%", json.dumps(exhibits))
                .replace("%%HORIZONS%%", json.dumps(HORIZONS)))
    left = sorted(set(re.findall(r"%%[A-Z_]+%%", html_str)))
    if left:
        raise RuntimeError(f"{out_path}: unsubstituted placeholders {left}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)


def main():
    ap = argparse.ArgumentParser(description="Build an interactive HTML bridge-forecast report.")
    ap.add_argument("--codes", default=None, help="Comma-separated NBI codes")
    ap.add_argument("--file", default=None, help="Text file with one NBI code per line")
    ap.add_argument("--district", default=None, help="Load every bridge in this district")
    ap.add_argument("--out", default="forecast_ui.html")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--review-dir", default="sme_review",
                    help="Directory holding the validation exhibit CSVs (optional)")
    args = ap.parse_args()

    codes = None
    if args.codes:
        codes = [c for c in args.codes.split(",") if c.strip()]
    elif args.file:
        with open(args.file) as f:
            codes = [ln.strip() for ln in f if ln.strip()]

    config = load_config(args.config)
    exhibits = load_validation_exhibits(args.review_dir)
    if not exhibits:
        print(f"Note: no validation exhibits found in {args.review_dir}/ -- "
              f"building the explorer without the KPI band or validation appendix.")

    bridges, targets, meta = build_data(config, codes=codes, district=args.district,
                                        driver_features=driver_feature_names(exhibits))
    if not targets:
        raise SystemExit("No deterioration models found in models/deterioration/. "
                         "Train them first with: python main.py")
    if not bridges:
        raise SystemExit("No matching bridges found (check the codes/district).")
    write_html(bridges, targets, meta, exhibits, args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Wrote {args.out} — {len(bridges):,} bridges, {size_mb:.1f} MB. Open it in any browser.")


if __name__ == "__main__":
    main()
