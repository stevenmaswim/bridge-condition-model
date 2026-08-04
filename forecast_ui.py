"""Generate a self-contained interactive HTML forecasting page for a set of bridges.

Runs the real deterioration model once over the requested bridges, precomputes each bridge's
forecast trajectory (most-likely + conservative) at several horizons for every condition rating it
has, and embeds it in a single offline HTML file with search, a target selector, a horizon slider,
and a click-to-view deterioration-curve chart.

Examples:
  python forecast_ui.py --codes 120200152401017,121700017714043
  python forecast_ui.py --file my_bridges.txt          # one NBI code per line
  python forecast_ui.py --district 12                  # every bridge in a district
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from src.data_loader import load_config, load_raw_data, rename_raw_columns, clean_data
from src.deterioration import build_inspection_events, predict_with_bundle, load_deterioration_model
from src.enrichment import attach_static_features

HORIZONS = [0, 2, 5, 10, 15, 20]


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


def build_data(config, codes=None, district=None):
    id_col = config["data"]["id_col"]
    insp_col = config["data"]["inspection_date_col"]
    dist_col = config["grouping"]["district_col"]
    hybrid = (config.get("deterioration", {}) or {}).get("hybrid_threshold_years", 3.0)

    # When reading live from Snowflake, push the district/bridge filter into the SQL so we only pull
    # the rows we need (seconds) instead of the whole ~1.7M-row table. Harmless for the CSV source.
    if config.get("data", {}).get("source") == "snowflake":
        if codes:
            config["_snowflake_filter"] = {"column": id_col, "values": [str(c).strip() for c in codes]}
        elif district is not None:
            config["_snowflake_filter"] = {"column": dist_col, "values": [str(district).strip()]}

    df = clean_data(rename_raw_columns(load_raw_data(config), config), config)
    events = build_inspection_events(df, config)
    latest = events.sort_values(insp_col).groupby(id_col, as_index=False).tail(1).copy()

    if codes:
        wanted = {str(c).strip() for c in codes}
        latest = latest[latest[id_col].astype(str).str.strip().isin(wanted)]
    elif district is not None and dist_col in latest.columns:
        latest = latest[latest[dist_col].astype(str).str.strip() == str(district).strip()]
    latest = latest.reset_index(drop=True)
    if latest.empty:
        return [], []

    latest["age_t0"] = (latest[insp_col].dt.year
                        - pd.to_numeric(latest["year_built"], errors="coerce")).clip(0, 130)
    base, _ = attach_static_features(latest, config)

    per_target = {}
    targets = []
    for target in config["targets"]:
        try:
            bundle = load_deterioration_model(target, config)
        except FileNotFoundError:
            continue
        targets.append(target)
        per_target[target] = _forecast_target(base, bundle, target, hybrid)

    bridges = []
    for i, row in latest.iterrows():
        entry = {
            "id": str(row[id_col]),
            "district": (str(row[dist_col]) if dist_col in latest.columns and pd.notna(row[dist_col])
                         else ""),
            "year_built": (int(row["year_built"]) if pd.notna(row.get("year_built")) else None),
            "last_year": int(row[insp_col].year),
            "targets": {},
        }
        for target in targets:
            if i in per_target[target]:
                entry["targets"][target] = per_target[target][i]
        if entry["targets"]:
            bridges.append(entry)
    return bridges, targets


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bridge Condition Forecast</title>
<style>
  :root{--ink:#1f2937;--muted:#6b7280;--line:#e5e7eb;--bg:#f9fafb;--card:#fff;
        --good:#2f9e44;--fair:#f08c00;--poor:#e03131;--blue:#4E79A7;--accent:#1c7ed6}
  *{box-sizing:border-box}
  body{margin:0;font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif;color:var(--ink);background:var(--bg)}
  header{padding:20px 24px;background:var(--card);border-bottom:1px solid var(--line)}
  h1{margin:0;font-size:20px}
  .sub{color:var(--muted);font-size:13px;margin-top:2px}
  .wrap{padding:18px 24px;max-width:820px;margin:0 auto}
  .controls{display:flex;gap:18px;flex-wrap:wrap;align-items:end;margin-bottom:16px}
  .controls label{display:block;font-size:12px;color:var(--muted);margin-bottom:4px;font-weight:600}
  select,input{font:inherit;padding:9px 11px;border:1px solid var(--line);border-radius:8px;background:#fff}
  select{min-width:240px}
  input[type=range]{padding:0;width:220px}
  .yearval{font-weight:600;color:var(--accent)}
  .prompt{background:var(--card);border:1px dashed var(--line);border-radius:10px;padding:26px;text-align:center;color:var(--muted)}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;display:none}
  .card h2{margin:0 0 2px;font-size:19px}
  .card .meta{color:var(--muted);font-size:13px;margin-bottom:14px}
  table{width:100%;border-collapse:collapse}
  th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
  th{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
  .pill{display:inline-block;min-width:36px;text-align:center;padding:2px 9px;border-radius:999px;font-weight:700;color:#fff}
  .muted{color:var(--muted)}
  .chart{margin-top:16px}
  .legend{font-size:12px;color:var(--muted);margin-top:6px}
  .legend span{display:inline-flex;align-items:center;gap:5px;margin-right:14px}
  .sw{width:22px;height:0;border-top:3px solid var(--blue);display:inline-block}
  .sw.d{border-top-style:dashed}
  footer{color:var(--muted);font-size:12px;padding:14px 24px;max-width:820px;margin:0 auto}
</style></head>
<body>
<header>
  <h1>Bridge Condition Forecast</h1>
  <div class="sub">Select one bridge number to forecast its condition ratings &middot; %%N%% bridges available</div>
</header>
<div class="wrap">
  <div class="controls">
    <div>
      <label>Bridge number</label>
      <input id="filter" placeholder="type to filter..." style="width:150px">
      <select id="bridge"><option value="">-- select a bridge --</option></select>
    </div>
    <div>
      <label>Forecast horizon: <span class="yearval" id="yearlabel">+10 years</span></label>
      <input type="range" id="horizon" min="0" max="20" step="1" value="10">
    </div>
  </div>

  <div class="prompt" id="prompt">Pick a bridge number above to see its forecast.</div>

  <div class="card" id="card">
    <h2 id="btitle"></h2>
    <div class="meta" id="bmeta"></div>
    <table>
      <thead><tr><th>Rating</th><th>Current</th><th>Most-likely</th><th>Plan-for (budget)</th><th>Risk of poor</th></tr></thead>
      <tbody id="brows"></tbody>
    </table>
    <div class="chart" id="chart"></div>
    <div class="legend">
      <span><i class="sw"></i>most-likely</span>
      <span><i class="sw d"></i>plan-for (conservative)</span>
      <span>ratings: <b style="color:var(--good)">7-9 good</b> &middot; <b style="color:var(--fair)">5-6 fair</b> &middot; <b style="color:var(--poor)">0-4 poor</b></span>
    </div>
  </div>
</div>
<footer>Each bridge shows only the ratings it actually has (deck/superstructure/substructure OR culvert &mdash; never both). Decision-support for screening, not a substitute for inspection; weakest on already-poor bridges and cannot foresee sudden failures.</footer>
<script>
const DATA = %%DATA%%;
const HZ = %%HORIZONS%%;
const NICE = {deck_cond_rating:"Deck",superstructure_cond_rating:"Superstructure",substructure_cond_rating:"Substructure",culvert_cond_rating:"Culvert"};
const COLORS = ["#4E79A7","#F28E2B","#59A14F","#B07AA1"];
const BY_ID = {}; DATA.forEach(b=>BY_ID[b.id]=b);
const sel = document.getElementById('bridge');
const ALL_IDS = DATA.map(b=>b.id).sort();
function fillOptions(list){
  sel.innerHTML = '<option value="">-- select a bridge --</option>' +
    list.slice(0,5000).map(id=>`<option value="${id}">${id}</option>`).join("");
}
fillOptions(ALL_IDS);
function ratingColor(v){if(v==null)return"#9ca3af";return v>=7?"#2f9e44":v>=5?"#f08c00":"#e03131";}
function interp(map,h){const ks=HZ; if(map[h]!=null)return map[h];
  let lo=ks[0],hi=ks[ks.length-1];
  for(let i=0;i<ks.length-1;i++){if(ks[i]<=h&&h<=ks[i+1]){lo=ks[i];hi=ks[i+1];break;}}
  const a=map[lo],b=map[hi]; if(a==null||b==null)return a??b; if(hi===lo)return a;
  return a+(b-a)*(h-lo)/(hi-lo);}
function pill(v){const s=v==null?"&mdash;":v.toFixed(1);return `<span class="pill" style="background:${ratingColor(v)}">${s}</span>`;}
function riskCell(v){if(v==null)return '<span class="muted">&mdash;</span>';const c=v>=50?"#e03131":v>=25?"#f08c00":"#6b7280";return `<span style="font-weight:700;color:${c}">${Math.round(v)}%</span>`;}
function render(){
  const h=+document.getElementById('horizon').value;
  document.getElementById('yearlabel').textContent = "+"+h+" years";
  const b = BY_ID[sel.value];
  document.getElementById('prompt').style.display = b ? "none" : "block";
  document.getElementById('card').style.display = b ? "block" : "none";
  if(!b) return;
  document.getElementById('btitle').textContent = "Bridge "+b.id;
  document.getElementById('bmeta').textContent =
    (b.district?"District "+b.district+" · ":"") + (b.year_built?"built "+b.year_built+" · ":"") +
    "last inspected "+b.last_year+" · forecast year "+(b.last_year+h);
  // only the ratings this bridge actually has (deck/super/sub OR culvert)
  const tb=document.getElementById('brows'); tb.innerHTML="";
  Object.keys(b.targets).forEach(t=>{const d=b.targets[t];
    const like=interp(d.likely,h), cons=interp(d.cons,h), rk=interp(d.risk,h);
    tb.innerHTML += `<tr><td><b>${NICE[t]||t}</b></td><td>${pill(d.current)}</td>`+
      `<td>${pill(like)}</td><td>${pill(cons)}</td><td>${riskCell(rk)}</td></tr>`;});
  drawChart(b);
}
function drawChart(b){
  const W=680,H=240,pl=38,pr=90,pt=14,pb=28, yrs=HZ;
  const x=h=>pl+(W-pl-pr)*(h/20); const y=v=>pt+(H-pt-pb)*(1-v/9);
  let svg=`<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px">`;
  for(let g=0;g<=9;g+=3){svg+=`<line x1=${pl} y1=${y(g)} x2=${W-pr} y2=${y(g)} stroke="#eee"/><text x=4 y=${y(g)+4} font-size=11 fill="#9ca3af">${g}</text>`;}
  yrs.forEach(h=>{svg+=`<text x=${x(h)} y=${H-8} font-size=11 fill="#9ca3af" text-anchor="middle">+${h}</text>`;});
  Object.keys(b.targets).forEach((t,idx)=>{const d=b.targets[t];const c=COLORS[idx%COLORS.length];
    const pL=yrs.map(h=>`${x(h)},${y(d.likely[h])}`).join(" ");
    const pC=yrs.map(h=>`${x(h)},${y(d.cons[h])}`).join(" ");
    svg+=`<polyline points="${pC}" fill="none" stroke="${c}" stroke-width="2" stroke-dasharray="5 4" opacity=".7"/>`;
    svg+=`<polyline points="${pL}" fill="none" stroke="${c}" stroke-width="2.5"/>`;
    svg+=`<text x=${W-pr+6} y=${y(d.likely[20])+4} font-size="11" fill="${c}">${NICE[t]||t}</text>`;});
  svg+=`</svg>`;
  document.getElementById('chart').innerHTML=svg;
}
document.getElementById('filter').oninput=function(){
  const q=this.value.trim().toLowerCase();
  fillOptions(q ? ALL_IDS.filter(id=>id.toLowerCase().includes(q)) : ALL_IDS);
  render();
};
sel.onchange=render;
document.getElementById('horizon').oninput=render;
render();
</script>
</body></html>"""


def write_html(bridges, targets, out_path):
    html_str = (HTML_TEMPLATE
                .replace("%%N%%", str(len(bridges)))
                .replace("%%DATA%%", json.dumps(bridges))
                .replace("%%TARGETS%%", json.dumps(targets))
                .replace("%%HORIZONS%%", json.dumps(HORIZONS)))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)


def main():
    ap = argparse.ArgumentParser(description="Build an interactive HTML bridge-forecast page.")
    ap.add_argument("--codes", default=None, help="Comma-separated NBI codes")
    ap.add_argument("--file", default=None, help="Text file with one NBI code per line")
    ap.add_argument("--district", default=None, help="Load every bridge in this district")
    ap.add_argument("--out", default="forecast_ui.html")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    codes = None
    if args.codes:
        codes = [c for c in args.codes.split(",") if c.strip()]
    elif args.file:
        with open(args.file) as f:
            codes = [ln.strip() for ln in f if ln.strip()]

    config = load_config(args.config)
    bridges, targets = build_data(config, codes=codes, district=args.district)
    if not targets:
        raise SystemExit("No deterioration models found in models/deterioration/. "
                         "Train them first with: python main.py")
    if not bridges:
        raise SystemExit("No matching bridges found (check the codes/district).")
    write_html(bridges, targets, args.out)
    print(f"Wrote {args.out} with {len(bridges)} bridges. Open it in any browser.")


if __name__ == "__main__":
    main()
