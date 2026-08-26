"""Presentation layer for the bridge-forecast report page (built by forecast_ui.py).

Everything visual lives here so forecast_ui.py stays data assembly + CLI. The output is a
single self-contained offline HTML file: no external CSS, fonts, scripts or images, because
it gets emailed around and opened from a file:// path on machines with no network.

BRAND
-----
Colours and layout follow the TxDOT poster/presentation templates: blue header band, 3px red
rule beneath it, "Connecting you with Texas" tagline, light-grey page with white cards, blue
section headings, blue-header/zebra tables.

BRAND SOURCE
------------
Colours, fonts and the logo are taken from the official TxDOT PowerPoint templates
(txdot-presentation-template.potx), not eyeballed from a rendering:

    ppt/theme/theme1.xml   "TxDOT Template Color Scheme" / "TxDOT Template fonts"
      accent1 #0056A9  accent2 #D90D0D  accent3 #196533
      accent4 #5F0F40  accent5 #002E69  accent6 #333F48   lt2 #EBEBEB
      major font "Verdana Bold", minor font "Verdana"
    ppt/media/image3.svg   the horizontal logo lockup (see src/txdot_logo.py)

"Verdana Bold" is not a CSS family, so headings are Verdana at font-weight 700. Verdana ships
on Windows and macOS, so the page needs no web font and stays offline-safe.

CHART PALETTE -- why these exact values
---------------------------------------
Series 1 and 2 are the official accent1 blue and accent2 red, unchanged.

Series 3 is #27713E, NOT the official accent3 #196533. The official trio hard-fails the
colour-vision gate: accent2 red against accent3 green is dE 4.5 under simulated protanopia
(Machado-Oliveira-Fernandes 2009, severity 1.0), below the floor of 6 -- so it cannot be
rescued by labels or texture, unlike a borderline pair. Following snap-to-passing, accent3
was moved one lightness step (same hue, same chroma) to the nearest value that clears:

    #27713E   L 0.488 (accent3 is 0.448), C 0.108 (unchanged), dE 4.1 from accent3

which is a barely perceptible shift on screen. The result clears every check on the
all-pairs gate (the honest test here, since all three series appear side by side):

    lightness band   PASS    chroma floor    PASS
    CVD separation   PASS    worst pair dE 8.3 (protan), target 8
    normal vision    PASS    worst pair dE 20.9, floor 15
    contrast         PASS    all three >= 3:1 on white

The official #196533 is still used verbatim for the "good" condition pill, which is a status
chip carrying its own numeral rather than a plotted series, so it does no identity work.

Do not change a series hex without re-running the validator; the margin is thin.

STATUS SCALE (condition ratings) is separate from the series palette and never plotted as a
series -- it only fills the small table pills, which always show the numeral, so condition is
never encoded by colour alone. White text on each fill clears WCAG 4.5:1 (good 7.1, fair 5.0,
poor 5.2). Note the TxDOT scheme has no warning/amber colour, so "fair" (#B45309) is the one
value here with no theme source.
"""

# The official mark, extracted from the template package -- see src/txdot_logo.py for provenance.
from src.txdot_logo import LOGO_SVG   # noqa: F401  (re-exported; forecast_ui imports it from here)


BRAND_CSS = """
:root{
  color-scheme: light;
  /* --- TxDOT brand --- */
  --tx-blue-900:#002E69; --tx-blue:#0056A9; --tx-blue-600:#0056A9; --tx-blue-050:#E6EFF7;
  --tx-band-top:#0057A8; --tx-band-bot:#003567;  /* slideLayout1-5 hard-code, matched pair */
  --tx-slate:#333F48;                    /* accent6 -- reference/baseline series */
  --tx-red:#D90D0D;                      /* accent2 -- the rule under the header band */
  --page:#F2F2F2; --card:#FFFFFF;
  --ink:#1A1A1A; --muted:#5A6470; --line:#D9D9D9; --line-soft:#E4E6EA;
  --zebra:#DADEE5;                       /* literal srgbClr, off-palette by design */
  /* --- chart series (validated; see module docstring before editing) --- */
  --s1:#0056A9; --s2:#D90D0D; --s3:#27713E;   /* accent1, accent2, snapped accent3 */
  --ref:#8A94A6; --ref-soft:#BFC6D1;     /* baseline/reference series */
  /* --- status scale: NBI condition classes, pills only, never a plotted series --- */
  --good:#196533; --fair:#B45309; --poor:#D90D0D; --none:#5A6470;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--page);color:var(--ink);
     font:15px/1.5 Verdana,Geneva,"DejaVu Sans",Tahoma,sans-serif;
     font-variant-numeric:normal}

/* ---------- header band ---------- */
.brandbar{background:linear-gradient(180deg,var(--tx-band-top),var(--tx-band-bot));color:#fff;
          padding:14px 32px;display:flex;align-items:center;justify-content:space-between;gap:24px}
.txdot-logo{height:42px;width:auto;display:block;flex:none}
.tagline{font-style:italic;font-size:17px;opacity:.96;white-space:nowrap}
.redrule{height:2px;background:var(--tx-red)}

/* ---------- title + KPI ---------- */
.hero{background:var(--card);border-bottom:1px solid var(--line);padding:26px 32px 22px}
.hero h1{margin:0;font-size:30px;line-height:1.2;color:var(--tx-blue-600);font-weight:700;
         letter-spacing:-.01em}
.hero .deck{margin:7px 0 0;color:var(--muted);font-size:15px;max-width:78ch}
.hero .scope{margin:12px 0 0;font-size:13px;color:var(--muted)}
.hero .scope b{color:var(--ink);font-weight:700}

.kpirow{display:grid;grid-template-columns:repeat(auto-fit,minmax(196px,1fr));gap:14px;
        padding:20px 32px 4px;max-width:1560px}
.kpi{background:var(--card);border:1px solid var(--line);border-top:4px solid var(--tx-blue-600);
     border-radius:0;padding:14px 16px 13px}
.kpi .val{font-size:34px;font-weight:700;line-height:1.05;color:var(--tx-blue-900);
          letter-spacing:-.02em}
.kpi .val .u{font-size:17px;font-weight:700;color:var(--muted);margin-left:2px}
.kpi .lbl{font-size:12.5px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;
          color:var(--ink);margin-top:6px}
.kpi .src{font-size:11.5px;color:var(--muted);margin-top:4px;line-height:1.35}

/* ---------- layout ---------- */
main{padding:22px 32px 8px;max-width:1560px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:0;margin-bottom:22px}
.panel > h2{margin:0;padding:15px 22px;font-size:20px;color:var(--tx-blue-600);font-weight:700;
            border-bottom:2px solid var(--tx-blue-600)}
.panel > h2 .n{font-weight:400;color:var(--muted);font-size:14px;margin-left:8px}
.panel .body{padding:20px 22px 22px}
h3{margin:0 0 10px;font-size:15px;font-weight:700;color:var(--ink);
   text-transform:uppercase;letter-spacing:.045em}
h3.sub{margin-top:26px}
p.note{margin:8px 0 0;font-size:13px;color:var(--muted);line-height:1.5;max-width:96ch}

/* ---------- controls ---------- */
.controls{display:flex;gap:26px;flex-wrap:wrap;align-items:flex-end;
          background:var(--tx-blue-050);border:1px solid var(--line);border-radius:0;
          padding:14px 18px;margin-bottom:20px}
.controls label{display:block;font-size:11.5px;font-weight:700;text-transform:uppercase;
                letter-spacing:.05em;color:var(--tx-blue-900);margin-bottom:5px}
select,input[type=text]{font:inherit;font-size:15px;padding:8px 11px;border:1px solid var(--line);
                        border-radius:0;background:#fff;color:var(--ink)}
select{min-width:250px}
input[type=range]{width:250px;accent-color:var(--tx-blue-600);vertical-align:middle}
.hz{font-weight:700;color:var(--tx-blue-600);font-variant-numeric:tabular-nums}

/* ---------- explorer grid ---------- */
.grid{display:grid;grid-template-columns:minmax(0,1fr) 372px;gap:22px;align-items:start}
@media(max-width:1120px){.grid{grid-template-columns:minmax(0,1fr)}}
.prompt{border:1px dashed var(--line);border-radius:0;padding:44px 26px;text-align:center;
        color:var(--muted);font-size:15px}

/* ---------- tables ---------- */
table{width:100%;border-collapse:collapse;font-size:14.5px}
thead th{background:var(--tx-blue-600);color:#fff;font-weight:700;font-size:12px;
         text-transform:uppercase;letter-spacing:.045em;text-align:left;padding:9px 12px;
         border-right:1px solid rgba(255,255,255,.22)}
thead th:last-child{border-right:0}
thead th.num,tbody td.num{text-align:right}
tbody td{padding:9px 12px;border-bottom:1px solid var(--line-soft);
         font-variant-numeric:tabular-nums}
tbody tr:nth-child(even){background:var(--zebra)}
tbody tr.pick{cursor:pointer}
tbody tr.pick:hover{background:var(--tx-blue-050)}
tbody tr.on{background:var(--tx-blue-050);box-shadow:inset 3px 0 0 var(--tx-blue-600)}
tbody td.name{font-weight:700}

.pill{display:inline-block;min-width:40px;text-align:center;padding:3px 10px;border-radius:0;
      font-weight:700;color:#fff;font-size:14px;font-variant-numeric:tabular-nums}
.delta{font-weight:700;font-variant-numeric:tabular-nums}
.muted{color:var(--muted)}
.tiny{font-size:12px;color:var(--muted)}

/* ---------- side rail ---------- */
.rail{border:1px solid var(--line);border-top:4px solid var(--tx-blue-900);border-radius:0;
      background:#FAFBFC;font-size:13.5px}
.rail .rh{padding:13px 16px 11px;border-bottom:1px solid var(--line)}
.rail .rh .t{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;
             color:var(--tx-blue-900)}
.rail .rh .s{font-size:12.5px;color:var(--muted);margin-top:3px}
.rail section{padding:13px 16px;border-bottom:1px solid var(--line-soft)}
.rail section:last-child{border-bottom:0}
.rail h4{margin:0 0 8px;font-size:11.5px;font-weight:700;text-transform:uppercase;
         letter-spacing:.06em;color:var(--muted)}
.badge{display:inline-block;padding:3px 9px;border-radius:0;font-size:11.5px;font-weight:700;
       text-transform:uppercase;letter-spacing:.05em;color:#fff}
.badge.model{background:var(--tx-blue-600)}
.badge.carry{background:var(--ref)}
.badge.interp{background:var(--fair)}
.badge.obs{background:var(--tx-blue-900)}
.rail .exp{margin:8px 0 0;line-height:1.5;color:var(--ink)}
.kv{width:100%;border-collapse:collapse;font-size:13px}
.kv td{padding:4px 0;border-bottom:1px dotted var(--line);vertical-align:top}
.kv td:first-child{color:var(--muted);padding-right:10px;white-space:nowrap}
.kv td:last-child{text-align:right;font-variant-numeric:tabular-nums;font-weight:700;
                  overflow-wrap:anywhere;word-break:break-word}
.kv{table-layout:fixed}
.kv td:first-child{width:47%}
.kv tr:last-child td{border-bottom:0}
.mono{font-family:Consolas,"Courier New",monospace;font-size:12.5px}
.est{margin:0;padding:0;list-style:none}
.est li{padding:6px 0;border-bottom:1px dotted var(--line);line-height:1.45}
.est li:last-child{border-bottom:0}
.est b{display:block;font-size:12.5px}
.est span{color:var(--muted);font-size:12.5px}
.swatch{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;
        vertical-align:baseline}

/* driver bars: one-hue magnitude ramp, value always printed */
.drv{display:grid;grid-template-columns:1fr auto;gap:2px 10px;align-items:center;margin-bottom:7px}
.drv .n{font-size:12.5px;line-height:1.25}
.drv .v{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.drv .track{grid-column:1/-1;height:7px;background:var(--line-soft);border-radius:0}
.drv .fill{height:7px;background:var(--tx-blue-600);border-radius:0}
.drv .own{font-size:11.5px;color:var(--muted)}

/* ---------- charts ---------- */
.chartwrap{margin-top:18px}
.chartwrap svg{display:block;width:100%;height:auto}
.legend{display:flex;flex-wrap:wrap;gap:8px 20px;margin-top:10px;font-size:12.5px;
        color:var(--muted);align-items:center}
.legend i{display:inline-block;vertical-align:middle;margin-right:6px}
.legend .ln{width:22px;height:0;border-top:2.5px solid var(--s1)}
.legend .ln.d{border-top-style:dashed}
.legend .bx{width:12px;height:12px;border-radius:0}
.scroll{overflow-x:auto}

/* ---------- callouts ---------- */
.callout{border-left:4px solid var(--tx-blue-600);background:var(--tx-blue-050);
         padding:12px 16px;margin:16px 0 0;font-size:13.5px;line-height:1.5}
.callout.warn{border-left-color:var(--fair);background:#FDF6EC}
.callout.prov{border-left-color:var(--ref);background:#F4F6F9}
.callout b{font-weight:700}
.callout ul{margin:7px 0 0 18px;padding:0}
.callout li{margin:3px 0}

.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:22px 26px}
footer{padding:18px 32px 34px;max-width:1560px;color:var(--muted);font-size:12.5px;
       line-height:1.55}
footer .fr{border-top:2px solid var(--tx-blue-600);padding-top:12px}
"""


# ---------------------------------------------------------------------------------------------
# The page. Placeholders (%%NAME%%) are filled by forecast_ui.write_html.
# ---------------------------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bridge Condition Forecast &mdash; TxDOT</title>
<style>%%CSS%%</style></head>
<body>

<div class="brandbar">%%LOGO%%<div class="tagline">Connecting you with Texas</div></div>
<div class="redrule"></div>

<div class="hero">
  <h1>Bridge Condition Forecast</h1>
  <p class="deck">Forecasts how a bridge&rsquo;s NBI condition ratings (0&ndash;9) are likely to
     change over the next 20 years, so deterioration can be priced into maintenance and capital
     budgets before a structure reaches poor condition. Decision support for screening &mdash;
     not a replacement for inspection.</p>
  <p class="scope">Scope: <b>%%N%% bridges</b> &middot; source <b>%%SOURCE%%</b>
     &middot; generated <b>%%GENERATED%%</b></p>
</div>

<div class="kpirow">%%KPIS%%</div>

<main>

  <section class="panel">
    <h2>Bridge forecast explorer</h2>
    <div class="body">

      <div class="controls">
        <div>
          <label for="filter">Filter bridge number</label>
          <input type="text" id="filter" placeholder="type to filter&hellip;" style="width:168px">
        </div>
        <div>
          <label for="bridge">Bridge</label>
          <select id="bridge"><option value="">&mdash; select a bridge &mdash;</option></select>
          <div class="tiny" id="listnote" style="margin-top:5px"></div>
        </div>
        <div>
          <label for="horizon">Forecast horizon &mdash; <span class="hz" id="hzlabel">+10 years</span></label>
          <input type="range" id="horizon" min="0" max="20" step="1" value="10">
        </div>
      </div>

      <div class="prompt" id="prompt">Select a bridge number above to see its forecast and the
        calculation behind it.</div>

      <div class="grid" id="grid" style="display:none">
        <div>
          <h3 id="btitle"></h3>
          <p class="tiny" id="bmeta" style="margin:-4px 0 12px"></p>
          <div class="scroll">
          <table>
            <thead><tr>
              <th>Component</th><th class="num">Current</th><th class="num">Most&#8209;likely</th>
              <th class="num">Change</th><th class="num">Plan&#8209;for (budget)</th>
              <th class="num">Risk of poor</th>
            </tr></thead>
            <tbody id="brows"></tbody>
          </table>
          </div>
          <p class="tiny" style="margin-top:8px">Select a row to explain that component in the
             panel at right. Each bridge shows only the components it actually carries &mdash;
             deck / superstructure / substructure <b>or</b> culvert, never both.</p>

          <div class="chartwrap" id="chart"></div>
          <div class="legend" id="chartlegend"></div>
        </div>

        <aside class="rail" id="rail"></aside>
      </div>

    </div>
  </section>

  <section class="panel">
    <h2>Methodology &amp; validation</h2>
    <div class="body">%%METHOD%%</div>
  </section>

</main>

<footer><div class="fr">
  <b>Limitations.</b> The model projects trends; it cannot foresee sudden events such as scour or
  vehicle impact, so inspection remains the safety net. It is weakest on bridges already rated 0&ndash;4
  and tends to over-predict their recovery. Rating increases from repair or rehabilitation are treated
  as noise rather than predicted. Long horizons rest on fewer historical examples than short ones.
  Not yet reviewed by a bridge engineer.
</div></footer>

<script>
const DATA = %%DATA%%;
const HZ   = %%HORIZONS%%;
const META = %%META%%;
const EX   = %%EXHIBITS%%;

const NICE = {deck_cond_rating:"Deck", superstructure_cond_rating:"Superstructure",
              substructure_cond_rating:"Substructure", culvert_cond_rating:"Culvert"};
const SERIES = ["#0056A9","#D90D0D","#27713E","#002E69"];
/* Marker shape is the second identity channel (the TxDOT template line charts use
   circle / square / triangle for exactly this reason) -- required because the brand
   red/green pair sits in the 6-8 CVD floor band. */
const MARKS  = ["circle","square","triangle","diamond"];

/* bridge.vals is positional and bridge.feat_missing holds indices -- the names live once in
   META rather than being repeated on every record (worth megabytes at district scale). */
const VAL_IX = {}; (META.val_keys || []).forEach((k, i) => VAL_IX[k] = i);
const ownValue = (b, feat) => {
  const i = VAL_IX[feat];
  return (i == null || !b.vals) ? null : b.vals[i];
};
const missingNames = b => (b.feat_missing || []).map(i => (META.attr_names || [])[i]).filter(Boolean);

const BY_ID = {}; DATA.forEach(b => BY_ID[b.id] = b);
const ALL_IDS = DATA.map(b => b.id).sort();
const sel = document.getElementById('bridge');
let focusTarget = null;   // which component the side rail explains

/* ---------- helpers ---------- */
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt = (v, d=1) => v == null ? "—" : Number(v).toFixed(d);
function ratingColor(v){ if(v==null) return "#8A94A6";
  return v >= 7 ? "#196533" : v >= 5 ? "#B45309" : "#D90D0D"; }
function ratingWord(v){ if(v==null) return "not rated";
  return v >= 7 ? "good" : v >= 5 ? "fair" : "poor"; }
function pill(v){ return '<span class="pill" style="background:'+ratingColor(v)+'" '+
  'title="'+fmt(v)+' — '+ratingWord(v)+'">'+fmt(v)+'</span>'; }

/* The payload holds six anchors only; anything between them is a straight-line blend.
   anchorInfo tells the rail which case it is looking at, so the page never implies a
   model call happened where one did not. */
function anchorInfo(h){
  if (HZ.indexOf(h) >= 0) return {exact:true, lo:h, hi:h};
  let lo = HZ[0], hi = HZ[HZ.length-1];
  for (let i=0; i<HZ.length-1; i++){ if (HZ[i] < h && h < HZ[i+1]){ lo=HZ[i]; hi=HZ[i+1]; break; } }
  return {exact:false, lo:lo, hi:hi};
}
function interp(map, h){
  if (map[h] != null) return map[h];
  const a = anchorInfo(h), lo = map[a.lo], hi = map[a.hi];
  if (lo == null || hi == null) return lo != null ? lo : hi;
  if (a.hi === a.lo) return lo;
  return lo + (hi - lo) * (h - a.lo) / (a.hi - a.lo);
}
/* No risk probability exists at or below the hybrid threshold -- the classifier is never
   called there. Interpolating one in from the first model anchor would put a number on screen
   that the model never produced. */
function riskAt(d, h){ return h <= META.hybrid ? null : interp(d.risk, h); }

/* route: what actually produced the number on screen at this horizon */
function route(b, h){
  const a = anchorInfo(h);
  if (h === 0) return {kind:"obs", label:"Observed",
    text:"Horizon 0 is the rating recorded at the last inspection ("+b.last_year+"). No forecast is involved."};
  if (h <= META.hybrid) return {kind:"carry", label:"Carry-forward",
    text:"At or below the "+META.hybrid+"-year hybrid threshold the model is <b>not called</b>. "+
         "The last inspection rating is held flat, because over short gaps that beats the model — "+
         "ratings rarely move between inspections. No risk probability is produced on this path."};
  if (a.exact) return {kind:"model", label:"Model · +"+h+"y anchor",
    text:"A direct model evaluation at the +"+h+"-year anchor. The three estimators below were each "+
         "run once on this bridge’s feature row."};
  return {kind:"interp", label:"Interpolated", lo:a.lo, hi:a.hi,
    text:"Forecasts are precomputed at "+HZ.join(", ")+" years. +"+h+" years is a straight-line blend "+
         "of the <b>+"+a.lo+"</b> and <b>+"+a.hi+"</b> model evaluations — not a separate model call. "+
         "Move the slider to an anchor for a direct evaluation."};
}

/* ---------- selector ---------- */
/* Cap exists so a very large scope cannot lock the browser building <option> nodes, but a
   silent cap is worse than a slow page: it looks complete while hiding most of the inventory.
   If it ever bites, the page says so and tells the reader how to reach the rest. */
const OPTION_CAP = 12000;
function fillOptions(list){
  const shown = list.slice(0, OPTION_CAP);
  sel.innerHTML = '<option value="">— select a bridge —</option>' +
    shown.map(id => '<option value="'+esc(id)+'">'+esc(id)+'</option>').join("");
  const note = document.getElementById('listnote');
  if (note){
    note.textContent = list.length > OPTION_CAP
      ? "Listing " + shown.length.toLocaleString() + " of " + list.length.toLocaleString() +
        " bridges — type in the filter box to reach the rest."
      : list.length.toLocaleString() + " bridges in this report.";
  }
}
fillOptions(ALL_IDS);

/* ---------- main render ---------- */
function render(){
  const h = +document.getElementById('horizon').value;
  document.getElementById('hzlabel').textContent = "+" + h + (h === 1 ? " year" : " years");
  const b = BY_ID[sel.value];
  document.getElementById('prompt').style.display = b ? "none" : "block";
  document.getElementById('grid').style.display  = b ? "grid" : "none";
  if (!b) return;

  const keys = Object.keys(b.targets);
  if (!focusTarget || keys.indexOf(focusTarget) < 0) focusTarget = keys[0];

  document.getElementById('btitle').textContent = "Bridge " + b.id;
  document.getElementById('bmeta').innerHTML =
    (b.district ? "District " + esc(b.district) + " &middot; " : "") +
    (b.year_built ? "built " + b.year_built + " &middot; " : "") +
    "last inspected " + esc(b.last_inspection || String(b.last_year)) +
    " &middot; forecast year <b>" + (b.last_year + h) + "</b>";

  const tb = document.getElementById('brows');
  tb.innerHTML = keys.map(t => {
    const d = b.targets[t];
    const like = interp(d.likely, h), cons = interp(d.cons, h), rk = riskAt(d, h);
    const ch = (like == null || d.current == null) ? null : like - d.current;
    const chTxt = ch == null ? '<span class="muted">—</span>'
      : '<span class="delta" style="color:' + (ch <= -1 ? "#D90D0D" : ch < -0.05 ? "#B45309" : "#5A6470") + '">' +
        (ch > 0 ? "+" : "") + ch.toFixed(1) + '</span>';
    const rkTxt = rk == null ? '<span class="muted" title="no risk probability on the carry-forward path">—</span>'
      : '<span class="delta" style="color:' + (rk >= 50 ? "#D90D0D" : rk >= 25 ? "#B45309" : "#5A6470") + '">' +
        Math.round(rk) + '%</span>';
    return '<tr class="pick' + (t === focusTarget ? ' on' : '') + '" data-t="' + t + '">' +
      '<td class="name">' + (NICE[t] || t) + '</td>' +
      '<td class="num">' + pill(d.current) + '</td><td class="num">' + pill(like) + '</td>' +
      '<td class="num">' + chTxt + '</td><td class="num">' + pill(cons) + '</td>' +
      '<td class="num">' + rkTxt + '</td></tr>';
  }).join("");
  Array.prototype.forEach.call(tb.querySelectorAll('tr.pick'), tr => {
    tr.onclick = () => { focusTarget = tr.getAttribute('data-t'); render(); };
  });

  drawChart(b, h);
  drawRail(b, h);
}

/* ---------- deterioration curve ---------- */
function drawChart(b, h){
  const W = 720, H = 300, pl = 44, pr = 132, pt = 18, pb = 44;
  const x = v => pl + (W - pl - pr) * (v / 20);
  const y = v => pt + (H - pt - pb) * (1 - v / 9);
  const keys = Object.keys(b.targets);
  let s = '<svg viewBox="0 0 ' + W + ' ' + H + '" style="max-width:' + W + 'px" role="img" ' +
          'aria-label="Forecast condition rating by year ahead">';

  // condition bands, palest possible so they never compete with the data
  s += '<rect x="'+pl+'" y="'+y(9)+'" width="'+(W-pl-pr)+'" height="'+(y(7)-y(9))+'" fill="#196533" opacity=".055"/>';
  s += '<rect x="'+pl+'" y="'+y(7)+'" width="'+(W-pl-pr)+'" height="'+(y(5)-y(7))+'" fill="#B45309" opacity=".055"/>';
  s += '<rect x="'+pl+'" y="'+y(5)+'" width="'+(W-pl-pr)+'" height="'+(y(0)-y(5))+'" fill="#D90D0D" opacity=".055"/>';

  for (let g = 0; g <= 9; g += 3){                       // solid hairline grid
    s += '<line x1="'+pl+'" y1="'+y(g)+'" x2="'+(W-pr)+'" y2="'+y(g)+'" stroke="#CFD4DA" stroke-width="1"/>';
    s += '<text x="'+(pl-9)+'" y="'+(y(g)+4)+'" font-size="11.5" fill="#5A6470" text-anchor="end">'+g+'</text>';
  }
  HZ.forEach(v => {
    s += '<text x="'+x(v)+'" y="'+(H-24)+'" font-size="11.5" fill="#5A6470" text-anchor="middle">+'+v+'</text>';
  });
  s += '<text x="'+((pl+W-pr)/2)+'" y="'+(H-6)+'" font-size="11.5" fill="#5A6470" text-anchor="middle">years ahead</text>';
  s += '<text transform="translate(13,'+((pt+H-pb)/2)+') rotate(-90)" font-size="11.5" '+
       'fill="#5A6470" text-anchor="middle">condition rating</text>';
  // current horizon marker
  s += '<line x1="'+x(h)+'" y1="'+pt+'" x2="'+x(h)+'" y2="'+(H-pb)+'" stroke="#0056A9" stroke-width="1" opacity=".45"/>';
  s += '<text x="'+x(h)+'" y="'+(pt-4)+'" font-size="11" fill="#0056A9" text-anchor="middle">+'+h+'</text>';

  keys.forEach((t, i) => {
    const d = b.targets[t], c = SERIES[i % SERIES.length], mk = MARKS[i % MARKS.length];
    const pL = HZ.map(v => x(v) + "," + y(d.likely[v])).join(" ");
    const pC = HZ.map(v => x(v) + "," + y(d.cons[v])).join(" ");
    s += '<polyline points="'+pC+'" fill="none" stroke="'+c+'" stroke-width="2" stroke-dasharray="6 4" opacity=".75"/>';
    s += '<polyline points="'+pL+'" fill="none" stroke="'+c+'" stroke-width="2"/>';
    HZ.forEach(v => { s += marker(x(v), y(d.likely[v]), mk, c); });
  });
  /* Direct labels are the mandatory secondary encoding for this palette, so they have to stay
     legible: lay them out in a second pass and push overlapping ones apart. */
  const labels = keys.map((t, i) => ({t:t, y:y(b.targets[t].likely[20]), c:SERIES[i % SERIES.length]}))
                     .sort((p, q) => p.y - q.y);
  for (let k = 1; k < labels.length; k++){
    if (labels[k].y - labels[k-1].y < 15) labels[k].y = labels[k-1].y + 15;
  }
  labels.forEach(L => {
    s += '<text x="'+(W-pr+8)+'" y="'+(L.y+4)+'" font-size="11.5" fill="'+L.c+'" font-weight="700">'+
         (NICE[L.t] || L.t)+'</text>';
  });
  s += '</svg>';
  document.getElementById('chart').innerHTML = s;

  document.getElementById('chartlegend').innerHTML =
    keys.map((t, i) => '<span>' + svgChip(MARKS[i % MARKS.length], SERIES[i % SERIES.length]) +
                       (NICE[t] || t) + '</span>').join("") +
    '<span><i class="ln"></i>most-likely</span>' +
    '<span><i class="ln d"></i>plan-for (25th percentile)</span>' +
    '<span>bands: good 7&ndash;9 &middot; fair 5&ndash;6 &middot; poor 0&ndash;4</span>';
}
function marker(cx, cy, kind, c){
  const r = 4.2, ring = ' stroke="#FFFFFF" stroke-width="2"';   // 2px surface ring, not a border
  if (kind === "square")   return '<rect x="'+(cx-r)+'" y="'+(cy-r)+'" width="'+(2*r)+'" height="'+(2*r)+'" fill="'+c+'"'+ring+'/>';
  if (kind === "triangle") return '<polygon points="'+cx+','+(cy-r-1)+' '+(cx+r+1)+','+(cy+r)+' '+(cx-r-1)+','+(cy+r)+'" fill="'+c+'"'+ring+'/>';
  if (kind === "diamond")  return '<polygon points="'+cx+','+(cy-r-1)+' '+(cx+r+1)+','+cy+' '+cx+','+(cy+r+1)+' '+(cx-r-1)+','+cy+'" fill="'+c+'"'+ring+'/>';
  return '<circle cx="'+cx+'" cy="'+cy+'" r="'+r+'" fill="'+c+'"'+ring+'/>';
}
function svgChip(kind, c){
  return '<svg width="14" height="14" style="vertical-align:-2px;margin-right:6px">' +
         marker(7, 7, kind, c).replace(/stroke="#FFFFFF" stroke-width="2"/, '') + '</svg>';
}

/* ---------- the side rail: how this number was produced ---------- */
function drawRail(b, h){
  const t = focusTarget, d = b.targets[t], r = route(b, h);
  const like = interp(d.likely, h), cons = interp(d.cons, h), rk = riskAt(d, h);
  const drivers = (EX.drivers && EX.drivers[t]) || [];
  const modelPath = (r.kind === "model" || r.kind === "interp");

  let s = '<div class="rh"><div class="t">How this was calculated</div>' +
          '<div class="s">' + (NICE[t] || t) + ' &middot; bridge ' + esc(b.id) + ' &middot; +' + h + 'y</div></div>';

  s += '<section><h4>Path taken</h4><span class="badge ' + r.kind + '">' + r.label + '</span>' +
       '<p class="exp">' + r.text + '</p></section>';

  s += '<section><h4>Result at +' + h + ' years</h4><table class="kv">' +
       row("Most-likely rating", fmt(like) + " (" + ratingWord(like) + ")") +
       row("Plan-for (25th pctile)", fmt(cons)) +
       row("Risk of reaching ≤ " + META.risk_threshold, rk == null ? "not produced" : Math.round(rk) + "%") +
       '</table></section>';

  s += '<section><h4>Inputs fed to the model</h4><table class="kv">' +
       row("r0 &mdash; rating at last inspection", fmt(d.current)) +
       row("horizon &mdash; years ahead", modelPath ? (r.kind === "interp" ? r.lo + " &amp; " + r.hi : h) : "n/a") +
       row("age_t0 &mdash; age at inspection", b.age_t0 == null ? "—" : b.age_t0 + " yr") +
       row("attributes present", b.feat_present + " of " + b.feat_total) +
       '</table>' +
       '<p class="exp tiny" style="margin-top:8px">age_t0 is clipped to 0&ndash;130 years to match ' +
       'training. Attributes the record does not carry are passed as missing and handled natively ' +
       'by the tree ensemble &mdash; the forecast is still produced, from less information.' +
       (missingNames(b).length
          ? ' Missing here: <span class="mono">' + missingNames(b).map(esc).join(", ") + '</span>.' : '') +
       '</p></section>';

  if (modelPath){
    s += '<section><h4>Estimators &mdash; three separate models</h4><ul class="est">' +
      '<li><b><span class="swatch" style="background:var(--s1)"></span>Most-likely</b>' +
      '<span>Gradient-boosted regression tree ensemble, squared-error objective. Predicts the ' +
      'expected future rating.</span></li>' +
      '<li><b><span class="swatch" style="background:var(--s2)"></span>Plan-for</b>' +
      '<span>Separate ensemble fit with a quantile objective at &alpha;&nbsp;=&nbsp;' +
      META.conservative_quantile + '. Deliberately errs toward worse condition so a budget ' +
      'watch-list catches more of the fast decliners.</span></li>' +
      '<li><b><span class="swatch" style="background:var(--s3)"></span>Risk of poor</b>' +
      '<span>Gradient-boosted classifier, logistic objective, trained on the label ' +
      '<span class="mono">future rating &le; ' + META.risk_threshold + '</span>. Reported as its ' +
      'predicted probability. Not post-hoc calibrated.</span></li></ul>' +
      '<p class="exp tiny" style="margin-top:8px">All three are evaluated on the same feature row ' +
      'and share one set of hyper-parameters. Outputs are clipped to the valid 0&ndash;9 rating ' +
      'range, and the plan-for value is held at or below the most-likely value so the two can ' +
      'never cross.</p></section>';

    if (drivers.length){
      s += '<section><h4>What drives this model</h4>' +
           drivers.map(dr => {
             const ov = ownValue(b, dr.feature);
             const own = ov != null
               ? '<div class="own">this bridge: <b>' + esc(ov) + '</b></div>' : '';
             return '<div class="drv"><div class="n">' + esc(prettyFeat(dr.feature)) + '</div>' +
                    '<div class="v">' + (dr.importance * 100).toFixed(1) + '%</div>' +
                    '<div class="track"><div class="fill" style="width:' +
                    Math.max(2, dr.importance / drivers[0].importance * 100).toFixed(1) + '%"></div></div>' +
                    own + '</div>';
           }).join("") +
           '<p class="exp tiny">Model-wide importance across the whole training set &mdash; how much ' +
           'each input shapes the model in general. It is <b>not</b> an attribution for this one ' +
           'bridge; no per-prediction attribution is computed.</p></section>';
    }
  }

  s += '<section><h4>Provenance</h4><table class="kv">' +
       row("Data source", esc(META.source_label)) +
       row("Last inspection", esc(b.last_inspection || String(b.last_year))) +
       row("Model trained", esc(META.model_trained || "—")) +
       row("Page generated", esc(META.generated)) +
       '</table></section>';

  document.getElementById('rail').innerHTML = s;
}
function row(k, v){ return '<tr><td>' + k + '</td><td>' + v + '</td></tr>'; }
function prettyFeat(f){
  const m = {r0:"Rating at last inspection", horizon:"Years ahead", age_t0:"Age at last inspection",
    load_posting_status:"Load posting status", span_continuity:"Span continuity",
    wearing_surface:"Wearing surface", structure_kind:"Structure material",
    structure_type:"Structure type", deck_type:"Deck type", deck_protection:"Deck protection",
    inventory_load_rating_factor:"Inventory load rating", operating_load_rating_factor:"Operating load rating",
    txdot_district:"District", maintenance_resp:"Maintenance responsibility", owner:"Owner",
    num_beam_lines:"Beam lines", num_spans_main:"Main spans", adt:"Average daily traffic",
    adt_truck:"Truck traffic", adt_year:"Traffic count year", deck_width:"Deck width",
    roadway_width:"Roadway width", max_span_length:"Max span length", structure_length:"Structure length",
    approach_roadway_width:"Approach width", functional_class:"Functional class",
    design_load:"Design load", scour_vulnerability:"Scour vulnerability", skew_angle:"Skew angle",
    latitude:"Latitude", longitude:"Longitude"};
  return m[f] || f.replace(/_/g, " ");
}

/* ---------- wiring ---------- */
document.getElementById('filter').oninput = function(){
  const q = this.value.trim().toLowerCase();
  fillOptions(q ? ALL_IDS.filter(id => id.toLowerCase().indexOf(q) >= 0) : ALL_IDS);
  render();
};
sel.onchange = function(){ focusTarget = null; render(); };
document.getElementById('horizon').oninput = render;
render();
</script>
</body></html>"""
