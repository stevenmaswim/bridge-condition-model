"""Build one TxDOT-branded forecast report per district, plus an index page linking them.

    ./venv/Scripts/python.exe build_district_reports.py
    ./venv/Scripts/python.exe build_district_reports.py --districts 12,15,18
    ./venv/Scripts/python.exe build_district_reports.py --on-system-only

Why per district rather than one statewide page: at ~1 KB of embedded forecast per bridge, a
statewide report is 40-70 MB -- too large to email and slow to parse before first paint. One
file per district is ~7 MB, opens instantly, and matches how responsibility is actually
assigned. The index page is what you send; each district page is what that district opens.

The source panel is pulled ONCE and reused for every district. That matters with Snowflake:
each connection under externalbrowser auth triggers an interactive SSO round trip, so a naive
loop would ask you to log in 25 times.
"""
import argparse
import datetime as dt
import html
import os
import re
import zipfile

import pandas as pd

from src.data_loader import load_config
from src.report_template import BRAND_CSS, LOGO_SVG
from forecast_ui import (build_data, driver_feature_names, load_panel,
                         load_validation_exhibits, write_html)


def _assert_filled(page, what):
    """Fail loudly on an unsubstituted %%PLACEHOLDER%%.

    These render as a literal "%%LO%%" on the page: visible to any reader, invisible to the
    tests, and embarrassing on something going to a director. Cheaper to catch here.
    """
    left = sorted(set(re.findall(r"%%[A-Z_]+%%", page)))
    if left:
        raise RuntimeError(f"{what}: unsubstituted placeholders {left}")


def _safe(name):
    """District identifiers come from the data, so keep them to a filename-safe subset."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(name).strip())


INDEX_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bridge Condition Forecast &mdash; District Reports</title>
<style>%%CSS%%
.dgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(268px,1fr));gap:14px}
a.dcard{display:block;background:var(--card);border:1px solid var(--line);
        border-left:4px solid var(--tx-blue-600);padding:15px 17px;text-decoration:none;color:inherit}
a.dcard:hover{background:var(--tx-blue-050);border-left-color:var(--tx-blue-900)}
a.dcard .d{font-size:20px;font-weight:700;color:var(--tx-blue-900)}
a.dcard .n{font-size:13px;color:var(--muted);margin-top:3px}
a.dcard .w{font-size:12.5px;margin-top:7px;font-weight:700}
</style></head>
<body>
<div class="brandbar">%%LOGO%%<div class="tagline">Connecting you with Texas</div></div>
<div class="redrule"></div>

<div class="hero">
  <h1>Bridge Condition Forecast &mdash; District Reports</h1>
  <p class="deck">Each district has its own report: pick a bridge, set a horizon, and see the
     forecast condition of every component it carries alongside an explanation of exactly how
     that number was produced. Decision support for screening &mdash; not a replacement for
     inspection.</p>
  <p class="scope">Scope: <b>%%NDIST%% districts</b> &middot; <b>%%NBRIDGE%% bridges</b>
     &middot; source <b>%%SOURCE%%</b> &middot; generated <b>%%GENERATED%%</b></p>
</div>

<main>
  <section class="panel">
    <h2>Districts</h2>
    <div class="body">
      <p class="note" style="margin-top:0">Bridges shown are those with enough inspection history
         to forecast. <b>Watch-list</b> counts bridges currently rated %%LO%%&ndash;%%HI%% whose
         conservative (25th-percentile) %%HORIZON%%-year forecast reaches %%THRESH%% or below in at
         least one component &mdash; the same screen <code>build_watchlist.py</code> applies.
         Already-poor structures are inspection priorities rather than forecasts, and near-new ones
         are just aging normally, so both are excluded.</p>
      <div class="callout warn" style="margin:14px 0 0"><b>Read the watch-list within a district,
         not across them.</b> The rate tracks how complete each district&rsquo;s attribute data is
         far more closely than how worn its bridges are &mdash; across districts the two correlate
         <b>&minus;0.96</b>. Where physical attributes are missing the conservative model leans
         pessimistic, so a district with patchy records looks worse than one with tidy records
         regardless of actual condition. Attribute coverage is shown on each card for that reason.
         Ranking districts against each other on this number would be reading the data-entry
         backlog, not the bridge inventory.</div>
      <div class="dgrid" style="margin-top:16px">%%CARDS%%</div>
    </div>
  </section>
</main>

<footer><div class="fr">
  <b>Limitations.</b> The model projects trends; it cannot foresee sudden events such as scour or
  vehicle impact, so inspection remains the safety net. It is weakest on bridges already rated
  0&ndash;4 and tends to over-predict their recovery. Rating increases from repair or rehabilitation
  are treated as noise rather than predicted. Not yet reviewed by a bridge engineer.
</div></footer>
</body></html>"""


def _coverage(bridges):
    """Mean share of model attributes actually present, per district. This is reported on the
    index because it is a confounder, not a footnote: across districts it correlates -0.96 with
    the watch-list rate. Where attributes are missing the conservative model leans pessimistic,
    so a low-coverage district looks like it has worse bridges when it may just have worse data.
    """
    vals = [b["feat_present"] / b["feat_total"] for b in bridges if b.get("feat_total")]
    return 100.0 * sum(vals) / len(vals) if vals else 0.0


def _trending_poor(bridges, watchlist):
    """Count bridges on the same screen build_watchlist.py uses, so the headline on the index
    matches the tool anyone would act on.

    Screening on "conservative forecast <= 5.0" alone is not that screen: it trivially catches
    every bridge already rated 5 or below, which is not a bridge trending to poor but one that
    is already there. On this data that inflated district 4 from a plausible figure to 61% of
    its inventory -- the kind of number that gets a deck thrown out. The config's own band
    (currently 5-7, excluding already-poor and near-new) is the defensible one.
    """
    lo = watchlist.get("current_min", 5)
    hi = watchlist.get("current_max", 7)
    horizon = watchlist.get("horizon_years", 10)
    threshold = watchlist.get("poor_threshold", 5.0)
    n = 0
    for b in bridges:
        for d in b["targets"].values():
            current = d.get("current")
            v = d["cons"].get(str(horizon), d["cons"].get(horizon))
            if current is None or v is None:
                continue
            if lo <= current <= hi and v <= threshold:
                n += 1
                break
    return n


def write_index(rows, meta, out_dir, watchlist):
    cards = []
    for r in sorted(rows, key=lambda x: (len(str(x["district"])), str(x["district"]))):
        pct = (100.0 * r["poor"] / r["bridges"]) if r["bridges"] else 0.0
        # Deliberately NOT colour-scaled by rate: colouring it would invite exactly the
        # cross-district ranking the caveat above says the number cannot support.
        cards.append(
            f'<a class="dcard" href="{html.escape(r["file"])}">'
            f'<div class="d">District {html.escape(str(r["district"]))}</div>'
            f'<div class="n">{r["bridges"]:,} bridges &middot; {r["size_mb"]:.1f} MB</div>'
            f'<div class="w">{r["poor"]:,} on the watch-list '
            f'<span class="muted" style="font-weight:400">({pct:.0f}%)</span></div>'
            f'<div class="n">attribute coverage {r["coverage"]:.0f}%</div></a>')
    page = (INDEX_TEMPLATE
            .replace("%%CSS%%", BRAND_CSS)
            .replace("%%LOGO%%", LOGO_SVG)
            .replace("%%NDIST%%", f"{len(rows):,}")
            .replace("%%NBRIDGE%%", f"{sum(r['bridges'] for r in rows):,}")
            .replace("%%SOURCE%%", html.escape(meta.get("source_label", "—")))
            .replace("%%GENERATED%%", html.escape(meta.get("generated", "")))
            .replace("%%HORIZON%%", str(watchlist.get("horizon_years", 10)))
            .replace("%%THRESH%%", str(watchlist.get("poor_threshold", 5.0)))
            .replace("%%LO%%", str(watchlist.get("current_min", 5)))
            .replace("%%HI%%", str(watchlist.get("current_max", 7)))
            .replace("%%CARDS%%", "\n".join(cards)))
    _assert_filled(page, "index.html")
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


README = """BRIDGE CONDITION FORECAST - DISTRICT REPORTS
Texas Department of Transportation
Generated {generated}

WHAT THIS IS
------------
A forecast of how each bridge's NBI condition ratings (0-9) are likely to change
over the next 20 years, so deterioration can be priced into maintenance and
capital budgets before a structure reaches poor condition.

It is decision support for screening. It does not replace inspection.

HOW TO OPEN IT
--------------
Unzip the folder, then double-click:

    index.html

That lists all {ndist} districts. Click a district to open its report. Everything
works offline in any browser - no install, no network, no login.

USING A DISTRICT REPORT
-----------------------
  1. Pick a bridge by its NBI number (type in the filter box to narrow the list).
  2. Drag the horizon slider to the year you care about.
  3. The table shows, for every component that bridge carries:
       Current       the rating at its last inspection
       Most-likely   the expected future rating
       Plan-for      a deliberately conservative (25th percentile) figure for
                     budgeting - it errs toward worse condition on purpose
       Risk of poor  the chance that component reaches {thresh} or below
  4. The panel on the right explains how that specific number was produced -
     which model ran, what was fed into it, and what the model relies on.
  5. Scroll down for the methodology and the accuracy measurements behind it.

WHAT "WATCH-LIST" MEANS ON THE INDEX
------------------------------------
Bridges currently rated {lo}-{hi} whose conservative {horizon}-year forecast reaches
{thresh} or below in at least one component. Bridges already rated below {lo} are
inspection priorities rather than forecasting problems, and near-new bridges are
simply aging normally, so both are excluded.

IMPORTANT - DO NOT RANK DISTRICTS AGAINST EACH OTHER ON THAT NUMBER
Across districts, the watch-list rate tracks how complete each district's
attribute records are far more closely than how worn its bridges are (they
correlate -0.96). Where physical attributes are missing, the model leans
pessimistic. A district with patchy records will look worse than one with tidy
records regardless of actual condition. Each card shows its attribute coverage
for that reason. The number is meaningful WITHIN a district, not between them.

DATA SOURCE
-----------
{source}
{nbridge} bridges across {ndist} districts.

WHAT THE MODEL CANNOT DO
------------------------
  - It cannot foresee sudden events such as scour or vehicle impact. Those are
    events, not trends. Inspection remains the safety net.
  - It is weakest on bridges already rated 0-4 and tends to over-predict their
    recovery. It should not be used to justify deferring work on those.
  - It treats rating increases from repair or rehabilitation as noise rather
    than predicting them.
  - Longer horizons rest on fewer historical examples than short ones.
  - It has not yet been reviewed by a bridge engineer.

Accuracy figures, the comparison against the two methods currently used in
practice, and the known weaknesses are all in the "Methodology & validation"
section at the bottom of every district report.
"""


def write_zip(rows, meta, out_dir, watchlist, zip_path):
    """Package the folder plus a plain-text guide, ready to email."""
    folder = "Bridge_Condition_Forecast"
    readme = README.format(
        generated=meta.get("generated", ""), source=meta.get("source_label", "—"),
        nbridge=f"{sum(r['bridges'] for r in rows):,}", ndist=len(rows),
        lo=watchlist.get("current_min", 5), hi=watchlist.get("current_max", 7),
        horizon=watchlist.get("horizon_years", 10), thresh=watchlist.get("poor_threshold", 5.0))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr(f"{folder}/README.txt", readme)
        z.write(os.path.join(out_dir, "index.html"), f"{folder}/index.html")
        for r in rows:
            z.write(os.path.join(out_dir, r["file"]), f"{folder}/{r['file']}")
    return zip_path


def main():
    ap = argparse.ArgumentParser(description="Build one forecast report per district.")
    ap.add_argument("--out-dir", default="reports",
                    help="Folder to write the reports and index into (default: reports)")
    ap.add_argument("--districts", default=None,
                    help="Comma-separated districts (default: every district in the data)")
    ap.add_argument("--on-system-only", action="store_true",
                    help="Restrict to state-maintained (on-system) bridges")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--review-dir", default="sme_review")
    ap.add_argument("--zip", dest="zip_path", nargs="?", const="auto", default=None,
                    metavar="PATH",
                    help="Also package the folder as a .zip ready to send (default name if "
                         "no path given)")
    args = ap.parse_args()

    config = load_config(args.config)
    id_col = config["data"]["id_col"]
    dist_col = config["grouping"]["district_col"]
    system_col = config.get("data", {}).get("system_col")
    os.makedirs(args.out_dir, exist_ok=True)

    # Clear previously generated pages. Without this a rerun that produces different district
    # keys leaves the old files sitting beside the new ones -- and since every page states its
    # own source and date only in small print, a stale CSV-sourced district silently ships
    # alongside fresh live ones. Only files this script writes are removed.
    stale = [f for f in os.listdir(args.out_dir)
             if f == "index.html" or re.fullmatch(r"district_[A-Za-z0-9_-]+\.html", f)]
    for f in stale:
        os.remove(os.path.join(args.out_dir, f))
    if stale:
        print(f"Cleared {len(stale)} previously generated file(s) from {args.out_dir}/")

    exhibits = load_validation_exhibits(args.review_dir)
    if not exhibits:
        print(f"Note: no validation exhibits in {args.review_dir}/ -- "
              f"pages will omit the KPI band and validation appendix.")

    print("Pulling the inspection panel once (reused for every district) ...")
    events = load_panel(config)
    if system_col and system_col in events.columns and args.on_system_only:
        before = events[id_col].nunique()
        events = events[events[system_col].astype(str).str.strip().str.upper().eq("ON")]
        print(f"  on-system only: {events[id_col].nunique():,} of {before:,} bridges")

    if args.districts:
        districts = [d.strip() for d in args.districts.split(",") if d.strip()]
    else:
        districts = sorted(events[dist_col].dropna().astype(str).str.strip().unique(),
                           key=lambda d: (len(d), d))
    print(f"Building {len(districts)} district reports into {args.out_dir}/ ...\n")

    watchlist = (config.get("deterioration", {}) or {}).get("watchlist", {}) or {}
    drivers = driver_feature_names(exhibits)
    rows, meta, skipped = [], {}, []
    for i, d in enumerate(districts, 1):
        bridges, targets, dmeta = build_data(config, district=d, driver_features=drivers,
                                             events=events)
        if not bridges:
            skipped.append(d)
            print(f"  [{i:>2}/{len(districts)}] district {d:<4} no forecastable bridges, skipped")
            continue
        meta = meta or dmeta
        fname = f"district_{_safe(d)}.html"
        path = os.path.join(args.out_dir, fname)
        write_html(bridges, targets, dmeta, exhibits, path)
        size_mb = os.path.getsize(path) / 1e6
        poor = _trending_poor(bridges, watchlist)
        cov = _coverage(bridges)
        rows.append({"district": d, "file": fname, "bridges": len(bridges),
                     "size_mb": size_mb, "poor": poor, "coverage": cov})
        print(f"  [{i:>2}/{len(districts)}] district {d:<4} {len(bridges):>6,} bridges  "
              f"{size_mb:>5.1f} MB  {poor:>5,} watch-list  {cov:>4.0f}% coverage")

    if not rows:
        raise SystemExit("No district produced a report -- check the source and filters.")

    index = write_index(rows, meta, args.out_dir, watchlist)
    total = sum(r["bridges"] for r in rows)
    disk = sum(r["size_mb"] for r in rows)
    print(f"\nWrote {len(rows)} district reports + index: {total:,} bridges, {disk:.1f} MB total")
    if skipped:
        print(f"Skipped (no forecastable bridges): {', '.join(map(str, skipped))}")
    if args.zip_path:
        name = (args.zip_path if args.zip_path != "auto"
                else f"TxDOT_Bridge_Condition_Forecast_"
                     f"{dt.datetime.now().strftime('%Y-%m-%d')}.zip")
        write_zip(rows, meta, args.out_dir, watchlist, name)
        print(f"Packaged {name} ({os.path.getsize(name)/1e6:.1f} MB) -- ready to send")
    print(f"Open {index}")


if __name__ == "__main__":
    main()
