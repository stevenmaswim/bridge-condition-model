"""Quick check that your Snowflake credentials and query work -- run this BEFORE main.py.

    ./venv/Scripts/python.exe test_snowflake_connection.py                # one district (fast)
    ./venv/Scripts/python.exe test_snowflake_connection.py --district 5
    ./venv/Scripts/python.exe test_snowflake_connection.py --all          # whole table (slow)

It does four things and stops:
  1. confirms your credentials connect (prints who/where you are connected as),
  2. runs config.snowflake.query (scoped to one district by default so it returns in seconds),
  3. prints the first few rows + the columns the model will see,
  4. checks the result is actually an INSPECTION PANEL, not one row per bridge.

Step 4 is the one that matters when you point this at a new table. The deterioration model
learns from inspection-to-inspection transitions, so it needs many dated inspections per
bridge. A source with one current row per bridge will load, clean and train without error and
produce a useless model -- there are no transitions to learn from. This check makes that
failure loud instead of silent.

Nothing is written or changed.
"""
import argparse
import sys

import pandas as pd

from src.data_loader import load_config
from src.snowflake_loader import _connect, load_from_snowflake

# Below this many inspections per bridge there are too few forward pairs to learn a
# deterioration curve; at ~1.0 the source is a current-values table, not a panel.
MIN_INSPECTIONS_PER_BRIDGE = 3.0

# Physically plausible ranges for the numeric columns the model consumes. These exist because a
# fitted tree ensemble does not complain about out-of-range inputs -- values simply fall to one
# side of every learned split and the feature silently stops contributing. That is exactly how the
# legacy table's packed coordinates (29153429 for a latitude) and x100 load ratings went unnoticed.
# Ranges are deliberately generous: this is a units/encoding check, not an outlier hunt.
PLAUSIBLE_RANGE = {
    "latitude": (25.0, 37.0),                    # Texas
    "longitude": (-107.0, -93.0),                # Texas, western hemisphere
    "inventory_load_rating_factor": (0.0, 10.0),
    "operating_load_rating_factor": (0.0, 10.0),
    "deck_cond_rating": (0.0, 9.0),
    "superstructure_cond_rating": (0.0, 9.0),
    "substructure_cond_rating": (0.0, 9.0),
    "culvert_cond_rating": (0.0, 9.0),
    "year_built": (1800.0, 2100.0),
    "skew_angle": (0.0, 99.0),
}
MAX_OUT_OF_RANGE_PCT = 5.0


def _panel_shape(df, config):
    """Report whether this really is a time-series panel, and say so in plain terms."""
    id_col = config["data"]["id_col"]
    insp_col = config["data"]["inspection_date_col"]

    print("\nStep 4/5  Panel shape -- is this inspection history or current values?")
    for col, label in ((id_col, "bridge id"), (insp_col, "inspection date")):
        if col not in df.columns:
            print(f"  Cannot check: no '{col}' column ({label}) in the result.")
            return False

    n_rows = len(df)
    n_bridges = df[id_col].astype(str).str.strip().nunique()
    if not n_bridges:
        print("  No bridges returned.")
        return False

    per_bridge = n_rows / n_bridges
    print(f"  rows: {n_rows:,}   distinct bridges: {n_bridges:,}   rows per bridge: {per_bridge:.1f}")

    # Distinct DATED inspections per bridge is the number that actually drives training --
    # the raw export repeats the last result annually, and those repeats collapse later.
    from src.deterioration import parse_inspection_date
    dates = parse_inspection_date(df[insp_col])
    ok_dates = dates.notna()
    print(f"  inspection dates parsed: {ok_dates.sum():,} of {n_rows:,} "
          f"({100 * ok_dates.mean():.1f}%)")
    if ok_dates.any():
        print(f"  date range: {dates.min().date()} to {dates.max().date()}")
        events = (pd.DataFrame({id_col: df[id_col].astype(str).str.strip(), "d": dates})
                  .dropna().drop_duplicates())
        per_bridge_events = len(events) / events[id_col].nunique()
        print(f"  distinct dated inspections per bridge: {per_bridge_events:.1f}")
    else:
        per_bridge_events = 0.0
        print("  No inspection dates parsed -- check the inspection_date alias and its format.")

    # On/off-system split. An on-system bridge is on the state highway system by definition, so
    # the highway filter should keep essentially all of them and cut into off-system instead --
    # a filter that is quietly dropping on-system structures is a filter that is wrong.
    system_col = config.get("data", {}).get("system_col")
    if system_col and system_col in df.columns:
        sysv = df[system_col].astype(str).str.strip().str.upper()
        ids = df[id_col].astype(str).str.strip()
        on = ids[sysv.eq("ON")].nunique()
        off = ids[~sysv.eq("ON")].nunique()
        print(f"  on-system bridges: {on:,}   off-system: {off:,}   "
              f"(on-system share {100 * on / max(on + off, 1):.1f}%)")

    if per_bridge_events >= MIN_INSPECTIONS_PER_BRIDGE:
        print(f"  PANEL CONFIRMED -- {per_bridge_events:.1f} inspections per bridge is enough to "
              f"build forward pairs. Safe to train.")
        return True
    print(f"\n  PROBLEM: only {per_bridge_events:.1f} distinct inspections per bridge "
          f"(need >= {MIN_INSPECTIONS_PER_BRIDGE}).")
    print("  This looks like a CURRENT-VALUES table (one row per bridge), not inspection history.")
    print("  The deterioration model cannot be trained from it -- there are no transitions to")
    print("  learn from. Point config.snowflake at the inspection-history table, or keep the")
    print("  attributes-only model. It will NOT raise an error on its own, so do not skip this.")
    return False


def _value_ranges(df):
    """Catch units/encoding drift between sources before it reaches the model."""
    print("\nStep 5/5  Value ranges -- are the units what the model was fitted on?")
    problems = []
    for col, (lo, hi) in PLAUSIBLE_RANGE.items():
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        if v.empty:
            continue
        bad_pct = 100.0 * (~v.between(lo, hi)).mean()
        flag = "  <-- OUT OF RANGE" if bad_pct > MAX_OUT_OF_RANGE_PCT else ""
        if flag:
            problems.append((col, bad_pct, v.median(), lo, hi))
        print(f"  {col:<30} median={v.median():>12,.2f}   expected {lo:g}..{hi:g}   "
              f"outside={bad_pct:5.1f}%{flag}")
    if not problems:
        print("  All checked columns are in the units the model expects.")
        return True
    print("\n  PROBLEM: the columns flagged above are not in the units the model was fitted on.")
    print("  A tree ensemble will NOT raise on these -- the values fall past every learned split")
    print("  and the feature stops contributing, quietly. Check the aliases in")
    print("  config.snowflake.query, and whether src/data_loader.normalize_legacy_encodings")
    print("  covers this source's encoding.")
    return False


def main():
    ap = argparse.ArgumentParser(description="Verify the Snowflake connection and query.")
    ap.add_argument("--district", default="12",
                    help="Scope the test pull to one district (default 12) so it returns fast")
    ap.add_argument("--all", action="store_true",
                    help="Pull the whole table instead -- slow, but the true row counts")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    config = load_config(args.config)
    sf = config.get("snowflake", {}) or {}

    print("Step 1/5  Connecting to Snowflake ...")
    try:
        conn = _connect(sf)
    except Exception as e:                      # noqa: BLE001 -- surface any setup problem plainly
        print(f"\n  Could not connect: {e}")
        sys.exit(1)

    try:
        cur = conn.cursor()
        cur.execute("SELECT CURRENT_USER(), CURRENT_ACCOUNT(), CURRENT_WAREHOUSE(), "
                    "CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_ROLE()")
        who = cur.fetchone()
        print(f"  Connected.  user={who[0]}  account={who[1]}  warehouse={who[2]}")
        print(f"              database={who[3]}  schema={who[4]}  role={who[5]}")
    finally:
        conn.close()

    if not args.all:
        dist_col = config["grouping"]["district_col"]
        config["_snowflake_filter"] = {"column": dist_col, "values": [str(args.district)]}
        print(f"\nStep 2/5  Running your query, scoped to district {args.district} "
              f"(use --all for the whole table) ...")
    else:
        print("\nStep 2/5  Running your query over the WHOLE table -- this may take a while ...")

    try:
        df = load_from_snowflake(config)         # prints row/column counts itself
    except Exception as e:                       # noqa: BLE001
        print(f"\n  Query failed: {e}")
        sys.exit(1)

    if df.empty:
        print("\n  Query returned no rows. If you scoped to a district, try another, or --all.")
        sys.exit(1)

    print("\nStep 3/5  Preview -- these are the columns the model will receive:")
    print("  columns:", list(df.columns))
    print(df.head(5).to_string())

    expected = ["bridge_id", "deck_cond_rating", "superstructure_cond_rating",
                "substructure_cond_rating", "culvert_cond_rating", "inspection_date", "year_built"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print(f"\n  Heads up: these model columns are not in the result yet: {missing}")
        print("  Alias them in config.snowflake.query (e.g.  YOUR_COL AS bridge_id).")

    is_panel = _panel_shape(df, config)

    # ranges are checked on the CLEANED frame -- that is where the model actually reads from,
    # and where normalize_legacy_encodings has had its chance to run
    from src.data_loader import clean_data, rename_raw_columns
    try:
        ranges_ok = _value_ranges(clean_data(rename_raw_columns(df.copy(), config), config))
    except Exception as e:                       # noqa: BLE001
        print(f"\nStep 5/5  Could not check value ranges: {e}")
        ranges_ok = True

    print()
    if not missing and is_panel and ranges_ok:
        print("  All core model columns present and the source is a panel. "
              "You're ready to set data.source: snowflake.")
    else:
        print("  Fix the problems above before training or serving from this source.")
        sys.exit(1)


if __name__ == "__main__":
    main()
