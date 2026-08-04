"""Quick check that your Snowflake credentials and query work -- run this BEFORE main.py.

    ./venv/Scripts/python.exe test_snowflake_connection.py

It does three things and stops:
  1. confirms the connector is installed,
  2. confirms your credentials connect (prints who/where you are connected as),
  3. runs config.snowflake.query (or table) and prints the first few rows + the columns
     the model will see.

Nothing is written or changed. Fix any error it reports, then run the real pipeline with
data.source set to "snowflake" in config.yaml.
"""
import sys

from src.data_loader import load_config
from src.snowflake_loader import _connect, load_from_snowflake


def main():
    config = load_config()
    sf = config.get("snowflake", {}) or {}

    print("Step 1/3  Connecting to Snowflake ...")
    try:
        conn = _connect(sf)
    except Exception as e:                      # noqa: BLE001 -- surface any setup problem plainly
        print(f"\n  Could not connect: {e}")
        sys.exit(1)

    try:
        cur = conn.cursor()
        cur.execute("SELECT CURRENT_USER(), CURRENT_ACCOUNT(), CURRENT_WAREHOUSE(), "
                    "CURRENT_DATABASE(), CURRENT_SCHEMA()")
        who = cur.fetchone()
        print(f"  Connected.  user={who[0]}  account={who[1]}  warehouse={who[2]}  "
              f"database={who[3]}  schema={who[4]}")
    finally:
        conn.close()

    print("\nStep 2/3  Running your query (config.snowflake.query / table) ...")
    df = load_from_snowflake(config)            # prints row/column counts itself

    print("\nStep 3/3  Preview -- these are the columns the model will receive:")
    print("  columns:", list(df.columns))
    print(df.head(5).to_string())

    # Friendly reminder: the model expects these snake_case names (alias to them in your query).
    expected = ["bridge_id", "deck_cond_rating", "superstructure_cond_rating",
                "substructure_cond_rating", "culvert_cond_rating", "inspection_date", "year_built"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print(f"\n  Heads up: these model columns are not in the result yet: {missing}")
        print("  Alias them in config.snowflake.query (e.g.  YOUR_COL AS bridge_id).")
    else:
        print("\n  All core model columns are present. You're ready to set data.source: snowflake.")


if __name__ == "__main__":
    main()
