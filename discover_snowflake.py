"""Explore what's in your Snowflake so we can point the model at the right table.

Run it with no arguments to list the databases you can see:
    ./venv/Scripts/python.exe discover_snowflake.py

Then drill down by passing what you found:
    ./venv/Scripts/python.exe discover_snowflake.py MY_DATABASE            # lists its schemas
    ./venv/Scripts/python.exe discover_snowflake.py MY_DATABASE MY_SCHEMA  # lists its tables
    ./venv/Scripts/python.exe discover_snowflake.py MY_DATABASE MY_SCHEMA MY_TABLE  # lists its columns

Copy whatever it prints and hand it back -- that's everything needed to fill in config.yaml.
Nothing is written or changed; it only reads catalog listings.
"""
import sys

from src.data_loader import load_config
from src.snowflake_loader import _connect


def main():
    args = [a for a in sys.argv[1:] if a.strip()]
    config = load_config()
    conn = _connect(config.get("snowflake", {}) or {})
    cur = conn.cursor()

    try:
        if len(args) == 0:
            cur.execute("SHOW DATABASES")
            names = [r[1] for r in cur.fetchall()]
            print(f"\nDatabases you can see ({len(names)}):")
            for n in names:
                print(f"   {n}")
            print("\nNext: re-run with one of these, e.g.")
            print(f"   ./venv/Scripts/python.exe discover_snowflake.py {names[0] if names else 'DBNAME'}")

        elif len(args) == 1:
            db = args[0]
            cur.execute(f'SHOW SCHEMAS IN DATABASE "{db}"')
            names = [r[1] for r in cur.fetchall()]
            print(f"\nSchemas in {db} ({len(names)}):")
            for n in names:
                print(f"   {n}")
            print("\nNext: re-run with a schema, e.g.")
            print(f"   ./venv/Scripts/python.exe discover_snowflake.py {db} {names[0] if names else 'SCHEMA'}")

        elif len(args) == 2:
            db, schema = args
            cur.execute(f'SHOW TABLES IN SCHEMA "{db}"."{schema}"')
            tables = [r[1] for r in cur.fetchall()]
            cur.execute(f'SHOW VIEWS IN SCHEMA "{db}"."{schema}"')
            views = [r[1] for r in cur.fetchall()]
            print(f"\nTables in {db}.{schema} ({len(tables)}):")
            for n in tables:
                print(f"   {n}")
            print(f"\nViews in {db}.{schema} ({len(views)}):")
            for n in views:
                print(f"   {n}")
            print("\nNext: pick the one with bridge inspection data and list its columns, e.g.")
            first = (tables or views or ['TABLE'])[0]
            print(f"   ./venv/Scripts/python.exe discover_snowflake.py {db} {schema} {first}")

        else:
            db, schema, table = args[:3]
            cur.execute(f'DESCRIBE TABLE "{db}"."{schema}"."{table}"')
            rows = cur.fetchall()
            print(f"\nColumns in {db}.{schema}.{table} ({len(rows)}):")
            for r in rows:
                print(f"   {r[0]:<40} {r[1]}")     # name, type
            print("\nCopy this whole list back -- I'll map these to the model's column names.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
