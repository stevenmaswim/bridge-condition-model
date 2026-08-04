"""Load bridge data live from Snowflake instead of a local CSV.

SECURITY: credentials come from environment variables (or a .env file), never from code or config,
because this repo is public. `.env` is already in .gitignore. See the README / walkthrough.

Enable it by setting `data.source: snowflake` in config.yaml and filling the `snowflake:` block.
"""
import os

import pandas as pd

# Optional: load a local .env file if python-dotenv is installed (pip install python-dotenv).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _connect(sf):
    """Open a Snowflake connection. Account/user come from env vars; auth is password or SSO."""
    try:
        import snowflake.connector
    except ImportError as e:
        raise ImportError(
            "The Snowflake connector is not installed. Run:\n"
            "    ./venv/Scripts/python.exe -m pip install \"snowflake-connector-python[pandas]\"") from e

    params = dict(
        account=os.environ.get("SNOWFLAKE_ACCOUNT", sf.get("account")),
        user=os.environ.get("SNOWFLAKE_USER", sf.get("user")),
        warehouse=sf.get("warehouse"),
        database=sf.get("database"),
        schema=sf.get("schema"),
        role=sf.get("role"),
    )
    missing = [k for k in ("account", "user") if not params.get(k)]
    if missing:
        raise RuntimeError(f"Missing Snowflake {', '.join(missing)} -- set SNOWFLAKE_ACCOUNT / "
                           f"SNOWFLAKE_USER (env vars) or config.snowflake.")

    auth = sf.get("authenticator", "password")
    if auth == "externalbrowser":      # single sign-on (opens a browser to authenticate)
        params["authenticator"] = "externalbrowser"
    else:                              # username / password
        params["password"] = os.environ.get("SNOWFLAKE_PASSWORD")
        if not params["password"]:
            raise RuntimeError("Set SNOWFLAKE_PASSWORD (env var), or use authenticator: externalbrowser.")
    return snowflake.connector.connect(**params)


def _apply_filter(query, flt):
    """Wrap the aliased query as a subquery and add a WHERE, so Snowflake only returns the rows we
    want (e.g. one district or a handful of bridge_ids) instead of the whole 1.7M-row table.
    `flt` = {"column": <aliased name>, "values": [...]}. Filtering happens on the OUTER query, so it
    references the aliases (bridge_id, txdot_district) -- not the raw Snowflake column names."""
    if not flt or not flt.get("values"):
        return query
    column = flt["column"]                                  # our own alias name, not user input
    # escape single quotes in the values so a stray apostrophe can't break (or inject into) the SQL
    vals = ", ".join("'" + str(v).replace("'", "''") + "'" for v in flt["values"])
    return f"SELECT * FROM (\n{query}\n) AS _sub WHERE {column} IN ({vals})"


def load_from_snowflake(config):
    """Run the configured query and return a pandas DataFrame. Use `query` (with column aliases that
    match the model's expected names) or `table` in config.snowflake. An optional runtime filter at
    config["_snowflake_filter"] narrows the pull to specific bridges/districts (set by forecast_ui)."""
    sf = config.get("snowflake", {}) or {}
    query = sf.get("query")
    if not query:
        table = sf.get("table")
        if not table:
            raise RuntimeError("config.snowflake needs either a `query` or a `table`.")
        query = f"SELECT * FROM {table}"

    query = _apply_filter(query, config.get("_snowflake_filter"))

    conn = _connect(sf)
    try:
        cur = conn.cursor()
        cur.execute(query)
        df = cur.fetch_pandas_all()   # requires snowflake-connector-python[pandas]
    finally:
        conn.close()

    # Snowflake folds unquoted identifiers/aliases to UPPERCASE (BRIDGE_ID), but the model's
    # config uses lowercase snake_case (bridge_id). Normalise so the query's `AS name` aliases
    # line up with the model's expected column names regardless of Snowflake's casing.
    df.columns = [str(c).lower() for c in df.columns]

    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns from Snowflake")
    return df
