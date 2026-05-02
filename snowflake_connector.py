"""
snowflake_connector.py
----------------------
Handles all Snowflake connections and raw data extraction via SnowSQL-style
queries. Uses snowflake-connector-python under the hood.
"""

import os
import logging
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
from contextlib import contextmanager

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_connection_params() -> dict:
    """Read Snowflake credentials from environment variables."""
    return {
        "account":   os.getenv("SNOWFLAKE_ACCOUNT"),
        "user":      os.getenv("SNOWFLAKE_USER"),
        "password":  os.getenv("SNOWFLAKE_PASSWORD"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        "database":  os.getenv("SNOWFLAKE_DATABASE"),
        "schema":    os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
        "role":      os.getenv("SNOWFLAKE_ROLE", "SYSADMIN"),
    }


@contextmanager
def snowflake_connection():
    """Context manager that opens/closes a Snowflake connection safely."""
    params = get_connection_params()
    conn = snowflake.connector.connect(**params)
    logger.info("Snowflake connection established.")
    try:
        yield conn
    finally:
        conn.close()
        logger.info("Snowflake connection closed.")


def run_query(sql: str) -> pd.DataFrame:
    """
    Execute a SQL query against Snowflake and return results as a DataFrame.

    Parameters
    ----------
    sql : str  — Any valid Snowflake SQL statement.

    Returns
    -------
    pd.DataFrame with query results. Empty DataFrame on error.
    """
    with snowflake_connection() as conn:
        try:
            cur = conn.cursor()
            cur.execute(sql)
            df = cur.fetch_pandas_all()
            logger.info(f"Query returned {len(df)} rows.")
            return df
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def fetch_query_history(lookback_hours: int = 24) -> pd.DataFrame:
    """
    Pull query execution history from SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY.
    Captures query text, execution time, bytes scanned, status, and user.

    This view has ~45 min latency; use INFORMATION_SCHEMA for near-real-time.
    """
    sql = f"""
        SELECT
            QUERY_ID,
            QUERY_TEXT,
            DATABASE_NAME,
            SCHEMA_NAME,
            QUERY_TYPE,
            USER_NAME,
            WAREHOUSE_NAME,
            EXECUTION_STATUS,
            ERROR_MESSAGE,
            START_TIME,
            END_TIME,
            TOTAL_ELAPSED_TIME,          -- milliseconds
            BYTES_SCANNED,
            ROWS_PRODUCED,
            COMPILATION_TIME,
            EXECUTION_TIME,
            QUEUED_OVERLOAD_TIME
        FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
        WHERE START_TIME >= DATEADD(HOUR, -{lookback_hours}, CURRENT_TIMESTAMP())
          AND EXECUTION_STATUS IN ('SUCCESS', 'FAIL')
        ORDER BY START_TIME DESC;
    """
    logger.info(f"Fetching query history for the last {lookback_hours} hours...")
    return run_query(sql)


def fetch_table_statistics(database: str, schema: str) -> pd.DataFrame:
    """
    Pull row counts and byte sizes for all tables in a given schema.
    Used to detect sudden data volume spikes or drops.
    """
    sql = f"""
        SELECT
            TABLE_CATALOG,
            TABLE_SCHEMA,
            TABLE_NAME,
            ROW_COUNT,
            BYTES,
            LAST_ALTERED,
            CREATED
        FROM {database}.INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = '{schema}'
          AND TABLE_TYPE = 'BASE TABLE'
        ORDER BY BYTES DESC;
    """
    logger.info(f"Fetching table statistics for {database}.{schema}...")
    return run_query(sql)


def fetch_column_metadata(database: str, schema: str) -> pd.DataFrame:
    """
    Pull column-level metadata for schema drift detection.
    Tracks column names, data types, ordinal positions.
    """
    sql = f"""
        SELECT
            TABLE_NAME,
            COLUMN_NAME,
            ORDINAL_POSITION,
            DATA_TYPE,
            IS_NULLABLE,
            CHARACTER_MAXIMUM_LENGTH,
            NUMERIC_PRECISION
        FROM {database}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{schema}'
        ORDER BY TABLE_NAME, ORDINAL_POSITION;
    """
    logger.info(f"Fetching column metadata for {database}.{schema}...")
    return run_query(sql)


def fetch_null_rates(database: str, schema: str, table: str) -> pd.DataFrame:
    """
    Dynamically compute per-column NULL rates for a given table.
    Builds a UNION ALL query over all columns to get null % in one pass.
    """
    # First get column list
    col_sql = f"""
        SELECT COLUMN_NAME
        FROM {database}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table}'
        ORDER BY ORDINAL_POSITION;
    """
    cols_df = run_query(col_sql)
    if cols_df.empty:
        return pd.DataFrame()

    columns = cols_df["COLUMN_NAME"].tolist()

    # Build dynamic UNION ALL to count NULLs per column
    union_parts = [
        f"""SELECT '{col}' AS COLUMN_NAME,
                   COUNT(*) AS TOTAL_ROWS,
                   SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS NULL_COUNT,
                   ROUND(SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS NULL_PCT
            FROM {database}.{schema}.{table}"""
        for col in columns
    ]

    null_sql = "\nUNION ALL\n".join(union_parts) + ";"
    logger.info(f"Computing NULL rates for {table} across {len(columns)} columns...")
    return run_query(null_sql)


def fetch_load_history(lookback_hours: int = 24) -> pd.DataFrame:
    """
    Pull COPY INTO load history to detect load failures and partial loads.
    """
    sql = f"""
        SELECT
            TABLE_NAME,
            SCHEMA_NAME,
            FILE_NAME,
            LAST_LOAD_TIME,
            STATUS,
            ROW_COUNT,
            ROW_PARSED,
            FIRST_ERROR_MESSAGE,
            FIRST_ERROR_LINE_NUMBER,
            ERROR_COUNT,
            ERROR_LIMIT
        FROM SNOWFLAKE.ACCOUNT_USAGE.LOAD_HISTORY
        WHERE LAST_LOAD_TIME >= DATEADD(HOUR, -{lookback_hours}, CURRENT_TIMESTAMP())
        ORDER BY LAST_LOAD_TIME DESC;
    """
    logger.info(f"Fetching load history for the last {lookback_hours} hours...")
    return run_query(sql)