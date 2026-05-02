"""
alerting_engine.py
------------------
Automated alerting logic that fires on:
  1. Schema drift   — columns added / removed / type-changed vs. last known snapshot
  2. NULL surges    — a column's null rate exceeds threshold vs. rolling baseline
  3. Load failures  — COPY INTO jobs that errored or partially loaded

All incidents are logged to a structured Snowflake audit table via SnowSQL INSERT.
"""

import json
import logging
import os
import hashlib
from datetime import datetime, timezone
from typing import Optional
import pandas as pd

from snowflake_connector import run_query, snowflake_connection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audit table DDL — run once to bootstrap the audit table
# ---------------------------------------------------------------------------

AUDIT_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS OBSERVABILITY.AUDIT.INCIDENT_LOG (
    INCIDENT_ID       VARCHAR(64)    PRIMARY KEY,
    INCIDENT_TYPE     VARCHAR(50)    NOT NULL,       -- SCHEMA_DRIFT | NULL_SURGE | LOAD_FAILURE | QUERY_ANOMALY
    SEVERITY          VARCHAR(10)    NOT NULL,       -- HIGH | MEDIUM | LOW
    DATABASE_NAME     VARCHAR(255),
    SCHEMA_NAME       VARCHAR(255),
    TABLE_NAME        VARCHAR(255),
    COLUMN_NAME       VARCHAR(255),
    DESCRIPTION       TEXT           NOT NULL,
    METADATA          VARIANT,                       -- structured JSON payload
    DETECTED_AT       TIMESTAMP_NTZ  NOT NULL,
    RESOLVED          BOOLEAN        DEFAULT FALSE,
    RESOLVED_AT       TIMESTAMP_NTZ
);
"""


def bootstrap_audit_table():
    """Create the incident log table if it doesn't exist."""
    run_query(AUDIT_TABLE_DDL)
    logger.info("Audit table bootstrapped.")


# ---------------------------------------------------------------------------
# Incident writer
# ---------------------------------------------------------------------------

def _make_incident_id(incident_type: str, description: str, ts: datetime) -> str:
    """Deterministic ID so re-runs don't duplicate identical incidents."""
    raw = f"{incident_type}|{description}|{ts.date()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def log_incident(
    incident_type: str,
    description: str,
    severity: str = "MEDIUM",
    database_name: str = None,
    schema_name: str = None,
    table_name: str = None,
    column_name: str = None,
    metadata: dict = None,
) -> str:
    """
    Insert a single incident record into the Snowflake audit table.

    Parameters
    ----------
    incident_type : SCHEMA_DRIFT | NULL_SURGE | LOAD_FAILURE | QUERY_ANOMALY
    description   : Human-readable description of the incident.
    severity      : HIGH | MEDIUM | LOW
    metadata      : Optional dict of extra structured context (stored as VARIANT).

    Returns
    -------
    incident_id : str
    """
    ts = datetime.now(timezone.utc)
    incident_id = _make_incident_id(incident_type, description, ts)
    meta_json = json.dumps(metadata or {}).replace("'", "\\'")

    sql = f"""
        INSERT INTO OBSERVABILITY.AUDIT.INCIDENT_LOG
            (INCIDENT_ID, INCIDENT_TYPE, SEVERITY,
             DATABASE_NAME, SCHEMA_NAME, TABLE_NAME, COLUMN_NAME,
             DESCRIPTION, METADATA, DETECTED_AT)
        SELECT
            '{incident_id}',
            '{incident_type}',
            '{severity}',
            {f"'{database_name}'" if database_name else 'NULL'},
            {f"'{schema_name}'"   if schema_name   else 'NULL'},
            {f"'{table_name}'"    if table_name     else 'NULL'},
            {f"'{column_name}'"   if column_name    else 'NULL'},
            '{description}',
            PARSE_JSON('{meta_json}'),
            '{ts.isoformat()}'
        WHERE NOT EXISTS (
            SELECT 1 FROM OBSERVABILITY.AUDIT.INCIDENT_LOG
            WHERE INCIDENT_ID = '{incident_id}'
        );
    """

    with snowflake_connection() as conn:
        conn.cursor().execute(sql)

    logger.info(f"[{severity}] {incident_type} logged → {incident_id}: {description}")
    return incident_id


# ---------------------------------------------------------------------------
# 1. Schema drift detection
# ---------------------------------------------------------------------------

def detect_schema_drift(
    current_meta: pd.DataFrame,
    snapshot_path: str = "schema_snapshot.json",
    database: str = None,
    schema: str = None,
) -> list[dict]:
    """
    Compare current column metadata against a persisted snapshot.
    Flags: added columns, dropped columns, data type changes.

    Parameters
    ----------
    current_meta  : DataFrame from fetch_column_metadata()
    snapshot_path : Path to JSON file storing the last known schema state.

    Returns
    -------
    List of drift incident dicts.
    """
    incidents = []

    # Build current schema map: {table_name: {col_name: data_type}}
    current_map = {}
    for _, row in current_meta.iterrows():
        tbl = row["TABLE_NAME"]
        col = row["COLUMN_NAME"]
        dtype = row["DATA_TYPE"]
        current_map.setdefault(tbl, {})[col] = dtype

    # Load previous snapshot
    if not os.path.exists(snapshot_path):
        # First run — save snapshot and return
        with open(snapshot_path, "w") as f:
            json.dump(current_map, f, indent=2)
        logger.info(f"Schema snapshot created at {snapshot_path}. No drift to compare.")
        return incidents

    with open(snapshot_path) as f:
        previous_map = json.load(f)

    # Compare
    all_tables = set(current_map) | set(previous_map)
    for table in all_tables:
        curr_cols = current_map.get(table, {})
        prev_cols = previous_map.get(table, {})

        # Dropped columns
        for col in set(prev_cols) - set(curr_cols):
            desc = f"Column '{col}' DROPPED from {table}"
            inc_id = log_incident(
                "SCHEMA_DRIFT", desc, "HIGH",
                database_name=database, schema_name=schema, table_name=table,
                column_name=col,
                metadata={"previous_type": prev_cols[col]},
            )
            incidents.append({"incident_id": inc_id, "type": "SCHEMA_DRIFT", "detail": desc})

        # Added columns
        for col in set(curr_cols) - set(prev_cols):
            desc = f"Column '{col}' ADDED to {table} with type {curr_cols[col]}"
            inc_id = log_incident(
                "SCHEMA_DRIFT", desc, "MEDIUM",
                database_name=database, schema_name=schema, table_name=table,
                column_name=col,
                metadata={"new_type": curr_cols[col]},
            )
            incidents.append({"incident_id": inc_id, "type": "SCHEMA_DRIFT", "detail": desc})

        # Type changes
        for col in set(curr_cols) & set(prev_cols):
            if curr_cols[col] != prev_cols[col]:
                desc = (
                    f"Column '{col}' in {table} type changed: "
                    f"{prev_cols[col]} → {curr_cols[col]}"
                )
                inc_id = log_incident(
                    "SCHEMA_DRIFT", desc, "HIGH",
                    database_name=database, schema_name=schema, table_name=table,
                    column_name=col,
                    metadata={"from": prev_cols[col], "to": curr_cols[col]},
                )
                incidents.append({"incident_id": inc_id, "type": "SCHEMA_DRIFT", "detail": desc})

    # Update snapshot
    with open(snapshot_path, "w") as f:
        json.dump(current_map, f, indent=2)

    logger.info(f"Schema drift check complete. {len(incidents)} drift(s) detected.")
    return incidents


# ---------------------------------------------------------------------------
# 2. NULL surge detection
# ---------------------------------------------------------------------------

def detect_null_surges(
    null_rates_df: pd.DataFrame,
    threshold_pct: float = None,
    table_name: str = None,
    database: str = None,
    schema: str = None,
) -> list[dict]:
    """
    Flag columns where NULL percentage exceeds the configured threshold.

    Parameters
    ----------
    null_rates_df : DataFrame from fetch_null_rates() with columns
                    [COLUMN_NAME, TOTAL_ROWS, NULL_COUNT, NULL_PCT]
    threshold_pct : Alert if NULL_PCT > this value (default from .env: 30%)
    """
    if threshold_pct is None:
        threshold_pct = float(os.getenv("ALERT_NULL_SURGE_PCT", 0.30)) * 100

    incidents = []
    surged = null_rates_df[null_rates_df["NULL_PCT"] > threshold_pct]

    for _, row in surged.iterrows():
        col = row["COLUMN_NAME"]
        pct = row["NULL_PCT"]
        desc = f"NULL surge: '{col}' in {table_name} has {pct:.1f}% NULLs (threshold: {threshold_pct}%)"
        inc_id = log_incident(
            "NULL_SURGE", desc,
            severity="HIGH" if pct > 80 else "MEDIUM",
            database_name=database, schema_name=schema, table_name=table_name,
            column_name=col,
            metadata={
                "null_pct": float(pct),
                "null_count": int(row["NULL_COUNT"]),
                "total_rows": int(row["TOTAL_ROWS"]),
                "threshold_pct": threshold_pct,
            },
        )
        incidents.append({"incident_id": inc_id, "type": "NULL_SURGE", "detail": desc})

    logger.info(f"NULL surge check: {len(incidents)} column(s) flagged.")
    return incidents


# ---------------------------------------------------------------------------
# 3. Load failure detection
# ---------------------------------------------------------------------------

def detect_load_failures(load_history_df: pd.DataFrame) -> list[dict]:
    """
    Flag COPY INTO jobs with errors or partial loads from LOAD_HISTORY.
    """
    incidents = []
    if load_history_df.empty:
        return incidents

    # Complete failures
    failed = load_history_df[load_history_df["STATUS"] == "LOAD_FAILED"]
    for _, row in failed.iterrows():
        desc = (
            f"Load FAILED for {row.get('TABLE_NAME', 'UNKNOWN')} "
            f"file: {row.get('FILE_NAME', 'N/A')} — {row.get('FIRST_ERROR_MESSAGE', '')}"
        )
        inc_id = log_incident(
            "LOAD_FAILURE", desc, "HIGH",
            table_name=row.get("TABLE_NAME"),
            schema_name=row.get("SCHEMA_NAME"),
            metadata={
                "file": row.get("FILE_NAME"),
                "error_count": int(row.get("ERROR_COUNT", 0)),
                "first_error": row.get("FIRST_ERROR_MESSAGE"),
            },
        )
        incidents.append({"incident_id": inc_id, "type": "LOAD_FAILURE", "detail": desc})

    # Partial loads (loaded with errors)
    partial = load_history_df[
        (load_history_df["STATUS"] == "LOADED") &
        (load_history_df["ERROR_COUNT"].fillna(0) > 0)
    ]
    for _, row in partial.iterrows():
        desc = (
            f"Partial load for {row.get('TABLE_NAME')} — "
            f"{int(row.get('ERROR_COUNT', 0))} row errors in {row.get('FILE_NAME', 'N/A')}"
        )
        inc_id = log_incident(
            "LOAD_FAILURE", desc, "MEDIUM",
            table_name=row.get("TABLE_NAME"),
            schema_name=row.get("SCHEMA_NAME"),
            metadata={"error_count": int(row.get("ERROR_COUNT", 0))},
        )
        incidents.append({"incident_id": inc_id, "type": "LOAD_FAILURE", "detail": desc})

    logger.info(f"Load failure check: {len(incidents)} incident(s) logged.")
    return incidents


# ---------------------------------------------------------------------------
# 4. Query anomaly alerting (wraps IF results)
# ---------------------------------------------------------------------------

def alert_on_query_anomalies(anomaly_result, top_n: int = 20) -> list[dict]:
    """
    Log the top-N most anomalous queries detected by Isolation Forest
    into the Snowflake audit table.
    """
    incidents = []
    df = anomaly_result.df_with_scores
    flagged = df[df["is_anomaly"]].sort_values("anomaly_score").head(top_n)

    for _, row in flagged.iterrows():
        elapsed_s = row.get("TOTAL_ELAPSED_TIME", 0) / 1000
        desc = (
            f"Anomalous query by {row.get('USER_NAME', 'UNKNOWN')} "
            f"— elapsed {elapsed_s:.1f}s, scanned {row.get('BYTES_SCANNED', 0)/1e9:.2f} GB"
        )
        inc_id = log_incident(
            "QUERY_ANOMALY", desc,
            severity="HIGH" if row["anomaly_score"] < -0.3 else "MEDIUM",
            database_name=row.get("DATABASE_NAME"),
            schema_name=row.get("SCHEMA_NAME"),
            metadata={
                "query_id": row.get("QUERY_ID"),
                "anomaly_score": float(row["anomaly_score"]),
                "elapsed_ms": float(row.get("TOTAL_ELAPSED_TIME", 0)),
                "bytes_scanned": float(row.get("BYTES_SCANNED", 0)),
                "user": row.get("USER_NAME"),
            },
        )
        incidents.append({"incident_id": inc_id, "type": "QUERY_ANOMALY", "detail": desc})

    logger.info(f"Query anomaly alerting: {len(incidents)} incident(s) logged.")
    return incidents