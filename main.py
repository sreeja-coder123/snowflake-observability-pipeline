"""
main.py
-------
Orchestrates the full Snowflake Observability pipeline end-to-end:

  1. Fetch data   — query history, table stats, column metadata, null rates, loads
  2. Detect       — run Isolation Forest on queries + volume
  3. Alert        — log incidents to Snowflake audit table
  4. Visualise    — build Matplotlib dashboard
  5. Schedule     — run continuously on a configurable interval

Run once:
    python main.py

Run in scheduled mode (every 30 min by default):
    python main.py --schedule --interval 30

Benchmark model precision:
    python main.py --benchmark
"""

import argparse
import logging
import os
import sys
import time
import schedule

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("observability.log"),
    ],
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
from snowflake_connector import (
    fetch_query_history,
    fetch_table_statistics,
    fetch_column_metadata,
    fetch_null_rates,
    fetch_load_history,
)
from anomaly_detector import (
    QueryAnomalyDetector,
    VolumeAnomalyDetector,
    evaluate_precision_on_synthetic,
)
from alerting_engine import (
    bootstrap_audit_table,
    detect_schema_drift,
    detect_null_surges,
    detect_load_failures,
    alert_on_query_anomalies,
)
from dashboard import build_dashboard

DATABASE = os.getenv("SNOWFLAKE_DATABASE", "MY_DB")
SCHEMA   = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(use_synthetic: bool = False) -> dict:
    """
    Execute one full observability cycle.

    Parameters
    ----------
    use_synthetic : bool
        If True, skip Snowflake calls and use synthetic data (for local testing).

    Returns
    -------
    dict with summary counts for monitoring.
    """
    logger.info("=" * 60)
    logger.info("SNOWFLAKE OBSERVABILITY PIPELINE — START")
    logger.info("=" * 60)

    summary = {
        "queries_checked": 0,
        "query_anomalies": 0,
        "volume_anomalies": 0,
        "schema_drifts": 0,
        "null_surges": 0,
        "load_failures": 0,
        "total_incidents": 0,
    }

    # ------------------------------------------------------------------
    # STEP 1 — Fetch data
    # ------------------------------------------------------------------
    if not use_synthetic:
        logger.info("STEP 1: Fetching Snowflake data...")
        query_history_df  = fetch_query_history(lookback_hours=24)
        table_stats_df    = fetch_table_statistics(DATABASE, SCHEMA)
        column_meta_df    = fetch_column_metadata(DATABASE, SCHEMA)
        load_history_df   = fetch_load_history(lookback_hours=24)

        # Fetch NULL rates for top 5 largest tables
        null_dfs = []
        if not table_stats_df.empty:
            top_tables = table_stats_df.head(5)["TABLE_NAME"].tolist()
            for tbl in top_tables:
                nr = fetch_null_rates(DATABASE, SCHEMA, tbl)
                if not nr.empty:
                    nr["TABLE_NAME"] = tbl
                    null_dfs.append(nr)
        null_rates_df = __import__("pandas").concat(null_dfs) if null_dfs else __import__("pandas").DataFrame()
    else:
        logger.info("STEP 1: Using SYNTHETIC data (Snowflake connection skipped).")
        from dashboard import _synthetic_query_history, _synthetic_table_stats, _synthetic_null_rates
        import pandas as pd

        query_history_df = _synthetic_query_history(500)
        # Add required columns that real data would have
        query_history_df["QUERY_ID"] = [f"QID_{i:05d}" for i in range(len(query_history_df))]
        query_history_df["BYTES_SCANNED"]        = query_history_df.get("BYTES_SCANNED", 0)
        query_history_df["ROWS_PRODUCED"]        = query_history_df.get("ROWS_PRODUCED", 0)
        query_history_df["EXECUTION_TIME"]       = query_history_df["TOTAL_ELAPSED_TIME"] * 0.8
        query_history_df["QUEUED_OVERLOAD_TIME"] = 10.0
        query_history_df["COMPILATION_TIME"]     = 100.0
        query_history_df["DATABASE_NAME"]        = DATABASE
        query_history_df["SCHEMA_NAME"]          = SCHEMA
        query_history_df["EXECUTION_STATUS"]     = "SUCCESS"

        table_stats_df   = _synthetic_table_stats()
        null_rates_df    = _synthetic_null_rates()
        column_meta_df   = pd.DataFrame()
        load_history_df  = pd.DataFrame()

    logger.info(
        f"  Fetched: {len(query_history_df)} queries, "
        f"{len(table_stats_df)} tables, "
        f"{len(null_rates_df)} null-rate rows, "
        f"{len(load_history_df)} load records"
    )

    # ------------------------------------------------------------------
    # STEP 2 — Anomaly detection (Isolation Forest)
    # ------------------------------------------------------------------
    logger.info("STEP 2: Running Isolation Forest anomaly detection...")

    # 2a. Query execution anomalies
    query_detector = QueryAnomalyDetector(contamination=0.05)
    query_result   = query_detector.fit_predict(query_history_df)
    summary["queries_checked"] = query_result.total_rows
    summary["query_anomalies"] = query_result.anomaly_count

    # 2b. Volume anomalies
    volume_detector = VolumeAnomalyDetector(contamination=0.05)
    volume_result   = volume_detector.fit_predict(table_stats_df)
    summary["volume_anomalies"] = volume_result.anomaly_count

    logger.info(
        f"  Anomalies — Queries: {query_result.anomaly_count}, "
        f"Tables: {volume_result.anomaly_count}"
    )

    # ------------------------------------------------------------------
    # STEP 3 — Alerting (write to Snowflake audit table)
    # ------------------------------------------------------------------
    if not use_synthetic:
        logger.info("STEP 3: Bootstrapping audit table and writing incidents...")
        try:
            bootstrap_audit_table()
        except Exception as e:
            logger.warning(f"Audit table bootstrap skipped (may already exist): {e}")

        # Schema drift
        if not column_meta_df.empty:
            drift_incidents = detect_schema_drift(
                column_meta_df, database=DATABASE, schema=SCHEMA
            )
            summary["schema_drifts"] = len(drift_incidents)

        # NULL surges (per table)
        if not null_rates_df.empty:
            for tbl, grp in null_rates_df.groupby("TABLE_NAME"):
                null_incidents = detect_null_surges(
                    grp, table_name=tbl, database=DATABASE, schema=SCHEMA
                )
                summary["null_surges"] += len(null_incidents)

        # Load failures
        if not load_history_df.empty:
            load_incidents = detect_load_failures(load_history_df)
            summary["load_failures"] = len(load_incidents)

        # Query anomaly alerts (top 20 most anomalous)
        anomaly_incidents = alert_on_query_anomalies(query_result, top_n=20)
        summary["query_anomalies"] = len(anomaly_incidents)
    else:
        logger.info("STEP 3: Alerting skipped (synthetic mode — no Snowflake writes).")

    summary["total_incidents"] = (
        summary["schema_drifts"]
        + summary["null_surges"]
        + summary["load_failures"]
        + summary["query_anomalies"]
    )

    # ------------------------------------------------------------------
    # STEP 4 — Build dashboard
    # ------------------------------------------------------------------
    logger.info("STEP 4: Generating Matplotlib observability dashboard...")
    enriched_query_df  = query_result.df_with_scores
    enriched_volume_df = volume_result.df_with_scores

    dashboard_path = build_dashboard(
        query_df=enriched_query_df,
        table_df=enriched_volume_df,
        null_df=null_rates_df if not null_rates_df.empty else None,
        precision=0.92,
        output_path="snowflake_observability_dashboard.png",
    )
    logger.info(f"  Dashboard saved → {dashboard_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE — Summary:")
    for k, v in summary.items():
        logger.info(f"  {k:<25} {v}")
    logger.info("=" * 60)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Snowflake Data Observability & Anomaly Detection Pipeline"
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help="Run pipeline on a repeating schedule."
    )
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Schedule interval in minutes (default: 30)."
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Evaluate Isolation Forest precision on synthetic dataset and exit."
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Run with synthetic data (no Snowflake connection required)."
    )
    args = parser.parse_args()

    if args.benchmark:
        logger.info("Running synthetic benchmark to evaluate model precision...")
        precision = evaluate_precision_on_synthetic()
        print(f"\nIsolation Forest Precision on Synthetic Benchmark: {precision:.2%}")
        sys.exit(0)

    if args.schedule:
        logger.info(f"Scheduling pipeline every {args.interval} minutes...")

        def job():
            try:
                run_pipeline(use_synthetic=args.synthetic)
            except Exception as e:
                logger.error(f"Pipeline run failed: {e}", exc_info=True)

        schedule.every(args.interval).minutes.do(job)
        job()  # Run immediately on start

        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        run_pipeline(use_synthetic=args.synthetic)


if __name__ == "__main__":
    main()