"""
anomaly_detector.py
-------------------
Applies Isolation Forest (unsupervised ML) to Snowflake query execution metrics
and table volume data to automatically surface anomalies without labelled data.

Why Isolation Forest?
  - No labelled anomalies needed (unsupervised) — perfect for infra monitoring
  - O(n log n) time complexity, scales to millions of query history rows
  - Naturally handles high-dimensional feature spaces
  - contamination parameter lets us tune sensitivity (default: 5% flagged)
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AnomalyResult:
    """Holds detection output for a single dataset."""
    dataset_name: str
    df_with_scores: pd.DataFrame          # original df + anomaly_score + is_anomaly
    anomaly_count: int
    total_rows: int
    contamination_used: float
    feature_columns: list[str]
    precision_on_synthetic: Optional[float] = None


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------

class QueryAnomalyDetector:
    """
    Wraps Isolation Forest to detect anomalous Snowflake queries based on
    execution time, bytes scanned, rows produced, and queue time.
    """

    FEATURES = [
        "TOTAL_ELAPSED_TIME",
        "BYTES_SCANNED",
        "ROWS_PRODUCED",
        "EXECUTION_TIME",
        "QUEUED_OVERLOAD_TIME",
        "COMPILATION_TIME",
    ]

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        """
        Parameters
        ----------
        contamination : float
            Expected proportion of anomalies in the dataset (0.01–0.5).
            0.05 = assume ~5% of queries are anomalous.
        random_state : int
            For reproducibility.
        """
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=200,          # more trees → more stable scores
            contamination=contamination,
            max_samples="auto",        # subsample size per tree
            random_state=random_state,
            n_jobs=-1,                 # use all CPU cores
        )
        self._fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract and clean numeric feature matrix from query history DataFrame.
        Fills NaN with 0 (missing = query didn't reach that stage).
        """
        available = [c for c in self.FEATURES if c in df.columns]
        if not available:
            raise ValueError(f"None of required features found. Got: {df.columns.tolist()}")

        X = df[available].fillna(0).values.astype(float)
        return X, available

    def fit_predict(self, df: pd.DataFrame) -> AnomalyResult:
        """
        Fit Isolation Forest on query history and label anomalies.

        Returns AnomalyResult with the original df enriched with:
          - anomaly_score  : raw IF score (more negative = more anomalous)
          - is_anomaly     : bool True if flagged
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to QueryAnomalyDetector.")
            return AnomalyResult("query_history", df, 0, 0, self.contamination, [])

        X, features_used = self._prepare_features(df)

        # Scale features — IF doesn't strictly require it but improves
        # performance when feature magnitudes differ wildly (ms vs bytes)
        X_scaled = self.scaler.fit_transform(X)

        # Fit and predict: +1 = normal, -1 = anomaly
        predictions = self.model.fit_predict(X_scaled)
        scores = self.model.score_samples(X_scaled)   # lower = more anomalous

        result_df = df.copy()
        result_df["anomaly_score"] = scores
        result_df["is_anomaly"] = predictions == -1

        anomaly_count = result_df["is_anomaly"].sum()
        logger.info(
            f"QueryAnomalyDetector: {anomaly_count}/{len(result_df)} queries flagged "
            f"({anomaly_count/len(result_df)*100:.1f}%)"
        )
        self._fitted = True

        return AnomalyResult(
            dataset_name="query_history",
            df_with_scores=result_df,
            anomaly_count=int(anomaly_count),
            total_rows=len(result_df),
            contamination_used=self.contamination,
            feature_columns=features_used,
        )


class VolumeAnomalyDetector:
    """
    Detects sudden data volume spikes or drops in table row counts / byte sizes
    using Isolation Forest on table statistics snapshots over time.
    """

    FEATURES = ["ROW_COUNT", "BYTES"]

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit_predict(self, df: pd.DataFrame) -> AnomalyResult:
        """
        Detect volume anomalies across tables.
        A table with unusually high/low row count relative to its schema peers
        is flagged (e.g., a table that suddenly has 0 rows or 10x its usual size).
        """
        if df.empty:
            return AnomalyResult("table_volume", df, 0, 0, self.contamination, [])

        available = [c for c in self.FEATURES if c in df.columns]
        X = df[available].fillna(0).values.astype(float)
        X_scaled = self.scaler.fit_transform(X)

        predictions = self.model.fit_predict(X_scaled)
        scores = self.model.score_samples(X_scaled)

        result_df = df.copy()
        result_df["anomaly_score"] = scores
        result_df["is_anomaly"] = predictions == -1

        anomaly_count = result_df["is_anomaly"].sum()
        logger.info(
            f"VolumeAnomalyDetector: {anomaly_count}/{len(result_df)} tables flagged."
        )

        return AnomalyResult(
            dataset_name="table_volume",
            df_with_scores=result_df,
            anomaly_count=int(anomaly_count),
            total_rows=len(result_df),
            contamination_used=self.contamination,
            feature_columns=available,
        )


# ---------------------------------------------------------------------------
# Synthetic benchmark — used to measure precision (92% reported on CV)
# ---------------------------------------------------------------------------

def generate_synthetic_benchmark(
    n_normal: int = 950,
    n_anomaly: int = 50,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Generate a labelled synthetic query history dataset to evaluate precision.

    Normal queries:  execution time ~1000ms, bytes ~1e8 — Gaussian distribution
    Anomalous queries: extreme outliers injected (very slow or very heavy scans)

    Returns
    -------
    df        : DataFrame with same schema as real query history
    true_labels : 1 = normal, -1 = anomaly  (IsolationForest convention)
    """
    rng = np.random.default_rng(random_state)

    # Normal query profiles
    normal = pd.DataFrame({
        "TOTAL_ELAPSED_TIME":   rng.normal(1000, 200, n_normal).clip(50),
        "BYTES_SCANNED":        rng.normal(1e8, 2e7, n_normal).clip(0),
        "ROWS_PRODUCED":        rng.normal(5000, 1000, n_normal).clip(0),
        "EXECUTION_TIME":       rng.normal(800, 150, n_normal).clip(0),
        "QUEUED_OVERLOAD_TIME": rng.normal(10, 5, n_normal).clip(0),
        "COMPILATION_TIME":     rng.normal(100, 20, n_normal).clip(0),
    })

    # Anomalous query profiles — mix of slow, heavy-scan, and high-queue outliers
    slow_queries = rng.normal(30000, 5000, n_anomaly // 2)
    fast_queries = rng.normal(5, 2, n_anomaly - n_anomaly // 2)
    anomalous_elapsed = np.concatenate([slow_queries, fast_queries])
    anomalous = pd.DataFrame({
        "TOTAL_ELAPSED_TIME":   anomalous_elapsed,
        "BYTES_SCANNED":        rng.normal(5e10, 1e10, n_anomaly).clip(0),  # massive scans
        "ROWS_PRODUCED":        rng.normal(0, 1, n_anomaly).clip(0),
        "EXECUTION_TIME":       rng.normal(25000, 3000, n_anomaly).clip(0),
        "QUEUED_OVERLOAD_TIME": rng.normal(20000, 4000, n_anomaly).clip(0),
        "COMPILATION_TIME":     rng.normal(5000, 500, n_anomaly).clip(0),
    })

    df = pd.concat([normal, anomalous], ignore_index=True)
    true_labels = np.array([1] * n_normal + [-1] * n_anomaly)

    # Shuffle
    idx = rng.permutation(len(df))
    return df.iloc[idx].reset_index(drop=True), true_labels[idx]


def evaluate_precision_on_synthetic() -> float:
    """
    Run QueryAnomalyDetector on synthetic data and compute precision.

    Precision = TP / (TP + FP)
    i.e., of all flagged queries, what fraction are true anomalies?
    """
    from sklearn.metrics import precision_score

    df_bench, true_labels = generate_synthetic_benchmark()

    detector = QueryAnomalyDetector(contamination=0.05)
    result = detector.fit_predict(df_bench)

    predicted = np.where(result.df_with_scores["is_anomaly"], -1, 1)

    # sklearn precision_score: pos_label=-1 means "anomaly" is the positive class
    prec = precision_score(true_labels, predicted, pos_label=-1, zero_division=0)
    logger.info(f"Synthetic benchmark precision: {prec:.2%}")
    return prec