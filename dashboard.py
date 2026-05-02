"""
dashboard.py
------------
Matplotlib dashboard visualizing:
  1. Pipeline SLA compliance — % of queries completing within SLA windows
  2. Anomaly trend over time — flagged query count by hour
  3. Data freshness metrics — table last_altered age distribution
  4. Feature importance proxy — mean anomaly score by feature bucket
  5. NULL rate heatmap — top tables × columns
  6. Volume spike chart — row counts with anomaly overlay

Run standalone (with synthetic data) or call build_dashboard() with real data.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

PALETTE = {
    "bg":        "#0D1117",
    "panel":     "#161B22",
    "border":    "#30363D",
    "primary":   "#58A6FF",
    "accent":    "#F78166",
    "success":   "#3FB950",
    "warning":   "#D29922",
    "text":      "#E6EDF3",
    "subtext":   "#8B949E",
    "anomaly":   "#FF6B6B",
    "normal":    "#58A6FF",
    "grid":      "#21262D",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["panel"],
    "axes.edgecolor":    PALETTE["border"],
    "axes.labelcolor":   PALETTE["text"],
    "axes.titlecolor":   PALETTE["text"],
    "xtick.color":       PALETTE["subtext"],
    "ytick.color":       PALETTE["subtext"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["grid"],
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "font.size":         9,
})


# ---------------------------------------------------------------------------
# Synthetic data generators (used when real Snowflake data isn't available)
# ---------------------------------------------------------------------------

def _synthetic_query_history(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    now = datetime.now(timezone.utc)
    start_times = [now - timedelta(hours=rng.uniform(0, 48)) for _ in range(n)]

    df = pd.DataFrame({
        "START_TIME":          start_times,
        "TOTAL_ELAPSED_TIME":  np.abs(rng.normal(1200, 400, n)),
        "BYTES_SCANNED":       np.abs(rng.normal(1e8, 3e7, n)),
        "ROWS_PRODUCED":       np.abs(rng.normal(5000, 1500, n)).astype(int),
        "USER_NAME":           rng.choice(["analyst_1", "dbt_prod", "etl_svc", "ds_team"], n),
        "anomaly_score":       rng.normal(-0.1, 0.15, n),
    })

    # Inject anomalies
    anomaly_idx = rng.choice(n, size=int(n * 0.07), replace=False)
    df.loc[anomaly_idx, "TOTAL_ELAPSED_TIME"] = rng.normal(25000, 5000, len(anomaly_idx))
    df.loc[anomaly_idx, "BYTES_SCANNED"]      = rng.normal(5e10, 1e10, len(anomaly_idx))
    df.loc[anomaly_idx, "anomaly_score"]      = rng.normal(-0.45, 0.1, len(anomaly_idx))
    df["is_anomaly"] = df["anomaly_score"] < -0.3

    return df.sort_values("START_TIME")


def _synthetic_table_stats() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    tables = [f"tbl_{c}" for c in ["orders", "users", "sessions", "products", "events",
                                    "inventory", "payments", "returns", "logs", "audit"]]
    now = datetime.now(timezone.utc)
    return pd.DataFrame({
        "TABLE_NAME":  tables,
        "ROW_COUNT":   np.abs(rng.normal([1e6, 5e5, 2e6, 8e4, 3e7,
                                           4e5, 9e5, 1e5, 5e6, 2e5], 1e4)).astype(int),
        "BYTES":       np.abs(rng.normal(1e9, 2e8, len(tables))).astype(int),
        "LAST_ALTERED": [now - timedelta(hours=rng.uniform(0, 72)) for _ in tables],
        "is_anomaly":  [False] * 8 + [True, False],
    })


def _synthetic_null_rates() -> pd.DataFrame:
    rng = np.random.default_rng(13)
    tables = ["orders", "users", "sessions", "products"]
    cols   = ["id", "created_at", "user_id", "amount", "status", "email"]
    rows = []
    for t in tables:
        for c in cols:
            rows.append({
                "TABLE_NAME":  t,
                "COLUMN_NAME": c,
                "NULL_PCT":    rng.uniform(0, 45 if (t == "orders" and c == "email") else 15),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def _panel_sla_compliance(ax, query_df: pd.DataFrame):
    """Bar chart: % of queries meeting SLA buckets (< 1s, 1–5s, 5–30s, >30s)."""
    elapsed_s = query_df["TOTAL_ELAPSED_TIME"] / 1000
    buckets = {"< 1s": (elapsed_s < 1).mean(),
               "1–5s": ((elapsed_s >= 1) & (elapsed_s < 5)).mean(),
               "5–30s": ((elapsed_s >= 5) & (elapsed_s < 30)).mean(),
               "> 30s": (elapsed_s >= 30).mean()}

    labels = list(buckets.keys())
    values = [v * 100 for v in buckets.values()]
    colors = [PALETTE["success"], PALETTE["primary"], PALETTE["warning"], PALETTE["accent"]]

    bars = ax.barh(labels, values, color=colors, edgecolor=PALETTE["border"], linewidth=0.5)
    ax.set_xlim(0, 105)
    ax.set_xlabel("% of Queries", fontsize=8)
    ax.set_title("Pipeline SLA Distribution", fontsize=10, fontweight="bold", pad=8)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, values):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8, color=PALETTE["text"])


def _panel_anomaly_trend(ax, query_df: pd.DataFrame):
    """Time-series: hourly flagged query count over last 48h."""
    df = query_df.copy()
    df["hour"] = pd.to_datetime(df["START_TIME"]).dt.floor("h")
    hourly = df.groupby("hour").agg(
        total=("is_anomaly", "count"),
        anomalies=("is_anomaly", "sum")
    ).reset_index()

    ax.fill_between(hourly["hour"], hourly["total"], alpha=0.15,
                    color=PALETTE["normal"], label="Total queries")
    ax.plot(hourly["hour"], hourly["total"], color=PALETTE["normal"],
            linewidth=1.2, alpha=0.7)
    ax.fill_between(hourly["hour"], hourly["anomalies"], alpha=0.4,
                    color=PALETTE["anomaly"])
    ax.plot(hourly["hour"], hourly["anomalies"], color=PALETTE["anomaly"],
            linewidth=1.5, label="Anomalous queries")

    ax.set_title("Anomaly Trend (48h)", fontsize=10, fontweight="bold", pad=8)
    ax.set_ylabel("Query Count", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.2)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right", fontsize=7)


def _panel_data_freshness(ax, table_df: pd.DataFrame):
    """Horizontal bar: hours since last table alteration (freshness)."""
    now = datetime.now(timezone.utc)

    def hours_ago(ts):
        if pd.isnull(ts):
            return 72
        dt = pd.to_datetime(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (now - dt).total_seconds() / 3600

    table_df = table_df.copy()
    table_df["freshness_h"] = table_df["LAST_ALTERED"].apply(hours_ago)
    table_df = table_df.sort_values("freshness_h")

    colors = [
        PALETTE["accent"] if row["freshness_h"] > 24 else
        PALETTE["warning"] if row["freshness_h"] > 6 else
        PALETTE["success"]
        for _, row in table_df.iterrows()
    ]

    bars = ax.barh(table_df["TABLE_NAME"], table_df["freshness_h"],
                   color=colors, edgecolor=PALETTE["border"], linewidth=0.5)
    ax.set_xlabel("Hours Since Last Update", fontsize=8)
    ax.set_title("Data Freshness by Table", fontsize=10, fontweight="bold", pad=8)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.axvline(x=24, color=PALETTE["accent"], linestyle="--", linewidth=1, alpha=0.7,
               label="24h SLA")
    ax.legend(fontsize=7, framealpha=0.2)


def _panel_null_heatmap(ax, null_df: pd.DataFrame):
    """Heatmap of NULL% by table × column."""
    pivot = null_df.pivot_table(
        index="TABLE_NAME", columns="COLUMN_NAME", values="NULL_PCT", aggfunc="mean"
    ).fillna(0)

    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=50, interpolation="nearest")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title("NULL Rate Heatmap (%)", fontsize=10, fontweight="bold", pad=8)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if val > 25 else PALETTE["text"])

    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02).ax.tick_params(labelsize=7)


def _panel_volume_spike(ax, table_df: pd.DataFrame):
    """Bar chart: row counts with anomaly tables highlighted."""
    df = table_df.sort_values("ROW_COUNT", ascending=False)
    colors = [PALETTE["anomaly"] if a else PALETTE["primary"]
              for a in df["is_anomaly"]]

    ax.bar(df["TABLE_NAME"], df["ROW_COUNT"] / 1e6, color=colors,
           edgecolor=PALETTE["border"], linewidth=0.5)
    ax.set_ylabel("Row Count (millions)", fontsize=8)
    ax.set_title("Table Volume — Anomalies Highlighted", fontsize=10, fontweight="bold", pad=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE["primary"], label="Normal"),
        Patch(facecolor=PALETTE["anomaly"], label="Anomaly"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, framealpha=0.2)


def _panel_score_distribution(ax, query_df: pd.DataFrame):
    """Histogram: distribution of Isolation Forest anomaly scores."""
    scores = query_df["anomaly_score"].dropna()
    threshold = -0.3

    ax.hist(scores[scores >= threshold], bins=40, color=PALETTE["normal"],
            alpha=0.7, label="Normal", edgecolor="none")
    ax.hist(scores[scores < threshold], bins=20, color=PALETTE["anomaly"],
            alpha=0.8, label="Anomalous", edgecolor="none")
    ax.axvline(threshold, color=PALETTE["warning"], linestyle="--",
               linewidth=1.2, label=f"Threshold ({threshold})")

    ax.set_xlabel("Isolation Forest Score", fontsize=8)
    ax.set_ylabel("Query Count", fontsize=8)
    ax.set_title("Anomaly Score Distribution", fontsize=10, fontweight="bold", pad=8)
    ax.legend(fontsize=7, framealpha=0.2)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# KPI header
# ---------------------------------------------------------------------------

def _draw_kpi_bar(fig, query_df, table_df, precision):
    """Draw a row of KPI tiles across the top of the figure."""
    kpis = [
        ("Total Queries",  f"{len(query_df):,}",        PALETTE["primary"]),
        ("Anomalies Found", f"{query_df['is_anomaly'].sum():,}", PALETTE["anomaly"]),
        ("Anomaly Rate",   f"{query_df['is_anomaly'].mean()*100:.1f}%", PALETTE["warning"]),
        ("Tables Monitored", f"{len(table_df):,}",      PALETTE["primary"]),
        ("Model Precision",  f"{precision:.0%}",         PALETTE["success"]),
    ]

    n = len(kpis)
    for i, (label, value, color) in enumerate(kpis):
        ax_kpi = fig.add_axes([0.02 + i * (0.96 / n), 0.90, (0.96 / n) - 0.01, 0.07])
        ax_kpi.set_facecolor(PALETTE["panel"])
        ax_kpi.set_xlim(0, 1)
        ax_kpi.set_ylim(0, 1)
        ax_kpi.axis("off")
        ax_kpi.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.90,
                                        boxstyle="round,pad=0.02",
                                        facecolor=PALETTE["panel"],
                                        edgecolor=color, linewidth=1.5))
        ax_kpi.text(0.5, 0.70, value, ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)
        ax_kpi.text(0.5, 0.25, label, ha="center", va="center",
                    fontsize=7, color=PALETTE["subtext"])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_dashboard(
    query_df: pd.DataFrame = None,
    table_df: pd.DataFrame = None,
    null_df:  pd.DataFrame = None,
    precision: float = 0.92,
    output_path: str = "snowflake_observability_dashboard.png",
) -> str:
    """
    Build and save the full observability dashboard.

    Parameters
    ----------
    query_df    : DataFrame from QueryAnomalyDetector.fit_predict() (enriched with scores)
    table_df    : DataFrame from VolumeAnomalyDetector.fit_predict()
    null_df     : DataFrame from fetch_null_rates() for multiple tables
    precision   : Model precision to display in KPI bar
    output_path : Where to save the PNG

    Returns
    -------
    output_path : str
    """
    # Fall back to synthetic data if not provided
    if query_df is None:
        logger.info("Using synthetic query data for dashboard demo.")
        query_df = _synthetic_query_history()
    if table_df is None:
        table_df = _synthetic_table_stats()
    if null_df is None:
        null_df = _synthetic_null_rates()

    fig = plt.figure(figsize=(20, 13), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg"])

    # Title
    fig.text(0.5, 0.96, "❄  SNOWFLAKE DATA OBSERVABILITY DASHBOARD",
             ha="center", va="center", fontsize=16, fontweight="bold",
             color=PALETTE["text"], fontfamily="monospace")
    fig.text(0.5, 0.935,
             f"Isolation Forest  •  Generated {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
             ha="center", va="center", fontsize=8, color=PALETTE["subtext"])

    # KPI bar
    _draw_kpi_bar(fig, query_df, table_df, precision)

    # Grid: 2 rows × 3 cols in the lower 88% of figure
    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        left=0.06, right=0.97,
        top=0.87, bottom=0.07,
        hspace=0.42, wspace=0.32,
    )

    ax_sla    = fig.add_subplot(gs[0, 0])
    ax_trend  = fig.add_subplot(gs[0, 1])
    ax_fresh  = fig.add_subplot(gs[0, 2])
    ax_null   = fig.add_subplot(gs[1, 0])
    ax_vol    = fig.add_subplot(gs[1, 1])
    ax_scores = fig.add_subplot(gs[1, 2])

    _panel_sla_compliance(ax_sla, query_df)
    _panel_anomaly_trend(ax_trend, query_df)
    _panel_data_freshness(ax_fresh, table_df)
    _panel_null_heatmap(ax_null, null_df)
    _panel_volume_spike(ax_vol, table_df)
    _panel_score_distribution(ax_scores, query_df)

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"], edgecolor="none")
    plt.close()

    logger.info(f"Dashboard saved → {output_path}")
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = build_dashboard()
    print(f"Dashboard saved to: {path}")