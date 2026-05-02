"""
demo_run.py
===========
Snowflake Data Observability & Anomaly Detection — Live Demo
Run this in front of recruiters. No Snowflake connection needed.

    python demo_run.py

What it does (end-to-end in ~5 seconds):
  1. Simulates 700 Snowflake query history rows (realistic distributions)
  2. Injects 50 real anomalies (slow queries, massive scans, zero-row failures)
  3. Runs Isolation Forest to detect them — unsupervised, no labels
  4. Computes precision against ground truth
  5. Fires mock alerting logic (schema drift, NULL surges, load failures)
  6. Renders a 6-panel Matplotlib dashboard and saves it as PNG

Author: [Your Name]
Stack:  Python · Pandas · scikit-learn · Matplotlib
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime, timedelta, timezone

# ─────────────────────────────────────────────
#  STEP 1 — Generate realistic Snowflake data
# ─────────────────────────────────────────────

def generate_query_history(n_normal=650, n_anomaly=50, seed=42):
    """
    Simulate Snowflake ACCOUNT_USAGE.QUERY_HISTORY rows.
    Normal queries follow realistic distributions.
    Anomalies cover three real failure modes:
      A) Runaway queries   — execution > 30 000 ms
      B) Full-table scans  — bytes scanned > 50 GB
      C) Ghost queries     — 0 rows produced, near-instant (misconfigured filters)
    """
    rng = np.random.default_rng(seed)
    now = datetime.now(timezone.utc)

    users = ["dbt_prod", "analyst_1", "analyst_2", "etl_service", "ds_team", "bi_service"]
    warehouses = ["COMPUTE_WH", "TRANSFORM_WH", "REPORTING_WH"]

    # Normal query profiles
    n_elapsed   = np.abs(rng.normal(1100, 350, n_normal)).clip(80)
    n_bytes     = np.abs(rng.normal(0.8e8, 2e7, n_normal)).clip(0)
    n_rows      = np.abs(rng.normal(4800, 1200, n_normal)).astype(int)
    n_exec      = n_elapsed * rng.uniform(0.75, 0.90, n_normal)
    n_queue     = np.abs(rng.normal(12, 5, n_normal))
    n_compile   = np.abs(rng.normal(95, 22, n_normal))

    # Anomaly profiles
    n_a = n_anomaly // 3
    # A: runaway
    a_elapsed   = rng.normal(32000, 4000, n_a).clip(28000)
    a_bytes_run = rng.normal(1.5e9, 2e8, n_a).clip(0)
    a_rows_run  = rng.normal(4000, 500, n_a).clip(0)
    a_exec_run  = a_elapsed * 0.88
    a_queue_run = rng.normal(8000, 1000, n_a).clip(0)
    a_comp_run  = rng.normal(800, 100, n_a).clip(0)

    # B: full-table scan
    b_elapsed   = rng.normal(18000, 2000, n_a).clip(12000)
    b_bytes     = rng.normal(6e10, 5e9, n_a).clip(4e10)    # 40–70 GB
    b_rows      = rng.normal(50000, 5000, n_a).clip(0)
    b_exec      = b_elapsed * 0.92
    b_queue     = rng.normal(200, 40, n_a).clip(0)
    b_comp      = rng.normal(300, 50, n_a).clip(0)

    # C: ghost queries — remaining anomalies
    n_c = n_anomaly - 2 * n_a
    c_elapsed   = rng.normal(15, 3, n_c).clip(5)
    c_bytes     = rng.normal(50, 10, n_c).clip(0)
    c_rows      = np.zeros(n_c)
    c_exec      = c_elapsed * 0.5
    c_queue     = rng.normal(5, 1, n_c).clip(0)
    c_comp      = rng.normal(10, 2, n_c).clip(0)

    elapsed   = np.concatenate([n_elapsed, a_elapsed, b_elapsed, c_elapsed])
    bytes_sc  = np.concatenate([n_bytes,   a_bytes_run, b_bytes, c_bytes])
    rows      = np.concatenate([n_rows,    a_rows_run, b_rows, c_rows])
    exec_t    = np.concatenate([n_exec,    a_exec_run, b_exec, c_exec])
    queue_t   = np.concatenate([n_queue,   a_queue_run, b_queue, c_queue])
    compile_t = np.concatenate([n_compile, a_comp_run, b_comp, c_comp])

    total = n_normal + n_anomaly
    true_labels = np.array([1] * n_normal + [-1] * n_anomaly)  # 1=normal, -1=anomaly

    idx = rng.permutation(total)
    start_times = [now - timedelta(hours=rng.uniform(0, 48)) for _ in range(total)]

    df = pd.DataFrame({
        "QUERY_ID":             [f"QID_{i:05d}" for i in idx],
        "START_TIME":           start_times,
        "USER_NAME":            rng.choice(users, total),
        "WAREHOUSE_NAME":       rng.choice(warehouses, total),
        "DATABASE_NAME":        "PROD_DB",
        "SCHEMA_NAME":          "PUBLIC",
        "TOTAL_ELAPSED_TIME":   elapsed[idx],
        "BYTES_SCANNED":        bytes_sc[idx],
        "ROWS_PRODUCED":        rows[idx],
        "EXECUTION_TIME":       exec_t[idx],
        "QUEUED_OVERLOAD_TIME": queue_t[idx],
        "COMPILATION_TIME":     compile_t[idx],
        "EXECUTION_STATUS":     "SUCCESS",
    })

    return df, true_labels[idx]


def generate_table_stats(seed=7):
    rng = np.random.default_rng(seed)
    now = datetime.now(timezone.utc)
    tables = ["orders", "users", "sessions", "products", "events",
              "inventory", "payments", "returns", "logs", "audit"]
    row_counts = np.abs(rng.normal(
        [1e6, 5e5, 2e6, 8e4, 3e7, 4e5, 9e5, 1e5, 5e6, 2e5],
        1e4
    )).astype(int)
    bytes_vals = np.abs(rng.normal(1e9, 2e8, len(tables))).astype(int)
    freshness  = [now - timedelta(hours=rng.uniform(0, 72)) for _ in tables]
    is_anomaly = [False] * 8 + [True, False]

    return pd.DataFrame({
        "TABLE_NAME":  tables,
        "ROW_COUNT":   row_counts,
        "BYTES":       bytes_vals,
        "LAST_ALTERED": freshness,
        "is_anomaly":  is_anomaly,
    })


def generate_null_rates(seed=13):
    rng = np.random.default_rng(seed)
    tables = ["orders", "users", "sessions", "products"]
    cols   = ["id", "created_at", "user_id", "amount", "status", "email"]
    rows = []
    for t in tables:
        for c in cols:
            spike = (t == "orders" and c == "email")  # intentional NULL surge
            rows.append({
                "TABLE_NAME":  t,
                "COLUMN_NAME": c,
                "NULL_PCT":    round(rng.uniform(32, 44) if spike else rng.uniform(0, 12), 1),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  STEP 2 — Isolation Forest detection
# ─────────────────────────────────────────────

FEATURES = ["TOTAL_ELAPSED_TIME", "BYTES_SCANNED", "ROWS_PRODUCED",
            "EXECUTION_TIME", "QUEUED_OVERLOAD_TIME", "COMPILATION_TIME"]

def run_isolation_forest(df, contamination=0.07):
    X = df[FEATURES].fillna(0).values.astype(float)
    X_scaled = StandardScaler().fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    predictions = model.fit_predict(X_scaled)
    scores      = model.score_samples(X_scaled)

    df = df.copy()
    df["anomaly_score"] = scores
    df["is_anomaly"]    = predictions == -1
    return df, model


# ─────────────────────────────────────────────
#  STEP 3 — Alerting logic
# ─────────────────────────────────────────────

def run_alerting(df_scored, null_df):
    """
    In production this writes to a Snowflake VARIANT audit table.
    Here we log incidents to console and return structured dicts.
    """
    incidents = []

    # A) Query anomaly alerts — top 10 most anomalous
    flagged = df_scored[df_scored["is_anomaly"]].sort_values("anomaly_score").head(10)
    for _, row in flagged.iterrows():
        elapsed_s = row["TOTAL_ELAPSED_TIME"] / 1000
        severity  = "HIGH" if row["anomaly_score"] < -0.35 else "MEDIUM"
        incidents.append({
            "type":     "QUERY_ANOMALY",
            "severity": severity,
            "detail":   f"[{severity}] QUERY_ANOMALY | user={row['USER_NAME']} | "
                        f"elapsed={elapsed_s:.1f}s | "
                        f"scanned={row['BYTES_SCANNED']/1e9:.1f} GB | "
                        f"score={row['anomaly_score']:.3f}",
        })

    # B) NULL surge alerts — threshold 25%
    surges = null_df[null_df["NULL_PCT"] > 25]
    for _, row in surges.iterrows():
        incidents.append({
            "type":     "NULL_SURGE",
            "severity": "HIGH",
            "detail":   f"[HIGH]   NULL_SURGE   | table={row['TABLE_NAME']} | "
                        f"column={row['COLUMN_NAME']} | null_pct={row['NULL_PCT']}%",
        })

    # C) Schema drift (simulated — in prod compares vs. persisted JSON snapshot)
    incidents.append({
        "type":     "SCHEMA_DRIFT",
        "severity": "MEDIUM",
        "detail":   "[MEDIUM] SCHEMA_DRIFT | table=orders | "
                    "column='promo_code' ADDED (VARCHAR)",
    })

    return incidents


# ─────────────────────────────────────────────
#  STEP 4 — Dashboard
# ─────────────────────────────────────────────

PALETTE = {
    "bg":      "#0D1117",
    "panel":   "#161B22",
    "border":  "#30363D",
    "primary": "#58A6FF",
    "accent":  "#F78166",
    "success": "#3FB950",
    "warning": "#D29922",
    "text":    "#E6EDF3",
    "subtext": "#8B949E",
    "anomaly": "#FF6B6B",
    "grid":    "#21262D",
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor":   PALETTE["panel"],
    "axes.edgecolor":   PALETTE["border"],
    "axes.labelcolor":  PALETTE["text"],
    "axes.titlecolor":  PALETTE["text"],
    "xtick.color":      PALETTE["subtext"],
    "ytick.color":      PALETTE["subtext"],
    "text.color":       PALETTE["text"],
    "grid.color":       PALETTE["grid"],
    "grid.linewidth":   0.5,
    "font.family":      "monospace",
    "font.size":        9,
})


def panel_sla(ax, df):
    elapsed_s = df["TOTAL_ELAPSED_TIME"] / 1000
    pcts = {
        "< 1s":  (elapsed_s < 1).mean() * 100,
        "1–5s":  ((elapsed_s >= 1) & (elapsed_s < 5)).mean() * 100,
        "5–30s": ((elapsed_s >= 5) & (elapsed_s < 30)).mean() * 100,
        "> 30s": (elapsed_s >= 30).mean() * 100,
    }
    colors = [PALETTE["success"], PALETTE["primary"], PALETTE["warning"], PALETTE["anomaly"]]
    bars = ax.barh(list(pcts.keys()), list(pcts.values()), color=colors,
                   edgecolor=PALETTE["border"], linewidth=0.5)
    ax.set_xlim(0, 108)
    ax.set_title("Pipeline SLA Distribution", fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("% of Queries", fontsize=8)
    ax.grid(axis="x", alpha=0.3); ax.set_axisbelow(True)
    for bar, val in zip(bars, pcts.values()):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8, color=PALETTE["text"])


def panel_trend(ax, df):
    df = df.copy()
    df["hour"] = pd.to_datetime(df["START_TIME"]).dt.floor("h")
    hourly = df.groupby("hour").agg(
        total=("is_anomaly", "count"),
        anomalies=("is_anomaly", "sum")
    ).reset_index()
    ax.fill_between(hourly["hour"], hourly["total"],
                    alpha=0.15, color=PALETTE["primary"])
    ax.plot(hourly["hour"], hourly["total"],
            color=PALETTE["primary"], linewidth=1.2, alpha=0.8, label="Total queries")
    ax.fill_between(hourly["hour"], hourly["anomalies"],
                    alpha=0.45, color=PALETTE["anomaly"])
    ax.plot(hourly["hour"], hourly["anomalies"],
            color=PALETTE["anomaly"], linewidth=1.5, label="Anomalous queries")
    ax.set_title("Anomaly Trend (48h)", fontsize=10, fontweight="bold", pad=8)
    ax.set_ylabel("Query Count", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.2)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right", fontsize=7)


def panel_freshness(ax, table_df):
    now = datetime.now(timezone.utc)
    def hours_ago(ts):
        dt = pd.to_datetime(ts)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        return (now - dt).total_seconds() / 3600
    df = table_df.copy()
    df["freshness_h"] = df["LAST_ALTERED"].apply(hours_ago)
    df = df.sort_values("freshness_h")
    colors = [
        PALETTE["anomaly"] if r > 24 else
        PALETTE["warning"] if r > 6 else
        PALETTE["success"]
        for r in df["freshness_h"]
    ]
    ax.barh(df["TABLE_NAME"], df["freshness_h"], color=colors,
            edgecolor=PALETTE["border"], linewidth=0.5)
    ax.axvline(24, color=PALETTE["anomaly"], linestyle="--",
               linewidth=1, alpha=0.7, label="24h SLA")
    ax.set_title("Data Freshness by Table", fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Hours Since Last Update", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.2)
    ax.grid(axis="x", alpha=0.3); ax.set_axisbelow(True)


def panel_null_heatmap(ax, null_df):
    pivot = null_df.pivot_table(
        index="TABLE_NAME", columns="COLUMN_NAME", values="NULL_PCT", aggfunc="mean"
    ).fillna(0)
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=50, interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title("NULL Rate Heatmap (%)", fontsize=10, fontweight="bold", pad=8)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=7, color="white" if v > 25 else PALETTE["text"])
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02).ax.tick_params(labelsize=7)


def panel_volume(ax, table_df):
    df = table_df.sort_values("ROW_COUNT", ascending=False)
    colors = [PALETTE["anomaly"] if a else PALETTE["primary"] for a in df["is_anomaly"]]
    ax.bar(df["TABLE_NAME"], df["ROW_COUNT"] / 1e6, color=colors,
           edgecolor=PALETTE["border"], linewidth=0.5)
    ax.set_title("Table Volume — Anomalies Highlighted", fontsize=10, fontweight="bold", pad=8)
    ax.set_ylabel("Row Count (millions)", fontsize=8)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=PALETTE["primary"], label="Normal"),
        Patch(facecolor=PALETTE["anomaly"], label="Volume anomaly"),
    ], fontsize=7, framealpha=0.2)


def panel_scores(ax, df):
    scores    = df["anomaly_score"].dropna()
    threshold = -0.30
    ax.hist(scores[scores >= threshold], bins=40,
            color=PALETTE["primary"], alpha=0.7, label="Normal", edgecolor="none")
    ax.hist(scores[scores < threshold], bins=20,
            color=PALETTE["anomaly"], alpha=0.85, label="Anomalous", edgecolor="none")
    ax.axvline(threshold, color=PALETTE["warning"], linestyle="--",
               linewidth=1.2, label=f"Threshold ({threshold})")
    ax.set_title("Anomaly Score Distribution", fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Isolation Forest Score", fontsize=8)
    ax.set_ylabel("Query Count", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.2)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)


def build_dashboard(df_scored, table_df, null_df, precision, recall, f1, out):
    n_queries   = len(df_scored)
    n_anomalies = df_scored["is_anomaly"].sum()
    n_tables    = len(table_df)

    fig = plt.figure(figsize=(20, 13), dpi=150)
    fig.patch.set_facecolor(PALETTE["bg"])

    fig.text(0.5, 0.965,
             "SNOWFLAKE DATA OBSERVABILITY & ANOMALY DETECTION",
             ha="center", fontsize=16, fontweight="bold", color=PALETTE["text"])
    fig.text(0.5, 0.945,
             f"Isolation Forest (unsupervised)  |  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             ha="center", fontsize=8, color=PALETTE["subtext"])

    # KPI tiles
    kpis = [
        ("Queries Analysed",  f"{n_queries:,}",          PALETTE["primary"]),
        ("Anomalies Detected", f"{n_anomalies:,}",       PALETTE["anomaly"]),
        ("Anomaly Rate",       f"{n_anomalies/n_queries*100:.1f}%", PALETTE["warning"]),
        ("Tables Monitored",  f"{n_tables:,}",            PALETTE["primary"]),
        ("Model Precision",   f"{precision:.0%}",         PALETTE["success"]),
    ]
    for i, (label, value, color) in enumerate(kpis):
        kax = fig.add_axes([0.02 + i * 0.192, 0.895, 0.185, 0.042])
        kax.set_facecolor(PALETTE["panel"])
        kax.axis("off")
        kax.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.88,
                                     boxstyle="round,pad=0.02",
                                     facecolor=PALETTE["panel"],
                                     edgecolor=color, linewidth=1.5))
        kax.text(0.5, 0.68, value,  ha="center", va="center",
                 fontsize=14, fontweight="bold", color=color)
        kax.text(0.5, 0.22, label,  ha="center", va="center",
                 fontsize=7.5, color=PALETTE["subtext"])

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.06, right=0.97,
                           top=0.88, bottom=0.07,
                           hspace=0.42, wspace=0.32)

    panels = [
        (gs[0, 0], panel_sla,         [df_scored]),
        (gs[0, 1], panel_trend,       [df_scored]),
        (gs[0, 2], panel_freshness,   [table_df]),
        (gs[1, 0], panel_null_heatmap,[null_df]),
        (gs[1, 1], panel_volume,      [table_df]),
        (gs[1, 2], panel_scores,      [df_scored]),
    ]
    for spec, fn, args in panels:
        ax = fig.add_subplot(spec)
        fn(ax, *args)

    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"], edgecolor="none")
    plt.close()
    return out


# ─────────────────────────────────────────────
#  MAIN — orchestrate all steps with timing
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  SNOWFLAKE OBSERVABILITY PIPELINE — DEMO RUN")
    print("=" * 62)

    t0 = time.time()

    # Step 1
    print("\n[1/4] Generating synthetic Snowflake query history...")
    df_raw, true_labels = generate_query_history(n_normal=650, n_anomaly=50)
    table_df = generate_table_stats()
    null_df  = generate_null_rates()
    print(f"      {len(df_raw)} query records  |  {len(table_df)} tables  |  {len(null_df)} null-rate rows")
    print(f"      True anomalies injected: {(true_labels == -1).sum()}")

    # Step 2
    print("\n[2/4] Running Isolation Forest (200 trees, contamination=7%)...")
    df_scored, model = run_isolation_forest(df_raw, contamination=0.07)
    predicted = np.where(df_scored["is_anomaly"], -1, 1)
    prec = precision_score(true_labels, predicted, pos_label=-1, zero_division=0)
    rec  = recall_score(true_labels, predicted, pos_label=-1, zero_division=0)
    f1   = f1_score(true_labels, predicted, pos_label=-1, zero_division=0)
    print(f"      Flagged: {df_scored['is_anomaly'].sum()} queries")
    print(f"      Precision : {prec:.2%}")
    print(f"      Recall    : {rec:.2%}")
    print(f"      F1-Score  : {f1:.2%}")

    # Step 3
    print("\n[3/4] Running alerting logic...")
    incidents = run_alerting(df_scored, null_df)
    print(f"      {len(incidents)} incident(s) logged (→ Snowflake audit table in prod):")
    for inc in incidents:
        print(f"        {inc['detail']}")

    # Step 4
    print("\n[4/4] Building Matplotlib dashboard...")
    out_path = "snowflake_observability_dashboard.png"
    build_dashboard(df_scored, table_df, null_df, prec, rec, f1, out_path)
    elapsed = time.time() - t0

    print(f"\n{'=' * 62}")
    print(f"  DONE in {elapsed:.1f}s  —  dashboard saved to:")
    print(f"  {out_path}")
    print(f"{'=' * 62}\n")