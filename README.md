# 🚀 Snowflake Observability & Anomaly Detection (ML Project)

## 📌 Overview

This project simulates a **real-world Snowflake data observability system** that monitors query performance, detects anomalies using Machine Learning, and generates a production-style dashboard.

It is designed to demonstrate how modern data teams track:

* Query performance issues
* Data quality problems
* Pipeline failures

---

## 🎯 Key Features

* ✅ Synthetic Snowflake query history generation (700+ queries)
* ✅ Injection of real-world anomalies:

  * Slow queries (latency spikes)
  * Full-table scans (high data usage)
  * Zero-row failures (misconfigured filters)
* ✅ **Isolation Forest (ML)** for anomaly detection

---

## 🧠 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

## ⚙️ How to Run (Live Demo)

### 🔹 Step 1: Clone the Repository

```bash
git clone https://github.com/sreeja-coder123/snowflake-observability-pipeline.git
cd snowflake-observability-pipeline
```

### 🔹 Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔹 Step 3: Run Demo

```bash
python demo_run.py
```

---

## 📊 Output

After running, you will get:

* ✔ Console output with anomaly detection metrics
* ✔ Alerts for anomalies and data quality issues
* ✔ Dashboard image.
---

## 📈 Model Performance

* Precision: ~100%
* Recall: ~98%
* F1 Score: ~99%

---

## 📸 Dashboard Preview

<img width="1060" height="674" alt="image" src="https://github.com/user-attachments/assets/4ead519a-5f62-4cc0-bb06-912b90012a93" />


---

## 🧩 Project Structure

```
snowflake_observability/
│── main.py
│── demo_run.py
│── anomaly_detector.py
│── alerting_engine.py
│── dashboard.py
│── snowflake_connector.py
│── requirements.txt
│── .gitignore
```

---

## 💡 How It Works

1. Generates synthetic Snowflake query data
2. Injects anomalies into dataset
3. Applies Isolation Forest for anomaly detection
4. Runs alerting logic (query issues + data quality)
5. Builds a visual observability dashboard

---

## 🎯 Use Case

This project simulates a **production-grade data observability system** used by:

* Data Engineers
* Analytics Teams
* ML Engineers

to monitor pipelines and ensure data reliability.

---


## 🚀 Future Improvements

* Add Streamlit live dashboard
* Deploy on cloud (Render / HuggingFace)
* Integrate with real Snowflake account
* Add real-time alerting (Slack / Email)


This project is for educational and demonstration purposes.
