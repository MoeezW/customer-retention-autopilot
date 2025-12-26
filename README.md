# Customer Retention Autopilot (Churn + Automation)

A small, end-to-end “internal tool”-style system that predicts customer churn, segments customers into risk tiers, generates retention actions via a rules engine, simulates outreach, and produces a weekly manager report + dashboard.

---

## Executive Summary (10 lines)

- **Problem:** Telcos/utilities lose revenue due to customer churn; retention teams can’t manually prioritize who to contact each week.
- **Solution:** A weekly pipeline that scores churn risk, segments customers, and generates retention actions with clear “reason codes”.
- **Data:** Telco churn dataset (7,043 customers) loaded into Postgres as a warehouse-like source.
- **Model:** Logistic Regression baseline using a sklearn preprocessing pipeline (one-hot categoricals + numeric features).
- **Tracking:** MLflow logs params/metrics/artifacts so runs are reproducible and auditable.
- **Batch scoring:** Stores `churn_prob` per customer into `churn_scores` with a batch `run_id`.
- **Segmentation:** Converts probabilities into HIGH/MEDIUM/LOW segments (quantile or threshold policy).
- **Rules engine:** Generates retention actions (EMAIL templates) based on segment + key attributes (charges, tenure, contract).
- **Reporting:** Auto-generates weekly `summary.md` + plots + exports of segments/actions for stakeholders.
- **Dashboard:** Streamlit app to explore runs, risk distribution, targeting list, and weekly report.

---

## Architecture

```mermaid
flowchart LR
    A[Raw CSV] --> B[ETL: extract/transform/load]
    B --> C[(Postgres Warehouse)]
    C --> D[Feature Builder]
    D --> E[Train Model]
    E --> F[MLflow Tracking]
    C --> G[Batch Scoring Job]
    F --> G
    G --> H[(Postgres Tables: scores/segments/actions)]
    H --> I[Automation: send_actions (simulated)]
    H --> J[Weekly Report Generator]
    H --> K[Streamlit Dashboard]
    J --> L[reports/weekly/<run_id>/summary.md + plots]
    I --> M[email_previews/ + action_dispatch_log]


-- Quickstart --
----------------
0 - Prereqs
Python 3.11
Docker (for Postgres)

1 - Create a virtualenv and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

2 - Start Postgres (Docker)
docker run --name churn-postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres:16
docker exec -it churn-postgres psql -U postgres -c "CREATE DATABASE churn;"

3 - Create '.env' in repo root
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/churn
MLFLOW_TRACKING_URI=http://127.0.0.1:5000

4 - Run ETL (raw → cleaned → Postgres)
python -m src.etl.run_etl --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
    Outputs:
            data/processed/churn_clean.parquet
            Postgres tables: churn_dataset, customer_dim

5 - Start MLflow UI (separate terminal)
mlflow ui --host 127.0.0.1 --port 5000
    Open: http://127.0.0.1:5000

6 - Train a model (logs to MLflow)
python -m src.modeling.train
    This logs:
            metrics: ROC-AUC, PR-AUC, precision/recall/F1
            artifacts: reports, plots, model pipeline

7 - Run weekly scoring + actions
python -m src.scoring.score_batch
# copy the printed run_id, then:
python -m src.automation.send_actions --run-id <RUN_ID>
python -m src.reporting.weekly_report --run-id <RUN_ID>
    Outputs:
            Postgres: churn_scores, churn_segments, retention_actions, action_dispatch_log
            reports/weekly/<RUN_ID>/summary.md
            reports/weekly/<RUN_ID>/risk_distribution.png
            reports/weekly/<RUN_ID>/actions_by_template.png
            reports/weekly/<RUN_ID>/top_drivers.csv
            reports/weekly/<RUN_ID>/email_previews/

8 - Run the dashboard
streamlit run dashboards/app.py
----------------

Repo map:
customer-retention-autopilot/
  dashboards/
    app.py                       # Streamlit dashboard
  data/
    raw/                         # raw CSV input
    processed/                   # cleaned parquet snapshot
  reports/
    modeling/                    # training artifacts (local)
    weekly/<run_id>/             # weekly outputs: summary.md + plots + previews
  src/
    etl/
      extract.py                 # read raw CSV + normalize columns
      transform.py               # cleaning + type fixes + target encoding
      load.py                    # write to Postgres
      run_etl.py                 # CLI entry: raw -> clean -> load
    features/
      build_features.py          # reads churn_dataset, builds train/val/test splits + pipeline
    modeling/
      train.py                   # trains model + logs to MLflow
      evaluate.py                # threshold scans + business-ish metrics
    scoring/
      schema.py                  # ensures scoring tables
      score_batch.py             # batch scoring + segmentation + action generation
      segment.py                 # segment logic
      rules_engine.py            # action selection + reason codes
    automation/
      send_actions.py            # simulates sending + writes email previews + dispatch log
    reporting/
      weekly_report.py           # generates weekly markdown report + plots + exports
    utils/
      logging.py                 # structured logger
      validation.py              # ETL validations (fail fast)
  .env.example                   # template config
  README.md


Where to look
    MLflow runs: http://127.0.0.1:5000
        See metrics, params, artifacts, and the stored model pipeline.

    Weekly outputs: reports/weekly/<run_id>/summary.md
        The summary + plots + top drivers.

    Dashboard: streamlit run dashboards/app.py
        Overview + risk distribution + targeting list + report viewer.
    
    DB tables: churn_dataset, churn_scores, churn_segments, retention_actions, action_dispatch_log


Notes / Limitations
This is a demo pipeline using a public dataset; real production churn systems require:
    time-aware splits and backtesting (avoid leakage),
    stronger monitoring + drift detection,
    consent/compliance constraints for outreach,
    experimentation to measure true uplift from interventions.
The business impact estimate in the weekly report is intentionally labeled as an estimate with explicit assumptions.