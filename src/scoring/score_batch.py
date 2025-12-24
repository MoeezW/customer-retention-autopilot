from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import uuid

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sqlalchemy import create_engine, text

from src.config import settings
from src.scoring.db_schema import ensure_scoring_tables
from src.scoring.segment import segment_by_quantiles
from src.scoring.rules_engine import decide_action
from src.utils.logging import get_logger

logger = get_logger("scoring.batch")

EXPERIMENT_NAME = "Customer Retention Autopilot - Churn Model"


@dataclass(frozen=True)
class LatestModelInfo:
    model_run_id: str
    model_uri: str
    chosen_threshold: float


def get_latest_model_info() -> LatestModelInfo:
    """
    Pull the most recent MLflow run in our experiment and build a model URI.
    Also retrieve chosen_threshold from params.
    """
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    client = MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise ValueError(f"MLflow experiment not found: {EXPERIMENT_NAME}")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError("No MLflow runs found. Train a model first.")

    run = runs[0]
    run_id = run.info.run_id

    # We logged the model to artifact path "sklearn-model"
    model_uri = f"runs:/{run_id}/sklearn-model"

    # chosen_threshold is stored as a param (string)
    chosen_threshold_str = run.data.params.get("chosen_threshold", "0.5")
    chosen_threshold = float(chosen_threshold_str)

    return LatestModelInfo(model_run_id=run_id, model_uri=model_uri, chosen_threshold=chosen_threshold)


def main() -> None:
    if not settings.database_url:
        raise ValueError("DATABASE_URL is empty. Check your .env file.")

    engine = create_engine(settings.database_url, future=True)
    ensure_scoring_tables(engine)

    # --- Determine this week's batch run id (operational run)
    batch_run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    logger.info(f"Batch run_id: {batch_run_id}")

    # --- Load latest model from MLflow
    model_info = get_latest_model_info()
    logger.info(f"Using MLflow model run_id: {model_info.model_run_id}")
    logger.info(f"Using chosen_threshold: {model_info.chosen_threshold:.2f}")

    clf = mlflow.sklearn.load_model(model_info.model_uri)

    # --- Pull customer data to score (for now, score full churn_dataset snapshot)
    logger.info("Loading customers from churn_dataset...")
    df = pd.read_sql_query(text("SELECT * FROM churn_dataset;"), con=engine)
    logger.info(f"Loaded {len(df)} customers for scoring.")

    # Features for model: match training (drop churn + customer_id)
    required_cols = {"customer_id", "churn"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Expected columns missing from churn_dataset: {required_cols - set(df.columns)}")

    customer_ids = df["customer_id"].astype(str)
    X = df.drop(columns=["churn", "customer_id"])

    # Predict churn probability
    logger.info("Scoring churn probabilities...")
    probs = clf.predict_proba(X)[:, 1]

    # Segment (quantiles)
    segments = segment_by_quantiles(probs, high_quantile=0.90, med_quantile=0.70)

    # Decide actions per row
    logger.info("Applying rules engine to generate actions...")
    actions = []
    for i, row in df.iterrows():
        seg = str(segments.iloc[i])
        decision = decide_action(row=row, segment=seg)
        actions.append(
            {
                "run_id": batch_run_id,
                "customer_id": str(customer_ids.iloc[i]),
                "action_type": decision.action_type,
                "template_id": decision.template_id,
                "reason_code": decision.reason_code,
            }
        )
    actions_df = pd.DataFrame(actions)

    # Persist scores, segments, actions
    scores_df = pd.DataFrame(
        {
            "run_id": batch_run_id,
            "model_run_id": model_info.model_run_id,
            "customer_id": customer_ids,
            "churn_prob": probs,
        }
    )

    segments_df = pd.DataFrame(
        {
            "run_id": batch_run_id,
            "customer_id": customer_ids,
            "segment": segments.astype(str),
        }
    )

    logger.info("Writing churn_scores / churn_segments / retention_actions...")
    scores_df.to_sql("churn_scores", engine, if_exists="append", index=False)
    segments_df.to_sql("churn_segments", engine, if_exists="append", index=False)
    actions_df.to_sql("retention_actions", engine, if_exists="append", index=False)

    # Also write a quick local snapshot for debugging
    out_dir = Path("reports/weekly") / batch_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(out_dir / "scores.csv", index=False)
    segments_df.to_csv(out_dir / "segments.csv", index=False)
    actions_df.to_csv(out_dir / "actions.csv", index=False)

    logger.info(f"Wrote run artifacts to: {out_dir}")
    logger.info("Batch scoring complete ✅")
    logger.info(f"Next: python -m src.automation.send_actions --run-id {batch_run_id}")


if __name__ == "__main__":
    main()
