from __future__ import annotations

from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.config import settings
from src.features.build_features import FeatureSpec, make_splits, build_preprocessor
from src.modeling.evaluate import threshold_sweep, topk_targeting, plot_calibration_curve, metrics_at_threshold
from src.utils.logging import get_logger

logger = get_logger("modeling.train")


def choose_threshold_from_validation(sweep_df, max_contact_rate: float = 0.25) -> float:
    """
    Choose a threshold intentionally.

    Strategy:
    - Prefer thresholds where positive_rate <= max_contact_rate (e.g. contact at most 25% of customers)
    - Among those, pick the one with highest F1

    If none satisfy the contact constraint, pick the overall best F1.
    """
    constrained = sweep_df[sweep_df["positive_rate"] <= max_contact_rate]
    if len(constrained) > 0:
        best = constrained.sort_values("f1", ascending=False).iloc[0]
        return float(best["threshold"])
    best = sweep_df.sort_values("f1", ascending=False).iloc[0]
    return float(best["threshold"])


def main() -> None:
    # --- MLflow setup
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    mlflow.set_experiment("Customer Retention Autopilot - Churn Model")

    # --- Load data splits and build preprocessor
    spec = FeatureSpec()
    splits = make_splits(spec)

    preprocessor = build_preprocessor(splits.categorical_cols, splits.numeric_cols)

    # --- Model: Logistic Regression baseline (strong, interpretable, common in legacy)
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",  # helps imbalance
        solver="lbfgs",
    )

    # Pipeline = preprocessing + model (this is how you productionize tabular ML)
    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # Output directory for artifacts we will log to MLflow
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("reports/modeling") / run_stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"logreg_baseline_{run_stamp}") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        # Log basic params
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 2000)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("split_random_state", spec.random_state)
        mlflow.log_param("max_contact_rate", 0.25)

        # --- Train
        logger.info("Training model...")
        clf.fit(splits.X_train, splits.y_train)

        # --- Validation predictions (for threshold selection)
        val_proba = clf.predict_proba(splits.X_val)[:, 1]
        val_y = splits.y_val.to_numpy()

        # Sweep thresholds
        sweep_df = threshold_sweep(val_y, val_proba)
        sweep_path = out_dir / "threshold_sweep_val.csv"
        sweep_df.to_csv(sweep_path, index=False)
        mlflow.log_artifact(str(sweep_path), artifact_path="evaluation")

        # Top-k analysis (val)
        topk_df = topk_targeting(val_y, val_proba, ks=(0.05, 0.10, 0.20, 0.30, 0.40))
        topk_path = out_dir / "topk_targeting_val.csv"
        topk_df.to_csv(topk_path, index=False)
        mlflow.log_artifact(str(topk_path), artifact_path="evaluation")

        # Choose threshold
        chosen_threshold = choose_threshold_from_validation(sweep_df, max_contact_rate=0.25)
        mlflow.log_param("chosen_threshold", chosen_threshold)
        logger.info(f"Chosen threshold (from validation): {chosen_threshold:.2f}")

        # Validation metrics at chosen threshold
        val_summary = metrics_at_threshold(val_y, val_proba, chosen_threshold)
        mlflow.log_metric("val_roc_auc", val_summary.roc_auc)
        mlflow.log_metric("val_pr_auc", val_summary.pr_auc)
        mlflow.log_metric("val_precision", val_summary.precision)
        mlflow.log_metric("val_recall", val_summary.recall)
        mlflow.log_metric("val_f1", val_summary.f1)

        # --- Test evaluation (final holdout)
        test_proba = clf.predict_proba(splits.X_test)[:, 1]
        test_y = splits.y_test.to_numpy()

        test_summary = metrics_at_threshold(test_y, test_proba, chosen_threshold)
        mlflow.log_metric("test_roc_auc", test_summary.roc_auc)
        mlflow.log_metric("test_pr_auc", test_summary.pr_auc)
        mlflow.log_metric("test_precision", test_summary.precision)
        mlflow.log_metric("test_recall", test_summary.recall)
        mlflow.log_metric("test_f1", test_summary.f1)

        # Confusion matrix components are often helpful for managers
        mlflow.log_metric("test_tp", test_summary.tp)
        mlflow.log_metric("test_fp", test_summary.fp)
        mlflow.log_metric("test_tn", test_summary.tn)
        mlflow.log_metric("test_fn", test_summary.fn)

        # Calibration curve plot (test)
        calib_path = out_dir / "calibration_curve_test.png"
        plot_calibration_curve(test_y, test_proba, calib_path)
        mlflow.log_artifact(str(calib_path), artifact_path="plots")

        # Save the pipeline locally (artifact)
        model_path = Path("data/artifacts") / f"churn_model_pipeline_{run_id}.joblib"
        joblib.dump(clf, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model_artifacts")

        # Log model to MLflow model registry-style artifact
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="sklearn-model",
            input_example=splits.X_train.head(3),
        )

        logger.info("Training + logging complete ✅")
        logger.info(f"Artifacts written locally to: {out_dir}")


if __name__ == "__main__":
    main()
