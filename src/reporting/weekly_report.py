from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sqlalchemy import create_engine, text

from src.config import settings
from src.utils.logging import get_logger

logger = get_logger("reporting.weekly")

EXPERIMENT_NAME = "Customer Retention Autopilot - Churn Model"


# -----------------------------
# Data classes for clean structure
# -----------------------------

@dataclass(frozen=True)
class RunContext:
    batch_run_id: str
    model_run_id: str


@dataclass(frozen=True)
class ModelSummary:
    model_run_id: str
    params: dict
    metrics: dict


# -----------------------------
# MLflow helpers
# -----------------------------

def get_model_summary_by_run_id(model_run_id: str) -> ModelSummary:
    """
    Load a specific MLflow run by run_id.

    This ties the weekly batch report to the *exact* model that produced the scores,
    which is the audit-correct enterprise pattern.
    """
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    client = MlflowClient()
    run = client.get_run(model_run_id)

    return ModelSummary(
        model_run_id=run.info.run_id,
        params=dict(run.data.params),
        metrics=dict(run.data.metrics),
    )


# -----------------------------
# DB helpers
# -----------------------------

def get_run_context(engine, batch_run_id: str) -> RunContext:
    """
    Determine which model_run_id was used for this batch run.
    We stored model_run_id in churn_scores when scoring.
    """
    q = text("""
        SELECT model_run_id
        FROM churn_scores
        WHERE run_id = :run_id
        LIMIT 1;
    """)
    df = pd.read_sql_query(q, con=engine, params={"run_id": batch_run_id})
    if df.empty:
        raise ValueError(f"No churn_scores found for run_id={batch_run_id}. Did you run score_batch?")
    model_run_id = str(df.loc[0, "model_run_id"])
    return RunContext(batch_run_id=batch_run_id, model_run_id=model_run_id)


def load_scores(engine, run_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        text("SELECT customer_id, churn_prob FROM churn_scores WHERE run_id = :run_id;"),
        con=engine,
        params={"run_id": run_id},
    )


def load_segments(engine, run_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        text("SELECT customer_id, segment FROM churn_segments WHERE run_id = :run_id;"),
        con=engine,
        params={"run_id": run_id},
    )


def load_actions(engine, run_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        text("""
            SELECT customer_id, action_type, template_id, reason_code
            FROM retention_actions
            WHERE run_id = :run_id;
        """),
        con=engine,
        params={"run_id": run_id},
    )


def load_dispatch_log(engine, run_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        text("""
            SELECT customer_id, template_id, status, sent_at
            FROM action_dispatch_log
            WHERE run_id = :run_id;
        """),
        con=engine,
        params={"run_id": run_id},
    )


def get_actual_churn_rate_in_segment(engine, run_id: str, segment: str) -> float:
    """
    Demo-only: because this dataset contains labels, we can compute the realized churn rate
    within a segment (HIGH/MEDIUM/LOW) as a sanity check for the impact estimate.

    In a real weekly scoring pipeline, you usually won't have churn outcomes yet for the current week.
    """
    q = text("""
        SELECT AVG(d.churn::float) AS churn_rate
        FROM churn_segments s
        JOIN churn_dataset d ON s.customer_id = d.customer_id
        WHERE s.run_id = :run_id AND s.segment = :segment;
    """)
    df = pd.read_sql_query(q, con=engine, params={"run_id": run_id, "segment": segment})
    if df.empty or df.loc[0, "churn_rate"] is None:
        return 0.0
    return float(df.loc[0, "churn_rate"])


# -----------------------------
# Top drivers (LogReg coefficients)
# -----------------------------

def get_logreg_top_drivers(mlflow_model_uri: str, top_n: int = 12) -> pd.DataFrame:
    """
    Extract 'top drivers' for a logistic regression pipeline:
    - Load the sklearn Pipeline from MLflow
    - Grab one-hot feature names from the ColumnTransformer
    - Grab LogisticRegression coefficients
    - Rank by absolute coefficient magnitude

    Interpretation:
    - Positive coefficient -> higher churn risk
    - Negative coefficient -> lower churn risk

    Notes:
    - Coefficients are associative, not causal.
    - With one-hot, each category becomes a separate indicator feature.
    """
    clf = mlflow.sklearn.load_model(mlflow_model_uri)

    # Our training pipeline is expected to have these named steps
    if not hasattr(clf, "named_steps"):
        raise ValueError("Expected an sklearn Pipeline with named_steps (preprocess, model).")

    if "preprocess" not in clf.named_steps or "model" not in clf.named_steps:
        raise ValueError("Expected pipeline steps named 'preprocess' and 'model'. Check your training code.")

    pre = clf.named_steps["preprocess"]
    model = clf.named_steps["model"]

    if not hasattr(model, "coef_"):
        raise ValueError("Model does not expose coef_. Top drivers not available for this model type.")

    # ColumnTransformer can emit expanded names like: cat__contract_Month-to-month
    feature_names = pre.get_feature_names_out()
    coefs = model.coef_.ravel()

    df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    df["abs_coef"] = df["coef"].abs()
    df = df.sort_values("abs_coef", ascending=False).head(top_n).copy()

    df["direction"] = df["coef"].apply(lambda x: "↑ churn risk" if x > 0 else "↓ churn risk")
    return df[["feature", "coef", "direction"]]


# -----------------------------
# Plot helpers
# -----------------------------

def plot_risk_distribution(scores_df: pd.DataFrame, outpath: Path) -> None:
    """
    Plot distribution of churn probabilities.
    Managers like seeing the risk curve shift over time.
    """
    probs = scores_df["churn_prob"].to_numpy(dtype=float)

    plt.figure()
    plt.hist(probs, bins=30)
    plt.title("Churn Risk Distribution")
    plt.xlabel("Predicted churn probability")
    plt.ylabel("Customer count")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_actions_by_template(actions_df: pd.DataFrame, outpath: Path) -> None:
    """
    Plot action counts by template_id (excluding NONE).
    """
    df = actions_df[actions_df["action_type"] != "NONE"].copy()
    counts = df["template_id"].value_counts()

    plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Actions Sent by Template")
    plt.xlabel("Template ID")
    plt.ylabel("Count")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# -----------------------------
# Business impact estimation (toy but coherent)
# -----------------------------

@dataclass(frozen=True)
class ImpactAssumptions:
    """
    Keep assumptions explicit.
    This is what makes the report defensible.
    """
    arpu_monthly: float = 60.0                 # Average revenue per user per month
    contacted_segment: str = "HIGH"            # We estimate impact for a targeted segment
    contact_rate_in_segment: float = 1.0       # fraction of that segment contacted
    baseline_churn_rate_in_segment: float = 0.08  # expected churn without intervention
    save_rate_from_intervention: float = 0.20  # fraction of churners saved by intervention


def estimate_business_impact(
    segments_df: pd.DataFrame,
    assumptions: ImpactAssumptions = ImpactAssumptions(),
) -> dict:
    """
    Calculate a simple revenue-protected estimate:

    Let N = number of customers in contacted segment (e.g., HIGH)
    We contact N * contact_rate customers

    Baseline churners among contacted = contacted * baseline_churn_rate
    Saved churners = baseline_churners * save_rate

    Monthly revenue protected = saved_churners * ARPU
    """
    seg = assumptions.contacted_segment
    N = int((segments_df["segment"] == seg).sum())
    contacted = int(np.round(N * assumptions.contact_rate_in_segment))

    baseline_churners = contacted * assumptions.baseline_churn_rate_in_segment
    saved_churners = baseline_churners * assumptions.save_rate_from_intervention
    revenue_protected = saved_churners * assumptions.arpu_monthly

    return {
        "segment": seg,
        "segment_count": N,
        "contacted": contacted,
        "baseline_churn_rate": assumptions.baseline_churn_rate_in_segment,
        "save_rate": assumptions.save_rate_from_intervention,
        "arpu_monthly": assumptions.arpu_monthly,
        "baseline_churners_est": float(baseline_churners),
        "saved_churners_est": float(saved_churners),
        "monthly_revenue_protected_est": float(revenue_protected),
    }


# -----------------------------
# Markdown report writer
# -----------------------------

def write_summary_markdown(
    outpath: Path,
    run_ctx: RunContext,
    model_summary: ModelSummary,
    totals: dict,
    top_drivers_df: pd.DataFrame,
    impact: dict,
) -> None:
    """
    Write a clean manager-friendly weekly summary.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Pull a few key metrics (might not exist if you change training later)
    m = model_summary.metrics
    p = model_summary.params

    lines = []
    lines.append("# Weekly Retention Autopilot Report")
    lines.append("")
    lines.append(f"- Generated at: **{now}**")
    lines.append(f"- Batch run_id: **{run_ctx.batch_run_id}**")
    lines.append(f"- Model run_id: **{run_ctx.model_run_id}**")
    lines.append("")

    lines.append("## 1) Operational Summary")
    lines.append("")
    lines.append(f"- Total customers scored: **{totals['total_scored']}**")
    lines.append("- Segments:")
    lines.append(f"  - HIGH: **{totals['seg_high']}**")
    lines.append(f"  - MEDIUM: **{totals['seg_medium']}**")
    lines.append(f"  - LOW: **{totals['seg_low']}**")
    lines.append("")
    lines.append("- Actions:")
    lines.append(f"  - Emails queued/sent (simulated): **{totals['emails_sent']}**")
    lines.append("")
    lines.append("### Actions by template")
    for k, v in totals["actions_by_template"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## 2) Model Performance (model used for this batch)")
    lines.append("")
    lines.append(
        f"- Chosen threshold policy: **{p.get('chosen_threshold', 'N/A')}** "
        f"(max_contact_rate={p.get('max_contact_rate', 'N/A')})"
    )
    lines.append(f"- Validation ROC-AUC: **{m.get('val_roc_auc', float('nan')):.4f}**")
    lines.append(f"- Validation PR-AUC: **{m.get('val_pr_auc', float('nan')):.4f}**")
    lines.append(f"- Test ROC-AUC: **{m.get('test_roc_auc', float('nan')):.4f}**")
    lines.append(f"- Test PR-AUC: **{m.get('test_pr_auc', float('nan')):.4f}**")
    lines.append(
        f"- Test Precision / Recall / F1: **"
        f"{m.get('test_precision', float('nan')):.3f} / "
        f"{m.get('test_recall', float('nan')):.3f} / "
        f"{m.get('test_f1', float('nan')):.3f}**"
    )
    lines.append("")

    lines.append("## 3) Top Drivers (Logistic Regression coefficients)")
    lines.append("")
    lines.append("Top features by absolute coefficient magnitude (post one-hot encoding).")
    lines.append("")
    lines.append("| Feature | Coef | Direction |")
    lines.append("|---|---:|---|")
    for _, r in top_drivers_df.iterrows():
        lines.append(f"| {r['feature']} | {float(r['coef']):.4f} | {r['direction']} |")
    lines.append("")
    lines.append("Note: coefficients are not causal; they reflect association under this dataset + preprocessing.")
    lines.append("")

    lines.append("## 4) Business Impact Estimate (toy model, assumptions explicit)")
    lines.append("")
    lines.append("Assumptions:")
    lines.append(f"- Segment targeted: **{impact['segment']}**")
    lines.append(f"- Baseline churn rate within targeted group: **{impact['baseline_churn_rate']:.2%}**")
    lines.append(f"- Save rate from intervention: **{impact['save_rate']:.2%}**")
    lines.append(f"- ARPU (monthly): **${impact['arpu_monthly']:.2f}**")
    lines.append("")
    lines.append("Estimated outcomes:")
    lines.append(f"- Customers in segment: **{impact['segment_count']}**")
    lines.append(f"- Contacted customers: **{impact['contacted']}**")
    lines.append(f"- Baseline churners (est): **{impact['baseline_churners_est']:.1f}**")
    lines.append(f"- Saved churners (est): **{impact['saved_churners_est']:.1f}**")
    lines.append(f"- Estimated monthly revenue protected: **${impact['monthly_revenue_protected_est']:.2f}**")
    lines.append("")
    lines.append(
        "Realism note: these numbers are small because the demo dataset is ~7k customers. "
        "In production, impact scales roughly linearly with customer base size and campaign scope "
        "(e.g., for 100k customers, multiply estimates by ~14× under similar segment proportions)."
    )
    lines.append("")

    outpath.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# CLI entry
# -----------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a weekly retention report for a given batch run_id.")
    p.add_argument("--run-id", required=True, help="Batch run_id from score_batch")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = args.run_id

    if not settings.database_url:
        raise ValueError("DATABASE_URL is empty. Check your .env file.")

    engine = create_engine(settings.database_url, future=True)

    # Figure out what model was used for this batch run (this is correct; no extra work needed)
    run_ctx = get_run_context(engine, run_id)

    # Load tables
    scores_df = load_scores(engine, run_id)
    segments_df = load_segments(engine, run_id)
    actions_df = load_actions(engine, run_id)
    dispatch_df = load_dispatch_log(engine, run_id)

    if scores_df.empty or segments_df.empty or actions_df.empty:
        raise ValueError("Expected scoring tables are empty for this run_id. Did score_batch succeed?")

    # Compute summary counts
    totals = {
        "total_scored": int(len(scores_df)),
        "seg_high": int((segments_df["segment"] == "HIGH").sum()),
        "seg_medium": int((segments_df["segment"] == "MEDIUM").sum()),
        "seg_low": int((segments_df["segment"] == "LOW").sum()),
        "emails_sent": int(len(dispatch_df)),
        "actions_by_template": actions_df[actions_df["action_type"] != "NONE"]["template_id"].value_counts().to_dict(),
    }

    # Model performance summary (for the exact model used by this batch)
    model_summary = get_model_summary_by_run_id(run_ctx.model_run_id)

    # Build a model URI to load the exact pipeline used (for top drivers)
    model_uri = f"runs:/{run_ctx.model_run_id}/sklearn-model"
    top_drivers_df = get_logreg_top_drivers(model_uri, top_n=12)

    # Output folder for this run
    out_dir = Path("reports/weekly") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save top drivers as a CSV artifact too
    top_drivers_df.to_csv(out_dir / "top_drivers.csv", index=False)

    # Plots
    plot_risk_distribution(scores_df, out_dir / "risk_distribution.png")
    plot_actions_by_template(actions_df, out_dir / "actions_by_template.png")

    # Business impact estimate:
    # For demo credibility we anchor baseline churn rate in HIGH to the realized churn rate in that segment.
    # (In real weekly scoring you wouldn't know outcomes yet.)
    actual_high_churn_rate = get_actual_churn_rate_in_segment(engine, run_id, "HIGH")
    impact = estimate_business_impact(
        segments_df,
        assumptions=ImpactAssumptions(
            arpu_monthly=60.0,
            contacted_segment="HIGH",
            contact_rate_in_segment=1.0,
            baseline_churn_rate_in_segment=actual_high_churn_rate if actual_high_churn_rate > 0 else 0.08,
            save_rate_from_intervention=0.20,
        ),
    )

    # Write summary.md
    write_summary_markdown(out_dir / "summary.md", run_ctx, model_summary, totals, top_drivers_df, impact)

    logger.info(f"Weekly report generated ✅ at: {out_dir}")
    logger.info(f"- {out_dir / 'summary.md'}")
    logger.info(f"- {out_dir / 'top_drivers.csv'}")
    logger.info(f"- {out_dir / 'risk_distribution.png'}")
    logger.info(f"- {out_dir / 'actions_by_template.png'}")


if __name__ == "__main__":
    main()
