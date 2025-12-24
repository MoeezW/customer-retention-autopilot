from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


@dataclass(frozen=True)
class EvalSummary:
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int


def metrics_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> EvalSummary:
    """
    Convert probabilities to class predictions using a threshold,
    then compute core classification metrics + confusion matrix.
    """
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return EvalSummary(
        roc_auc=float(roc_auc),
        pr_auc=float(pr_auc),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def threshold_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Iterable[float] = np.arange(0.10, 0.91, 0.05),
) -> pd.DataFrame:
    """
    Evaluate metrics across a range of thresholds.
    This is how you choose a threshold intentionally instead of defaulting to 0.5.
    """
    rows = []
    for t in thresholds:
        s = metrics_at_threshold(y_true, y_proba, float(t))
        positive_rate = float((y_proba >= t).mean())  # % of customers you'd contact at this threshold
        rows.append(
            {
                "threshold": float(t),
                "roc_auc": s.roc_auc,
                "pr_auc": s.pr_auc,
                "precision": s.precision,
                "recall": s.recall,
                "f1": s.f1,
                "positive_rate": positive_rate,
                "tp": s.tp,
                "fp": s.fp,
                "tn": s.tn,
                "fn": s.fn,
            }
        )
    return pd.DataFrame(rows)


def topk_targeting(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    ks: Iterable[float] = (0.05, 0.10, 0.20, 0.30),
) -> pd.DataFrame:
    """
    Business-ish evaluation:
    "If we can only contact the top K% highest-risk customers, what recall do we get?"

    - Sort customers by churn probability descending
    - Take top K% as 'targeted'
    - Compute how many churners are captured (recall among churners)
    """
    n = len(y_true)
    order = np.argsort(-y_proba)  # descending
    y_true_sorted = y_true[order]

    total_churners = int(y_true.sum())
    rows = []

    for k in ks:
        top_n = int(np.ceil(k * n))
        targeted = y_true_sorted[:top_n]
        captured = int(targeted.sum())

        recall = captured / total_churners if total_churners > 0 else 0.0
        precision = captured / top_n if top_n > 0 else 0.0

        rows.append(
            {
                "k": float(k),
                "top_n": int(top_n),
                "captured_churners": int(captured),
                "total_churners": int(total_churners),
                "recall_among_churners": float(recall),
                "precision_in_targeted": float(precision),
            }
        )

    return pd.DataFrame(rows)


def plot_calibration_curve(y_true: np.ndarray, y_proba: np.ndarray, outpath: Path) -> None:
    """
    Calibration curve answers:
    "When we predict 0.80 churn probability, does ~80% actually churn?"

    This matters a lot in operational systems because thresholds depend on probability trustworthiness.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("Calibration Curve (Quantile Bins)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
