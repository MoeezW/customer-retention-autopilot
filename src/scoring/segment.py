from __future__ import annotations

import numpy as np
import pandas as pd


def segment_by_quantiles(
    probs: np.ndarray,
    high_quantile: float = 0.90,
    med_quantile: float = 0.70,
) -> pd.Series:
    """
    Segment customers into LOW / MEDIUM / HIGH churn risk by quantiles.

    Why quantiles are realistic:
    - Most companies have capacity constraints (call center, email budget, incentives).
    - They often target the top X% each week rather than a fixed probability threshold.

    Defaults:
    - HIGH = top 10% (>= 90th percentile)
    - MEDIUM = next 20% (70th to 90th percentile)
    - LOW = bottom 70%
    """
    probs = np.asarray(probs, dtype=float)

    high_cut = float(np.quantile(probs, high_quantile))
    med_cut = float(np.quantile(probs, med_quantile))

    # HIGH if prob >= high_cut
    # MEDIUM if med_cut <= prob < high_cut
    # LOW otherwise
    segments = np.where(
        probs >= high_cut,
        "HIGH",
        np.where(probs >= med_cut, "MEDIUM", "LOW"),
    )
    return pd.Series(segments)


def segment_by_thresholds(
    probs: np.ndarray,
    high: float = 0.70,
    med: float = 0.50,
) -> pd.Series:
    """
    Alternative segmentation using fixed thresholds.
    This matches many interview prompts and is simpler to explain.

    - HIGH if >= 0.70
    - MEDIUM if 0.50–0.70
    - LOW otherwise
    """
    probs = np.asarray(probs, dtype=float)
    segments = np.where(
        probs >= high,
        "HIGH",
        np.where(probs >= med, "MEDIUM", "LOW"),
    )
    return pd.Series(segments)
