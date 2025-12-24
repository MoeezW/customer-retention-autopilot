from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("utils.validation")


REQUIRED_COLUMNS = [
    "customer_id",
    "tenure",
    "monthly_charges",
    "total_charges",
    "contract",
    "payment_method",
    "churn",
]


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_target(df: pd.DataFrame) -> None:
    if "churn" not in df.columns:
        raise ValueError("Missing 'churn' column.")
    if df["churn"].isna().any():
        raise ValueError("Target 'churn' contains nulls.")
    allowed = {0, 1}
    vals = set(df["churn"].unique().tolist())
    if not vals.issubset(allowed):
        raise ValueError(f"Target 'churn' has unexpected values: {vals}")


def validate_customer_id_uniqueness(df: pd.DataFrame) -> None:
    if "customer_id" not in df.columns:
        raise ValueError("Missing 'customer_id' column.")
    dupes = df["customer_id"].duplicated().sum()
    if dupes > 0:
        raise ValueError(f"Found {dupes} duplicate customer_id values. ETL should dedupe.")


def validate_numeric_not_all_null(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    for c in numeric_cols:
        if c not in df.columns:
            continue
        if df[c].isna().all():
            raise ValueError(f"Numeric column '{c}' is all null after transform.")


def validate_clean_df(df: pd.DataFrame) -> None:
    """
    Run all validations that we consider 'must pass' for downstream steps.
    """
    logger.info("Running validation checks...")
    if df.empty:
        raise ValueError("DataFrame is empty.")

    validate_required_columns(df)
    validate_target(df)
    validate_customer_id_uniqueness(df)
    validate_numeric_not_all_null(df, numeric_cols=["tenure", "monthly_charges", "total_charges"])

    logger.info("All validations passed ✅")
