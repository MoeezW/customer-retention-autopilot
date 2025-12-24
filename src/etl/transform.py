from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("etl.transform")


LEAKAGE_KEYWORDS = (
    "churn_reason",
    "cancel",
    "cancellation",
    "termination",
    "closed_account",
    "winback",
)


def _detect_potential_leakage_columns(columns: list[str]) -> list[str]:
    """
    A simple heuristic: if column names suggest they might encode
    information too close to the churn event, we warn.
    (This dataset doesn't include obvious leakage columns, but
    having this logic signals 'production thinking'.)
    """
    lower_cols = [c.lower() for c in columns]
    flagged = []
    for c in lower_cols:
        if c == "churn":
            continue
        if any(k in c for k in LEAKAGE_KEYWORDS):
            flagged.append(c)
    return flagged


def transform_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize raw telco churn dataset into a model-ready table.

    Rules:
    - Drop duplicates (customer_id unique)
    - Convert target churn: Yes/No -> 1/0
    - Fix TotalCharges blank strings -> NaN -> numeric
    - Ensure dtypes
    - Fill missing numeric with median, missing categoricals with "unknown"
    """
    df = df.copy()
    # --- Defensive canonicalization: map common variants to canonical column names
    # This ensures we tolerate minor differences in column normalization.
    col_map = {}
    if "customer_i_d" in df.columns and "customer_id" not in df.columns:
        col_map["customer_i_d"] = "customer_id"
    if "customerid" in df.columns and "customer_id" not in df.columns:
        col_map["customerid"] = "customer_id"
    if col_map:
        logger.info(f"Renaming columns for canonicalization: {col_map}")
        df = df.rename(columns=col_map)


    # --- Drop duplicates (keep first occurrence)
    if "customer_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["customer_id"], keep="first")
        after = len(df)
        if after != before:
            logger.info(f"Dropped duplicates by customer_id: {before - after}")

    # --- Strip whitespace from object columns (safe normalization)
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # --- Target conversion: churn Yes/No -> 1/0
    if "churn" not in df.columns:
        raise ValueError("Expected 'churn' column missing after extraction/normalization.")

    churn_map = {"Yes": 1, "No": 0}
    if not set(df["churn"].unique()).issubset(set(churn_map.keys())):
        raise ValueError(f"Unexpected churn values: {df['churn'].unique().tolist()}")
    df["churn"] = df["churn"].map(churn_map).astype("int64")

    # --- Fix TotalCharges: dataset has blank strings that pandas doesn't count as NaN
    if "total_charges" in df.columns:
        # Replace blank strings with real missing values
        df["total_charges"] = df["total_charges"].replace("", pd.NA)
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

    # --- Ensure numeric columns are numeric
    # tenure should be integer-like; monthly_charges float; total_charges float
    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    if "monthly_charges" in df.columns:
        df["monthly_charges"] = pd.to_numeric(df["monthly_charges"], errors="coerce")
    if "senior_citizen" in df.columns:
        df["senior_citizen"] = pd.to_numeric(df["senior_citizen"], errors="coerce")

    # --- Fill missing values
    # Numeric: median; Categorical: "unknown"
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    # (We don't want to fill churn target)
    numeric_fill_cols = [c for c in numeric_cols if c != "churn"]
    for c in numeric_fill_cols:
        if df[c].isna().any():
            med = df[c].median()
            df[c] = df[c].fillna(med)
            logger.info(f"Filled numeric NaNs in '{c}' with median={med}")

    for c in cat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna("unknown")
            logger.info(f"Filled categorical NaNs in '{c}' with 'unknown'")

    # --- Final dtype tightening
    # After fill, enforce expected types
    if "tenure" in df.columns:
        df["tenure"] = df["tenure"].astype("int64")
    if "senior_citizen" in df.columns:
        df["senior_citizen"] = df["senior_citizen"].astype("int64")

    # --- Leakage warnings (heuristic)
    flagged = _detect_potential_leakage_columns(list(df.columns))
    if flagged:
        logger.warning(f"Potential leakage-like columns detected: {flagged}")

    logger.info("Transform complete.")
    return df
