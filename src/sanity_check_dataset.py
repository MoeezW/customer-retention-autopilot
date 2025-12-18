from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("sanity")


def main() -> None:
    path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(path)

    logger.info(f"Loaded: {path}")
    logger.info(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    logger.info(f"Columns: {list(df.columns)}")

    # Basic target sanity
    if "Churn" not in df.columns:
        raise ValueError("Expected column 'Churn' not found. Dataset format changed?")

    churn_counts = df["Churn"].value_counts(dropna=False)
    logger.info("Churn value counts:")
    for k, v in churn_counts.items():
        logger.info(f"  {k}: {v}")

    # Check missingness quickly
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) == 0:
        logger.info("No missing values detected by pandas isna().")
    else:
        logger.info("Missing values by column (top):")
        for col, cnt in missing.head(15).items():
            logger.info(f"  {col}: {cnt}")

    # A known quirk: TotalCharges is sometimes stored as string with blanks
    if "TotalCharges" in df.columns:
        blanks = (df["TotalCharges"].astype(str).str.strip() == "").sum()
        logger.info(f"TotalCharges blank-string rows: {blanks}")

    logger.info("Sanity check complete ✅")


if __name__ == "__main__":
    main()
