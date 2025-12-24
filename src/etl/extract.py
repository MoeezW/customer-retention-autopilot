from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("etl.extract")


def _to_snake_case(name: str) -> str:
    """
    Robust camelCase / PascalCase / ALLCAPS -> snake_case conversion.

    This two-step regex approach handles acronyms like "ID" properly so that
    "customerID" becomes "customer_id" (not "customer_i_d").
    """
    s = name.strip()

    # Insert underscores between a lower-to-upper boundary where the upper is followed by lowercase letters:
    # e.g. "CustomerName" -> "Customer_Name"
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s)

    # Insert underscores between a lower/number and upper boundary:
    # e.g. "customerID" -> "customer_ID"
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)

    # Replace any non-alphanumeric characters with underscore
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)

    # Normalize multiple underscores, lowercase, strip
    s = re.sub(r"_+", "_", s).lower().strip("_")

    return s


def read_raw_csv(path: str | Path) -> pd.DataFrame:
    """
    Read the raw churn CSV and normalize column names.
    We keep raw values as-is here; cleaning happens in transform().
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path}")

    logger.info(f"Reading raw CSV: {path}")
    df = pd.read_csv(path)

    # Normalize column names to snake_case
    df.columns = [_to_snake_case(c) for c in df.columns]

    logger.info(f"Loaded raw dataframe: {df.shape[0]} rows x {df.shape[1]} cols")
    logger.info(f"Normalized columns: {list(df.columns)}")
    return df
