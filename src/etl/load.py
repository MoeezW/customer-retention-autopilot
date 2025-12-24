from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.utils.logging import get_logger

logger = get_logger("etl.load")


@dataclass(frozen=True)
class LoadResult:
    rows_dataset: int
    rows_customer_dim: int


def _get_engine(database_url: str) -> Engine:
    if not database_url:
        raise ValueError("DATABASE_URL is empty. Did you create a .env file?")
    return create_engine(database_url, future=True)


def load_to_postgres(df_clean: pd.DataFrame, database_url: str) -> LoadResult:
    """
    Load the cleaned dataset into Postgres.

    We write:
    - churn_dataset: the full cleaned snapshot (all features + target)
    - customer_dim: a small dimension table (unique customers + basic demographics)
    """
    engine = _get_engine(database_url)

    # Full table for modeling/analytics
    table_dataset = "churn_dataset"

    # A minimal dimension table (looks more “warehouse-ish”)
    dim_cols = [c for c in ["customer_id", "gender", "senior_citizen", "partner", "dependents"] if c in df_clean.columns]
    df_dim = df_clean[dim_cols].drop_duplicates(subset=["customer_id"]).copy() if dim_cols else pd.DataFrame()

    logger.info(f"Writing table '{table_dataset}' (replace) ...")
    df_clean.to_sql(table_dataset, engine, if_exists="replace", index=False)

    if not df_dim.empty:
        logger.info("Writing table 'customer_dim' (replace) ...")
        df_dim.to_sql("customer_dim", engine, if_exists="replace", index=False)

    # Create useful indexes (safe/idempotent in Postgres with IF NOT EXISTS)
    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_churn_dataset_customer_id ON churn_dataset(customer_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_churn_dataset_churn ON churn_dataset(churn);"))
        if not df_dim.empty:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_customer_dim_customer_id ON customer_dim(customer_id);"))

    logger.info("Load complete.")
    return LoadResult(rows_dataset=len(df_clean), rows_customer_dim=len(df_dim))
