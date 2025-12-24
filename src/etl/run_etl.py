from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import settings
from src.etl.extract import read_raw_csv
from src.etl.transform import transform_raw
from src.etl.load import load_to_postgres
from src.utils.logging import get_logger
from src.utils.validation import validate_clean_df, validate_required_columns

logger = get_logger("etl.run")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ETL for Telco churn dataset.")
    p.add_argument(
        "--input",
        required=True,
        help="Path to raw churn CSV.",
    )
    p.add_argument(
        "--save-parquet",
        default="data/processed/churn_clean.parquet",
        help="Where to write the cleaned parquet snapshot.",
    )
    p.add_argument(
        "--skip-db",
        action="store_true",
        help="If set, do not load into Postgres (still writes parquet).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    input_path = Path(args.input)
    out_parquet = Path(args.save_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    # Extract
    df_raw = read_raw_csv(input_path)

    # Validate presence of expected raw columns (by normalized names)
    # (We validate required set early so we fail fast if dataset changes.)
    validate_required_columns(df_raw)

    # Transform
    df_clean = transform_raw(df_raw)

    # Validate cleaned output
    validate_clean_df(df_clean)

    # Save parquet snapshot (fast local dev + reproducibility)
    logger.info(f"Writing cleaned parquet snapshot: {out_parquet}")
    df_clean.to_parquet(out_parquet, index=False)

    # Load to Postgres
    if args.skip_db:
        logger.info("Skipping DB load (per --skip-db).")
        return

    logger.info("Loading to Postgres...")
    res = load_to_postgres(df_clean, settings.database_url)
    logger.info(f"Loaded churn_dataset rows: {res.rows_dataset}")
    logger.info(f"Loaded customer_dim rows: {res.rows_customer_dim}")
    logger.info("ETL finished ✅")


if __name__ == "__main__":
    main()
