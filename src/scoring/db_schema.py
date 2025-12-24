from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.utils.logging import get_logger

logger = get_logger("scoring.schema")


def ensure_scoring_tables(engine: Engine) -> None:
    """
    Creates the operational tables used by the weekly pipeline.

    Why we do this:
    - In real teams, your batch jobs must be idempotent: they can run on a fresh DB.
    - Tables store history per run_id (auditability).
    """
    ddl_statements = [
        """
        CREATE TABLE IF NOT EXISTS churn_scores (
            run_id TEXT NOT NULL,
            model_run_id TEXT NOT NULL,
            customer_id TEXT NOT NULL,
            churn_prob DOUBLE PRECISION NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            PRIMARY KEY (run_id, customer_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS churn_segments (
            run_id TEXT NOT NULL,
            customer_id TEXT NOT NULL,
            segment TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            PRIMARY KEY (run_id, customer_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS retention_actions (
            run_id TEXT NOT NULL,
            customer_id TEXT NOT NULL,
            action_type TEXT NOT NULL,
            template_id TEXT NOT NULL,
            reason_code TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            PRIMARY KEY (run_id, customer_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS action_dispatch_log (
            run_id TEXT NOT NULL,
            customer_id TEXT NOT NULL,
            template_id TEXT NOT NULL,
            status TEXT NOT NULL,
            sent_at TIMESTAMP NOT NULL DEFAULT NOW(),
            preview_path TEXT NOT NULL,
            PRIMARY KEY (run_id, customer_id)
        );
        """,
    ]

    with engine.begin() as conn:
        for ddl in ddl_statements:
            conn.execute(text(ddl))

        # Helpful indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_scores_run_id ON churn_scores(run_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_actions_run_id ON retention_actions(run_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_dispatch_run_id ON action_dispatch_log(run_id);"))

    logger.info("Scoring tables ensured ✅")
