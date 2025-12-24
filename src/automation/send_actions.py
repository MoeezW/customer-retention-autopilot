from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import settings
from src.automation.action_templates import TEMPLATES
from src.utils.logging import get_logger

logger = get_logger("automation.send")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Send retention actions (simulated).")
    p.add_argument("--run-id", required=True, help="Batch run_id to send actions for.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = args.run_id

    if not settings.database_url:
        raise ValueError("DATABASE_URL is empty. Check your .env file.")

    engine = create_engine(settings.database_url, future=True)

    # Find actions for this run that:
    # - are action_type != NONE
    # - are not already in action_dispatch_log (not sent yet)
    query = text("""
        SELECT a.run_id, a.customer_id, a.action_type, a.template_id, a.reason_code
        FROM retention_actions a
        LEFT JOIN action_dispatch_log d
          ON a.run_id = d.run_id AND a.customer_id = d.customer_id
        WHERE a.run_id = :run_id
          AND a.action_type != 'NONE'
          AND d.customer_id IS NULL;
    """)

    actions = pd.read_sql_query(query, con=engine, params={"run_id": run_id})
    logger.info(f"Found {len(actions)} unsent actions for run_id={run_id}")

    out_dir = Path("reports/weekly") / run_id / "email_previews"
    out_dir.mkdir(parents=True, exist_ok=True)

    sent_rows = 0
    with engine.begin() as conn:
        for _, row in actions.iterrows():
            customer_id = str(row["customer_id"])
            template_id = str(row["template_id"])
            reason_code = str(row["reason_code"])

            if template_id not in TEMPLATES:
                logger.warning(f"Unknown template_id={template_id} for customer_id={customer_id}. Skipping.")
                continue

            tmpl = TEMPLATES[template_id]
            subject = tmpl.subject
            body = tmpl.body.format(customer_id=customer_id, reason_code=reason_code)

            preview_path = out_dir / f"{customer_id}_{template_id}.txt"
            preview_path.write_text(f"SUBJECT: {subject}\n\n{body}", encoding="utf-8")

            # Write dispatch log row (this is your "sent" record)
            conn.execute(
                text("""
                    INSERT INTO action_dispatch_log (run_id, customer_id, template_id, status, preview_path)
                    VALUES (:run_id, :customer_id, :template_id, :status, :preview_path)
                """),
                {
                    "run_id": run_id,
                    "customer_id": customer_id,
                    "template_id": template_id,
                    "status": "SENT_SIMULATED",
                    "preview_path": str(preview_path),
                }
            )

            sent_rows += 1

    logger.info(f"Simulated sending complete ✅ Sent {sent_rows} emails.")
    logger.info(f"Previews written to: {out_dir}")


if __name__ == "__main__":
    main()
