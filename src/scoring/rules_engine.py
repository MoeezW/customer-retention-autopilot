from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class ActionDecision:
    action_type: str     # e.g. EMAIL or NONE
    template_id: str     # e.g. DISCOUNT_10, ONBOARDING_HELP
    reason_code: str     # e.g. HIGH_RISK_HIGH_CHARGES


def decide_action(row: pd.Series, segment: str) -> ActionDecision:
    """
    A tiny rules engine that converts risk + customer attributes into an action.

    This is a huge differentiator vs 'just a churn model':
    companies care about the *workflow* after scoring.

    Inputs we’ll use:
    - segment (HIGH/MEDIUM/LOW)
    - monthly_charges (proxy for revenue / price sensitivity)
    - tenure (new customers churn for different reasons)
    - contract (month-to-month is churn-prone; longer contracts less so)
    """
    tenure = int(row.get("tenure", 0))
    monthly = float(row.get("monthly_charges", 0.0))
    contract = str(row.get("contract", "")).lower()

    if segment == "HIGH":
        # Rule 1: high charges -> offer discount
        if monthly >= 70:
            return ActionDecision("EMAIL", "DISCOUNT_10", "HIGH_RISK_HIGH_CHARGES")

        # Rule 2: very new customer -> onboarding support
        if tenure <= 6:
            return ActionDecision("EMAIL", "ONBOARDING_HELP", "HIGH_RISK_LOW_TENURE")

        # Rule 3: month-to-month -> lock-in offer
        if "month-to-month" in contract or "month to month" in contract:
            return ActionDecision("EMAIL", "LOCK_IN_OFFER", "HIGH_RISK_MONTH_TO_MONTH")

        return ActionDecision("EMAIL", "LOYALTY_PERK", "HIGH_RISK_GENERIC")

    if segment == "MEDIUM":
        # Medium risk: softer nudges
        if tenure <= 6:
            return ActionDecision("EMAIL", "WELCOME_TIPS", "MEDIUM_RISK_NEW_CUSTOMER")
        return ActionDecision("EMAIL", "BENEFITS_REMINDER", "MEDIUM_RISK_GENERIC")

    # LOW risk: no action (don’t spam people)
    return ActionDecision("NONE", "NONE", "LOW_RISK_NO_ACTION")
