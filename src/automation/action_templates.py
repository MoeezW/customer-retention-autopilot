from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmailTemplate:
    subject: str
    body: str


TEMPLATES = {
    "DISCOUNT_10": EmailTemplate(
        subject="A small thank-you — 10% off your next bill",
        body=(
            "Hi {customer_id},\n\n"
            "We noticed you’ve been with us and want to make sure you’re getting the best value.\n"
            "Here’s a 10% discount on your next bill if you stay with us.\n\n"
            "Reply to this email or click the link in your account portal to apply it.\n\n"
            "— Retention Team\n"
            "Reason: {reason_code}\n"
        ),
    ),
    "ONBOARDING_HELP": EmailTemplate(
        subject="Need help getting set up? We can help.",
        body=(
            "Hi {customer_id},\n\n"
            "We want to make sure everything is working smoothly.\n"
            "If you’ve had any issues setting up your service, we can help right away.\n\n"
            "Reply to this email and our support team will assist you.\n\n"
            "— Support Team\n"
            "Reason: {reason_code}\n"
        ),
    ),
    "LOCK_IN_OFFER": EmailTemplate(
        subject="Lock in your rate with a longer-term plan",
        body=(
            "Hi {customer_id},\n\n"
            "If you’d like a more stable monthly rate, we can offer a longer-term contract option.\n"
            "This can reduce monthly variability and include extra benefits.\n\n"
            "Reply if you want details.\n\n"
            "— Plans Team\n"
            "Reason: {reason_code}\n"
        ),
    ),
    "LOYALTY_PERK": EmailTemplate(
        subject="A loyalty perk for being with us",
        body=(
            "Hi {customer_id},\n\n"
            "We appreciate your business. Here’s a small loyalty perk available in your account.\n"
            "Check your portal for details.\n\n"
            "— Retention Team\n"
            "Reason: {reason_code}\n"
        ),
    ),
    "WELCOME_TIPS": EmailTemplate(
        subject="Quick tips to get the most out of your service",
        body=(
            "Hi {customer_id},\n\n"
            "Here are a few quick tips to make sure you’re getting the best experience.\n"
            "If you have any issues, reply and we’ll help.\n\n"
            "— Support Team\n"
            "Reason: {reason_code}\n"
        ),
    ),
    "BENEFITS_REMINDER": EmailTemplate(
        subject="Reminder: benefits you may be missing",
        body=(
            "Hi {customer_id},\n\n"
            "Just a reminder: your plan includes benefits you may not be using.\n"
            "Log in to your portal to see what’s included.\n\n"
            "— Customer Success\n"
            "Reason: {reason_code}\n"
        ),
    ),
}
