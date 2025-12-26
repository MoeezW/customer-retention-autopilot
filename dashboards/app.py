from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text

# Make imports robust no matter where streamlit is launched from
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402


# ----------------------------
# Page setup
# ----------------------------

st.set_page_config(
    page_title="Customer Retention Autopilot",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Customer Retention Autopilot (Churn + Automation)")
st.caption("An internal-style tool for weekly churn scoring, segmentation, and retention actions.")


# ----------------------------
# DB helpers (cached)
# ----------------------------

@st.cache_resource
def get_engine():
    """
    cache_resource is the right tool for heavy, long-lived objects like DB engines.
    Streamlit will create it once per session (until code changes).
    """
    if not settings.database_url:
        raise ValueError("DATABASE_URL is empty. Check your .env file.")
    return create_engine(settings.database_url, future=True)


@st.cache_data(ttl=60)
def get_latest_run_id(_engine) -> Optional[str]:
    """
    The latest run is the one with the most recent created_at in churn_scores.
    NOTE: leading underscore tells Streamlit not to hash the engine.
    """
    q = text("""
        SELECT run_id
        FROM churn_scores
        ORDER BY created_at DESC
        LIMIT 1;
    """)
    df = pd.read_sql_query(q, con=_engine)
    if df.empty:
        return None
    return str(df.loc[0, "run_id"])


@st.cache_data(ttl=60)
def load_segment_counts(_engine, run_id: str) -> pd.DataFrame:
    q = text("""
        SELECT segment, COUNT(*) AS n
        FROM churn_segments
        WHERE run_id = :run_id
        GROUP BY segment
        ORDER BY segment;
    """)
    return pd.read_sql_query(q, con=_engine, params={"run_id": run_id})


@st.cache_data(ttl=60)
def load_actions_by_template(_engine, run_id: str) -> pd.DataFrame:
    q = text("""
        SELECT template_id, COUNT(*) AS n
        FROM retention_actions
        WHERE run_id = :run_id AND action_type <> 'NONE'
        GROUP BY template_id
        ORDER BY n DESC;
    """)
    return pd.read_sql_query(q, con=_engine, params={"run_id": run_id})


@st.cache_data(ttl=60)
def load_scores(_engine, run_id: str) -> pd.DataFrame:
    q = text("""
        SELECT customer_id, churn_prob, created_at
        FROM churn_scores
        WHERE run_id = :run_id;
    """)
    return pd.read_sql_query(q, con=_engine, params={"run_id": run_id})


@st.cache_data(ttl=60)
def load_targeting_list(_engine, run_id: str, limit: int = 500) -> pd.DataFrame:
    """
    Join model outputs + segment + action + a few customer attributes.
    This is what stakeholders want: who do we target, what do we do, and why?
    """
    q = text("""
        SELECT
            s.customer_id,
            sc.churn_prob,
            s.segment,
            a.action_type,
            a.template_id,
            a.reason_code,
            d.tenure,
            d.contract,
            d.monthly_charges,
            d.total_charges,
            d.internet_service,
            d.payment_method,
            d.paperless_billing
        FROM churn_segments s
        JOIN churn_scores sc
          ON s.customer_id = sc.customer_id AND s.run_id = sc.run_id
        JOIN retention_actions a
          ON s.customer_id = a.customer_id AND s.run_id = a.run_id
        JOIN churn_dataset d
          ON s.customer_id = d.customer_id
        WHERE s.run_id = :run_id
        ORDER BY sc.churn_prob DESC
        LIMIT :lim;
    """)
    return pd.read_sql_query(q, con=_engine, params={"run_id": run_id, "lim": limit})


@st.cache_data(ttl=60)
def load_summary_md(run_id: str) -> Optional[str]:
    p = PROJECT_ROOT / "reports" / "weekly" / run_id / "summary.md"
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


# ----------------------------
# Simple plotting utilities
# ----------------------------

def plot_histogram(probs: pd.Series, threshold: float):
    fig = plt.figure()
    plt.hist(probs, bins=30)
    plt.axvline(threshold, linestyle="--")
    plt.title("Churn Probability Distribution")
    plt.xlabel("Predicted churn probability")
    plt.ylabel("Customer count")
    plt.tight_layout()
    return fig


def plot_bar(counts_df: pd.DataFrame, x_col: str, y_col: str, title: str):
    fig = plt.figure()
    plt.bar(counts_df[x_col].astype(str), counts_df[y_col].astype(int))
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    return fig


# ----------------------------
# Sidebar controls
# ----------------------------

engine = get_engine()
latest_run = get_latest_run_id(engine)

with st.sidebar:
    st.header("Controls")

    if latest_run is None:
        st.error("No scoring runs found. Run: python -m src.scoring.score_batch")
        st.stop()

    run_id = st.text_input("Run ID", value=latest_run)

    page = st.radio(
        "Page",
        ["Overview", "Risk distribution", "Targeting list", "Report viewer"],
        index=0,
    )

    st.divider()
    st.caption("Tip: paste an older run_id to compare runs later.")


# ----------------------------
# Page: Overview
# ----------------------------

if page == "Overview":
    st.subheader("Overview")

    seg_counts = load_segment_counts(engine, run_id)
    actions_by_template = load_actions_by_template(engine, run_id)
    scores = load_scores(engine, run_id)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Run ID", run_id)
    c2.metric("Customers scored", len(scores))

    seg_map = {r["segment"]: int(r["n"]) for _, r in seg_counts.iterrows()}
    c3.metric("HIGH", seg_map.get("HIGH", 0))
    c4.metric("Emails sent (simulated)", int(actions_by_template["n"].sum()) if not actions_by_template.empty else 0)

    st.markdown("### Segment counts")
    st.dataframe(seg_counts, use_container_width=True)

    st.markdown("### Actions by template")
    if actions_by_template.empty:
        st.info("No actions found (or only NONE actions).")
    else:
        st.dataframe(actions_by_template, use_container_width=True)
        st.pyplot(plot_bar(actions_by_template, "template_id", "n", "Actions by template"))

    st.markdown("### Quick notes")
    st.write(
        "- Reads directly from Postgres tables populated by your ETL + scoring pipeline.\n"
        "- The Targeting List is the stakeholder view: who gets contacted, what template, and why.\n"
        "- The Report Viewer shows the weekly markdown report in-app."
    )


# ----------------------------
# Page: Risk distribution
# ----------------------------

elif page == "Risk distribution":
    st.subheader("Risk distribution")

    scores = load_scores(engine, run_id)
    probs = scores["churn_prob"].astype(float)

    threshold = st.slider("Exploration threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.01)

    above = (probs >= threshold).sum()
    st.metric("Customers above threshold", int(above))

    left, right = st.columns([2, 1])
    with left:
        st.pyplot(plot_histogram(probs, threshold))

    with right:
        st.markdown("### Top risky customers (by probability)")
        top = scores.sort_values("churn_prob", ascending=False).head(25)[["customer_id", "churn_prob"]]
        st.dataframe(top, use_container_width=True)
        st.caption("Use this to communicate policy tradeoffs: higher threshold = fewer contacts.")


# ----------------------------
# Page: Targeting list
# ----------------------------

elif page == "Targeting list":
    st.subheader("Targeting list")

    limit = st.slider("Rows to display", min_value=50, max_value=1000, value=500, step=50)
    df = load_targeting_list(engine, run_id, limit=int(limit))

    if df.empty:
        st.warning("No targeting rows found for this run_id.")
        st.stop()

    # Normalize types defensively
    df = df.copy()
    df["churn_prob"] = df["churn_prob"].astype(float)

    st.markdown("### Filters")

    # Build options safely
    seg_options = sorted(df["segment"].dropna().unique().tolist())
    tmpl_options = sorted(df["template_id"].dropna().unique().tolist())

    # Choose defaults that actually exist
    seg_default = [s for s in ["HIGH", "MEDIUM"] if s in seg_options]
    if not seg_default and seg_options:
        seg_default = [seg_options[0]]

    tmpl_default = [t for t in tmpl_options if t != "NONE"]
    # If everything is NONE, fall back to whatever exists
    if not tmpl_default and tmpl_options:
        tmpl_default = [tmpl_options[0]]

    f1, f2, f3 = st.columns(3)
    with f1:
        seg_filter = st.multiselect("Segment", options=seg_options, default=seg_default)
    with f2:
        tmpl_filter = st.multiselect("Template", options=tmpl_options, default=tmpl_default)
    with f3:
        min_prob = st.slider("Min churn_prob", 0.0, 1.0, 0.50, 0.01)

    view = df[df["churn_prob"] >= float(min_prob)].copy()
    if seg_filter:
        view = view[view["segment"].isin(seg_filter)]
    if tmpl_filter:
        view = view[view["template_id"].isin(tmpl_filter)]

    view = view.sort_values("churn_prob", ascending=False)

    st.markdown("### Ranked targeting table")
    st.dataframe(view, use_container_width=True, height=500)

    # Download button (internal-tool vibe)
    csv_bytes = view.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download targeting list (CSV)",
        data=csv_bytes,
        file_name=f"targeting_list_{run_id}.csv",
        mime="text/csv",
    )

    st.caption(
        "Exportable ops view: risk score → segment → action/template → reason + key customer attributes."
    )


# ----------------------------
# Page: Report viewer
# ----------------------------

elif page == "Report viewer":
    st.subheader("Weekly report viewer")

    md = load_summary_md(run_id)
    if md is None:
        st.warning(
            "summary.md not found for this run. Generate it with: "
            "python -m src.reporting.weekly_report --run-id <RUN_ID>"
        )
        st.stop()

    st.markdown(md)
    st.caption("Source: reports/weekly/<run_id>/summary.md")
