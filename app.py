"""PF Insights Q2 — Campaign Audience Demographics

Streamlit app: Select campaigns → FreeWheel impressions →
identity resolution → Experian demographic profiling (Age, Gender,
Ethnicity, HHI, Education).

UX matches Market Demographics app (main-body steps, no sidebar).
"""

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from databricks import sql as dbsql
from dotenv import load_dotenv

load_dotenv()

# ── Palette ──────────────────────────────────────────────────────────────────────
NAVY = "#1B2A4A"
CYAN = "#00BCD4"
LIGHT_CYAN = "#80DEEA"
LIME = "#C5E063"
DARK_BG = "#0d1f3a"
BORDER = "#2a3d5e"


# ── Configuration (matches Market Demographics pattern) ───────────────────────
def _cfg(env_key: str, secret_key: str | None = None) -> str:
    """Try st.secrets (lowercase then uppercase), fall back to env var."""
    for key in [secret_key or env_key.lower(), env_key]:
        try:
            val = st.secrets.get(key, "")
            if val:
                return val
        except Exception:
            pass
    return os.environ.get(env_key, "")


SERVER_HOSTNAME = _cfg("DATABRICKS_SERVER_HOSTNAME")
HTTP_PATH = _cfg("DATABRICKS_HTTP_PATH")
TOKEN = _cfg("DATABRICKS_TOKEN")


# ── DB helpers ─────────────────────────────────────────────────────────────────────
def _conn():
    for name, val in [
        ("DATABRICKS_SERVER_HOSTNAME", SERVER_HOSTNAME),
        ("DATABRICKS_HTTP_PATH", HTTP_PATH),
        ("DATABRICKS_TOKEN", TOKEN),
    ]:
        if not val:
            raise ValueError(f"{name} secret is missing or empty.")
    return dbsql.connect(
        server_hostname=SERVER_HOSTNAME.strip(),
        http_path=HTTP_PATH.strip(),
        access_token=TOKEN.strip(),
    )


def _run_query(query: str) -> pd.DataFrame:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


# ── CSS (matches Market Demographics exactly) ─────────────────────────────────
CSS = f"""
<style>
.block-container {{ padding-top: 1.25rem; }}
.header-bar {{
    background: linear-gradient(90deg, {NAVY} 0%, {DARK_BG} 100%);
    padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 1.25rem;
    border-left: 4px solid {CYAN};
}}
.header-bar h1 {{ color: {CYAN}; margin: 0; font-size: 1.7rem; }}
.header-bar p  {{ color: {LIGHT_CYAN}; margin: 0.2rem 0 0 0; font-size: 0.85rem; }}
.step-pill {{
    display: inline-block; background: {NAVY}; color: {CYAN};
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.09em; padding: 0.2rem 0.65rem;
    border-radius: 999px; border: 1px solid {BORDER};
    margin-bottom: 0.5rem;
}}
</style>
"""


# ── Campaign IDs (Planet Fitness Q2 2025 from Operative) ──────────────────────
CAMPAIGN_IDS = [
    '98126', '98416', '98425', '98118', '98121', '98123', '98124', '98125',
    '98397', '98415', '98421', '98530', '98127', '98119', '98417', '98423',
    '98447', '98173', '98422', '98420', '98597', '96690',
]


# ── Cached Loaders ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_campaign_list() -> pd.DataFrame:
    """Load campaign metadata from placement mapping for the known IDs."""
    id_list = ", ".join(f"\'{c}\'" for c in CAMPAIGN_IDS)
    df = _run_query(f"""
        SELECT
            locality_campaign_id,
            locality_advertiser,
            locality_campaign,
            locality_campaign_start_date,
            locality_campaign_end_date,
            COUNT(DISTINCT fw_placement_id) AS placement_count
        FROM locality_dev.silver.freewheel_placement_mapping
        WHERE locality_campaign_id IN ({id_list})
        GROUP BY locality_campaign_id, locality_advertiser, locality_campaign,
                 locality_campaign_start_date, locality_campaign_end_date
        ORDER BY locality_campaign_start_date, locality_campaign
    """)
    return df


def _resolve_cte(campaign_ids: list) -> str:
    """Shared CTE: campaign IDs → FW logs → identity resolution → LUIDs."""
    campaign_list = ", ".join(f"\'{c}\'" for c in campaign_ids)
    return f"""
        WITH campaign_impressions AS (
            SELECT DISTINCT fw.ip_address, fw.device_id, fw.device_id_prefix
            FROM locality_dev.gold.freewheel_logs_gold fw
            WHERE fw.locality_campaign_id IN ({campaign_list})
        ),
        resolved_luids AS (
            SELECT DISTINCT idm.luid
            FROM campaign_impressions ci
            JOIN locality_dev.silver.experian_consolidated_id_map idm
                ON ((ci.ip_address = idm.identity AND idm.id_type = 'ip')
                    OR (ci.device_id = idm.identity AND idm.id_type IN ('ctv','aaid','idfa')
                        AND ci.device_id_prefix != 'corrupted'))
            WHERE idm.luid IS NOT NULL
        )"""


@st.cache_data(ttl=600, show_spinner=False)
def load_summary(campaign_ids: tuple) -> dict:
    cte = _resolve_cte(list(campaign_ids))
    df = _run_query(f"""
        {cte}
        SELECT
            COUNT(*) AS total_persons,
            MEDIAN(ma.exact_age) AS median_age,
            MEDIAN(ma.est_income_amt_thousands) AS median_income
        FROM resolved_luids rl
        JOIN locality_dev.gold.experian_marketing_attributes ma ON ma.recd_luid = rl.luid
        WHERE ma.reliability_code BETWEEN 1 AND 4
    """)
    row = df.iloc[0]
    return {
        "total_persons": int(row["total_persons"]),
        "median_age": int(row["median_age"]) if pd.notna(row["median_age"]) else None,
        "median_income": int(row["median_income"]) if pd.notna(row["median_income"]) else None,
    }


@st.cache_data(ttl=600, show_spinner=False)
def load_age_dist(campaign_ids: tuple) -> pd.DataFrame:
    cte = _resolve_cte(list(campaign_ids))
    return _run_query(f"""
        {cte}
        SELECT
            CASE
                WHEN ma.exact_age < 18 THEN '<18'
                WHEN ma.exact_age < 25 THEN '18-24'
                WHEN ma.exact_age < 35 THEN '25-34'
                WHEN ma.exact_age < 45 THEN '35-44'
                WHEN ma.exact_age < 55 THEN '45-54'
                WHEN ma.exact_age < 65 THEN '55-64'
                WHEN ma.exact_age < 75 THEN '65-74'
                ELSE '75+'
            END AS age_band,
            COUNT(*) AS cnt
        FROM resolved_luids rl
        JOIN locality_dev.gold.experian_marketing_attributes ma ON ma.recd_luid = rl.luid
        WHERE ma.reliability_code BETWEEN 1 AND 4 AND ma.exact_age IS NOT NULL
        GROUP BY 1
        ORDER BY MIN(ma.exact_age)
    """)


@st.cache_data(ttl=600, show_spinner=False)
def load_gender_dist(campaign_ids: tuple) -> pd.DataFrame:
    cte = _resolve_cte(list(campaign_ids))
    return _run_query(f"""
        {cte}
        SELECT ma.gender, COUNT(*) AS cnt
        FROM resolved_luids rl
        JOIN locality_dev.gold.experian_marketing_attributes ma ON ma.recd_luid = rl.luid
        WHERE ma.reliability_code BETWEEN 1 AND 4
          AND ma.gender IS NOT NULL AND ma.gender != 'Unknown'
        GROUP BY ma.gender
        ORDER BY cnt DESC
    """)


@st.cache_data(ttl=600, show_spinner=False)
def load_ethnicity_dist(campaign_ids: tuple) -> pd.DataFrame:
    cte = _resolve_cte(list(campaign_ids))
    return _run_query(f"""
        {cte}
        SELECT ma.ethnic_group, COUNT(*) AS cnt
        FROM resolved_luids rl
        JOIN locality_dev.gold.experian_marketing_attributes ma ON ma.recd_luid = rl.luid
        WHERE ma.reliability_code BETWEEN 1 AND 4
          AND ma.ethnic_group IS NOT NULL AND ma.ethnic_group != 'Uncoded'
        GROUP BY ma.ethnic_group
        ORDER BY cnt DESC
        LIMIT 10
    """)


@st.cache_data(ttl=600, show_spinner=False)
def load_income_dist(campaign_ids: tuple) -> pd.DataFrame:
    cte = _resolve_cte(list(campaign_ids))
    return _run_query(f"""
        {cte}
        SELECT
            CASE
                WHEN ma.est_income_amt_thousands < 25  THEN '<$25K'
                WHEN ma.est_income_amt_thousands < 50  THEN '$25-50K'
                WHEN ma.est_income_amt_thousands < 75  THEN '$50-75K'
                WHEN ma.est_income_amt_thousands < 100 THEN '$75-100K'
                WHEN ma.est_income_amt_thousands < 150 THEN '$100-150K'
                WHEN ma.est_income_amt_thousands < 200 THEN '$150-200K'
                WHEN ma.est_income_amt_thousands < 300 THEN '$200-300K'
                ELSE '$300K+'
            END AS income_band,
            COUNT(*) AS cnt
        FROM resolved_luids rl
        JOIN locality_dev.gold.experian_marketing_attributes ma ON ma.recd_luid = rl.luid
        WHERE ma.reliability_code BETWEEN 1 AND 4
          AND ma.est_income_amt_thousands IS NOT NULL
        GROUP BY 1
        ORDER BY MIN(ma.est_income_amt_thousands)
    """)


@st.cache_data(ttl=600, show_spinner=False)
def load_education_dist(campaign_ids: tuple) -> pd.DataFrame:
    cte = _resolve_cte(list(campaign_ids))
    return _run_query(f"""
        {cte}
        SELECT ma.education_level, COUNT(*) AS cnt
        FROM resolved_luids rl
        JOIN locality_dev.gold.experian_marketing_attributes ma ON ma.recd_luid = rl.luid
        WHERE ma.reliability_code BETWEEN 1 AND 4
          AND ma.education_level IS NOT NULL
        GROUP BY ma.education_level
    """)


# ── Chart Builders (light theme matching Market Demographics) ─────────────────
def _layout(title: str, height: int = 380, **kwargs) -> dict:
    base = dict(
        title=dict(text=title, font_color=NAVY, font_size=15),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font_color="#333333", height=height,
        margin=dict(t=55, b=40),
    )
    base.update(kwargs)
    return base


def chart_age(df: pd.DataFrame) -> go.Figure:
    total = df["cnt"].astype(int).sum()
    fig = go.Figure(go.Bar(
        x=df["age_band"].tolist(), y=df["cnt"].astype(int).tolist(),
        marker_color=CYAN,
        text=[f"{v/total*100:.1f}%" for v in df["cnt"].astype(int)],
        textposition="outside",
    ))
    fig.update_layout(**_layout(
        "Age Distribution",
        xaxis=dict(title="Age Band", gridcolor="#eee", title_font_color="#555"),
        yaxis=dict(title="Persons", gridcolor="#eee", title_font_color="#555"),
    ))
    return fig


def chart_gender(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=df["gender"].tolist(), values=df["cnt"].astype(int).tolist(),
        marker_colors=[CYAN, LIME], hole=0.4,
        textinfo="label+percent", textfont_color="#333",
    ))
    fig.update_layout(**_layout("Gender Split"))
    return fig


def chart_ethnicity(df: pd.DataFrame) -> go.Figure:
    total = df["cnt"].astype(int).sum()
    fig = go.Figure(go.Bar(
        x=df["cnt"].astype(int).tolist(), y=df["ethnic_group"].tolist(),
        orientation="h", marker_color=CYAN,
        text=[f"{v/total*100:.1f}%" for v in df["cnt"].astype(int)],
        textposition="outside",
    ))
    fig.update_layout(**_layout(
        "Ethnicity (Top 10)", height=420,
        xaxis=dict(title="Persons", gridcolor="#eee", title_font_color="#555"),
        yaxis=dict(gridcolor="#eee"),
        margin=dict(l=180, t=55, b=40),
    ))
    return fig


def chart_income(df: pd.DataFrame) -> go.Figure:
    total = df["cnt"].astype(int).sum()
    fig = go.Figure(go.Bar(
        x=df["income_band"].tolist(), y=df["cnt"].astype(int).tolist(),
        marker_color=LIME,
        text=[f"{v/total*100:.1f}%" for v in df["cnt"].astype(int)],
        textposition="outside",
    ))
    fig.update_layout(**_layout(
        "Household Income Distribution", height=400,
        xaxis=dict(title="Income Band", gridcolor="#eee", title_font_color="#555", tickangle=-30),
        yaxis=dict(title="Persons", gridcolor="#eee", title_font_color="#555"),
        margin=dict(t=55, b=80),
    ))
    return fig


def chart_education(df: pd.DataFrame) -> go.Figure:
    order = ["Less Than High School Diploma", "High School Diploma",
             "Some College", "Completed College", "Graduate Degree"]
    df = df.set_index("education_level").reindex(order).dropna().reset_index()
    total = df["cnt"].astype(int).sum()
    fig = go.Figure(go.Bar(
        x=df["education_level"].tolist(), y=df["cnt"].astype(int).tolist(),
        marker_color=CYAN,
        text=[f"{v/total*100:.1f}%" for v in df["cnt"].astype(int)],
        textposition="outside",
    ))
    fig.update_layout(**_layout(
        "Education Level", height=400,
        xaxis=dict(title="Education", gridcolor="#eee", title_font_color="#555", tickangle=-20),
        yaxis=dict(title="Persons", gridcolor="#eee", title_font_color="#555"),
        margin=dict(t=55, b=80),
    ))
    return fig


# ── Main App ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="PF Insights Q2 — Campaign Demographics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # ── Header banner ──
    st.markdown(
        '<div class="header-bar"><h1>📊 PF Insights Q2 — Campaign Audience Demographics</h1>'
        '<p>Planet Fitness Q2 2025 campaigns | FreeWheel → Identity Resolution → Experian</p></div>',
        unsafe_allow_html=True,
    )

    # ── Step 1: Campaign Selection ──
    st.markdown('<div class="step-pill">Step 1 · Select Campaigns</div>', unsafe_allow_html=True)
    st.markdown("Choose one or more Planet Fitness campaigns:")

    with st.spinner("Loading campaign list..."):
        campaign_df = load_campaign_list()

    # Build display labels: short campaign name + date range
    campaign_df["label"] = (
        campaign_df["locality_campaign_id"] + " — "
        + campaign_df["locality_campaign"].str.extract(r"_(.*?)_O-", expand=False).fillna(campaign_df["locality_campaign"])
        + " (" + campaign_df["locality_campaign_start_date"].astype(str).str[:10]
        + " to " + campaign_df["locality_campaign_end_date"].astype(str).str[:10] + ")"
    )
    campaign_options = dict(zip(campaign_df["label"], campaign_df["locality_campaign_id"]))

    # Select All checkbox
    select_all = st.checkbox("Select all campaigns", value=True)

    if select_all:
        selected_campaign_ids = tuple(campaign_df["locality_campaign_id"].tolist())
        st.caption(f"✓ All {len(selected_campaign_ids)} campaigns selected")
    else:
        selected_labels = st.multiselect(
            "Search or select campaigns...",
            options=list(campaign_options.keys()),
            default=[],
            label_visibility="collapsed",
            placeholder="Search or select campaigns...",
        )
        selected_campaign_ids = tuple(campaign_options[lbl] for lbl in selected_labels)

    if not selected_campaign_ids:
        st.info("Select at least one campaign to continue.")
        return

    # ── Step 2: Demographic Profile ──
    st.markdown('<div class="step-pill">Step 2 · Demographic Profile</div>', unsafe_allow_html=True)

    with st.spinner("Querying campaign impressions & resolving identities..."):
        summary = load_summary(selected_campaign_ids)
        age_df = load_age_dist(selected_campaign_ids)
        gender_df = load_gender_dist(selected_campaign_ids)
        eth_df = load_ethnicity_dist(selected_campaign_ids)
        inc_df = load_income_dist(selected_campaign_ids)
        edu_df = load_education_dist(selected_campaign_ids)

    if summary["total_persons"] == 0:
        st.warning(
            "⚠️ No audience members matched. "
            "This may occur if campaign impressions are not yet available "
            "in the FreeWheel logs (data starts 2025-09-28)."
        )
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Persons", f"{summary['total_persons']:,}")
    col2.metric("Median Age", str(summary["median_age"]) if summary["median_age"] else "N/A")
    col3.metric("Median HHI", f"${summary['median_income']}K" if summary["median_income"] else "N/A")
    col4.metric("Campaigns Selected", str(len(selected_campaign_ids)))

    # Charts in 2-column layout
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        if not age_df.empty:
            st.plotly_chart(chart_age(age_df), use_container_width=True)
        if not eth_df.empty:
            st.plotly_chart(chart_ethnicity(eth_df), use_container_width=True)
        if not edu_df.empty:
            st.plotly_chart(chart_education(edu_df), use_container_width=True)
    with right:
        if not gender_df.empty:
            st.plotly_chart(chart_gender(gender_df), use_container_width=True)
        if not inc_df.empty:
            st.plotly_chart(chart_income(inc_df), use_container_width=True)


if __name__ == "__main__":
    main()
