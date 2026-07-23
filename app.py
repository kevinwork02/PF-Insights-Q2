"""PF Insights Q2 — Campaign Audience Demographics

Streamlit app: Select campaigns → FreeWheel impressions →
identity resolution → Experian demographic profiling (Age, Gender,
Ethnicity, HHI, Education).

PERF: Single query returns all demographics; pandas does aggregation client-side.
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


# ── Configuration ─────────────────────────────────────────────────────────────────
def _cfg(env_key: str, secret_key: str | None = None) -> str:
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


# ── CSS ───────────────────────────────────────────────────────────────────────────
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


# ── Campaign IDs (Planet Fitness Q2 2026 from Operative) ──────────────────────
CAMPAIGN_IDS = [
    '103261', '103277', '103278', '103279', '103280', '103281', '103282', '103283',
    '103284', '103289', '103290', '103311', '103330', '103331', '103332', '103389',
    '103390', '103391', '103392', '103560', '103619', '103744',
]


# ── Single query: campaigns → identity resolution → demographics ──────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_campaign_list() -> pd.DataFrame:
    id_list = ", ".join(f"\'{c}\'" for c in CAMPAIGN_IDS)
    return _run_query(f"""
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


@st.cache_data(ttl=600, show_spinner=False)
def load_demographics(campaign_ids: tuple) -> pd.DataFrame:
    """Single heavy query — runs identity resolution ONCE, returns all demo cols."""
    campaign_list = ", ".join(f"\'{c}\'" for c in campaign_ids)
    return _run_query(f"""
        WITH campaign_impressions AS (
            SELECT DISTINCT ip_address, device_id, device_id_prefix
            FROM locality_dev.gold.freewheel_logs_gold
            WHERE locality_campaign_id IN ({campaign_list})
        ),
        resolved_luids AS (
            SELECT DISTINCT idm.luid
            FROM campaign_impressions ci
            JOIN locality_dev.silver.experian_consolidated_id_map idm
                ON ((ci.ip_address = idm.identity AND idm.id_type = 'ip')
                    OR (ci.device_id = idm.identity AND idm.id_type IN ('ctv','aaid','idfa')
                        AND ci.device_id_prefix != 'corrupted'))
            WHERE idm.luid IS NOT NULL
        )
        SELECT
            ma.exact_age,
            ma.gender,
            ma.ethnic_group,
            ma.est_income_amt_thousands,
            ma.education_level
        FROM resolved_luids rl
        JOIN locality_dev.gold.experian_marketing_attributes ma ON ma.recd_luid = rl.luid
        WHERE ma.reliability_code BETWEEN 1 AND 4
    """)


# ── Chart Builders (light theme) ─────────────────────────────────────────────────
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
    age_df = df[df["exact_age"].notna()].copy()
    bins = [0, 18, 25, 35, 45, 55, 65, 75, 120]
    labels = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
    age_df["age_band"] = pd.cut(age_df["exact_age"], bins=bins, labels=labels, right=False)
    counts = age_df["age_band"].value_counts().sort_index()
    total = counts.sum()
    fig = go.Figure(go.Bar(
        x=counts.index.astype(str).tolist(), y=counts.values.tolist(),
        marker_color=CYAN,
        text=[f"{v/total*100:.1f}%" for v in counts.values],
        textposition="outside",
    ))
    fig.update_layout(**_layout(
        "Age Distribution",
        xaxis=dict(title="Age Band", gridcolor="#eee", title_font_color="#555"),
        yaxis=dict(title="Persons", gridcolor="#eee", title_font_color="#555"),
    ))
    return fig


def chart_gender(df: pd.DataFrame) -> go.Figure:
    gdf = df[df["gender"].notna() & (df["gender"] != "Unknown")]
    counts = gdf["gender"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(), values=counts.values.tolist(),
        marker_colors=[CYAN, LIME], hole=0.4,
        textinfo="label+percent", textfont_color="#333",
    ))
    fig.update_layout(**_layout("Gender Split"))
    return fig


def chart_ethnicity(df: pd.DataFrame) -> go.Figure:
    edf = df[df["ethnic_group"].notna() & (df["ethnic_group"] != "Uncoded")]
    counts = edf["ethnic_group"].value_counts().head(10)
    total = counts.sum()
    fig = go.Figure(go.Bar(
        x=counts.values.tolist(), y=counts.index.tolist(),
        orientation="h", marker_color=CYAN,
        text=[f"{v/total*100:.1f}%" for v in counts.values],
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
    idf = df[df["est_income_amt_thousands"].notna()].copy()
    bins = [0, 25, 50, 75, 100, 150, 200, 300, 5000]
    labels = ["<$25K", "$25-50K", "$50-75K", "$75-100K", "$100-150K", "$150-200K", "$200-300K", "$300K+"]
    idf["income_band"] = pd.cut(idf["est_income_amt_thousands"], bins=bins, labels=labels, right=False)
    counts = idf["income_band"].value_counts().sort_index()
    total = counts.sum()
    fig = go.Figure(go.Bar(
        x=counts.index.astype(str).tolist(), y=counts.values.tolist(),
        marker_color=LIME,
        text=[f"{v/total*100:.1f}%" for v in counts.values],
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
    edf = df[df["education_level"].notna()]
    order = ["Less Than High School Diploma", "High School Diploma",
             "Some College", "Completed College", "Graduate Degree"]
    counts = edf["education_level"].value_counts().reindex(order).dropna()
    total = counts.sum()
    fig = go.Figure(go.Bar(
        x=counts.index.tolist(), y=counts.values.tolist(),
        marker_color=CYAN,
        text=[f"{v/total*100:.1f}%" for v in counts.values],
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
        page_icon="\U0001f4ca",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # ── Header banner ──
    st.markdown(
        '<div class="header-bar"><h1>\U0001f4ca PF Insights Q2 — Campaign Audience Demographics</h1>'
        '<p>Planet Fitness Q2 2026 campaigns | FreeWheel \u2192 Identity Resolution \u2192 Experian</p></div>',
        unsafe_allow_html=True,
    )

    # ── Step 1: Campaign Selection ──
    st.markdown('<div class="step-pill">Step 1 \u00b7 Select Campaigns</div>', unsafe_allow_html=True)
    st.markdown("Choose one or more Planet Fitness campaigns:")

    with st.spinner("Loading campaign list..."):
        campaign_df = load_campaign_list()

    # Build display labels
    campaign_df["label"] = (
        campaign_df["locality_campaign_id"] + " \u2014 "
        + campaign_df["locality_campaign"].str.extract(r"_(.*?)_O-", expand=False).fillna(campaign_df["locality_campaign"])
        + " (" + campaign_df["locality_campaign_start_date"].astype(str).str[:10]
        + " to " + campaign_df["locality_campaign_end_date"].astype(str).str[:10] + ")"
    )
    campaign_options = dict(zip(campaign_df["label"], campaign_df["locality_campaign_id"]))

    select_all = st.checkbox("Select all campaigns", value=True)

    if select_all:
        selected_campaign_ids = tuple(campaign_df["locality_campaign_id"].tolist())
        st.caption(f"\u2713 All {len(selected_campaign_ids)} campaigns selected")
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

    # ── Step 2: Demographic Profile (SINGLE QUERY) ──
    st.markdown('<div class="step-pill">Step 2 \u00b7 Demographic Profile</div>', unsafe_allow_html=True)

    with st.spinner("Running identity resolution & demographics (one query)..."):
        df = load_demographics(selected_campaign_ids)

    if df.empty:
        st.warning(
            "\u26a0\ufe0f No audience members matched for the selected campaigns. "
            "This may occur if campaign impressions have not yet been "
            "ingested into FreeWheel logs or identity resolution yielded no matches."
        )
        return

    # Summary metrics (computed from the single DataFrame — instant)
    total = len(df)
    median_age = int(df["exact_age"].median()) if df["exact_age"].notna().any() else None
    median_income = int(df["est_income_amt_thousands"].median()) if df["est_income_amt_thousands"].notna().any() else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Persons", f"{total:,}")
    col2.metric("Median Age", str(median_age) if median_age else "N/A")
    col3.metric("Median HHI", f"${median_income}K" if median_income else "N/A")
    col4.metric("Campaigns Selected", str(len(selected_campaign_ids)))

    # Charts in 2-column layout (all pandas aggregation — instant)
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(chart_age(df), use_container_width=True)
        st.plotly_chart(chart_ethnicity(df), use_container_width=True)
        st.plotly_chart(chart_education(df), use_container_width=True)
    with right:
        st.plotly_chart(chart_gender(df), use_container_width=True)
        st.plotly_chart(chart_income(df), use_container_width=True)


if __name__ == "__main__":
    main()
