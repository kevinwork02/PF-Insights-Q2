"""PF Insights Q2 — Campaign Audience Demographics

Streamlit app: DMA/zip filter → FreeWheel campaign impressions →
identity resolution → Experian demographic profiling (Age, Gender,
Ethnicity, HHI, Education).

Architecture mirrors ispot-ACR template.
"""

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from databricks import sql as dbsql
from dotenv import load_dotenv

load_dotenv()

# ── Color palette (navy/cyan/lime dark theme) ─────────────────────────────────
NAVY = "#1B2A4A"
CYAN = "#00BCD4"
LIGHT_CYAN = "#80DEEA"
LIME = "#C5E063"
DARK_BG = "#0d1f3a"
BORDER = "#2a3d5e"

# ── Campaign IDs (Planet Fitness Q2 2025 from Operative) ──────────────────────
CAMPAIGN_IDS = [
    '98126', '98416', '98425', '98118', '98121', '98123', '98124', '98125',
    '98397', '98415', '98421', '98530', '98127', '98119', '98417', '98423',
    '98447', '98173', '98422', '98420', '98597', '96690',
]


# ── Database connection ───────────────────────────────────────────────────────
def _get_creds():
    """Resolve Databricks SQL credentials (secrets → env)."""
    try:
        host = st.secrets["DATABRICKS_SERVER_HOSTNAME"]
        http_path = st.secrets["DATABRICKS_HTTP_PATH"]
        token = st.secrets["DATABRICKS_TOKEN"]
    except Exception:
        host = os.environ.get("DATABRICKS_SERVER_HOSTNAME", "")
        http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")
        token = os.environ.get("DATABRICKS_TOKEN", "")
    return host, http_path, token


def get_connection():
    host, http_path, token = _get_creds()
    return dbsql.connect(
        server_hostname=host,
        http_path=http_path,
        access_token=token,
    )


def run_query(query: str) -> pd.DataFrame:
    """Execute SQL and return pandas DataFrame."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


# ── Cached data loaders ───────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_dma_list() -> pd.DataFrame:
    """Load DMA code → name lookup."""
    return run_query("""
        SELECT CAST(dma_code AS STRING) AS dma_code, dma_name
        FROM locality_dev.default.dma_codes_v3
        ORDER BY dma_name
    """)


@st.cache_data(ttl=3600)
def load_zipcodes_for_dmas(dma_codes: tuple) -> pd.DataFrame:
    """Load zip codes available in selected DMAs."""
    dma_list = ", ".join(f"'{d}'" for d in dma_codes)
    return run_query(f"""
        SELECT DISTINCT zipcode, dma
        FROM locality_dev.silver.experian_location
        WHERE dma IN ({dma_list})
          AND zipcode IS NOT NULL
        ORDER BY zipcode
    """)


def query_demographics(dma_codes: list, zip_codes: list) -> pd.DataFrame:
    """Campaign impressions → identity resolution → demographics."""
    campaign_list = ", ".join(f"'{c}'" for c in CAMPAIGN_IDS)

    dma_clause = ""
    if dma_codes:
        dma_list = ", ".join(f"'{d}'" for d in dma_codes)
        dma_clause = f"AND fw.visitor_dma IN ({dma_list})"

    zip_clause = ""
    if zip_codes:
        zip_list = ", ".join(f"'{z}'" for z in zip_codes)
        zip_clause = f"AND fw.visitor_postal_code IN ({zip_list})"

    query = f"""
    WITH campaign_impressions AS (
        SELECT DISTINCT
            fw.ip_address,
            fw.device_id,
            fw.device_id_prefix
        FROM locality_dev.gold.freewheel_logs_gold fw
        WHERE fw.locality_campaign_id IN ({campaign_list})
          {dma_clause}
          {zip_clause}
    ),
    resolved_luids AS (
        SELECT DISTINCT idm.luid
        FROM campaign_impressions ci
        JOIN locality_dev.silver.experian_consolidated_id_map idm
            ON (
                (ci.ip_address = idm.identity AND idm.id_type = 'ip')
                OR (ci.device_id = idm.identity AND idm.id_type IN ('ctv', 'aaid', 'idfa')
                    AND ci.device_id_prefix != 'corrupted')
            )
        WHERE idm.luid IS NOT NULL
    )
    SELECT
        ma.exact_age,
        ma.gender,
        ma.ethnic_group,
        ma.est_income_amt_thousands,
        ma.education_level
    FROM resolved_luids rl
    JOIN locality_dev.gold.experian_marketing_attributes ma
        ON ma.recd_luid = rl.luid
    WHERE ma.reliability_code BETWEEN 1 AND 4
    """
    return run_query(query)


# ── Chart renderers ───────────────────────────────────────────────────────────
def _chart_layout(fig, title: str, height: int = 380, **kwargs):
    """Apply standard dark theme layout."""
    fig.update_layout(
        title=dict(text=title, font_color=LIGHT_CYAN),
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_BG,
        font_color=LIGHT_CYAN,
        height=height,
        margin=dict(t=55, b=40, l=kwargs.get("margin_l", 40)),
    )
    fig.update_xaxes(gridcolor=BORDER, title_font_color=LIGHT_CYAN)
    fig.update_yaxes(gridcolor=BORDER, title_font_color=LIGHT_CYAN)
    return fig


def render_age_chart(df: pd.DataFrame):
    age_df = df[df["exact_age"].notna()].copy()
    if age_df.empty:
        st.info("No age data available.")
        return
    bins = [0, 18, 25, 35, 45, 55, 65, 75, 120]
    labels = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
    age_df["age_band"] = pd.cut(age_df["exact_age"].astype(float), bins=bins, labels=labels, right=False)
    counts = age_df["age_band"].value_counts().sort_index()
    total = counts.sum()
    fig = go.Figure(go.Bar(
        x=counts.index.astype(str).tolist(),
        y=counts.values.tolist(),
        marker_color=CYAN,
        text=[f"{v/total*100:.1f}%" for v in counts.values],
        textposition="outside",
    ))
    fig = _chart_layout(fig, "Age Distribution")
    fig.update_xaxes(title_text="Age Band")
    fig.update_yaxes(title_text="Persons")
    st.plotly_chart(fig, use_container_width=True)


def render_gender_chart(df: pd.DataFrame):
    gdf = df[df["gender"].notna() & (df["gender"] != "Unknown")]
    if gdf.empty:
        st.info("No gender data available.")
        return
    counts = gdf["gender"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        marker_colors=[CYAN, LIME],
        hole=0.4,
        textinfo="label+percent",
        textfont_color=LIGHT_CYAN,
    ))
    fig = _chart_layout(fig, "Gender Split")
    st.plotly_chart(fig, use_container_width=True)


def render_ethnicity_chart(df: pd.DataFrame):
    edf = df[df["ethnic_group"].notna() & (df["ethnic_group"] != "Uncoded")]
    if edf.empty:
        st.info("No ethnicity data available.")
        return
    counts = edf["ethnic_group"].value_counts().head(10)
    total = counts.sum()
    fig = go.Figure(go.Bar(
        x=counts.values.tolist(),
        y=counts.index.tolist(),
        orientation="h",
        marker_color=CYAN,
        text=[f"{v/total*100:.1f}%" for v in counts.values],
        textposition="outside",
    ))
    fig = _chart_layout(fig, "Ethnicity (Top 10)", height=420, margin_l=180)
    fig.update_xaxes(title_text="Persons")
    st.plotly_chart(fig, use_container_width=True)


def render_income_chart(df: pd.DataFrame):
    idf = df[df["est_income_amt_thousands"].notna()].copy()
    if idf.empty:
        st.info("No income data available.")
        return
    bins = [0, 25, 50, 75, 100, 150, 200, 300, 5000]
    labels = ["<$25K", "$25-50K", "$50-75K", "$75-100K", "$100-150K", "$150-200K", "$200-300K", "$300K+"]
    idf["income_band"] = pd.cut(idf["est_income_amt_thousands"].astype(float), bins=bins, labels=labels, right=False)
    counts = idf["income_band"].value_counts().sort_index()
    total = counts.sum()
    fig = go.Figure(go.Bar(
        x=counts.index.astype(str).tolist(),
        y=counts.values.tolist(),
        marker_color=LIME,
        text=[f"{v/total*100:.1f}%" for v in counts.values],
        textposition="outside",
    ))
    fig = _chart_layout(fig, "Household Income Distribution", height=400)
    fig.update_xaxes(title_text="Income Band", tickangle=-30)
    fig.update_yaxes(title_text="Persons")
    st.plotly_chart(fig, use_container_width=True)


def render_education_chart(df: pd.DataFrame):
    edf = df[df["education_level"].notna()]
    if edf.empty:
        st.info("No education data available.")
        return
    edu_order = [
        "Less Than High School Diploma", "High School Diploma",
        "Some College", "Completed College", "Graduate Degree",
    ]
    counts = edf["education_level"].value_counts().reindex(edu_order).dropna()
    total = counts.sum()
    fig = go.Figure(go.Bar(
        x=counts.index.tolist(),
        y=counts.values.tolist(),
        marker_color=CYAN,
        text=[f"{v/total*100:.1f}%" for v in counts.values],
        textposition="outside",
    ))
    fig = _chart_layout(fig, "Education Level", height=400)
    fig.update_xaxes(title_text="Education", tickangle=-20)
    fig.update_yaxes(title_text="Persons")
    st.plotly_chart(fig, use_container_width=True)


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="PF Insights Q2 — Campaign Demographics",
        page_icon="\U0001f4ca",
        layout="wide",
    )

    # Custom CSS for dark theme
    st.markdown(f"""
    <style>
        .stApp {{ background-color: {NAVY}; }}
        .stSidebar {{ background-color: {DARK_BG}; }}
        h1, h2, h3 {{ color: {LIGHT_CYAN} !important; }}
        .stMetric label {{ color: {LIGHT_CYAN} !important; }}
        .stMetric [data-testid="stMetricValue"] {{ color: {CYAN} !important; }}
    </style>
    """, unsafe_allow_html=True)

    st.title("\U0001f4ca PF Insights Q2 — Campaign Audience Demographics")
    st.caption("Planet Fitness Q2 2025 campaigns | FreeWheel → Identity Resolution → Experian")

    # ── Sidebar: Filters ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("\U0001f50d Filters")

        # Step 1: DMA selection
        st.subheader("Step 1: Select DMA(s)")
        try:
            dma_df = load_dma_list()
            dma_options = dict(zip(
                dma_df["dma_name"] + " (" + dma_df["dma_code"] + ")",
                dma_df["dma_code"],
            ))
            selected_dma_labels = st.multiselect(
                "DMA Markets",
                options=list(dma_options.keys()),
                help="Leave empty for all DMAs",
            )
            selected_dmas = [dma_options[lbl] for lbl in selected_dma_labels]
        except Exception as e:
            st.error(f"Could not load DMA list: {e}")
            selected_dmas = []

        # Step 2: Zip code selection (optional, scoped to selected DMAs)
        st.subheader("Step 2: Zip Codes (optional)")
        selected_zips = []
        if selected_dmas:
            try:
                zip_df = load_zipcodes_for_dmas(tuple(selected_dmas))
                zip_options = sorted(zip_df["zipcode"].unique().tolist())
                selected_zips = st.multiselect(
                    "Filter by zip codes",
                    options=zip_options,
                    help="Leave empty for all zips in selected DMAs",
                )
            except Exception as e:
                st.warning(f"Could not load zip codes: {e}")
        else:
            st.info("Select DMAs above to enable zip filtering.")

        # Campaign info
        st.divider()
        st.subheader("Campaigns")
        st.write(f"**{len(CAMPAIGN_IDS)}** Planet Fitness campaigns")
        with st.expander("Campaign IDs"):
            st.code(", ".join(CAMPAIGN_IDS))

        # Run button
        st.divider()
        run_clicked = st.button(
            "\U0001f680 Profile Audience",
            type="primary",
            use_container_width=True,
        )

    # ── Main content area ─────────────────────────────────────────────────────
    if run_clicked:
        with st.spinner("Querying campaign impressions & resolving identities..."):
            df = query_demographics(selected_dmas, selected_zips)

        if df.empty:
            st.warning(
                "\u26a0\ufe0f No audience members matched. "
                "This may occur if campaign impressions are not yet available "
                "in the FreeWheel logs (data starts 2025-09-28)."
            )
            return

        # Summary metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Persons Matched", f"{len(df):,}")
        col2.metric("DMA Filter", ", ".join(selected_dmas) if selected_dmas else "All")
        col3.metric("Zip Filter", f"{len(selected_zips)} zips" if selected_zips else "All")
        col4.metric("Campaigns", str(len(CAMPAIGN_IDS)))

        st.markdown("---")

        # Render 5 demographic panels
        c1, c2 = st.columns(2)
        with c1:
            render_age_chart(df)
        with c2:
            render_gender_chart(df)

        render_ethnicity_chart(df)

        c3, c4 = st.columns(2)
        with c3:
            render_income_chart(df)
        with c4:
            render_education_chart(df)

    else:
        st.info(
            "\U0001f449 Use the sidebar to select DMA(s) and optionally zip codes, "
            "then click **Profile Audience** to generate demographic charts."
        )


if __name__ == "__main__":
    main()
