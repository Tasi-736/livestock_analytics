"""LivestockLens - responsive Streamlit dashboard.

This interface reuses the project's existing app.py backend for the source
dataset loader, data-cleaning rules, four model specifications, train/validation/
test split, conditional SMOTENC procedure, evaluation metrics, and 2026
High/Low-category predictions. Keep app.py beside this file.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import LabelEncoder

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import app as backend  # Existing Colab-aligned data and ML backend


# ---------------------------------------------------------------------------
# Page and theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LivestockLens | Philippine Livestock Analytics",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --ll-forest: #153f34;
      --ll-forest-2: #1d6048;
      --ll-green: #2f9b68;
      --ll-mint: #eaf4ed;
      --ll-pale: #f6f8f5;
      --ll-ink: #20392f;
      --ll-muted: #687c72;
      --ll-border: #e2eae3;
      --ll-gold: #c99a4b;
    }

    html, body, [data-testid="stAppViewContainer"] {
      background: var(--ll-pale) !important;
      color: var(--ll-ink) !important;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    }
    /* Keep Streamlit's own top toolbar opaque so page content never shows through it. */
    [data-testid="stHeader"],
    header[data-testid="stHeader"] {
      background: #F6F8F5 !important;
      background-color: #F6F8F5 !important;
      opacity: 1 !important;
      border-bottom: 1px solid var(--ll-border) !important;
      box-shadow: 0 1px 5px rgba(21,63,52,.05) !important;
      z-index: 1000 !important;
      backdrop-filter: none !important;
    }
    [data-testid="stToolbar"] {
      background: #F6F8F5 !important;
      background-color: #F6F8F5 !important;
      opacity: 1 !important;
    }
    [data-testid="stAppViewContainer"] > .main {
      background: var(--ll-pale) !important;
    }
    .block-container {
      padding-top: 1.45rem !important;
      padding-bottom: 3rem !important;
      max-width: 1560px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #153f34 0%, #12392f 100%) !important;
      border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] * { color: #f5fbf6; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] .stCaption {
      color: #c8ddd0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
      color: #edf7ef !important;
      border-radius: 10px;
      padding: .46rem .58rem;
      transition: background .18s ease, transform .18s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
      background: rgba(255,255,255,.10);
      transform: translateX(2px);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
      background: #28694f;
      box-shadow: inset 3px 0 #c5dfa0;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
      background: rgba(255,255,255,.08) !important;
      border: 1px dashed rgba(255,255,255,.28) !important;
    }

    /* Headings and page header */
    h1, h2, h3, h4, p, label, span, small {
      color: var(--ll-ink);
    }
    h1 { font-weight: 780 !important; letter-spacing: -.035em; }
    h2, h3 { font-weight: 700 !important; letter-spacing: -.02em; }
    .ll-brand {
      color: #fff; font-size: 1.55rem; line-height: 1.2;
      font-weight: 820; letter-spacing: -.045em;
    }
    .ll-brand-sub {
      color: #c7ddcf !important; font-size: .78rem;
      margin-top: .25rem; letter-spacing: .015em;
    }
    .ll-topline {
      display:flex; align-items:center; justify-content:space-between;
      gap: .8rem; flex-wrap:wrap; margin-bottom:1.1rem;
    }
    .ll-eyebrow {
      color:#789084 !important; text-transform:uppercase;
      font-size:.68rem; font-weight:800; letter-spacing:.16em;
      margin-bottom:.24rem;
    }
    .ll-page-sub {
      color:var(--ll-muted) !important; font-size:.94rem;
      margin-top:-.45rem; margin-bottom:1.2rem;
    }
    .ll-status {
      display:inline-flex; align-items:center; gap:.45rem;
      color:#286449; background:#e9f4eb; border:1px solid #d6e9d9;
      border-radius:999px; padding:.42rem .76rem; font-size:.76rem;
      font-weight:700; white-space:nowrap;
    }
    .ll-dot { width:7px; height:7px; background:#36a66d; border-radius:50%; display:inline-block; }

    /* Dashboard cards */
    [data-testid="stMetric"] {
      background:#fff !important;
      border:1px solid var(--ll-border) !important;
      border-radius:15px !important;
      padding:1rem 1.1rem !important;
      min-height:112px;
      box-shadow:0 3px 12px rgba(28,66,46,.035);
      transition:transform .2s ease, box-shadow .2s ease, border-color .2s ease;
      animation:ll-rise .42s ease both;
    }
    [data-testid="stMetric"]:hover {
      transform:translateY(-3px);
      box-shadow:0 9px 22px rgba(28,66,46,.09);
      border-color:#c9dfcf !important;
    }
    [data-testid="stMetricLabel"] p {
      color:#718579 !important; font-size:.76rem !important;
      font-weight:700 !important;
    }
    [data-testid="stMetricValue"] {
      color:#1d503c !important; font-weight:780 !important;
      letter-spacing:-.035em;
    }
    [data-testid="stMetricDelta"] { font-size:.76rem !important; }

    /* Inputs: explicit light controls so browser dark mode cannot make labels vanish */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
    [data-testid="stTextArea"] textarea {
      background:#fff !important; color:#243b32 !important;
      border:1px solid #dce7de !important; border-radius:10px !important;
    }
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder { color:#8b9a91 !important; }
    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
      background:#fff !important; color:#243b32 !important;
    }
    [role="option"], [data-baseweb="option"] { color:#243b32 !important; }
    [data-testid="stSelectbox"] label,
    [data-testid="stMultiSelect"] label,
    [data-testid="stTextInput"] label,
    [data-testid="stNumberInput"] label { color:#52685d !important; font-weight:650; }

    /* Buttons and downloads */
    .stButton > button, .stDownloadButton > button {
      border-radius:10px !important; font-weight:700 !important;
      transition:transform .17s ease, box-shadow .17s ease, background .17s ease;
      min-height:2.55rem;
    }
    .stButton > button[kind="primary"] {
      background:#24694e !important; color:#fff !important;
      border:1px solid #24694e !important;
    }
    .stButton > button[kind="primary"]:hover {
      background:#1b553e !important; transform:translateY(-2px);
      box-shadow:0 7px 16px rgba(31,100,73,.18);
    }
    .stButton > button:not([kind="primary"]),
    .stDownloadButton > button {
      background:#fff !important; color:#285b45 !important;
      border:1px solid #d7e5da !important;
    }
    .stButton > button:not([kind="primary"]):hover,
    .stDownloadButton > button:hover {
      background:#eff7f0 !important; border-color:#a9cbb3 !important;
      transform:translateY(-1px);
    }

    /* Alerts, tables, expanders */
    [data-testid="stAlert"] {
      border-radius:12px !important; border:1px solid var(--ll-border);
    }
    [data-testid="stAlert"] p { color:inherit !important; }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
      background:#fff !important; border:1px solid var(--ll-border);
      border-radius:12px; overflow:hidden;
    }
    [data-testid="stExpander"] {
      background:#fff; border:1px solid var(--ll-border);
      border-radius:12px; overflow:hidden;
    }
    hr { border-color:#e1e9e2 !important; }
    .ll-card-title { font-size:1.02rem; font-weight:760; color:#254437; margin:.05rem 0 .45rem; }
    .ll-caption { color:#74867c; font-size:.78rem; }

    /* Subtle motion. Respect users who request reduced motion. */
    @keyframes ll-rise {
      from { opacity:0; transform:translateY(8px); }
      to { opacity:1; transform:translateY(0); }
    }
    @keyframes ll-fade { from {opacity:0;} to {opacity:1;} }
    [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
      animation:ll-fade .35s ease both;
    }
    @media (max-width: 900px) {
      .block-container { padding:1rem 1rem 2rem !important; }
      .ll-brand { font-size:1.35rem; }
      [data-testid="stMetric"] { min-height:100px; padding:.85rem !important; }
      h1 { font-size:1.8rem !important; }
    }
    /* On mobile, leave clear space below Streamlit's fixed top toolbar. */
    @media (max-width: 768px) {
      [data-testid="stHeader"],
      header[data-testid="stHeader"] {
        background: #F6F8F5 !important;
        background-color: #F6F8F5 !important;
        opacity: 1 !important;
        z-index: 1000 !important;
        backdrop-filter: none !important;
      }
      [data-testid="stToolbar"] {
        background: #F6F8F5 !important;
        opacity: 1 !important;
      }
      [data-testid="stAppViewContainer"] > .main .block-container {
        padding-top: 5.25rem !important;
      }
    }
    @media (max-width: 600px) {
      [data-testid="stAppViewContainer"] > .main .block-container {
        padding: 5.25rem .75rem 1.5rem !important;
      }
      .ll-topline { align-items:flex-start; }
      [data-testid="stMetric"] { min-height:90px; padding:.72rem !important; }
      [data-testid="stMetricValue"] { font-size:1.45rem !important; }
      h1 { font-size:1.55rem !important; }
      h2 { font-size:1.22rem !important; }
      .ll-page-sub { font-size:.86rem; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        animation-duration:.01ms !important;
        animation-iteration-count:1 !important;
        scroll-behavior:auto !important;
        transition-duration:.01ms !important;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def chart_style(fig, height=400):
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=45, b=22),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#496256", family="Inter, Arial, sans-serif", size=12),
        colorway=["#2f9b68", "#85b99a", "#c99a4b", "#315d4b", "#a7cdb3"],
        legend_title_text="",
        hoverlabel=dict(bgcolor="#153f34", font_color="#ffffff"),
        transition_duration=450,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e8eee9", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e8eee9", zeroline=False)
    return fig


def csv_download(frame, filename, label):
    st.download_button(
        label,
        data=frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        width="content",
    )


def get_data():
    current = backend.STATE.get("df")
    return current.copy() if current is not None else None


def upload_dataset(uploaded):
    payload = uploaded.getvalue()
    name = Path(uploaded.name).name
    cleaned = backend._read_dataset_bytes(payload, name)
    suffix = Path(name).suffix.lower()
    target = backend.DATA_DIR / f"active_dataset{suffix}"
    for old in backend.DATA_DIR.glob("active_dataset.*"):
        if old != target:
            old.unlink(missing_ok=True)
    target.write_bytes(payload)
    backend.STATE.update(df=cleaned, source=name, error=None)
    backend.MODEL_STATE.clear()
    backend.MODEL_STATE.update(ready=False, error=None)


def build_future_table(data):
    if not backend.MODEL_STATE.get("ready"):
        raise ValueError("Train the four models before generating predictions.")
    latest = data[data["Year"].eq(2025)].dropna(subset=["Production"]).copy()
    if latest.empty:
        raise ValueError("No valid 2025 observations are available.")
    latest["Animal_Encoded"] = backend.MODEL_STATE["animal_encoder"].transform(latest["Animal Type"])
    latest["Province_Encoded"] = backend.MODEL_STATE["province_encoder"].transform(latest["Geolocation"])
    feature_columns = backend.MODEL_STATE["feature_columns"]
    future_x = latest[feature_columns].copy()
    future_x["Year"] = 2026
    output = latest[["Geolocation", "Animal Type", "Year", "Production"]].rename(
        columns={"Geolocation": "Province", "Year": "Current Year", "Production": "Production in 2025"}
    )
    for name, model in backend.MODEL_STATE["final_models"].items():
        prediction = model.predict(future_x)
        probability = model.predict_proba(future_x)[:, 1]
        output[f"{name} - Predicted 2026 Class"] = np.where(
            prediction == 1, "High Production", "Low Production"
        )
        output[f"{name} - Probability High"] = probability
    return output


def section_heading(eyebrow, title, subtitle):
    st.markdown(
        f"<div class='ll-eyebrow'>{eyebrow}</div><h1>{title}</h1>"
        f"<p class='ll-page-sub'>{subtitle}</p>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar navigation and global status
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding:.4rem .15rem .8rem">
      <div class="ll-brand">🐄 LivestockLens</div>
      <div class="ll-brand-sub">Philippine production analytics · 2010–2025</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "MAIN MENU",
    ["Overview", "Descriptive Analytics", "Model Evaluation", "Predictions", "Data Records", "Data Source"],
    format_func=lambda x: {
        "Overview": "▦  Dashboard",
        "Descriptive Analytics": "◫  Descriptive analytics",
        "Model Evaluation": "▥  Model evaluation",
        "Predictions": "✦  Predictions",
        "Data Records": "▤  Data records",
        "Data Source": "ⓘ  Data source & methodology",
    }[x],
)
st.sidebar.markdown("---")

data = get_data()
if data is not None:
    st.sidebar.markdown("**DATASET STATUS**")
    st.sidebar.success(f"{len(data):,} cleaned observations")
    st.sidebar.caption(f"{data['Geolocation'].nunique():,} geographic units · {data['Year'].min()}–{data['Year'].max()}")
else:
    st.sidebar.warning("Dataset not loaded")

with st.sidebar.expander("Dataset setup / upload", expanded=False):
    st.caption("Use the original repository ZIP, ARFF or CSV only if automatic loading fails.")
    uploaded = st.file_uploader("Dataset file", type=["zip", "arff", "csv"], key="sidebar_upload")
    if uploaded is not None and st.button("Load dataset", width="stretch"):
        try:
            upload_dataset(uploaded)
            st.success(f"Loaded {len(backend.STATE['df']):,} observations.")
            st.rerun()
        except Exception as exc:
            st.error(f"Dataset upload failed: {exc}")

st.sidebar.markdown("---")
st.sidebar.caption("LivestockLens · Research / academic demonstration")
st.sidebar.caption("Predictions are High/Low categories, not official PSA forecasts.")

data = get_data()
if data is None:
    st.title("LivestockLens")
    st.warning(backend.STATE.get("error") or "The dataset is not loaded.")
    st.info("Wait for automatic dataset loading, or use Dataset setup / upload in the sidebar.")
    st.stop()

animals = sorted(data["Animal Type"].dropna().unique().tolist())
provinces = sorted(data["Geolocation"].dropna().unique().tolist())
years = sorted(data["Year"].dropna().astype(int).unique().tolist())
latest_year = int(data["Year"].max())

st.markdown(
    "<div class='ll-topline'><div><div class='ll-eyebrow'>PHILIPPINE LIVESTOCK DATA</div>"
    "<div style='font-size:.88rem;color:#6d8176'>Provincial production intelligence workspace</div></div>"
    "<div class='ll-status'><span class='ll-dot'></span> Dataset connected</div></div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Overview dashboard
# ---------------------------------------------------------------------------
if page == "Overview":
    section_heading("DASHBOARD", "Livestock production overview", "A clear summary of Philippine provincial livestock production from 2010 to 2025.")
    total_production = float(data["Production"].sum())
    metric_cols = st.columns(4)
    metric_cols[0].metric("Cleaned observations", f"{len(data):,}")
    metric_cols[1].metric("Provinces / areas", f"{data['Geolocation'].nunique():,}")
    metric_cols[2].metric("Animal categories", f"{data['Animal Type'].nunique():,}")
    metric_cols[3].metric("Total production", f"{total_production:,.2f} MT")

    totals = data.groupby("Animal Type", as_index=False)["Production"].sum().sort_values("Production", ascending=False)
    totals["Share (%)"] = totals["Production"] / totals["Production"].sum() * 100
    annual = data.groupby(["Year", "Animal Type"], as_index=False)["Production"].sum()
    top_hog = (
        data[(data["Year"] == 2025) & (data["Animal Type"] == "Hog")]
        .nlargest(10, "Production").sort_values("Production")
    )

    left, right = st.columns([1.58, 1])
    with left:
        st.markdown("<div class='ll-card-title'>Annual production trend</div><div class='ll-caption'>Summed provincial records · metric tons</div>", unsafe_allow_html=True)
        fig = px.line(annual, x="Year", y="Production", color="Animal Type", markers=True)
        fig.update_xaxes(dtick=1, title="Year")
        fig.update_yaxes(title="Production (metric tons)", tickformat=",.0f")
        st.plotly_chart(chart_style(fig, 420), width="stretch")
    with right:
        st.markdown("<div class='ll-card-title'>Animal category contribution</div><div class='ll-caption'>Total production by animal type</div>", unsafe_allow_html=True)
        bars = totals.sort_values("Production")
        fig = px.bar(bars, x="Production", y="Animal Type", orientation="h", color="Animal Type",
                     text=bars["Share (%)"].map(lambda v: f"{v:.1f}%"),
                     color_discrete_sequence=["#9bc7a8", "#c99a4b", "#6eaa82", "#24694e"])
        fig.update_traces(textposition="inside")
        fig.update_xaxes(title="Metric tons", tickformat=",.0f")
        st.plotly_chart(chart_style(fig, 420), width="stretch")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("<div class='ll-card-title'>Production share</div><div class='ll-caption'>2010–2025, all available provinces</div>", unsafe_allow_html=True)
        fig = px.pie(totals, names="Animal Type", values="Production", hole=.55,
                     color_discrete_sequence=["#24694e", "#6eaa82", "#c99a4b", "#b8d1bd"])
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(chart_style(fig, 360), width="stretch")
    with right:
        st.markdown("<div class='ll-card-title'>Top 10 hog-producing provinces</div><div class='ll-caption'>Latest available year: 2025</div>", unsafe_allow_html=True)
        if top_hog.empty:
            st.info("No 2025 hog observations are available.")
        else:
            fig = px.bar(top_hog, x="Production", y="Geolocation", orientation="h",
                         color="Production", color_continuous_scale=["#cfe4d3", "#24694e"])
            fig.update_xaxes(title="Metric tons", tickformat=",.0f")
            fig.update_yaxes(title="")
            st.plotly_chart(chart_style(fig, 360), width="stretch")

    with st.expander("Preview cleaned source records"):
        st.dataframe(data.head(20), width="stretch", hide_index=True)
    csv_download(data, "cleaned_provincial_livestock_2010_2025.csv", "↓  Export cleaned dataset")


# ---------------------------------------------------------------------------
# Descriptive analytics
# ---------------------------------------------------------------------------
elif page == "Descriptive Analytics":
    section_heading("EXPLORE THE DATA", "Descriptive analytics", "Compare annual totals, province-level distributions and relationships in the dataset.")
    f1, f2, f3 = st.columns(3)
    animal_filter = f1.selectbox("Animal type", ["All animal types"] + animals)
    province_filter = f2.selectbox("Province", ["All provinces"] + provinces)
    year_filter = f3.selectbox("Year", ["All years"] + years)
    selected = data.copy()
    if animal_filter != "All animal types":
        selected = selected[selected["Animal Type"] == animal_filter]
    if province_filter != "All provinces":
        selected = selected[selected["Geolocation"] == province_filter]
    if year_filter != "All years":
        selected = selected[selected["Year"] == int(year_filter)]
    if selected.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    m1, m2, m3 = st.columns(3)
    m1.metric("Filtered observations", f"{len(selected):,}")
    m2.metric("Filtered production", f"{selected['Production'].sum():,.2f} MT")
    m3.metric("Median production", f"{selected['Production'].median():,.2f} MT")

    desc = selected.groupby("Animal Type")["Production"].agg(
        Count="count", Mean="mean", Median="median", Std_Dev="std", Minimum="min", Maximum="max"
    ).reset_index().sort_values("Mean", ascending=False)
    st.subheader("Descriptive statistics by animal type")
    st.dataframe(
        desc.style.format({c: "{:,.2f}" for c in ["Mean", "Median", "Std_Dev", "Minimum", "Maximum"]}),
        width="stretch", hide_index=True,
    )
    csv_download(desc, "descriptive_summary_by_animal.csv", "↓  Export descriptive statistics")

    totals = data.groupby("Animal Type", as_index=False)["Production"].sum().rename(
        columns={"Production": "Total_Production_2010_2025"}
    )
    totals["Percentage Share (%)"] = totals["Total_Production_2010_2025"] / totals["Total_Production_2010_2025"].sum() * 100
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Total production by animal type")
        fig = px.bar(totals, x="Animal Type", y="Total_Production_2010_2025",
                     color="Animal Type", text_auto=".3s",
                     color_discrete_sequence=["#24694e", "#c99a4b", "#82b596", "#b8d1bd"])
        fig.update_yaxes(title="Metric tons", tickformat=",.0f")
        st.plotly_chart(chart_style(fig, 390), width="stretch")
    with c2:
        st.subheader("Production share")
        fig = px.pie(totals, names="Animal Type", values="Total_Production_2010_2025", hole=.48,
                     color_discrete_sequence=["#24694e", "#c99a4b", "#82b596", "#b8d1bd"])
        st.plotly_chart(chart_style(fig, 390), width="stretch")

    annual = data.groupby(["Year", "Animal Type"], as_index=False)["Production"].sum()
    if animal_filter != "All animal types":
        annual = annual[annual["Animal Type"] == animal_filter]
    if province_filter != "All provinces":
        annual = data[data["Geolocation"] == province_filter].groupby(["Year", "Animal Type"], as_index=False)["Production"].sum()
        if animal_filter != "All animal types":
            annual = annual[annual["Animal Type"] == animal_filter]
    st.subheader("Annual production trend (2010–2025)")
    fig = px.line(annual, x="Year", y="Production", color="Animal Type", markers=True)
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(title="Metric tons", tickformat=",.0f")
    st.plotly_chart(chart_style(fig, 420), width="stretch")

    latest = data[data["Year"] == 2025]
    st.subheader("Distribution of province-level production (2025)")
    if not latest.empty:
        fig = px.box(latest, x="Animal Type", y="Production", points=False,
                     category_orders={"Animal Type": backend.ANIMALS},
                     color="Animal Type",
                     color_discrete_sequence=["#24694e", "#c99a4b", "#82b596", "#b8d1bd"])
        fig.update_yaxes(title="Production (metric tons)")
        st.plotly_chart(chart_style(fig, 410), width="stretch")
        hog = latest[latest["Animal Type"] == "Hog"].nlargest(10, "Production").sort_values("Production")
        st.subheader("Top 10 hog-producing provinces in 2025")
        if not hog.empty:
            fig = px.bar(hog, x="Production", y="Geolocation", orientation="h", text_auto=".3s",
                         color="Production", color_continuous_scale=["#d8eadb", "#24694e"])
            st.plotly_chart(chart_style(fig, 400), width="stretch")
            csv_download(hog, "top_10_hog_producers_2025.csv", "↓  Export top 10 hog producers")

    st.subheader("Production vs. year and Pearson correlation")
    scatter = data.copy()
    if animal_filter != "All animal types":
        scatter = scatter[scatter["Animal Type"] == animal_filter]
    if province_filter != "All provinces":
        scatter = scatter[scatter["Geolocation"] == province_filter]
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(scatter, x="Year", y="Production", color="Animal Type", opacity=.65,
                         color_discrete_sequence=["#24694e", "#c99a4b", "#82b596", "#b8d1bd"])
        fig.update_yaxes(title="Production (metric tons)")
        st.plotly_chart(chart_style(fig, 430), width="stretch")
    with c2:
        model_df = data.copy()
        model_df["Class"] = (model_df["Production"] >= model_df["Production"].median()).astype(int)
        le_a, le_p = LabelEncoder(), LabelEncoder()
        model_df["Animal_Encoded"] = le_a.fit_transform(model_df["Animal Type"])
        model_df["Province_Encoded"] = le_p.fit_transform(model_df["Geolocation"])
        corr = model_df[["Year", "Animal_Encoded", "Province_Encoded", "Production", "Class"]].corr(method="pearson")
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=["#b4534b", "#f7f8f5", "#24694e"],
                        zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(chart_style(fig, 430), width="stretch")

    st.subheader("Total production and percentage share")
    st.dataframe(totals, width="stretch", hide_index=True)
    csv_download(totals, "total_production_share_by_animal.csv", "↓  Export production share")


# ---------------------------------------------------------------------------
# Model evaluation - calls the existing backend unchanged
# ---------------------------------------------------------------------------
elif page == "Model Evaluation":
    section_heading("PREDICTIVE ANALYTICS", "Model evaluation", "Four classifiers, conditional SMOTENC augmentation and untouched test-set evaluation.")
    st.info(
        "Target: High Production when a provincial observation is at or above the cleaned-data median; otherwise Low Production. "
        "SMOTENC is attempted only when validation accuracy is below 80%, and the augmented model is retained only if validation accuracy improves."
    )
    if not backend.MODEL_STATE.get("ready"):
        st.warning("The four models have not been trained in this app session.")
        if st.button("Train four models", type="primary", width="stretch"):
            try:
                with st.spinner("Training Logistic Regression, Decision Tree, Random Forest and Gradient Boosting..."):
                    backend.train_models()
                st.success("All four models trained and evaluated successfully.")
                st.rerun()
            except Exception as exc:
                backend.MODEL_STATE["error"] = str(exc)
                st.error(f"Model training failed: {exc}")
    else:
        result = backend.model_results_payload()
        c = st.columns(4)
        c[0].metric("Median threshold", f"{result['median_threshold']:,.2f}")
        c[1].metric("Training rows", f"{result['train_count']:,}")
        c[2].metric("Validation rows", f"{result['validation_count']:,}")
        c[3].metric("Untouched test rows", f"{result['test_count']:,}")
        st.success(f"Best test-set accuracy in this run: {result['best_model']}")

        metrics = pd.DataFrame(result["metrics"])
        display = metrics.rename(columns={
            "model": "Model", "accuracy": "Accuracy", "precision": "Precision",
            "recall": "Recall", "f1": "F1-Score", "roc_auc": "ROC-AUC",
            "validation_before": "Validation Accuracy Before",
            "validation_after": "Validation Accuracy After",
            "augmentation_triggered": "Augmentation Triggered",
            "synthetic_records_created": "Synthetic Rows Created",
            "augmentation_selected": "Augmented Model Selected",
            "training_records_used": "Final Training Rows Used",
        })
        st.subheader("Model comparison on the untouched test set")
        st.dataframe(display.style.format({
            "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}",
            "F1-Score": "{:.2%}", "ROC-AUC": "{:.4f}",
            "Validation Accuracy Before": "{:.2%}",
            "Validation Accuracy After": lambda v: "N/A" if pd.isna(v) else f"{v:.2%}",
        }), width="stretch", hide_index=True)
        csv_download(display, "livestock_model_comparison.csv", "↓  Export model comparison")

        long = display.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                            var_name="Metric", value_name="Score")
        fig = px.bar(long, x="Model", y="Score", color="Metric", barmode="group", text_auto=".2f")
        fig.update_yaxes(range=[0, 1.12], title="Score")
        st.plotly_chart(chart_style(fig, 450), width="stretch")

        audit = pd.DataFrame(result["augmentation_log"])
        audit = audit.rename(columns={
            "model": "Model", "validation_before": "Validation Accuracy Before",
            "threshold": "80% Threshold", "triggered": "Augmentation Triggered",
            "training_rows_before": "Training Rows Before",
            "training_rows_after_attempt": "Training Rows After Attempt",
            "synthetic_rows_created": "Synthetic Rows Created",
            "validation_after": "Validation Accuracy After",
            "augmented_model_selected": "Augmented Model Selected",
            "final_training_rows_used": "Final Training Rows Used", "decision": "Decision",
        })
        st.subheader("Conditional SMOTENC augmentation audit")
        st.dataframe(audit, width="stretch", hide_index=True)
        csv_download(audit, "conditional_augmentation_record_log.csv", "↓  Export augmentation audit")
        st.caption("Synthetic rows are training-only samples, not official livestock observations. The final 2026 refit uses real observations only.")

        st.subheader("Confusion matrices")
        cm_cols = st.columns(2)
        for idx, (name, matrix) in enumerate(result["confusion"].items()):
            fig = px.imshow(np.array(matrix), text_auto=True,
                            x=["Predicted Low", "Predicted High"],
                            y=["Actual Low", "Actual High"],
                            color_continuous_scale=["#eaf4ed", "#24694e"], aspect="auto")
            fig.update_layout(title=name, coloraxis_showscale=False)
            with cm_cols[idx % 2]:
                st.plotly_chart(chart_style(fig, 320), width="stretch")

        st.subheader("ROC curves")
        roc = go.Figure()
        for name, values in result["roc"].items():
            points = values["points"]
            roc.add_trace(go.Scatter(
                x=[p["x"] for p in points], y=[p["y"] for p in points],
                mode="lines", name=f"{name} · AUC {values['auc']:.4f}",
                line=dict(width=3),
            ))
        roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="#9aa79f", width=1.5), name="No-skill baseline",
        ))
        roc.update_xaxes(title="False Positive Rate", range=[0, 1])
        roc.update_yaxes(title="True Positive Rate", range=[0, 1.05])
        st.plotly_chart(chart_style(roc, 450), width="stretch")


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
elif page == "Predictions":
    section_heading("FUTURE CATEGORY ESTIMATES", "2026 production predictions", "Generate High / Low production categories for a selected animal type, province and year.")
    st.warning("These are High/Low category estimates, not exact production-volume forecasts or official PSA forecasts.")
    if not backend.MODEL_STATE.get("ready"):
        st.info("Train the four models on the Model Evaluation page before generating predictions.")
        if st.button("Train the four models now", type="primary"):
            try:
                with st.spinner("Training the four classification models..."):
                    backend.train_models()
                st.success("Training completed. Open Model Evaluation to inspect the results.")
                st.rerun()
            except Exception as exc:
                st.error(f"Model training failed: {exc}")
    else:
        c1, c2, c3 = st.columns(3)
        animal = c1.selectbox("Animal type", animals)
        province = c2.selectbox("Province", provinces)
        pred_year = c3.number_input("Prediction year", min_value=2026, max_value=2050, value=2026)
        if st.button("Generate prediction", type="primary"):
            try:
                animal_code = int(backend.MODEL_STATE["animal_encoder"].transform([animal])[0])
                province_code = int(backend.MODEL_STATE["province_encoder"].transform([province])[0])
                row = pd.DataFrame([[int(pred_year), animal_code, province_code]],
                                   columns=backend.MODEL_STATE["feature_columns"])
                outputs = []
                for name, model in backend.MODEL_STATE["final_models"].items():
                    label = int(model.predict(row)[0])
                    probability = float(model.predict_proba(row)[0][1])
                    outputs.append({
                        "Model": name,
                        "Predicted Category": "High Production" if label == 1 else "Low Production",
                        "Probability High": probability,
                        "Confidence": probability if label == 1 else 1 - probability,
                    })
                st.session_state["single_prediction"] = {
                    "animal": animal, "province": province, "year": int(pred_year),
                    "outputs": outputs, "best": backend.MODEL_STATE["best_model"],
                }
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

        saved = st.session_state.get("single_prediction")
        if saved:
            st.subheader(f"{saved['animal']} · {saved['province']} · {saved['year']}")
            frame = pd.DataFrame(saved["outputs"])
            best = frame[frame["Model"] == saved["best"]].iloc[0]
            p1, p2, p3 = st.columns(3)
            p1.metric("Best model", saved["best"])
            p2.metric("Predicted category", best["Predicted Category"])
            p3.metric("Confidence", f"{best['Confidence']:.2%}")
            st.dataframe(frame.style.format({"Probability High": "{:.2%}", "Confidence": "{:.2%}"}),
                         width="stretch", hide_index=True)
            csv_download(frame, "livestock_single_prediction.csv", "↓  Export this prediction")

        st.markdown("---")
        st.subheader("Bulk 2026 prediction table")
        st.caption("Uses all province–animal combinations with valid 2025 observations.")
        if st.button("Generate all 2026 predictions"):
            try:
                with st.spinner("Generating future category estimates..."):
                    st.session_state["future_predictions"] = build_future_table(data)
            except Exception as exc:
                st.error(f"Bulk prediction failed: {exc}")
        future = st.session_state.get("future_predictions")
        if future is not None:
            st.dataframe(future.head(100), width="stretch", hide_index=True)
            st.caption(f"Showing the first 100 of {len(future):,} rows.")
            csv_download(future, "predicted_2026_high_low_by_province_animal.csv", "↓  Export complete 2026 predictions")


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------
elif page == "Data Records":
    section_heading("SOURCE OBSERVATIONS", "Provincial data records", "Search, filter, inspect and export cleaned livestock observations.")
    c1, c2, c3 = st.columns(3)
    animal_filter = c1.selectbox("Animal type", ["All"] + animals)
    province_filter = c2.selectbox("Province", ["All"] + provinces)
    year_filter = c3.selectbox("Year", ["All"] + years)
    query = st.text_input("Search animal, province or year", placeholder="e.g. Hog, Abra or 2025")
    page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
    view = data.copy()
    if animal_filter != "All":
        view = view[view["Animal Type"] == animal_filter]
    if province_filter != "All":
        view = view[view["Geolocation"] == province_filter]
    if year_filter != "All":
        view = view[view["Year"] == int(year_filter)]
    if query:
        mask = (
            view["Animal Type"].str.contains(query, case=False, na=False)
            | view["Geolocation"].str.contains(query, case=False, na=False)
            | view["Year"].astype(str).str.contains(query, case=False, na=False)
        )
        view = view[mask]
    view = view.sort_values(["Year", "Geolocation", "Animal Type"], ascending=[False, True, True]).reset_index(drop=True)
    pages = max(1, int(np.ceil(len(view) / page_size)))
    page_number = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1)
    start = (int(page_number) - 1) * page_size
    st.caption(f"{len(view):,} matching observations · Page {page_number} of {pages}")
    st.dataframe(view.iloc[start:start + page_size], width="stretch", hide_index=True)
    csv_download(view, "filtered_livestock_records.csv", "↓  Export filtered records")


# ---------------------------------------------------------------------------
# Data source and methodology
# ---------------------------------------------------------------------------
elif page == "Data Source":
    section_heading("PROJECT INFORMATION", "Data source & methodology", "Dataset provenance, cleaning rules and the modelling process behind LivestockLens.")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("### Dataset")
        st.success(f"{len(data):,} cleaned province-year observations loaded.")
        st.write("**Source:** [Livestock Annual Volume of Production by Animal Type and Province, 2010–2025](https://github.com/Tasi-736/Livestock-Annual-Volume-of-Production-by-Animal-Type-and-Province-for-2010-2025)")
        st.write("**Coverage:** 2010–2025 · **Unit:** metric tons")
    with right:
        st.markdown("### Dataset coverage")
        st.metric("Geographic units", f"{data['Geolocation'].nunique():,}")
        st.metric("Animal categories", f"{data['Animal Type'].nunique()}")
    st.markdown("### Data preparation")
    st.markdown(
        "1. Read the ARFF file's comma-separated rows after `@data`, or a compatible CSV.  \n"
        "2. Convert 2010–2025 production values to numeric; missing markers such as `..` remain missing.  \n"
        "3. Retain Carabao, Cattle, Goat and Hog.  \n"
        "4. Retain province-level rows identified by exactly four leading dots in source `Geolocation`.  \n"
        "5. Reshape annual columns to one row per province, animal type and year; remove missing observations."
    )
    st.markdown("### Classification methodology")
    st.markdown(
        "- **Target:** High Production when production is at or above the cleaned-data median; otherwise Low Production.  \n"
        "- **Features:** Year, encoded Animal Type and encoded Geolocation.  \n"
        "- **Split:** 64% training, 16% validation and 20% untouched testing.  \n"
        "- **Models:** Logistic Regression, Decision Tree, Random Forest and Gradient Boosting.  \n"
        "- **Conditional SMOTENC:** Attempted only when validation accuracy is below 80%; retain the augmented model only if validation accuracy improves.  \n"
        "- **Future refit:** Uses real observations only."
    )
    st.caption("Synthetic SMOTENC rows are training-only samples, not official livestock observations. Future outputs are High/Low category estimates, not exact-volume forecasts or official PSA predictions.")
    st.markdown("---")
    st.markdown("### Upload / replace dataset")
    uploaded = st.file_uploader("Select the original ZIP, ARFF or CSV dataset", type=["zip", "arff", "csv"], key="source_upload")
    if uploaded is not None and st.button("Load this dataset", type="primary"):
        try:
            upload_dataset(uploaded)
            st.success(f"Loaded {len(backend.STATE['df']):,} observations. Retrain models to refresh evaluation results.")
            st.rerun()
        except Exception as exc:
            st.error(f"Dataset upload failed: {exc}")


st.markdown(
    "<div style='border-top:1px solid #e2eae3;margin-top:2.2rem;padding-top:.9rem;"
    "color:#819187;font-size:.74rem'>LivestockLens · Philippine livestock production analytics · "
    "For academic analysis and demonstration.</div>",
    unsafe_allow_html=True,
)
