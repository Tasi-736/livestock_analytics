"""LivestockLens Streamlit application.

The data-cleaning, model definitions, train/validation/test split, conditional
SMOTENC augmentation, metrics, and 2026 refit are reused from app.py so the
Streamlit interface uses the same analytical workflow as the Colab notebook.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import LabelEncoder

import app as backend

st.set_page_config(
    page_title="LivestockLens | Philippine Livestock Analytics",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Theme and helpers
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    :root { --forest: #174b3b; --leaf: #2f7659; --mint: #e9f2eb; --ink: #243b32; }
    .stApp { background: #f6f8f5; color: var(--ink); }
    [data-testid="stSidebar"] { background: #153f34; }
    [data-testid="stSidebar"] * { color: #f7fbf7 !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #d8e8dd !important; }
    h1, h2, h3 { color: #183d31; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #e1e9e2; padding: 15px 18px; border-radius: 14px; }
    div[data-testid="stMetricLabel"] { color: #678174; }
    div[data-testid="stMetricValue"] { color: #1c503d; }
    div.stButton > button[kind="primary"] { background: #1d6048; border-color: #1d6048; }
    .ll-banner { background: #edf5ed; border: 1px solid #d7e7d8; padding: 13px 17px; border-radius: 12px; color: #315846; margin-bottom: 14px; }
    .ll-subtitle { color: #72877a; font-size: 0.9rem; margin-top: -8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_data() -> pd.DataFrame | None:
    data = backend.STATE.get("df")
    return data.copy() if data is not None else None


def money(value: float) -> str:
    return f"{value:,.2f}"


def chart_config(fig, height: int = 420):
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=55, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#385548", family="Arial"),
        legend_title_text="",
    )
    return fig


def download_csv(df: pd.DataFrame, filename: str, label: str):
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        use_container_width=False,
    )


def handle_dataset_upload(uploaded_file) -> bool:
    if uploaded_file is None:
        return False
    payload = uploaded_file.getvalue()
    name = Path(uploaded_file.name).name
    cleaned = backend._read_dataset_bytes(payload, name)
    ext = Path(name).suffix.lower()
    target = backend.DATA_DIR / f"active_dataset{ext}"
    for old in backend.DATA_DIR.glob("active_dataset.*"):
        if old != target:
            old.unlink(missing_ok=True)
    target.write_bytes(payload)
    backend.STATE.update(df=cleaned, source=name, error=None)
    backend.MODEL_STATE.clear()
    backend.MODEL_STATE.update(ready=False, error=None)
    return True


def prepare_future_predictions(data: pd.DataFrame) -> pd.DataFrame:
    if not backend.MODEL_STATE.get("ready"):
        raise ValueError("Train the four models first on the Model Evaluation page.")
    latest = data[data["Year"].eq(2025)].dropna(subset=["Production"]).copy()
    if latest.empty:
        raise ValueError("No valid 2025 observations are available for 2026 predictions.")
    enc_animal = backend.MODEL_STATE["animal_encoder"]
    enc_province = backend.MODEL_STATE["province_encoder"]
    latest["Animal_Encoded"] = enc_animal.transform(latest["Animal Type"])
    latest["Province_Encoded"] = enc_province.transform(latest["Geolocation"])
    features = backend.MODEL_STATE["feature_columns"]
    X_future = latest[features].copy()
    X_future["Year"] = 2026
    output = latest[["Geolocation", "Animal Type", "Year", "Production"]].rename(
        columns={
            "Geolocation": "Province",
            "Year": "Current Year",
            "Production": "Production in 2025",
        }
    )
    for model_name, model in backend.MODEL_STATE["final_models"].items():
        pred = model.predict(X_future)
        prob = model.predict_proba(X_future)[:, 1]
        output[f"{model_name} - Predicted 2026 Class"] = np.where(
            pred == 1, "High Production", "Low Production"
        )
        output[f"{model_name} - Probability High"] = prob
    return output


def sidebar_upload():
    st.sidebar.markdown("### Data source")
    st.sidebar.caption("Upload the original dataset if automatic loading is unavailable.")
    uploaded = st.sidebar.file_uploader(
        "Dataset file", type=["zip", "arff", "csv"], key="dataset_upload"
    )
    if uploaded is not None and st.sidebar.button("Load uploaded dataset", use_container_width=True):
        try:
            with st.spinner("Reading and cleaning the dataset..."):
                handle_dataset_upload(uploaded)
            st.sidebar.success(
                f"Loaded {len(backend.STATE['df']):,} provincial observations."
            )
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Could not load dataset: {exc}")


# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------

st.sidebar.markdown(
    "<div style='font-size:1.65rem;font-weight:800;letter-spacing:-.04em'>🐄 LivestockLens</div>"
    "<div style='font-size:.85rem;color:#d8e8dd'>PH production analytics · 2010–2025</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "WORKSPACE",
    ["Overview", "Descriptive Analytics", "Model Evaluation", "Predictions", "Data Records", "Data Source"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
sidebar_upload()

_data = get_data()

if _data is None:
    st.title("LivestockLens")
    st.markdown(
        "<div class='ll-banner'>The dashboard is ready, but the livestock dataset has not been loaded yet. "
        "The app attempts to download the source dataset automatically. If that fails, upload the original ZIP, ARFF, or CSV file in the sidebar.</div>",
        unsafe_allow_html=True,
    )
    st.error(backend.STATE.get("error") or "Dataset not loaded.")
    if page != "Data Source":
        st.info("Open **Data Source** or use the sidebar upload control to load the dataset.")
    st.stop()

# The authoritative cleaned data frame from the same loader used by the Flask app.
data = _data
animals = sorted(data["Animal Type"].dropna().unique().tolist())
provinces = sorted(data["Geolocation"].dropna().unique().tolist())
years = sorted(data["Year"].dropna().astype(int).unique().tolist())
latest_year = int(data["Year"].max())

# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------

if page == "Overview":
    st.title("Livestock production overview")
    st.markdown("<p class='ll-subtitle'>A provincial view of Philippine livestock production, 2010–2025.</p>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='ll-banner'>Dataset loaded successfully. The dashboard uses cleaned province-level observations from 2010–2025. Source: {backend.STATE.get('source') or 'GitHub dataset'}.</div>",
        unsafe_allow_html=True,
    )
    total = float(data["Production"].sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cleaned observations", f"{len(data):,}")
    k2.metric("Provinces / geographic units", f"{data['Geolocation'].nunique():,}")
    k3.metric("Animal categories", f"{data['Animal Type'].nunique():,}")
    k4.metric("Total production", f"{total:,.2f} metric tons")

    totals = data.groupby("Animal Type", as_index=False)["Production"].sum().sort_values("Production", ascending=False)
    totals["Share"] = totals["Production"] / totals["Production"].sum() * 100
    annual = data.groupby(["Year", "Animal Type"], as_index=False)["Production"].sum()
    top_hog = data[(data["Year"] == 2025) & (data["Animal Type"] == "Hog")].nlargest(10, "Production").sort_values("Production")

    left, right = st.columns([1.55, 1])
    with left:
        st.subheader("Annual production trend")
        fig = px.line(annual, x="Year", y="Production", color="Animal Type", markers=True)
        fig.update_xaxes(dtick=1, title="Year")
        fig.update_yaxes(title="Production (metric tons)", tickformat=",.0f")
        st.plotly_chart(chart_config(fig, 430), use_container_width=True)
    with right:
        st.subheader("Animal category contribution")
        fig = px.bar(totals.sort_values("Production"), x="Production", y="Animal Type", orientation="h", text=totals.sort_values("Production")["Share"].map(lambda x: f"{x:.1f}%"))
        fig.update_traces(textposition="inside")
        fig.update_xaxes(title="Total production (metric tons)", tickformat=",.0f")
        st.plotly_chart(chart_config(fig, 430), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Production share, 2010–2025")
        fig = px.pie(totals, names="Animal Type", values="Production", hole=.42)
        st.plotly_chart(chart_config(fig, 390), use_container_width=True)
    with right:
        st.subheader("Top 10 provinces by hog production, 2025")
        if top_hog.empty:
            st.info("No 2025 hog observations are available.")
        else:
            fig = px.bar(top_hog, x="Production", y="Geolocation", orientation="h", color="Production", color_continuous_scale="Greens")
            fig.update_xaxes(title="Production (metric tons)", tickformat=",.0f")
            fig.update_yaxes(title="Province")
            st.plotly_chart(chart_config(fig, 390), use_container_width=True)

    with st.expander("Dataset preview"):
        st.dataframe(data.head(15), use_container_width=True, hide_index=True)
        st.caption(f"Year coverage: {int(data['Year'].min())}–{latest_year} · Unit: metric tons")

# -----------------------------------------------------------------------------
# Descriptive analytics: preserve the notebook's charts and statistics
# -----------------------------------------------------------------------------

elif page == "Descriptive Analytics":
    st.title("Descriptive analytics")
    st.markdown("<p class='ll-subtitle'>Explore totals, distributions, geographic rankings and relationships in the provincial dataset.</p>", unsafe_allow_html=True)

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
    m2.metric("Filtered production", f"{selected['Production'].sum():,.2f}")
    m3.metric("Median production", f"{selected['Production'].median():,.2f}")

    desc = selected.groupby("Animal Type")["Production"].agg(
        Count="count", Mean="mean", Median="median", Std_Dev="std", Minimum="min", Maximum="max"
    ).reset_index().sort_values("Mean", ascending=False)
    st.subheader("Descriptive statistics by animal type")
    st.dataframe(desc.style.format({c: "{:.2f}" for c in ["Mean", "Median", "Std_Dev", "Minimum", "Maximum"]}), use_container_width=True, hide_index=True)
    download_csv(desc, "descriptive_summary_by_animal.csv", "Download descriptive statistics")

    totals = data.groupby("Animal Type", as_index=False)["Production"].sum().rename(columns={"Production": "Total_Production_2010_2025"})
    totals["Percentage Share (%)"] = totals["Total_Production_2010_2025"] / totals["Total_Production_2010_2025"].sum() * 100
    totals = totals.sort_values("Total_Production_2010_2025", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Total provincial production by animal type")
        fig = px.bar(totals, x="Animal Type", y="Total_Production_2010_2025", color="Animal Type", text_auto=".3s")
        fig.update_yaxes(title="Total production (metric tons)", tickformat=",.0f")
        st.plotly_chart(chart_config(fig, 400), use_container_width=True)
    with c2:
        st.subheader("Share of provincial livestock production")
        fig = px.pie(totals, names="Animal Type", values="Total_Production_2010_2025", hole=.35)
        st.plotly_chart(chart_config(fig, 400), use_container_width=True)

    annual = data.groupby(["Year", "Animal Type"], as_index=False)["Production"].sum()
    if animal_filter != "All animal types":
        annual = annual[annual["Animal Type"] == animal_filter]
    if province_filter != "All provinces":
        annual = data[data["Geolocation"] == province_filter].groupby(["Year", "Animal Type"], as_index=False)["Production"].sum()
        if animal_filter != "All animal types":
            annual = annual[annual["Animal Type"] == animal_filter]
    st.subheader("Provincial livestock production trends by animal type (2010–2025)")
    fig = px.line(annual, x="Year", y="Production", color="Animal Type", markers=True)
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(title="Summed provincial production", tickformat=",.0f")
    st.plotly_chart(chart_config(fig, 430), use_container_width=True)

    st.subheader("Distribution of province-level production by animal type (2025)")
    province_2025 = data[data["Year"] == 2025]
    if province_2025.empty:
        st.info("No 2025 observations available.")
    else:
        fig = px.box(province_2025, x="Animal Type", y="Production", points=False, category_orders={"Animal Type": backend.ANIMALS})
        fig.update_yaxes(title="Production volume (metric tons)")
        st.plotly_chart(chart_config(fig, 430), use_container_width=True)

        top_hog = province_2025[province_2025["Animal Type"] == "Hog"].nlargest(10, "Production").sort_values("Production")
        st.subheader("Top 10 provinces by hog production (2025)")
        if top_hog.empty:
            st.info("No hog data for 2025.")
        else:
            fig = px.bar(top_hog, x="Production", y="Geolocation", orientation="h", text_auto=".3s")
            fig.update_xaxes(title="Production volume", tickformat=",.0f")
            st.plotly_chart(chart_config(fig, 420), use_container_width=True)
            download_csv(top_hog, "top_10_hog_producers_2025.csv", "Download top 10 hog producers")

    st.subheader("Production vs. year and Pearson correlation")
    scatter_data = data.copy()
    if animal_filter != "All animal types":
        scatter_data = scatter_data[scatter_data["Animal Type"] == animal_filter]
    if province_filter != "All provinces":
        scatter_data = scatter_data[scatter_data["Geolocation"] == province_filter]
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(scatter_data, x="Year", y="Production", color="Animal Type", opacity=.6)
        fig.update_yaxes(title="Production (metric tons)")
        st.plotly_chart(chart_config(fig, 440), use_container_width=True)
    with c2:
        model_df = data.copy()
        model_df["Class"] = (model_df["Production"] >= model_df["Production"].median()).astype(int)
        animal_encoder = LabelEncoder()
        province_encoder = LabelEncoder()
        model_df["Animal_Encoded"] = animal_encoder.fit_transform(model_df["Animal Type"])
        model_df["Province_Encoded"] = province_encoder.fit_transform(model_df["Geolocation"])
        corr = model_df[["Year", "Animal_Encoded", "Province_Encoded", "Production", "Class"]].corr(method="pearson")
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(chart_config(fig, 440), use_container_width=True)

    st.subheader("Total production and percentage share")
    st.dataframe(totals, use_container_width=True, hide_index=True)
    download_csv(totals, "total_production_share_by_animal.csv", "Download production share table")

# -----------------------------------------------------------------------------
# Model evaluation: same backend implementation as the Colab workflow
# -----------------------------------------------------------------------------

elif page == "Model Evaluation":
    st.title("Model evaluation")
    st.markdown("<p class='ll-subtitle'>Four classifiers with validation-based conditional SMOTENC augmentation and untouched test-set evaluation.</p>", unsafe_allow_html=True)
    st.info(
        "Target: High Production when the provincial observation is at or above the cleaned-data median; otherwise Low Production. "
        "If validation accuracy is below 80%, SMOTENC attempts to augment training rows only. The augmented model is kept only if validation accuracy improves."
    )
    if not backend.MODEL_STATE.get("ready"):
        st.warning("The models have not been trained in this app session yet.")
        if st.button("Train four models", type="primary", use_container_width=True):
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
        a, b, c, d = st.columns(4)
        a.metric("Median class threshold", f"{result['median_threshold']:,.2f}")
        b.metric("Training rows", f"{result['train_count']:,}")
        c.metric("Validation rows", f"{result['validation_count']:,}")
        d.metric("Untouched test rows", f"{result['test_count']:,}")
        st.success(f"Best test-set accuracy in this run: {result['best_model']}")

        metrics = pd.DataFrame(result["metrics"])
        display_metrics = metrics.rename(columns={
            "model": "Model", "accuracy": "Accuracy", "precision": "Precision",
            "recall": "Recall", "f1": "F1-Score", "roc_auc": "ROC-AUC",
            "validation_before": "Validation Accuracy Before", "validation_after": "Validation Accuracy After",
            "augmentation_triggered": "Augmentation Triggered", "synthetic_records_created": "Synthetic Rows Created",
            "augmentation_selected": "Augmented Model Selected", "training_records_used": "Final Training Rows Used",
        })
        st.subheader("Model performance comparison on the untouched test set")
        st.dataframe(display_metrics.style.format({
            "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1-Score": "{:.2%}",
            "ROC-AUC": "{:.4f}", "Validation Accuracy Before": "{:.2%}",
            "Validation Accuracy After": lambda v: "N/A" if pd.isna(v) else f"{v:.2%}",
        }), use_container_width=True, hide_index=True)
        download_csv(display_metrics, "livestock_model_comparison.csv", "Download model comparison CSV")

        score_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        long_metrics = display_metrics.melt(id_vars="Model", value_vars=score_cols, var_name="Metric", value_name="Score")
        fig = px.bar(long_metrics, x="Model", y="Score", color="Metric", barmode="group", text_auto=".2f")
        fig.update_yaxes(range=[0, 1.12], title="Score")
        st.plotly_chart(chart_config(fig, 460), use_container_width=True)

        st.subheader("Conditional SMOTENC augmentation audit")
        audit = pd.DataFrame(result["augmentation_log"])
        audit = audit.rename(columns={
            "model": "Model", "validation_before": "Validation Accuracy Before", "threshold": "80% Threshold",
            "triggered": "Augmentation Triggered", "training_rows_before": "Training Rows Before",
            "training_rows_after_attempt": "Training Rows After Attempt", "synthetic_rows_created": "Synthetic Rows Created",
            "validation_after": "Validation Accuracy After", "augmented_model_selected": "Augmented Model Selected",
            "final_training_rows_used": "Final Training Rows Used", "decision": "Decision",
        })
        st.dataframe(audit, use_container_width=True, hide_index=True)
        download_csv(audit, "conditional_augmentation_record_log.csv", "Download augmentation audit")
        st.caption("Synthetic rows are generated only from training data. They are not official livestock observations. The final 2026 refit uses real observations only.")

        st.subheader("Confusion matrices")
        cm_cols = st.columns(2)
        for idx, (model_name, matrix) in enumerate(result["confusion"].items()):
            fig = px.imshow(
                np.array(matrix), text_auto=True, x=["Predicted Low", "Predicted High"],
                y=["Actual Low", "Actual High"], color_continuous_scale="Blues", aspect="auto",
            )
            fig.update_layout(title=model_name, coloraxis_showscale=False)
            with cm_cols[idx % 2]:
                st.plotly_chart(chart_config(fig, 330), use_container_width=True)

        st.subheader("ROC curves")
        roc_fig = go.Figure()
        for model_name, curve in result["roc"].items():
            points = curve["points"]
            roc_fig.add_trace(go.Scatter(
                x=[p["x"] for p in points], y=[p["y"] for p in points],
                mode="lines", name=f"{model_name} (AUC={curve['auc']:.4f})",
            ))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="No-skill baseline"))
        roc_fig.update_xaxes(title="False Positive Rate", range=[0, 1])
        roc_fig.update_yaxes(title="True Positive Rate", range=[0, 1.05])
        st.plotly_chart(chart_config(roc_fig, 470), use_container_width=True)
        st.caption("ROC-AUC values closer to 1.00 indicate better separation of High and Low production classes; 0.50 is near chance-level ranking.")

# -----------------------------------------------------------------------------
# Predictions: single observation and full 2026 table
# -----------------------------------------------------------------------------

elif page == "Predictions":
    st.title("2026 production-category predictions")
    st.markdown("<p class='ll-subtitle'>Predict High or Low production category for an animal type, province and year.</p>", unsafe_allow_html=True)
    st.warning("These are High/Low category estimates, not exact production-volume forecasts or official PSA forecasts.")
    if not backend.MODEL_STATE.get("ready"):
        st.info("Train the four models on the Model Evaluation page before generating predictions.")
        if st.button("Train the four models now", type="primary"):
            try:
                with st.spinner("Training the four classification models..."):
                    backend.train_models()
                st.success("Model training completed. Open Model Evaluation to inspect the scores.")
                st.rerun()
            except Exception as exc:
                st.error(f"Model training failed: {exc}")
    else:
        c1, c2, c3 = st.columns(3)
        animal = c1.selectbox("Animal type", animals)
        province = c2.selectbox("Province", provinces)
        pred_year = c3.number_input("Prediction year", min_value=2026, max_value=2050, value=2026, step=1)
        if st.button("Generate prediction", type="primary"):
            try:
                animal_enc = int(backend.MODEL_STATE["animal_encoder"].transform([animal])[0])
                province_enc = int(backend.MODEL_STATE["province_encoder"].transform([province])[0])
                row = pd.DataFrame([[int(pred_year), animal_enc, province_enc]], columns=backend.MODEL_STATE["feature_columns"])
                outputs = []
                for model_name, model in backend.MODEL_STATE["final_models"].items():
                    label = int(model.predict(row)[0])
                    p_high = float(model.predict_proba(row)[0][1])
                    confidence = p_high if label == 1 else 1 - p_high
                    outputs.append({
                        "Model": model_name,
                        "Predicted Category": "High Production" if label == 1 else "Low Production",
                        "Probability High": p_high,
                        "Confidence": confidence,
                    })
                st.session_state["single_prediction"] = {
                    "animal": animal, "province": province, "year": int(pred_year), "outputs": outputs,
                    "best": backend.MODEL_STATE["best_model"],
                }
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
        saved = st.session_state.get("single_prediction")
        if saved:
            st.subheader(f"Prediction for {saved['animal']} in {saved['province']} ({saved['year']})")
            frame = pd.DataFrame(saved["outputs"])
            best = frame[frame["Model"] == saved["best"]].iloc[0]
            p1, p2, p3 = st.columns(3)
            p1.metric("Best model", saved["best"])
            p2.metric("Predicted category", best["Predicted Category"])
            p3.metric("Confidence", f"{best['Confidence']:.2%}")
            st.dataframe(frame.style.format({"Probability High": "{:.2%}", "Confidence": "{:.2%}"}), use_container_width=True, hide_index=True)
            download_csv(frame, "livestock_single_prediction.csv", "Download this prediction")

        st.markdown("---")
        st.subheader("Bulk 2026 predictions for provinces with valid 2025 observations")
        if st.button("Generate full 2026 prediction table"):
            try:
                with st.spinner("Generating 2026 category predictions for all available 2025 province-animal records..."):
                    st.session_state["future_predictions"] = prepare_future_predictions(data)
            except Exception as exc:
                st.error(str(exc))
        future = st.session_state.get("future_predictions")
        if future is not None:
            st.dataframe(future.head(100), use_container_width=True, hide_index=True)
            st.caption(f"Showing first 100 of {len(future):,} province-animal records.")
            download_csv(future, "predicted_2026_high_low_by_province_animal.csv", "Download complete 2026 predictions")

# -----------------------------------------------------------------------------
# Data records
# -----------------------------------------------------------------------------

elif page == "Data Records":
    st.title("Provincial data records")
    st.markdown("<p class='ll-subtitle'>Search and filter the cleaned province-year observations.</p>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    af = c1.selectbox("Animal type", ["All"] + animals)
    pf = c2.selectbox("Province", ["All"] + provinces)
    yf = c3.selectbox("Year", ["All"] + years)
    query = st.text_input("Search animal, province or year")
    page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
    view = data.copy()
    if af != "All": view = view[view["Animal Type"] == af]
    if pf != "All": view = view[view["Geolocation"] == pf]
    if yf != "All": view = view[view["Year"] == int(yf)]
    if query:
        mask = (
            view["Animal Type"].str.contains(query, case=False, na=False)
            | view["Geolocation"].str.contains(query, case=False, na=False)
            | view["Year"].astype(str).str.contains(query, case=False, na=False)
        )
        view = view[mask]
    view = view.sort_values(["Year", "Geolocation", "Animal Type"], ascending=[False, True, True]).reset_index(drop=True)
    total_pages = max(1, int(np.ceil(len(view) / page_size)))
    page_no = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start = (int(page_no) - 1) * page_size
    st.caption(f"{len(view):,} matching records · Page {page_no} of {total_pages}")
    st.dataframe(view.iloc[start:start + page_size], use_container_width=True, hide_index=True)
    download_csv(view, "filtered_livestock_records.csv", "Download filtered records")

# -----------------------------------------------------------------------------
# Data source and methodology
# -----------------------------------------------------------------------------

elif page == "Data Source":
    st.title("Data source and methodology")
    st.markdown("<p class='ll-subtitle'>Dataset provenance, cleaning rules and model methodology used by the application.</p>", unsafe_allow_html=True)
    st.success(f"Dataset loaded: {len(data):,} cleaned province-year observations from {backend.STATE.get('source') or 'source dataset'}.")
    st.write("**Source repository:** [Livestock Annual Volume of Production by Animal Type and Province, 2010–2025](https://github.com/Tasi-736/Livestock-Annual-Volume-of-Production-by-Animal-Type-and-Province-for-2010-2025)")
    st.write("**Unit:** Metric tons")
    st.markdown("### Data preparation rules")
    st.markdown(
        "1. Read the ARFF file's comma-separated rows after `@data`, or a compatible CSV.\n"
        "2. Convert 2010–2025 production values to numeric; missing markers such as `..` remain missing.\n"
        "3. Retain Carabao, Cattle, Goat and Hog.\n"
        "4. Retain province-level rows identified by exactly four leading dots in the source `Geolocation` field.\n"
        "5. Reshape annual columns to one row per province, animal type and year; remove missing production observations."
    )
    st.markdown("### Classification and prediction methodology")
    st.markdown(
        "- Target: High Production when the value is at or above the median; otherwise Low Production.\n"
        "- Features: Year, encoded Animal Type and encoded Geolocation.\n"
        "- Split: 80% development / 20% final test, then 80% training / 20% validation within development (64% / 16% / 20% overall).\n"
        "- Models: Logistic Regression, Decision Tree (max depth 5), Random Forest (100 trees, max depth 6), Gradient Boosting (100 estimators, learning rate 0.1, max depth 4).\n"
        "- Conditional SMOTENC: attempted only when validation accuracy is below 80%; keep the augmented model only if validation accuracy improves.\n"
        "- Test set is untouched for final evaluation; future-model refits use real observations only."
    )
    st.caption("Synthetic records are training-only samples, not official livestock observations. Future predictions are High/Low category estimates, not exact volume forecasts or official PSA forecasts.")
    st.markdown("### Upload a dataset")
    file = st.file_uploader("Select a ZIP, ARFF or CSV dataset", type=["zip", "arff", "csv"], key="page_dataset_upload")
    if file is not None and st.button("Load dataset now", type="primary"):
        try:
            handle_dataset_upload(file)
            st.success(f"Successfully loaded {len(backend.STATE['df']):,} observations. Model results were reset and must be retrained.")
            st.rerun()
        except Exception as exc:
            st.error(f"Dataset upload failed: {exc}")

st.sidebar.markdown("---")
st.sidebar.caption("LivestockLens · Philippine livestock production analytics")
st.sidebar.caption("Classification categories are analytical estimates, not official production forecasts.")
