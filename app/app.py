# app/app.py

```python
import streamlit as st
import pandas as pd
import joblib
import mlflow
import plotly.express as px

from src.optimize import (
    optimize_budget,
    calculate_marginal_roi
)

from src.train import load_data
from src.features import engineer_features


# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(
    page_title="AdSpend ROI Forecaster Engine",
    page_icon="📈",
    layout="wide"
)


# -----------------------------------
# Load Models + Metadata
# -----------------------------------
metadata = joblib.load(
    "models/metadata.pkl"
)

ridge_model = joblib.load(
    "models/ridge_pipeline.pkl"
)

xgb_model = joblib.load(
    "models/xgb_pipeline.pkl"
)


# -----------------------------------
# Load Dataset
# -----------------------------------
df = load_data(
    "data/marketing_data.csv"
)

feature_df = engineer_features(df.copy())


# -----------------------------------
# Sidebar
# -----------------------------------
st.sidebar.title("⚙️ Control Panel")

budget = st.sidebar.slider(
    "Marketing Budget",
    min_value=1000,
    max_value=50000,
    value=10000,
    step=500
)

show_raw_data = st.sidebar.checkbox(
    "Show Raw Dataset"
)


# -----------------------------------
# Header
# -----------------------------------
st.title("📈 AdSpend ROI Forecaster Engine")

st.markdown(
    """
### AI-Powered Marketing Mix Optimization Platform

This platform demonstrates:
- Budget Optimization using Scipy
- Explainable AI with SHAP
- Time-Series Aware ML Validation
- Marketing Mix Modeling Concepts
- MLflow Experiment Tracking
"""
)


# -----------------------------------
# KPI Metrics
# -----------------------------------
latest_sales = round(df["Sales"].iloc[-1], 2)

avg_sales = round(df["Sales"].mean(), 2)

avg_tv = round(df["TV"].mean(), 2)

avg_social = round(df["Social"].mean(), 2)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Latest Sales",
        latest_sales
    )

with col2:
    st.metric(
        "Average Sales",
        avg_sales
    )

with col3:
    st.metric(
        "Average TV Spend",
        avg_tv
    )

with col4:
    st.metric(
        "Average Social Spend",
        avg_social
    )


# -----------------------------------
# Sales Trend Chart
# -----------------------------------
st.subheader("📊 Historical Sales Trend")

sales_fig = px.line(
    df,
    x="Date",
    y="Sales",
    title="Sales Over Time"
)

st.plotly_chart(
    sales_fig,
    use_container_width=True
)


# -----------------------------------
# Channel Spend Analysis
# -----------------------------------
st.subheader("📢 Advertising Channel Spend")

spend_df = df[["TV", "Social", "Newspaper"]].mean().reset_index()
spend_df.columns = ["Channel", "Average Spend"]

channel_fig = px.bar(
    spend_df,
    x="Channel",
    y="Average Spend",
    title="Average Spend by Channel"
)

st.plotly_chart(
    channel_fig,
    use_container_width=True
)


# -----------------------------------
# Budget Optimizer Section
# -----------------------------------
st.subheader("💰 Budget Allocation Optimizer")

allocation = optimize_budget(budget)

marginal_roi = calculate_marginal_roi(
    allocation
)

opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)

with opt_col1:
    st.metric(
        "TV Allocation",
        f"₹{allocation['TV']}"
    )

with opt_col2:
    st.metric(
        "Social Allocation",
        f"₹{allocation['Social']}"
    )

with opt_col3:
    st.metric(
        "Newspaper Allocation",
        f"₹{allocation['Newspaper']}"
    )

with opt_col4:
    st.metric(
        "Predicted Sales",
        allocation['Predicted_Sales']
    )


# -----------------------------------
# Marginal ROI Display
# -----------------------------------
st.info(
    f"💡 Marginal ROI: {marginal_roi}"
)


# -----------------------------------
# Allocation Pie Chart
# -----------------------------------
allocation_df = pd.DataFrame({
    "Channel": [
        "TV",
        "Social",
        "Newspaper"
    ],
    "Spend": [
        allocation["TV"],
        allocation["Social"],
        allocation["Newspaper"]
    ]
})

pie_fig = px.pie(
    allocation_df,
    names="Channel",
    values="Spend",
    title="Optimal Budget Allocation"
)

st.plotly_chart(
    pie_fig,
    use_container_width=True
)


# -----------------------------------
# SHAP Explainability Images
# -----------------------------------
st.subheader("🧠 Explainable AI Insights")

exp_col1, exp_col2 = st.columns(2)

with exp_col1:
    st.image(
        "reports/shap_summary.png",
        caption="SHAP Feature Importance"
    )

with exp_col2:
    st.image(
        "reports/shap_waterfall.png",
        caption="SHAP Waterfall Explanation"
    )


# -----------------------------------
# MLflow Information
# -----------------------------------
st.subheader("🧪 ML Experiment Tracking")

st.markdown(
    """
This project uses MLflow for:
- Experiment Tracking
- Metric Logging
- Model Versioning
- Reproducibility
"""
)


# -----------------------------------
# Raw Data Section
# -----------------------------------
if show_raw_data:

    st.subheader("🗂️ Raw Dataset Preview")

    st.dataframe(
        df.head(50),
        use_container_width=True
    )


# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")

st.caption(
    "Built with Streamlit, Scikit-learn, MLflow, SHAP, Plotly & Scipy"
)

