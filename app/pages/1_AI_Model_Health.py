# app/pages/1_Model_Health.py
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))
from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

from src.train import load_data
from src.features import engineer_features
from src.generate_data import generate_data


# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(
    page_title="AI Model Health",
    page_icon="🩺",
    layout="wide"
)


# -----------------------------------
# Load Models + Metadata
# -----------------------------------
ridge_pipeline = joblib.load(
    "models/ridge_pipeline.pkl"
)

metadata = joblib.load(
    "models/metadata.pkl"
)

feature_cols = metadata[
    "feature_names"
]


# -----------------------------------
# Header
# -----------------------------------
st.title("🩺 Model Health & Drift Monitoring")

st.markdown(
    """
This dashboard demonstrates:
- Concept Drift Simulation
- Model Stability Monitoring
- Residual Analysis
- Data Quality Validation
- Time-Series ML Evaluation
"""
)


# -----------------------------------
# Drift Simulation Slider
# -----------------------------------
st.sidebar.header("⚠️ Drift Simulation")

social_effect_strength = st.sidebar.slider(
    "Instagram Engagement Influence",
    min_value=0.10,
    max_value=0.80,
    value=0.55,
    step=0.05
)


# -----------------------------------
# Generate Drifted Dataset
# -----------------------------------
df = generate_data(
    rows=5000,
    social_effect=social_effect_strength
)

feature_df = engineer_features(
    df.copy()
)

X = feature_df[feature_cols]
y = feature_df["Engagement_Score"]


# -----------------------------------
# Model Prediction
# -----------------------------------
preds = ridge_pipeline.predict(X)


# -----------------------------------
# Evaluation Metrics
# -----------------------------------
r2 = r2_score(y, preds)
mae = mean_absolute_error(y, preds)

residuals = y - preds


# -----------------------------------
# KPI Cards
# -----------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "R² Score",
        round(r2, 4)
    )

with col2:
    st.metric(
        "MAE",
        round(mae, 4)
    )

with col3:
    drift_status = (
        "⚠️ Drift Detected"
        if r2 < 0.70
        else "✅ Audience Dynamics Stable"
    )

    st.metric(
        "Model Status",
        drift_status
    )


# -----------------------------------
# Predicted vs Actual
# -----------------------------------
st.subheader("📈 Predicted vs Actual Engagement")

comparison_df = pd.DataFrame({
    "Actual": y,
    "Predicted": preds
})

comparison_fig = px.scatter(
    comparison_df.sample(1000),
    x="Actual",
    y="Predicted",
    title="Predicted vs Actual"
)

st.plotly_chart(
    comparison_fig,
    use_container_width=True
)


# -----------------------------------
# Residual Distribution
# -----------------------------------
st.subheader("📉 Residual Distribution")

residual_fig = px.histogram(
    residuals,
    nbins=50,
    title="Residual Error Distribution"
)

st.plotly_chart(
    residual_fig,
    use_container_width=True
)


# -----------------------------------
# Residual Trend Over Time
# -----------------------------------
st.subheader("⏳ Residual Trend Over Time")

residual_df = pd.DataFrame({
    "Date": df["Date"],
    "Residuals": residuals
})

trend_fig = px.line(
    residual_df,
    x="Date",
    y="Residuals",
    title="Residual Drift Over Time"
)

st.plotly_chart(
    trend_fig,
    use_container_width=True
)


# -----------------------------------
# Data Quality Validation
# -----------------------------------
st.subheader("🧪 Data Quality Checks")

sample_yt = st.number_input(
    "Enter YouTube Spend",
    min_value=0.0,
    value=100.0
)

sample_insta = st.number_input(
    "Enter Instagram Spend",
    min_value=0.0,
    value=120.0
)

sample_twitter = st.number_input(
    "Enter Twitter Spend",
    min_value=0.0,
    value=30.0
)


# Threshold Checks
# -----------------------------------
outlier_detected = False

if sample_yt > df["YouTube"].quantile(0.99):
    outlier_detected = True

if sample_insta > df["Instagram"].quantile(0.99):
    outlier_detected = True

if sample_twitter > df["Twitter"].quantile(0.99):
    outlier_detected = True


if outlier_detected:

    st.warning(
        "⚠️ Input appears outside training distribution. Predictions may be unreliable."
    )

else:

    st.success(
        "✅ Input is within expected training distribution."
    )


# -----------------------------------
# Rolling Error Analysis
# -----------------------------------
st.subheader("📊 Rolling Error Analysis")

rolling_mae = (
    pd.Series(abs(residuals))
    .rolling(window=100)
    .mean()
)

rolling_df = pd.DataFrame({
    "Index": np.arange(len(rolling_mae)),
    "Rolling_MAE": rolling_mae
})

rolling_fig = px.line(
    rolling_df,
    x="Index",
    y="Rolling_MAE",
    title="Rolling MAE"
)

st.plotly_chart(
    rolling_fig,
    use_container_width=True
)


# -----------------------------------
# Feature Drift Analysis
# -----------------------------------
st.subheader("🔍 Feature Drift Analysis")

feature_choice = st.selectbox(
    "Select Feature",
    ["YouTube", "Instagram", "Twitter"]
)

hist_fig = go.Figure()

hist_fig.add_trace(
    go.Histogram(
        x=df[feature_choice],
        name="Current Distribution",
        opacity=0.7
    )
)

hist_fig.update_layout(
    title=f"Distribution Analysis: {feature_choice}",
    barmode="overlay"
)

st.plotly_chart(
    hist_fig,
    use_container_width=True
)


# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")

st.caption(
    "InfluenceIQ Monitoring Suite • Powered by Streamlit, Plotly & Scikit-learn"
)