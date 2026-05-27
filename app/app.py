# app/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

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
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------------
# Custom CSS
# -----------------------------------
st.markdown(
    """
<style>

.main {
    background-color: #0E1117;
}

.hero-box {
    padding: 2rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #111827, #1F2937);
    border: 1px solid #374151;
    margin-bottom: 1rem;
}

.metric-container {
    background-color: #161B22;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #30363D;
}

.small-text {
    color: #9CA3AF;
}

</style>
""",
    unsafe_allow_html=True
)


# -----------------------------------
# Sidebar Branding
# -----------------------------------
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2920/2920329.png",
    width=90
)

st.sidebar.title(
    "AdSpend Intelligence"
)

st.sidebar.markdown(
    """
### 🚀 AI Marketing Platform

Capabilities:
- Budget Optimization
- Explainable AI
- Drift Monitoring
- Retraining Pipelines
- Time-Series ML
"""
)




# -----------------------------------
# Load Models
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

feature_df = engineer_features(
    df.copy()
)


# -----------------------------------
# Sidebar Controls
# -----------------------------------
st.sidebar.header("⚙️ Scenario Simulator")

budget = st.sidebar.slider(
    "Total Marketing Budget",
    min_value=1000,
    max_value=50000,
    value=10000,
    step=500
)

market_condition = st.sidebar.selectbox(
    "Market Scenario",
    [
        "Stable Market",
        "Holiday Surge",
        "Social Media Boom",
        "Economic Slowdown"
    ]
)

competitor_mode = st.sidebar.checkbox(
    "Enable Competitor Benchmarking"
)

show_raw_data = st.sidebar.checkbox(
    "Show Raw Dataset"
)


# -----------------------------------
# Scenario Simulation
# -----------------------------------
scenario_multiplier = 1.0

if market_condition == "Holiday Surge":
    scenario_multiplier = 1.15

elif market_condition == "Social Media Boom":
    scenario_multiplier = 1.25

elif market_condition == "Economic Slowdown":
    scenario_multiplier = 0.85

# -----------------------------------
# Competitor Benchmark Logic
# -----------------------------------
if competitor_mode:

    benchmark_multiplier = 0.9

else:

    benchmark_multiplier = 1.0

# -----------------------------------
# Hero Section
# -----------------------------------
st.markdown(
    f"""
<div class="hero-box">

# 🚀 AdSpend ROI Forecaster Engine

### AI-Powered Marketing Optimization Platform

<p class="small-text">
Built using:

• Scipy Optimization
• SHAP Explainability
• Time-Series Validation
• Drift Monitoring
• MLflow Tracking
• Retraining Workflows
</p>

</div>
""",
    unsafe_allow_html=True
)


# -----------------------------------
# KPI Metrics
# -----------------------------------
latest_sales = round(
    df["Sales"].iloc[-1]
    * scenario_multiplier,
    2
)

avg_sales = round(
    df["Sales"].mean(),
    2
)

avg_tv = round(
    df["TV"].mean(),
    2
)

avg_social = round(
    df["Social"].mean(),
    2
)

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "📈 Latest Sales",
    latest_sales
)

metric2.metric(
    "💰 Avg Sales",
    avg_sales
)

metric3.metric(
    "📺 Avg TV Spend",
    avg_tv
)

metric4.metric(
    "📱 Avg Social Spend",
    avg_social
)


# -----------------------------------
# Tabs Layout
# -----------------------------------
analytics_tab, optimizer_tab, explain_tab = st.tabs([
    "📊 Analytics",
    "💰 Optimizer",
    "🧠 Explainability"
])


# ===================================
# Analytics Tab
# ===================================
with analytics_tab:

    st.subheader("📈 Historical Sales Trend")

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


    st.subheader("📢 Advertising Channel Spend")

    spend_df = df[[
        "TV",
        "Social",
        "Newspaper"
    ]].mean().reset_index()

    spend_df.columns = [
        "Channel",
        "Average Spend"
    ]

    spend_fig = px.bar(
        spend_df,
        x="Channel",
        y="Average Spend",
        title="Average Channel Spend"
    )

    st.plotly_chart(
        spend_fig,
        use_container_width=True
    )


# ===================================
# Optimizer Tab
# ===================================
with optimizer_tab:

    st.subheader("💰 Budget Allocation Optimizer")

    allocation = optimize_budget(
        budget
    )

    marginal_roi = calculate_marginal_roi(
        allocation
    )

    allocation["Predicted_Sales"] *= benchmark_multiplier

    opt1, opt2, opt3, opt4 = st.columns(4)

    opt1.metric(
        "📺 TV",
        f"₹{allocation['TV']}"
    )

    opt2.metric(
        "📱 Social",
        f"₹{allocation['Social']}"
    )

    opt3.metric(
        "📰 Newspaper",
        f"₹{allocation['Newspaper']}"
    )

    opt4.metric(
        "📈 Predicted Sales",
        allocation['Predicted_Sales']
    )


    st.success(
        f"💡 Marginal ROI: {marginal_roi}"
    )


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


# ===================================
# Explainability Tab
# ===================================
with explain_tab:

    st.subheader("🧠 Explainable AI")

    exp1, exp2 = st.columns(2)

    with exp1:
        st.image(
            "reports/shap_summary.png",
            caption="SHAP Feature Importance"
        )

    with exp2:
        st.image(
            "reports/shap_waterfall.png",
            caption="SHAP Waterfall Analysis"
        )


    st.info(
        "SHAP values explain how each feature contributes to the final prediction."
    )


# -----------------------------------
# Raw Dataset
# -----------------------------------
if show_raw_data:

    st.subheader("🗂️ Dataset Preview")

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