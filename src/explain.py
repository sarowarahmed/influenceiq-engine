# src/explain.py

import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

from src.train import load_data
from src.features import engineer_features


# -------------------------------
# Load Model + Metadata
# -------------------------------
xgb_pipeline = joblib.load(
    "models/xgb_pipeline.pkl"
)

metadata = joblib.load(
    "models/metadata.pkl"
)

feature_cols = metadata["feature_names"]


# -------------------------------
# Load & Prepare Dataset
# -------------------------------
df = load_data(
    "data/marketing_data.csv"
)

df = engineer_features(df)

X = df[feature_cols]


# -------------------------------
# Extract XGBoost Model
# -------------------------------
xgb_model = xgb_pipeline.named_steps[
    "model"
]


# -------------------------------
# Transform Features
# -------------------------------
scaler = xgb_pipeline.named_steps[
    "scaler"
]

X_scaled = scaler.transform(X)


# -------------------------------
# SHAP Explainer
# -------------------------------
explainer = shap.TreeExplainer(
    xgb_model
)

shap_values = explainer.shap_values(
    X_scaled
)


# -------------------------------
# Create Output Directory
# -------------------------------
os.makedirs(
    "reports",
    exist_ok=True
)


# -------------------------------
# SHAP Summary Plot
# -------------------------------
plt.figure()

shap.summary_plot(
    shap_values,
    X,
    show=False
)

plt.savefig(
    "reports/shap_summary.png",
    bbox_inches="tight"
)

print(
    "✅ SHAP summary plot saved"
)


# -------------------------------
# SHAP Waterfall Plot
# -------------------------------
sample_index = 0

explanation = shap.Explanation(
    values=shap_values[sample_index],
    base_values=explainer.expected_value,
    data=X.iloc[sample_index],
    feature_names=feature_cols
)

plt.figure()

shap.plots.waterfall(
    explanation,
    show=False
)

plt.savefig(
    "reports/shap_waterfall.png",
    bbox_inches="tight"
)

print(
    "✅ SHAP waterfall plot saved"
)