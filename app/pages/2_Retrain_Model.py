# app/pages/2_Retrain_Model.py

import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import joblib

from sklearn.metrics import (
    r2_score,
    mean_absolute_error
)

from src.train import (
    load_data,
    feature_target_split,
    train_final_models
)

from src.features import engineer_features


# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(
    page_title="Retraining Studio",
    page_icon="🔄",
    layout="wide"
)


# -----------------------------------
# Header
# -----------------------------------
st.title("🔄 Model Retraining System")

st.markdown(
    """
This page demonstrates:
- Uploading New Data
- Schema Validation
- Automated Retraining
- Model Comparison
- MLOps Workflow Concepts
"""
)


# -----------------------------------
# Load Existing Models
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
# Load Existing Dataset
# -----------------------------------
base_df = load_data(
    "data/marketing_data.csv"
)


# -----------------------------------
# File Upload
# -----------------------------------
st.subheader("📤 Upload New Marketing Data")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)


# -----------------------------------
# Schema Requirements
# -----------------------------------
required_columns = [
    "Date",
    "YouTube",
    "Instagram",
    "Twitter",
    "Engagement_Score"
]


# -----------------------------------
# Existing Model Metrics
# -----------------------------------
st.subheader("📊 Current Production Model Metrics")

base_feature_df = engineer_features(
    base_df.copy()
)

X_base = base_feature_df[feature_cols]
y_base = base_feature_df["Engagement_Score"]

base_preds = ridge_pipeline.predict(X_base)

base_r2 = r2_score(
    y_base,
    base_preds
)

base_mae = mean_absolute_error(
    y_base,
    base_preds
)

metric_col1, metric_col2 = st.columns(2)

with metric_col1:
    st.metric(
        "Current R²",
        round(base_r2, 4)
    )

with metric_col2:
    st.metric(
        "Current MAE",
        round(base_mae, 4)
    )


# -----------------------------------
# Process Uploaded File
# -----------------------------------
if uploaded_file is not None:

    st.subheader("🧪 Uploaded Data Validation")

    try:

        new_df = pd.read_csv(uploaded_file)

        # ---------------------------
        # Column Validation
        # ---------------------------
        missing_cols = [
            col for col in required_columns
            if col not in new_df.columns
        ]

        if len(missing_cols) > 0:

            st.error(
                f"❌ Missing Columns: {missing_cols}"
            )

        else:

            st.success(
                "✅ Schema Validation Passed"
            )

            # ---------------------------
            # Null Validation
            # ---------------------------
            null_count = new_df.isnull().sum().sum()

            if null_count > 0:

                st.warning(
                    f"⚠️ Dataset contains {null_count} null values"
                )

            else:

                st.success(
                    "✅ No Missing Values Detected"
                )


            # ---------------------------
            # Preview Uploaded Data
            # ---------------------------
            st.subheader("📄 Uploaded Data Preview")

            st.dataframe(
                new_df.head(20),
                use_container_width=True
            )


            # ---------------------------
            # Retraining Trigger
            # ---------------------------
            if st.button("🚀 Retrain Model"):

                with st.spinner("Training new models..."):

                    # -------------------
                    # Merge Data
                    # -------------------
                    combined_df = pd.concat(
                        [base_df, new_df],
                        ignore_index=True
                    )

                    combined_df["Date"] = pd.to_datetime(
                        combined_df["Date"]
                    )

                    combined_df = combined_df.sort_values(
                        "Date"
                    )


                    # -------------------
                    # Feature Engineering
                    # -------------------
                    feature_df = engineer_features(
                        combined_df.copy()
                    )

                    X_new = feature_df[feature_cols]
                    y_new = feature_df["Sales"]


                    # -------------------
                    # Train New Models
                    # -------------------
                    new_xgb_model, new_ridge_model = train_final_models(
                        X_new,
                        y_new
                    )


                    # -------------------
                    # Evaluate New Model
                    # -------------------
                    new_preds = new_ridge_model.predict(
                        X_new
                    )

                    new_r2 = r2_score(
                        y_new,
                        new_preds
                    )

                    new_mae = mean_absolute_error(
                        y_new,
                        new_preds
                    )


                    # -------------------
                    # Save Updated Models
                    # -------------------
                    os.makedirs(
                        "models",
                        exist_ok=True
                    )

                    joblib.dump(
                        new_ridge_model,
                        "models/ridge_pipeline.pkl"
                    )

                    joblib.dump(
                        new_xgb_model,
                        "models/xgb_pipeline.pkl"
                    )


                    # -------------------
                    # Retraining Success
                    # -------------------
                    st.success(
                        "✅ Model retraining completed successfully"
                    )


                    # -------------------
                    # Before vs After
                    # -------------------
                    st.subheader("📈 Model Performance Comparison")

                    compare_col1, compare_col2 = st.columns(2)

                    with compare_col1:
                        st.metric(
                            "Old R²",
                            round(base_r2, 4)
                        )

                        st.metric(
                            "Old MAE",
                            round(base_mae, 4)
                        )

                    with compare_col2:
                        st.metric(
                            "New R²",
                            round(new_r2, 4),
                            delta=round(new_r2 - base_r2, 4)
                        )

                        st.metric(
                            "New MAE",
                            round(new_mae, 4),
                            delta=round(base_mae - new_mae, 4)
                        )


                    # -------------------
                    # Drift Detection
                    # -------------------
                    if new_r2 < base_r2:

                        st.warning(
                            "⚠️ New data reduced model performance. Possible concept drift detected."
                        )

                    else:

                        st.success(
                            "✅ Model performance improved after retraining"
                        )


    except Exception as e:

        st.error(
            f"❌ Error processing file: {str(e)}"
        )


# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")

st.caption(
    "Retraining Pipeline powered by Streamlit, Scikit-learn & ML Engineering Best Practices"
)