# src/train.py

import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from src.features import engineer_features

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from xgboost import XGBRegressor


# -------------------------------
# 1. Load & Prepare Data
# -------------------------------
def load_data(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")

    return df


def feature_target_split(df):

    df = engineer_features(df)

    feature_cols = [
        "YouTube",
        "Instagram",
        "Twitter",

        "YouTube_Adstock",
        "Instagram_Adstock",
        "Twitter_Adstock",

        "YouTube_Instagram_Interaction",

        "Month",
        "Time_Index"
    ]

    X = df[feature_cols]
    y = df["Engagement_Score"]

    return X, y, feature_cols


# -------------------------------
# 2. Build Pipelines
# -------------------------------
def build_xgb_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.03,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
        ))
    ])


def build_ridge_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])


# -------------------------------
# 3. Cross Validation
# -------------------------------
def evaluate_model(pipeline, X, y):

    tscv = TimeSeriesSplit(n_splits=5)

    r2_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=tscv,
        scoring="r2"
    )

    mae_scores = -cross_val_score(
        pipeline,
        X,
        y,
        cv=tscv,
        scoring="neg_mean_absolute_error"
    )

    return {
        "r2_mean": np.mean(r2_scores),
        "r2_std": np.std(r2_scores),
        "mae_mean": np.mean(mae_scores)
    }


# -------------------------------
# 4. Train Final Models
# -------------------------------
def train_final_models(X, y):

    xgb_pipeline = build_xgb_pipeline()
    ridge_pipeline = build_ridge_pipeline()

    xgb_pipeline.fit(X, y)
    ridge_pipeline.fit(X, y)

    return xgb_pipeline, ridge_pipeline


# -------------------------------
# 5. Residual Analysis
# -------------------------------
def residual_analysis(model, X, y):

    preds = model.predict(X)
    residuals = y - preds

    return {
        "residuals": residuals,
        "predictions": preds
    }


# -------------------------------
# 6. Save Artifacts
# -------------------------------
def save_artifacts(xgb_model, ridge_model):

    os.makedirs("models", exist_ok=True)

    joblib.dump(xgb_model, "models/xgb_pipeline.pkl")
    joblib.dump(ridge_model, "models/ridge_pipeline.pkl")


# -------------------------------
# 7. Save Metadata
# -------------------------------
def save_metadata(feature_cols):

    metadata = {
        "feature_names": feature_cols
    }

    joblib.dump(metadata, "models/metadata.pkl")


# -------------------------------
# 8. Main Training Pipeline
# -------------------------------
def run_training(data_path):

    df = load_data(data_path)

    X, y, feature_cols = feature_target_split(df)

    with mlflow.start_run():

        print("📊 Evaluating XGBoost...")
        xgb_metrics = evaluate_model(
            build_xgb_pipeline(),
            X,
            y
        )

        print("📊 Evaluating Ridge...")
        ridge_metrics = evaluate_model(
            build_ridge_pipeline(),
            X,
            y
        )

        # -----------------------
        # Log Metrics
        # -----------------------
        mlflow.log_metric(
            "xgb_r2_mean",
            xgb_metrics["r2_mean"]
        )

        mlflow.log_metric(
            "xgb_mae_mean",
            xgb_metrics["mae_mean"]
        )

        mlflow.log_metric(
            "ridge_r2_mean",
            ridge_metrics["r2_mean"]
        )

        mlflow.log_metric(
            "ridge_mae_mean",
            ridge_metrics["mae_mean"]
        )

        print("\n--- Cross Validation Results ---")
        print("XGB:", xgb_metrics)
        print("Ridge:", ridge_metrics)

        print("\n🚀 Training final models...")

        xgb_model, ridge_model = train_final_models(X, y)

        print("📉 Running residual analysis...")

        residuals_info = residual_analysis(
            xgb_model,
            X,
            y
        )

        # -----------------------
        # Save Models
        # -----------------------
        save_artifacts(
            xgb_model,
            ridge_model
        )

        save_metadata(feature_cols)

        # -----------------------
        # Log Models
        # -----------------------
        mlflow.sklearn.log_model(
            xgb_model,
            "xgb_model"
        )

        mlflow.sklearn.log_model(
            ridge_model,
            "ridge_model"
        )

        print("✅ Models saved to /models")

    return {
        "xgb_metrics": xgb_metrics,
        "ridge_metrics": ridge_metrics,
        "residuals": residuals_info
    }


if __name__ == "__main__":

    results = run_training(
        "data/marketing_data.csv"
    )