# src/train.py

import pandas as pd
import numpy as np
import joblib
from src.features import engineer_features

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
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
        "TV",
        "Social",
        "Newspaper",

        "TV_Adstock",
        "Social_Adstock",
        "News_Adstock",

        "TV_Saturation",
        "Social_Saturation",
        "News_Saturation",

        "TV_Social_Interaction",

        "Month",
        "Quarter"
    ]

    X = df[feature_cols]
    y = df["Sales"]

    return X, y, feature_cols


# -------------------------------
# 2. Build Pipelines
# -------------------------------
def build_xgb_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
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
        pipeline, X, y,
        cv=tscv,
        scoring="r2"
    )

    mae_scores = -cross_val_score(
        pipeline, X, y,
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
    joblib.dump(xgb_model, "models/xgb_pipeline.pkl")
    joblib.dump(ridge_model, "models/ridge_pipeline.pkl")


# -------------------------------
# 7. Main Execution
# -------------------------------
def run_training(data_path):
    df = load_data(data_path)
    X, y, feature_cols = feature_target_split(df)

    print("📊 Evaluating XGBoost...")
    xgb_metrics = evaluate_model(build_xgb_pipeline(), X, y)

    print("📊 Evaluating Ridge...")
    ridge_metrics = evaluate_model(build_ridge_pipeline(), X, y)

    print("\n--- Cross Validation Results ---")
    print("XGB:", xgb_metrics)
    print("Ridge:", ridge_metrics)

    print("\n🚀 Training final models...")
    xgb_model, ridge_model = train_final_models(X, y)

    print("📉 Running residual analysis...")
    residuals_info = residual_analysis(xgb_model, X, y)

    save_artifacts(xgb_model, ridge_model)
    save_metadata(feature_cols)

    print("✅ Models saved to /models")

    def save_metadata(feature_cols):
        metadata = {
            "feature_names": feature_cols
        }

        joblib.dump(metadata, "models/metadata.pkl")

    return {
        "xgb_metrics": xgb_metrics,
        "ridge_metrics": ridge_metrics,
        "residuals": residuals_info
    }


if __name__ == "__main__":
    results = run_training("data/marketing_data.csv")