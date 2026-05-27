# src/optimize.py

import numpy as np
import joblib

from scipy.optimize import minimize

from src.features import engineer_features


# -------------------------------
# Load Models + Metadata
# -------------------------------
ridge_model = joblib.load(
    "models/ridge_pipeline.pkl"
)

metadata = joblib.load(
    "models/metadata.pkl"
)

feature_cols = metadata["feature_names"]


# -------------------------------
# Build Feature Row
# -------------------------------
def build_feature_row(youtube, instagram, twitter):

    import pandas as pd

    df = pd.DataFrame({
        "Date": [pd.Timestamp("2025-01-01")],
        "YouTube": [youtube],
        "Instagram": [instagram],
        "Twitter": [twitter]
    })

    df = engineer_features(df)
    df["Time_Index"] = 5000

    X = df[feature_cols]

    return X


# -------------------------------
# Objective Function
# -------------------------------
def objective(spends):

    youtube, instagram, twitter = spends

    X = build_feature_row(
        youtube,
        instagram,
        twitter
    )

    prediction = ridge_model.predict(X)[0]

    # NEGATIVE because scipy minimizes
    return -prediction


# -------------------------------
# Budget Constraint
# -------------------------------
def budget_constraint(spends, total_budget):

    return total_budget - np.sum(spends)


# -------------------------------
# Optimize Allocation
# -------------------------------
def optimize_budget(total_budget):

    # Initial Guess
    initial_guess = [
        total_budget / 3,
        total_budget / 3,
        total_budget / 3
    ]

    # Channel Bounds
    min_channel_spend = total_budget * 0.1

    bounds = [
        (
            min_channel_spend,
            total_budget * 0.6
        ),

        (
            min_channel_spend,
            total_budget * 0.6
        ),

        (
            min_channel_spend,
            total_budget * 0.6
        )
    ]

    # Constraint
    constraints = {
        "type": "ineq",
        "fun": budget_constraint,
        "args": (total_budget,)
    }

    result = minimize(
        objective,
        x0=initial_guess,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP"
    )

    optimal_youtube = result.x[0]
    optimal_instagram = result.x[1]
    optimal_twitter = result.x[2]

    # Final Prediction
    optimal_features = build_feature_row(
        optimal_youtube,
        optimal_instagram,
        optimal_twitter
    )

    predicted_sales = ridge_model.predict(
        optimal_features
    )[0]

    return {
        "YouTube": round(optimal_youtube, 2),
        "Instagram": round(optimal_instagram, 2),
        "Twitter": round(optimal_twitter, 2),
        "Predicted_Sales": round(predicted_sales, 2)
    }


# -------------------------------
# Marginal ROI Calculation
# -------------------------------
def calculate_marginal_roi(
    allocation,
    increment=100
):

    base_sales = allocation["Predicted_Sales"]

    updated_allocation = optimize_budget(
        allocation["YouTube"]
        + allocation["Instagram"]
        + allocation["Twitter"]
        + increment
    )

    new_sales = updated_allocation[
        "Predicted_Sales"
    ]

    marginal_roi = (
        new_sales - base_sales
    ) / increment

    return round(marginal_roi, 4)


# -------------------------------
# Manual Test
# -------------------------------
if __name__ == "__main__":

    budget = 10000

    allocation = optimize_budget(
        total_budget=budget
    )

    print("\n🎯 Optimal Allocation")
    print(allocation)

    mroi = calculate_marginal_roi(
        allocation
    )

    print(
        f"\n💰 Marginal ROI: {mroi}"
    )