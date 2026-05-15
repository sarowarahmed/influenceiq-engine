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
def build_feature_row(tv, social, newspaper):

    import pandas as pd

    df = pd.DataFrame({
        "Date": [pd.Timestamp("2024-01-01")],
        "TV": [tv],
        "Social": [social],
        "Newspaper": [newspaper]
    })

    df = engineer_features(df)
    df["Time_Index"] = 5000

    X = df[feature_cols]

    return X


# -------------------------------
# Objective Function
# -------------------------------
def objective(spends):

    tv, social, newspaper = spends

    X = build_feature_row(
        tv,
        social,
        newspaper
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

    optimal_tv = result.x[0]
    optimal_social = result.x[1]
    optimal_newspaper = result.x[2]

    # Final Prediction
    optimal_features = build_feature_row(
        optimal_tv,
        optimal_social,
        optimal_newspaper
    )

    predicted_sales = ridge_model.predict(
        optimal_features
    )[0]

    return {
        "TV": round(optimal_tv, 2),
        "Social": round(optimal_social, 2),
        "Newspaper": round(optimal_newspaper, 2),
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
        allocation["TV"]
        + allocation["Social"]
        + allocation["Newspaper"]
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