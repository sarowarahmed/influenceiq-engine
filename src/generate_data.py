# src/generate_data.py

import os
import pandas as pd
import numpy as np

from faker import Faker


# -------------------------------
# Random Seed
# -------------------------------
np.random.seed(42)

fake = Faker()


# -------------------------------
# Adstock Function
# -------------------------------
def adstock(x, decay=0.5):

    result = np.zeros(len(x))

    for i in range(len(x)):
        if i == 0:
            result[i] = x[i]
        else:
            result[i] = x[i] + decay * result[i - 1]

    return result


# -------------------------------
# Saturation Function
# -------------------------------
def saturation(x, alpha=0.01):

    return 1 / (1 + np.exp(-alpha * x))


# -------------------------------
# Main Data Generator
# -------------------------------
def generate_data(
    rows=5000,
    social_effect=0.55
):

    # Time Index
    days = pd.date_range(
        start="2010-01-01",
        periods=rows
    )

    df = pd.DataFrame({
        "Date": days
    })

    # -----------------------
    # Seasonality
    # -----------------------
    day_of_year = df["Date"].dt.dayofyear

    seasonality = (
        10 * np.sin(
            2 * np.pi * day_of_year / 365.25
        )
    )

    # -----------------------
    # Trend
    # -----------------------
    trend = 0.02 * np.arange(rows)

    # -----------------------
    # Ad Spend Features
    # -----------------------
    df["YouTube"] = np.random.gamma(
        shape=2,
        scale=50,
        size=rows
    )

    df["Social"] = np.random.uniform(
        10,
        200,
        size=rows
    )

    df["Newspaper"] = np.random.exponential(
        scale=30,
        size=rows
    )

    # -----------------------
    # Metadata
    # -----------------------
    df["Campaign_Manager"] = [
        fake.name()
        for _ in range(rows)
    ]

    df["Region"] = [
        fake.city()
        for _ in range(rows)
    ]

    # -----------------------
    # Adstock Effects
    # -----------------------
    tv_adstock = adstock(
        df["TV"].values,
        decay=0.6
    )

    social_adstock = adstock(
        df["Social"].values,
        decay=0.4
    )

    news_adstock = adstock(
        df["Newspaper"].values,
        decay=0.2
    )

    # -----------------------
    # Saturation Effects
    # -----------------------
    tv_effect = (
        8 * saturation(tv_adstock)
    )

    social_effects = (
        100 * saturation(social_adstock)
    )

    news_effect = (
        4 * saturation(news_adstock)
    )

    # -----------------------
    # Interaction Effect
    # -----------------------
    interaction = (
        0.0001
        * df["TV"]
        * df["Social"]
    )

    # -----------------------
    # Noise
    # -----------------------
    noise = np.random.normal(
        0,
        3,
        size=rows
    )

    # -----------------------
    # Final Sales Equation
    # -----------------------
    df["Sales"] = (
        50
        + tv_effect
        + social_effects
        + news_effect
        + interaction
        + seasonality
        + trend
        + noise
    )

    return df


# -------------------------------
# Script Execution
# -------------------------------
if __name__ == "__main__":

    df = generate_data()

    os.makedirs(
        "data",
        exist_ok=True
    )

    df.to_csv(
        "data/marketing_data.csv",
        index=False
    )

    print(
        "✅ Dataset saved to data/marketing_data.csv"
    )