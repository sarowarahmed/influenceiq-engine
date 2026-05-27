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
    rows=10000,
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

    df["Instagram"] = np.random.uniform(
        10,
        200,
        size=rows
    )

    df["Twitter"] = np.random.exponential(
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
    yt_adstock = adstock(
        df["YouTube"].values,
        decay=0.6
    )

    insta_adstock = adstock(
        df["Instagram"].values,
        decay=0.4
    )

    x_adstock = adstock(
        df["Twitter"].values,
        decay=0.2
    )

    # -----------------------
    # Saturation Effects
    # -----------------------
    yt_effect = (
        8 * saturation(yt_adstock)
    )

    insta_effects = (
        100 * saturation(insta_adstock)
    )

    x_effect = (
        4 * saturation(x_adstock)
    )

    # -----------------------
    # Interaction Effect
    # -----------------------
    interaction = (
        0.0001
        * df["YouTube"]
        * df["Instagram"]
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
    # Creator Niches
    # -----------------------
    niches = [
        "Beauty",
        "Gaming",
        "Tech",
        "Fitness",
        "Fashion"
    ]

    niche_weights = {
        "Beauty": 1.25,
        "Gaming": 1.15,
        "Tech": 1.10,
        "Fitness": 1.05,
        "Fashion": 1.20
    }


    df["Niche"] = np.random.choice(
        niches,
        size=rows
    )

    niche_multiplier = (
        df["Niche"]
        .map(niche_weights)
    )
    
    # -----------------------
    # Final Target Equation
    # -----------------------
    df["Engagement_Score"] = (
    (
        50
        + yt_effect
        + insta_effects
        + x_effect
        + interaction
        + seasonality
        + trend
        + noise
    )
    * niche_multiplier
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