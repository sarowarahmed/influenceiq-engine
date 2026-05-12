# src/data_generator.py

import pandas as pd
import numpy as np
from faker import Faker


def adstock(x, decay=0.5):
    result = np.zeros_like(x)
    for i in range(len(x)):
        result[i] = x[i] + (result[i-1] * decay if i > 0 else 0)
    return result


def saturation(x, alpha=0.01):
    return 1 / (1 + np.exp(-alpha * x))


def generate_data(n=5000, social_weight=0.55, seed=42):
    # Reproducibility
    np.random.seed(seed)
    Faker.seed(seed)
    fake = Faker()

    # Date
    days = pd.date_range(start="2012-01-01", periods=n)
    df = pd.DataFrame({'Date': days})

    # Time features
    day_of_year = df['Date'].dt.dayofyear
    seasonality = 10 * np.sin(2 * np.pi * day_of_year / 365.25)
    trend = 0.02 * np.arange(n)
    weekday = df['Date'].dt.weekday
    weekly = 3 * np.sin(2 * np.pi * weekday / 7 + np.pi/4)
    
    # Ad Spend
    df['TV'] = np.random.gamma(shape=2, scale=50, size=n)
    df['Social'] = np.random.uniform(10, 200, size=n)
    df['Newspaper'] = np.random.exponential(scale=30, size=n)

    # Metadata
    df['Campaign_Manager'] = [fake.name() for _ in range(n)]
    df['Region'] = [fake.city() for _ in range(n)]

    # Adstock
    df['TV_adstock'] = adstock(df['TV'].values, decay=0.6)
    df['Social_adstock'] = adstock(df['Social'].values, decay=0.4)
    df['News_adstock'] = adstock(df['Newspaper'].values, decay=0.2)

    # Saturation effects
    tv_effect = 40 * saturation(df['TV_adstock'])
    social_effect = (60 * social_weight) * saturation(df['Social_adstock'])
    news_effect = 15 * saturation(df['News_adstock'])

    # Interaction
    interaction = 0.02 * df['TV'] * df['Social']

    # Noise
    noise = np.random.normal(0, 3, size=n)

    # Final Sales
    df['Sales'] = (
        50 +
        tv_effect +
        social_effect +
        news_effect +
        interaction +
        seasonality +
        weekly +
        trend +
        noise
    )

    return df

import os

os.makedirs("data", exist_ok=True)

df.to_csv(
    "data/marketing_data.csv",
    index=False
)

print("✅ Dataset saved to data/marketing_data.csv")