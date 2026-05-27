# src/features.py

import pandas as pd
import numpy as np


# -------------------------------
# Adstock Transformation
# -------------------------------
def adstock(series, decay=0.5):
    result = np.zeros(len(series))

    for i in range(len(series)):
        if i == 0:
            result[i] = series.iloc[i]
        else:
            result[i] = series.iloc[i] + decay * result[i - 1]

    return result


# -------------------------------
# Saturation Transformation
# -------------------------------
def saturation(x, alpha=0.01):
    return 1 / (1 + np.exp(-alpha * x))


# -------------------------------
# Feature Engineering Pipeline
# -------------------------------
def engineer_features(df):

    # Adstock Features
    df["YouTube_Adstock"] = adstock(df["YouTube"], decay=0.6)
    df["Instagram_Adstock"] = adstock(df["Instagram"], decay=0.4)
    df["Twitter_Adstock"] = adstock(df["Twitter"], decay=0.2)


    # Interaction Feature
    df["YouTube_Instagram_Interaction"] = (
        df["YouTube"] * df["Instagram"]
    )

    # Time-Based Features
    df["Month"] = pd.to_datetime(df["Date"]).dt.month
    #df["Quarter"] = pd.to_datetime(df["Date"]).dt.quarter
    df["Time_Index"] = np.arange(len(df))

    return df