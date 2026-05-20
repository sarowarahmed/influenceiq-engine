import joblib
import pandas as pd

from src.optimize import optimize_budget


# -----------------------------------
# Test Model Loading
# -----------------------------------
def test_model_loading():

    ridge_model = joblib.load(
        "models/ridge_pipeline.pkl"
    )

    assert ridge_model is not None


# -----------------------------------
# Test Optimizer
# -----------------------------------
def test_optimizer_output():

    allocation = optimize_budget(10000)

    assert allocation["TV"] >= 0
    assert allocation["Social"] >= 0
    assert allocation["Newspaper"] >= 0


# -----------------------------------
# Test Budget Constraint
# -----------------------------------
def test_budget_constraint():

    allocation = optimize_budget(10000)

    total = (
        allocation["TV"]
        + allocation["Social"]
        + allocation["Newspaper"]
    )

    assert total <= 10000.5