"""Multiple Linear Regression: predicts Profit from planned sales inputs."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from preprocessing.data_processor import DataError, aggregate_monthly

NUMERIC_FEATURES = ["Selling Price", "Quantity", "Cost Price", "Discount"]
CATEGORY_FEATURES = ["Product"]
TARGET = "Profit"
# Revenue and Cost are NOT used: they are calculated from the same numbers as
# Profit, so they would leak the answer into the model.


def train_regression(df: pd.DataFrame, test_size=0.2, random_state=42) -> dict:
    """Train on 80% of the data, evaluate on the unseen 20%."""
    df = aggregate_monthly(df)
    if len(df) < 20:
        raise DataError("Not enough data to train the profit model. "
                        "Upload at least 20 product-per-month records.")
    X = df[NUMERIC_FEATURES + CATEGORY_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    prep = ColumnTransformer([
        ("product", OneHotEncoder(handle_unknown="ignore"), CATEGORY_FEATURES),
    ], remainder="passthrough")
    model = Pipeline([("prep", prep), ("regression", LinearRegression())])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = {
        "R2": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "Train R2": r2_score(y_train, model.predict(X_train)),
        "Train rows": len(X_train),
        "Test rows": len(X_test),
    }
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = pd.DataFrame({
        "Feature": [n.split("__")[-1] for n in names],
        "Coefficient": model.named_steps["regression"].coef_,
    })
    return {"model": model, "metrics": metrics, "coefficients": coefs,
            "test_actual": y_test.values, "test_predicted": pred}


def predict_profit_table(result: dict, inputs: pd.DataFrame) -> np.ndarray:
    """Predict profit for many rows. `inputs` needs the feature columns."""
    return result["model"].predict(inputs[NUMERIC_FEATURES + CATEGORY_FEATURES])


def predict_single(result: dict, product: str, price: float, quantity: float,
                   cost: float, discount: float = 0.0) -> dict:
    """Predict one product. Revenue and cost are plain arithmetic, profit is ML."""
    row = pd.DataFrame([{"Selling Price": price, "Quantity": quantity,
                         "Cost Price": cost, "Discount": discount, "Product": product}])
    profit = float(predict_profit_table(result, row)[0])
    revenue = price * (1 - discount) * quantity
    return {"Predicted Profit": profit, "Predicted Revenue": revenue,
            "Estimated Cost": cost * quantity,
            "Profit Margin": profit / revenue * 100 if revenue else 0.0}
