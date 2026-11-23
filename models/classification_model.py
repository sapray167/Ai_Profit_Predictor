"""Random Forest classifier: High / Medium / Low profit category.

Labeling strategy (explainable, learned from the data - no fixed cut-offs):
  1. Work out each product's overall profit margin.
  2. The 33rd and 67th percentiles of those margins become the thresholds.
  3. Every sales record gets a label from its own margin: Low / Medium / High.
The Random Forest then learns to predict the label from cost structure
(prices, cost, discount, quantity) - it never sees the margin itself.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split

from preprocessing.data_processor import DataError, aggregate_monthly

CATEGORIES = ["Low Profit", "Medium Profit", "High Profit"]
FEATURES = ["Selling Price", "Cost Price", "Discount", "Quantity", "Cost to Price Ratio"]


def _features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["Selling Price", "Cost Price", "Discount", "Quantity"]].copy()
    X["Cost to Price Ratio"] = X["Cost Price"] / (X["Selling Price"] * (1 - X["Discount"]))
    return X[FEATURES]


def _label(margin, low, high):
    return np.where(margin >= high, CATEGORIES[2], np.where(margin >= low, CATEGORIES[1], CATEGORIES[0]))


def make_labels(df: pd.DataFrame):
    """Return (row labels, thresholds) using percentiles of product margins."""
    totals = df.groupby("Product")[["Profit", "Revenue"]].sum()
    product_margin = totals["Profit"] / totals["Revenue"] * 100
    low, high = np.percentile(product_margin, [100 / 3, 200 / 3])
    row_margin = df["Profit"] / df["Revenue"] * 100
    return _label(row_margin, low, high), (float(low), float(high))


def train_classifier(df: pd.DataFrame, random_state=42) -> dict:
    df = aggregate_monthly(df)
    n_products = df["Product"].nunique()
    if n_products < 3:
        raise DataError("Profitability classification needs at least 3 different products.")

    labels, thresholds = make_labels(df)
    X, y = _features(df), pd.Series(labels, index=df.index)

    can_stratify = y.value_counts().min() >= 2
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y if can_stratify else None)
    model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    # Stricter test: hold out whole products the model has never seen
    folds = min(4, n_products)
    unseen = cross_val_score(
        RandomForestClassifier(n_estimators=200, random_state=random_state),
        X, y, groups=df["Product"], cv=GroupKFold(folds)).mean()

    # Final model learns from all the data
    final = RandomForestClassifier(n_estimators=200, random_state=random_state).fit(X, y)
    return {
        "model": final, "thresholds": thresholds,
        "metrics": {"Accuracy": accuracy_score(y_te, pred),
                    "Accuracy on unseen products": float(unseen),
                    "Train rows": len(X_tr), "Test rows": len(X_te)},
        "confusion": pd.DataFrame(confusion_matrix(y_te, pred, labels=CATEGORIES),
                                  index=CATEGORIES, columns=CATEGORIES),
        "importance": pd.Series(final.feature_importances_, index=FEATURES).sort_values(ascending=False),
    }


def classify_products(result: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Predicted category per product (average of the model's probabilities)."""
    model = result["model"]
    monthly = aggregate_monthly(df)
    proba = pd.DataFrame(model.predict_proba(_features(monthly)), columns=model.classes_, index=monthly.index)
    avg = proba.groupby(monthly["Product"]).mean()
    table = df.groupby("Product").agg(Quantity=("Quantity", "sum"), Revenue=("Revenue", "sum"),
                                      Cost=("Cost", "sum"), Profit=("Profit", "sum"))
    table["Profit Margin"] = table["Profit"] / table["Revenue"] * 100
    table["Predicted Category"] = avg.idxmax(axis=1)
    table["Confidence"] = avg.max(axis=1) * 100
    return table.reset_index().sort_values("Profit", ascending=False).reset_index(drop=True)


def classify_single(result: dict, price, cost, quantity, discount=0.0) -> dict:
    """Category for one what-if scenario."""
    row = pd.DataFrame([{"Selling Price": price, "Cost Price": cost,
                         "Discount": discount, "Quantity": quantity}])
    model = result["model"]
    proba = model.predict_proba(_features(row))[0]
    best = int(np.argmax(proba))
    return {"Category": model.classes_[best], "Confidence": float(proba[best] * 100)}
