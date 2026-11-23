"""Run from the project folder:  python -m pytest tests -q"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.profit_analysis import forecast_next_month, overall_kpis, product_defaults
from models.classification_model import classify_products, classify_single, train_classifier
from models.regression_model import NUMERIC_FEATURES, predict_single, train_regression
from preprocessing.data_processor import DataError, aggregate_monthly, load_plan, prepare

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = str(ROOT / "data" / "sample_sales_data.csv")
PLAN = str(ROOT / "data" / "sample_future_plan.csv")


@pytest.fixture(scope="module")
def df():
    return prepare(SAMPLE)


@pytest.fixture(scope="module")
def models(df):
    return train_regression(df), train_classifier(df)


def csv_file(frame, name="x.csv"):
    buf = io.StringIO(frame.to_csv(index=False))
    buf.name = name
    return buf


# ---------- preprocessing ----------
def test_sample_loads_and_profit_adds_up(df):
    assert len(df) == 288 and df["Product"].nunique() == 12
    k = overall_kpis(df)
    assert abs(k["Total Profit"] - (k["Total Revenue"] - k["Total Cost"])) < 1


def test_messy_file_is_cleaned():
    messy = pd.DataFrame({"Item": ["Laptop", "Mouse", None, "Mouse"], "Qty": [5, 10, 3, None],
                          "Price": [50000, 700, 100, 700], "Unit Cost": [40000, 600, 80, 600],
                          "Order Date": ["2025-01-05", "2025-01-09", "2025-01-10", "2025-01-11"],
                          "Disc": [10, None, 0, 5]})
    out = prepare(csv_file(messy))
    assert len(out) == 2 and out["Discount"].max() <= 1


def test_excel_upload_works(tmp_path):
    src = pd.read_csv(SAMPLE).head(30)
    path = tmp_path / "sales.xlsx"
    src.to_excel(path, index=False)
    assert len(prepare(str(path))) == 30


def test_incompatible_file_gives_clear_error():
    with pytest.raises(DataError, match="Missing required columns"):
        prepare(csv_file(pd.DataFrame({"foo": [1], "bar": [2]})))
    with pytest.raises(DataError, match="Unsupported"):
        prepare("notes.txt")


def test_transaction_level_data_is_aggregated(df):
    """Splitting each monthly row into 3 transactions must not change monthly totals."""
    raw = pd.read_csv(SAMPLE)
    parts = []
    for i in range(3):
        p = raw.copy()
        p["Quantity"] = p["Quantity"] / 3
        p["Profit"] = p["Profit"] / 3
        parts.append(p)
    tx = prepare(csv_file(pd.concat(parts)))
    a, b = aggregate_monthly(df), aggregate_monthly(tx)
    assert len(tx) == 864 and len(b) == len(a)
    assert np.allclose(a["Profit"].sum(), b["Profit"].sum(), rtol=1e-6)


# ---------- regression ----------
def test_regression_quality_and_no_leakage(models):
    reg, _ = models
    assert not {"Revenue", "Cost", "Profit"} & set(NUMERIC_FEATURES + ["Product"])
    assert reg["metrics"]["R2"] > 0.8
    assert abs(reg["metrics"]["R2"] - reg["metrics"]["Train R2"]) < 0.1


def test_regression_responds_to_inputs(models):
    reg, _ = models
    lo = predict_single(reg, "Laptop", 55000, 40, 42000, 0)
    hi = predict_single(reg, "Laptop", 55000, 80, 42000, 0)
    cheap = predict_single(reg, "Laptop", 50000, 40, 42000, 0)
    assert hi["Predicted Profit"] > lo["Predicted Profit"] > cheap["Predicted Profit"]
    assert lo["Predicted Revenue"] == 55000 * 40 and lo["Estimated Cost"] == 42000 * 40


def test_regression_needs_enough_data(df):
    with pytest.raises(DataError, match="Not enough data"):
        train_regression(df.head(10))


# ---------- classification ----------
def test_classifier_quality_and_outputs(df, models):
    _, clf = models
    assert clf["metrics"]["Accuracy"] > 0.85 and clf["metrics"]["Accuracy on unseen products"] > 0.8
    t = classify_products(clf, df)
    assert set(t["Predicted Category"]) <= {"High Profit", "Medium Profit", "Low Profit"}
    cat = dict(zip(t["Product"], t["Predicted Category"]))
    assert cat["Cables"] == "Low Profit" and cat["Headphones"] == "High Profit"
    assert classify_single(clf, 55000, 42000, 40)["Category"] == "High Profit"
    assert classify_single(clf, 700, 680, 300)["Category"] == "Low Profit"


def test_classifier_needs_three_products(df):
    with pytest.raises(DataError, match="at least 3"):
        train_classifier(df[df["Product"].isin(["Laptop", "Mouse"])])


# ---------- forecast ----------
def test_forecast_matches_and_totals(df, models):
    reg, clf = models
    table, unmatched, _ = forecast_next_month(reg, clf, df, load_plan(PLAN))
    assert len(table) == 12 and not unmatched
    assert table["Predicted Profit"].is_monotonic_decreasing
    total = table["Predicted Profit"].sum()
    typical = table["Typical Monthly Profit"].sum()
    assert 0.6 * typical < total < 1.6 * typical      # sane versus a normal month


def test_forecast_handles_unknown_case_and_duplicates(df, models):
    reg, clf = models
    plan = pd.DataFrame({"Product": ["laptop", "LAPTOP ", "Smartwatch", "Mouse"], "Planned Quantity": [20, 20, 50, 300]})
    table, unmatched, _ = forecast_next_month(reg, clf, df, load_plan(csv_file(plan)))
    assert unmatched == ["Smartwatch"]
    assert table.loc[table["Product"] == "Laptop", "Planned Quantity"].iloc[0] == 40


def test_forecast_warns_on_extreme_quantity(df, models):
    reg, clf = models
    plan = pd.DataFrame({"Product": ["Laptop"], "Planned Quantity": [5000]})
    _, _, warnings = forecast_next_month(reg, clf, df, load_plan(csv_file(plan)))
    assert warnings


def test_bad_plan_files_give_errors():
    with pytest.raises(DataError, match="Product and Planned Quantity"):
        load_plan(csv_file(pd.DataFrame({"a": [1], "b": [2]})))
    with pytest.raises(DataError, match="no valid rows"):
        load_plan(csv_file(pd.DataFrame({"Product": ["Laptop"], "Planned Quantity": [-5]})))


def test_product_defaults_are_recent(df):
    d = product_defaults(df, "Laptop")
    assert 45000 < d["Selling Price"] < 65000 and 30 < d["Quantity"] < 55


# ---------- the real app, clicked through ----------
def test_app_pages_work():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not at.exception
    assert len(at.metric) == 5 and any("SAMPLE" in w.value for w in at.warning)

    at.sidebar.radio[0].set_value("Product Analysis").run()
    assert not at.exception
    at.selectbox(key="analysis_product").set_value("Mouse").run()
    assert not at.exception
    assert any("Low Profit" in m.value and "red" in m.value for m in at.markdown)

    at.sidebar.radio[0].set_value("Profit Prediction").run()
    assert not at.exception
    at.selectbox(key="pred_product").set_value("Laptop").run()
    next(b for b in at.button if "Predict" in b.label).click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Predicted profit" in labels and "Total expected profit" in labels
    assert len(at.dataframe) >= 1
