"""Business calculations used by the dashboard (no UI code here)."""
import pandas as pd

from models.classification_model import classify_single
from models.regression_model import predict_profit_table
from preprocessing.data_processor import aggregate_monthly


def fmt_money(value: float) -> str:
    """Format as Indian rupees: 1,23,456 -> 1.23 L, 1.2 crore -> 1.20 Cr."""
    sign = "-" if value < 0 else ""
    v = abs(value)
    if v >= 1e7:
        return f"{sign}₹{v / 1e7:.2f} Cr"
    if v >= 1e5:
        return f"{sign}₹{v / 1e5:.2f} L"
    return f"{sign}₹{v:,.0f}"


def overall_kpis(df: pd.DataFrame) -> dict:
    revenue, cost, profit = df["Revenue"].sum(), df["Cost"].sum(), df["Profit"].sum()
    return {"Total Revenue": revenue, "Total Cost": cost, "Total Profit": profit,
            "Profit Margin": profit / revenue * 100 if revenue else 0.0,
            "Total Products": df["Product"].nunique()}


def monthly_profit(df: pd.DataFrame, product: str = None) -> pd.DataFrame:
    data = df if product is None else df[df["Product"] == product]
    return data.groupby("Month", as_index=False)[["Revenue", "Cost", "Profit"]].sum()


def product_summary(df: pd.DataFrame, product: str) -> dict:
    d = df[df["Product"] == product]
    revenue, profit = d["Revenue"].sum(), d["Profit"].sum()
    return {"Quantity Sold": int(d["Quantity"].sum()), "Revenue": revenue,
            "Cost": d["Cost"].sum(), "Profit": profit,
            "Profit Margin": profit / revenue * 100 if revenue else 0.0}


def product_defaults(df: pd.DataFrame, product: str, months: int = 3) -> dict:
    """Recent typical values for a product (average of its last few months)."""
    m = aggregate_monthly(df[df["Product"] == product])
    recent = m[m["Month"] >= m["Month"].max() - pd.DateOffset(months=months - 1)]
    return {"Selling Price": float(recent["Selling Price"].mean()),
            "Cost Price": float(recent["Cost Price"].mean()),
            "Discount": float(recent["Discount"].mean()),
            "Quantity": float(recent["Quantity"].mean())}


def history_warning(df: pd.DataFrame, product: str, price=None, quantity=None):
    """Return a caution message if inputs are far outside what the data has seen."""
    m = aggregate_monthly(df[df["Product"] == product])
    notes = []
    if quantity is not None and not m["Quantity"].min() * 0.5 <= quantity <= m["Quantity"].max() * 2:
        notes.append(f"quantity {quantity:,.0f} is far outside the historical range "
                     f"({m['Quantity'].min():,.0f} to {m['Quantity'].max():,.0f} per month)")
    if price is not None and not m["Selling Price"].min() * 0.5 <= price <= m["Selling Price"].max() * 2:
        notes.append("the selling price is far outside the historical range")
    return ("Prediction may be less reliable: " + "; ".join(notes) + ".") if notes else None


def next_month_label(df: pd.DataFrame) -> str:
    return (df["Month"].max() + pd.DateOffset(months=1)).strftime("%B %Y")


def forecast_next_month(reg: dict, clf: dict, df: pd.DataFrame, plan: pd.DataFrame):
    """Predict profit for every planned product. Returns (table, unmatched names, warnings)."""
    known = {p.casefold(): p for p in df["Product"].unique()}
    plan = plan.assign(key=plan["Product"].str.casefold())
    unmatched = plan.loc[~plan["key"].isin(known), "Product"].tolist()
    plan = plan[plan["key"].isin(known)]
    plan = plan.assign(Product=plan["key"].map(known)).groupby("Product", as_index=False)["Planned Quantity"].sum()
    if plan.empty:
        return pd.DataFrame(), unmatched, []

    rows, warnings = [], []
    for _, r in plan.iterrows():
        d = product_defaults(df, r["Product"])
        rows.append({"Product": r["Product"], "Quantity": r["Planned Quantity"], **{k: d[k] for k in
                     ["Selling Price", "Cost Price", "Discount"]}})
        note = history_warning(df, r["Product"], quantity=r["Planned Quantity"])
        if note:
            warnings.append(f"{r['Product']}: {note.replace('Prediction may be less reliable: ', '')}")
    inputs = pd.DataFrame(rows)
    inputs["Predicted Profit"] = predict_profit_table(reg, inputs)
    inputs["Predicted Revenue"] = inputs["Selling Price"] * (1 - inputs["Discount"]) * inputs["Quantity"]
    inputs["Estimated Cost"] = inputs["Cost Price"] * inputs["Quantity"]
    inputs["Profit Margin"] = inputs["Predicted Profit"] / inputs["Predicted Revenue"] * 100
    inputs["Category"] = [classify_single(clf, r["Selling Price"], r["Cost Price"], r["Quantity"],
                                          r["Discount"])["Category"] for _, r in inputs.iterrows()]
    monthly = aggregate_monthly(df).groupby("Product")["Profit"].mean()
    inputs["Typical Monthly Profit"] = inputs["Product"].map(monthly)
    inputs = inputs.rename(columns={"Quantity": "Planned Quantity"})
    return inputs.sort_values("Predicted Profit", ascending=False).reset_index(drop=True), unmatched, warnings
