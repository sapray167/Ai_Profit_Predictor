from pathlib import Path

import streamlit as st

from analysis.profit_analysis import (fmt_money, forecast_next_month, history_warning,
                                      next_month_label, product_defaults)
from models.classification_model import classify_single
from models.regression_model import predict_single
from preprocessing.data_processor import DataError, aggregate_monthly, load_plan
from visualization.charts import actual_vs_predicted_figure, forecast_chart

BADGE_COLORS = {"High Profit": "green", "Medium Profit": "blue", "Low Profit": "red"}
SAMPLE_PLAN = Path(__file__).parent.parent / "data" / "sample_future_plan.csv"


def render(ctx):
    st.title("Profit prediction")
    tab1, tab2, tab3 = st.tabs(["Single product", "Next-month forecast", "Model performance"])
    with tab1:
        single_product(ctx)
    with tab2:
        next_month(ctx)
    with tab3:
        model_performance(ctx)


def single_product(ctx):
    df, reg, clf = ctx["df"], ctx["reg"], ctx["clf"]
    st.caption("Enter a scenario. The regression model predicts profit and the Random Forest "
               "classifies the scenario. Fields start with the product's recent averages.")
    product = st.selectbox("Product", sorted(df["Product"].unique()), key="pred_product")
    d = product_defaults(df, product)

    with st.form("predict_form"):
        c1, c2, c3, c4 = st.columns(4)
        price = c1.number_input("Expected selling price (₹)", min_value=0.01, value=round(d["Selling Price"], 2),
                                key=f"price_{product}")
        qty = c2.number_input("Expected quantity", min_value=1, value=max(1, round(d["Quantity"])),
                              step=1, key=f"qty_{product}")
        cost = c3.number_input("Expected cost per unit (₹)", min_value=0.01, value=round(d["Cost Price"], 2),
                               key=f"cost_{product}")
        disc = c4.number_input("Discount (%)", min_value=0.0, max_value=90.0, value=round(d["Discount"] * 100, 1),
                               step=1.0, key=f"disc_{product}")
        submitted = st.form_submit_button("Predict profit", type="primary")

    if not submitted:
        return
    result = predict_single(reg, product, price, qty, cost, disc / 100)
    category = classify_single(clf, price, cost, qty, disc / 100)

    cols = st.columns(4)
    items = [("Predicted profit", fmt_money(result["Predicted Profit"])),
             ("Predicted revenue", fmt_money(result["Predicted Revenue"])),
             ("Estimated cost", fmt_money(result["Estimated Cost"])),
             ("Predicted margin", f"{result['Profit Margin']:.1f}%")]
    for col, (label, value) in zip(cols, items):
        with col.container(border=True):
            st.metric(label, value)
    with st.container(border=True):
        a, b = st.columns([1, 3])
        a.caption("Profitability category")
        a.badge(category["Category"], color=BADGE_COLORS[category["Category"]])
        b.caption(f"Random Forest confidence: {category['Confidence']:.0f}%")

    if result["Predicted Profit"] < 0:
        st.error("This scenario is predicted to make a loss.")
    note = history_warning(df, product, price=price, quantity=qty)
    if note:
        st.warning(note)


def next_month(ctx):
    df, reg, clf = ctx["df"], ctx["reg"], ctx["clf"]
    st.subheader(f"Forecast for {next_month_label(df)}")
    st.caption("Upload a plan with two columns: Product and Planned Quantity. Selling price, cost and "
               "discount come from each product's last three months.")

    typical = aggregate_monthly(df).groupby("Product")["Quantity"].mean().round().astype(int)
    template = typical.rename("Planned Quantity").reset_index().to_csv(index=False)
    left, right = st.columns([3, 1])
    upload = left.file_uploader("Upload sales plan", type=["csv", "xlsx", "xls"], key="plan_upload")
    right.download_button("Download template", template, "sales_plan_template.csv", "text/csv",
                          help="A plan pre-filled with each product's typical monthly quantity.")

    if upload is None and not ctx["is_sample"]:
        st.info("Upload a sales plan to see the forecast. The template has your products ready to edit.")
    else:
        try:
            plan = load_plan(upload if upload is not None else SAMPLE_PLAN)
            if upload is None:
                st.warning("Showing a SAMPLE plan (synthetic, for demonstration). Upload your own to replace it.")
            show_forecast(df, reg, clf, plan)
        except DataError as e:
            st.error(str(e))

    st.divider()
    st.subheader("Quick forecast for one product")
    c1, c2 = st.columns(2)
    product = c1.selectbox("Search for a product", sorted(df["Product"].unique()), key="quick_product")
    d = product_defaults(df, product)
    planned = c2.number_input("Planned quantity", min_value=1, value=max(1, round(d["Quantity"])), step=1,
                              key=f"quick_qty_{product}")
    one = predict_single(reg, product, d["Selling Price"], planned, d["Cost Price"], d["Discount"])
    with st.container(border=True):
        st.metric(f"Predicted profit for {planned:,} units of {product}", fmt_money(one["Predicted Profit"]))
    note = history_warning(df, product, quantity=planned)
    if note:
        st.warning(note)


def show_forecast(df, reg, clf, plan):
    forecast, unmatched, warnings = forecast_next_month(reg, clf, df, plan)
    if unmatched:
        st.warning("These products aren't in your sales history, so they were skipped: " + ", ".join(unmatched))
    if forecast.empty:
        st.error("None of the planned products match your sales history.")
        return
    total = forecast["Predicted Profit"].sum()
    typical = forecast["Typical Monthly Profit"].sum()

    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.metric("Total expected profit", fmt_money(total), delta=f"{(total / typical - 1) * 100:+.1f}% vs typical month")
    with c2.container(border=True):
        st.metric("Expected revenue", fmt_money(forecast["Predicted Revenue"].sum()))
    with c3.container(border=True):
        st.metric("Products planned", f"{len(forecast)}")

    top = forecast.head(3)
    top = forecast.head(3)
    st.markdown("**Highest expected profit:** " + ", ".join(
        f"{p} ({fmt_money(v)})" for p, v in zip(top["Product"], top["Predicted Profit"])))
    st.plotly_chart(forecast_chart(forecast), width="stretch")
    st.dataframe(
        forecast[["Product", "Planned Quantity", "Predicted Revenue", "Estimated Cost", "Predicted Profit",
                  "Profit Margin", "Category"]],
        hide_index=True, width="stretch",
        column_config={**{c: st.column_config.NumberColumn(f"{c} (₹)", format="%,.0f")
                          for c in ["Predicted Revenue", "Estimated Cost", "Predicted Profit"]},
                       "Planned Quantity": st.column_config.NumberColumn(format="%,d"),
                       "Profit Margin": st.column_config.NumberColumn(format="%.1f%%")})
    if warnings:
        st.warning("Less reliable predictions: " + "; ".join(warnings))


def model_performance(ctx):
    reg, clf = ctx["reg"], ctx["clf"]
    m = reg["metrics"]
    st.subheader("Profit model: Multiple Linear Regression")
    cols = st.columns(4)
    for col, (label, value) in zip(cols, [("R²", f"{m['R2']:.3f}"), ("MAE", fmt_money(m["MAE"])),
                                          ("RMSE", fmt_money(m["RMSE"])),
                                          ("Train / test rows", f"{m['Train rows']} / {m['Test rows']}")]):
        with col.container(border=True):
            st.metric(label, value)
    a, b = st.columns([1, 1])
    a.pyplot(actual_vs_predicted_figure(reg["test_actual"], reg["test_predicted"]))
    b.caption("Inputs: selling price, quantity, cost per unit, discount and product. Revenue and cost "
              "are not used because they are calculated from the same numbers as profit (target leakage). "
              "Scores come from data the model did not train on.")
    with b.expander("Model coefficients"):
        st.dataframe(reg["coefficients"].round(2), hide_index=True, width="stretch")

    st.subheader("Category model: Random Forest")
    cm = clf["metrics"]
    lo, hi = clf["thresholds"]
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.metric("Accuracy on held-out records", f"{cm['Accuracy'] * 100:.1f}%")
    with c2.container(border=True):
        st.metric("Accuracy on unseen products", f"{cm['Accuracy on unseen products'] * 100:.1f}%")
    st.caption(f"Labels come from your own data: product margins below {lo:.1f}% are Low, above {hi:.1f}% "
               "are High (the 33rd and 67th percentiles). The forest learns to predict these labels from "
               "prices, cost, discount and quantity. The cost-to-price ratio is the strongest signal, so "
               "the model closely follows the margin rule.")
    d1, d2 = st.columns(2)
    d1.dataframe(clf["confusion"], width="stretch")
    d2.bar_chart(clf["importance"], horizontal=True)
