import streamlit as st

from analysis.profit_analysis import fmt_money, monthly_profit, overall_kpis, product_summary
from visualization.charts import product_history_chart

BADGE_COLORS = {"High Profit": "green", "Medium Profit": "blue", "Low Profit": "red"}


def render(ctx):
    df, table = ctx["df"], ctx["product_table"]
    st.title("Product analysis")

    products = table["Product"].tolist()          # ordered by profit, best first
    product = st.selectbox("Search for a product", sorted(products), index=sorted(products).index(products[0]),
                           help="Click and type to search, for example Laptop or Mouse.", key="analysis_product")

    s = product_summary(df, product)
    row = table[table["Product"] == product].iloc[0]
    overall_margin = overall_kpis(df)["Profit Margin"]

    cols = st.columns(5)
    values = [("Quantity sold", f"{s['Quantity Sold']:,}", None),
              ("Total revenue", fmt_money(s["Revenue"]), None),
              ("Total cost", fmt_money(s["Cost"]), None),
              ("Total profit", fmt_money(s["Profit"]), None),
              ("Profit margin", f"{s['Profit Margin']:.1f}%", f"{s['Profit Margin'] - overall_margin:+.1f} pts vs all products")]
    for col, (label, value, delta) in zip(cols, values):
        with col.container(border=True):
            st.metric(label, value, delta=delta)

    with st.container(border=True):
        a, b = st.columns([1, 3])
        a.caption("Predicted profitability")
        a.badge(row["Predicted Category"], color=BADGE_COLORS[row["Predicted Category"]])
        b.caption(f"Random Forest confidence: {row['Confidence']:.0f}%. "
                  f"Rank {int(table.index[table['Product'] == product][0]) + 1} of {len(table)} by total profit.")

    monthly = monthly_profit(df, product)
    st.plotly_chart(product_history_chart(monthly, product), width="stretch")
    with st.expander("Monthly figures"):
        st.dataframe(monthly.assign(Month=monthly["Month"].dt.strftime("%b %Y")), hide_index=True,
                     width="stretch", column_config={c: st.column_config.NumberColumn(f"{c} (₹)", format="%,.0f")
                                                      for c in ["Revenue", "Cost", "Profit"]})
