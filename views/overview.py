import streamlit as st

from analysis.profit_analysis import fmt_money, monthly_profit, overall_kpis
from visualization.charts import distribution_chart, monthly_trend_chart, product_profit_chart


def render(ctx):
    df, table = ctx["df"], ctx["product_table"]
    st.title("Business overview")

    k = overall_kpis(df)
    cols = st.columns(5)
    items = [("Total revenue", fmt_money(k["Total Revenue"])),
             ("Total cost", fmt_money(k["Total Cost"])),
             ("Total profit", fmt_money(k["Total Profit"])),
             ("Profit margin", f"{k['Profit Margin']:.1f}%"),
             ("Total products", f"{k['Total Products']}")]
    for col, (label, value) in zip(cols, items):
        with col.container(border=True):
            st.metric(label, value)

    left, right = st.columns([3, 2])
    left.plotly_chart(monthly_trend_chart(monthly_profit(df)), width="stretch")
    right.plotly_chart(distribution_chart(table), width="stretch")

    st.plotly_chart(product_profit_chart(table), width="stretch")

    st.subheader("Product profitability (Random Forest prediction)")
    st.dataframe(
        table[["Product", "Profit", "Profit Margin", "Predicted Category", "Confidence"]],
        hide_index=True, width="stretch",
        column_config={
            "Profit": st.column_config.NumberColumn("Profit (₹)", format="%,.0f"),
            "Profit Margin": st.column_config.NumberColumn("Profit margin", format="%.1f%%"),
            "Confidence": st.column_config.NumberColumn("Confidence", format="%.0f%%"),
        })
    st.caption("Categories come from a Random Forest trained on your data. Low confidence "
               "means the product sits close to a category boundary.")
