"""Plotly charts. Colors: blue = main, green = high profit, red = low profit."""
import plotly.express as px
import plotly.graph_objects as go

BLUE = "#185FA5"
CATEGORY_COLORS = {"High Profit": "#2E9E5B", "Medium Profit": "#378ADD", "Low Profit": "#D64545"}
CATEGORY_ORDER = ["High Profit", "Medium Profit", "Low Profit"]


def _clean(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(size=13), legend_title_text="")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E5E7EB")
    return fig


def monthly_trend_chart(monthly):
    fig = px.line(monthly, x="Month", y="Profit", markers=True, title="Monthly profit trend")
    fig.update_traces(line=dict(color=BLUE, width=2.5))
    fig.update_yaxes(title="Profit (₹)")
    fig.update_xaxes(title="")
    return _clean(fig)


def product_profit_chart(table):
    data = table.sort_values("Profit")
    fig = px.bar(data, x="Profit", y="Product", orientation="h", color="Predicted Category",
                 color_discrete_map=CATEGORY_COLORS, category_orders={"Predicted Category": CATEGORY_ORDER},
                 title="Profit by product")
    fig.update_xaxes(title="Profit (₹)", showgrid=True, gridcolor="#E5E7EB")
    fig.update_yaxes(title="", gridcolor="white")
    return _clean(fig, height=max(340, 28 * len(data) + 120))


def distribution_chart(table):
    counts = table["Predicted Category"].value_counts().reindex(CATEGORY_ORDER).fillna(0)
    fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.55, sort=False,
                           marker=dict(colors=[CATEGORY_COLORS[c] for c in counts.index]),
                           textinfo="value"))
    fig.update_layout(title="Profitability distribution")
    return _clean(fig)


def product_history_chart(monthly, product):
    fig = go.Figure()
    for col, color in [("Revenue", BLUE), ("Cost", "#9CA3AF"), ("Profit", "#2E9E5B")]:
        fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly[col], name=col, mode="lines+markers",
                                 line=dict(color=color, width=2.5)))
    fig.update_layout(title=f"{product}: monthly performance", yaxis_title="Amount (₹)")
    return _clean(fig)


def forecast_chart(forecast):
    data = forecast.sort_values("Predicted Profit", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data["Product"], y=data["Typical Monthly Profit"], name="Typical monthly profit",
                         marker_color="#CBD5E1"))
    fig.add_trace(go.Bar(x=data["Product"], y=data["Predicted Profit"], name="Predicted profit",
                         marker_color=BLUE))
    fig.update_layout(barmode="group", title="Predicted profit vs typical month", yaxis_title="Profit (₹)",
                      legend=dict(orientation="h", y=1.12))
    return _clean(fig)


def actual_vs_predicted_figure(actual, predicted):
    """Matplotlib scatter used on the model performance tab."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(actual, predicted, color=BLUE, alpha=0.7, s=22)
    lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    ax.plot([lo, hi], [lo, hi], color="#D64545", linestyle="--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("Actual profit (₹)")
    ax.set_ylabel("Predicted profit (₹)")
    ax.set_title("Regression: actual vs predicted (test data)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
