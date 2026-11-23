"""Creates the 3 reference images in docs/. Run: python docs/make_reference_images.py
Image 3 uses numbers computed from the sample data by the real project code."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from analysis.profit_analysis import fmt_money, forecast_next_month, monthly_profit, overall_kpis
from models.classification_model import classify_products, train_classifier
from models.regression_model import train_regression
from preprocessing.data_processor import load_plan, prepare

BLUE, GREEN, RED, GREY, BG = "#185FA5", "#2E9E5B", "#D64545", "#6B7280", "#F5F7FA"
CAT = {"High Profit": GREEN, "Medium Profit": "#378ADD", "Low Profit": RED}

df = prepare(ROOT / "data" / "sample_sales_data.csv")
reg, clf = train_regression(df), train_classifier(df)
table = classify_products(clf, df)
kpi = overall_kpis(df)
fc, _, _ = forecast_next_month(reg, clf, df, load_plan(ROOT / "data" / "sample_future_plan.csv"))


def canvas(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    base = fig.add_axes([0, 0, 1, 1])
    base.set_xlim(0, 1); base.set_ylim(0, 1); base.axis("off")
    return fig, base


def card(base, x, y, w, h):
    base.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                  fc="white", ec="#E5E7EB", lw=1, transform=base.transAxes))


def kpi_card(base, x, y, w, h, label, value, color="#1F2937", size=17):
    card(base, x, y, w, h)
    base.text(x + 0.012, y + h * 0.66, label, fontsize=9, color=GREY, va="center")
    base.text(x + 0.012, y + h * 0.30, value, fontsize=size, color=color, va="center", fontweight="bold")


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7, colors=GREY)


# ---------- Image 1: KPI cards ----------
fig, b = canvas(9, 2.6)
b.text(0.03, 0.9, "Reference 1: simple KPI dashboard (revenue, profit, margin)", fontsize=10, color=GREY)
for i, (lab, val, col) in enumerate([("Total revenue", fmt_money(kpi["Total Revenue"]), "#1F2937"),
                                     ("Total profit", fmt_money(kpi["Total Profit"]), GREEN),
                                     ("Profit margin", f"{kpi['Profit Margin']:.1f}%", BLUE)]):
    kpi_card(b, 0.03 + i * 0.32, 0.12, 0.30, 0.62, lab, val, col, 22)
fig.savefig(ROOT / "docs" / "image1_kpi_cards.png", dpi=150)

# ---------- Image 2: product profit with categories ----------
fig, b = canvas(9, 5)
b.text(0.03, 0.95, "Reference 2: product-wise profit by High / Medium / Low category", fontsize=10, color=GREY)
ax = fig.add_axes([0.14, 0.14, 0.82, 0.72]); style(ax)
d = table.sort_values("Profit")
ax.barh(d["Product"], d["Profit"] / 1e5, color=[CAT[c] for c in d["Predicted Category"]], height=0.65)
ax.set_xlabel("Profit (₹ lakh)", fontsize=8, color=GREY)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in CAT.values()], labels=list(CAT),
          frameon=False, fontsize=8, loc="lower right")
fig.savefig(ROOT / "docs" / "image2_product_categories.png", dpi=150)

# ---------- Image 3: final dashboard mockup ----------
fig, b = canvas(14, 8.6)
b.add_patch(plt.Rectangle((0, 0), 0.13, 1, fc="white", ec="#E5E7EB", transform=b.transAxes))
b.text(0.012, 0.95, "Profit Predictor", fontsize=12, color=BLUE, fontweight="bold")
for i, name in enumerate(["Overview", "Product Analysis", "Profit Prediction"]):
    y = 0.86 - i * 0.065
    if i == 0:
        b.add_patch(FancyBboxPatch((0.008, y - 0.022), 0.114, 0.045, boxstyle="round,pad=0,rounding_size=0.006",
                                   fc="#E6F1FB", ec="none", transform=b.transAxes))
    b.text(0.016, y, name, fontsize=9.5, color=BLUE if i == 0 else GREY, va="center")
b.text(0.012, 0.58, "Upload CSV or Excel\n(sample data loaded)", fontsize=8, color=GREY, va="top")
b.text(0.15, 0.955, "Business overview", fontsize=17, fontweight="bold", color="#1F2937", va="center")
b.text(0.98, 0.955, "Mockup drawn from the project's SAMPLE data", fontsize=8, color=GREY, ha="right", va="center")

items = [("Total revenue", fmt_money(kpi["Total Revenue"]), "#1F2937"), ("Total cost", fmt_money(kpi["Total Cost"]), "#1F2937"),
         ("Total profit", fmt_money(kpi["Total Profit"]), GREEN), ("Profit margin", f"{kpi['Profit Margin']:.1f}%", BLUE),
         ("Total products", str(kpi["Total Products"]), "#1F2937")]
for i, (lab, val, col) in enumerate(items):
    kpi_card(b, 0.15 + i * 0.1700, 0.80, 0.16, 0.10, lab, val, col, 15)

card(b, 0.15, 0.45, 0.40, 0.30); card(b, 0.565, 0.45, 0.16, 0.30); card(b, 0.735, 0.45, 0.245, 0.30)
m = monthly_profit(df)
ax = fig.add_axes([0.17, 0.50, 0.37, 0.20]); style(ax)
ax.plot(range(len(m)), m["Profit"] / 1e5, color=BLUE, lw=2, marker="o", ms=3)
ax.set_xticks(range(0, len(m), 4)); ax.set_xticklabels(m["Month"].dt.strftime("%b %y")[::4])
ax.set_title("Monthly profit trend (₹ lakh)", fontsize=9, loc="left", color="#1F2937")
cnt = table["Predicted Category"].value_counts().reindex(list(CAT)).fillna(0)
ax = fig.add_axes([0.575, 0.47, 0.14, 0.22])
ax.pie(cnt, colors=list(CAT.values()), startangle=90, counterclock=False, wedgeprops=dict(width=0.38),
       labels=[f"{int(v)}" for v in cnt], textprops=dict(fontsize=8))
ax.set_title("Profitability mix", fontsize=9, loc="left", color="#1F2937")
b.text(0.575, 0.465, "● High  ", fontsize=7, color=GREEN); b.text(0.62, 0.465, "● Medium  ", fontsize=7, color="#378ADD")
b.text(0.675, 0.465, "● Low", fontsize=7, color=RED)
top = table.head(8).iloc[::-1]
ax = fig.add_axes([0.80, 0.50, 0.17, 0.20]); style(ax)
ax.barh(top["Product"], top["Profit"] / 1e5, color=[CAT[c] for c in top["Predicted Category"]], height=0.65)
ax.set_title("Profit by product", fontsize=9, loc="left", color="#1F2937")

card(b, 0.15, 0.04, 0.46, 0.37); card(b, 0.62, 0.04, 0.36, 0.37)
b.text(0.162, 0.385, "Product profitability (Random Forest)", fontsize=9.5, fontweight="bold", color="#1F2937", va="center")
for x, h in zip([0.162, 0.26, 0.34, 0.43], ["Product", "Profit", "Margin", "Predicted category"]):
    b.text(x, 0.345, h, fontsize=8, color=GREY, va="center")
for i, r in enumerate(table.head(8).itertuples()):
    y = 0.31 - i * 0.034
    b.plot([0.158, 0.602], [y + 0.017] * 2, color="#EEF0F3", lw=0.8)
    b.text(0.162, y, r.Product, fontsize=8, va="center")
    b.text(0.26, y, fmt_money(r.Profit), fontsize=8, va="center")
    b.text(0.34, y, f"{r._6:.1f}%", fontsize=8, va="center")
    b.text(0.43, y, r._7, fontsize=8, va="center", color=CAT[r._7], fontweight="bold")

b.text(0.632, 0.385, "Next-month forecast (sample plan)", fontsize=9.5, fontweight="bold", color="#1F2937", va="center")
b.text(0.632, 0.335, "Total expected profit", fontsize=8, color=GREY, va="center")
b.text(0.632, 0.295, fmt_money(fc["Predicted Profit"].sum()), fontsize=17, fontweight="bold", color=GREEN, va="center")
ax = fig.add_axes([0.65, 0.07, 0.32, 0.17]); style(ax)
t5 = fc.head(5)
x = range(len(t5))
ax.bar([i - 0.2 for i in x], t5["Typical Monthly Profit"] / 1e5, 0.4, color="#CBD5E1", label="Typical month")
ax.bar([i + 0.2 for i in x], t5["Predicted Profit"] / 1e5, 0.4, color=BLUE, label="Predicted")
ax.set_xticks(list(x)); ax.set_xticklabels(t5["Product"], fontsize=7); ax.legend(frameon=False, fontsize=7); ax.set_ylabel("₹ lakh", fontsize=7, color=GREY)
fig.savefig(ROOT / "docs" / "image3_final_dashboard_mockup.png", dpi=130)
print("Saved 3 images to docs/")
