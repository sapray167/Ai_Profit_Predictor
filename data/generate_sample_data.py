"""Creates SAMPLE sales data (synthetic, for demo only). Run: python data/generate_sample_data.py"""
import numpy as np
import pandas as pd

# product: (base selling price, base cost price, typical monthly units)
PRODUCTS = {
    "Laptop": (55000, 42000, 40), "Mobile": (22000, 17000, 90),
    "Monitor": (12000, 9800, 60), "Keyboard": (1500, 1200, 150),
    "Mouse": (700, 640, 300), "Headphones": (2500, 1800, 120),
    "Printer": (9000, 7600, 35), "Webcam": (3000, 2200, 70),
    "Tablet": (28000, 22500, 50), "Router": (2200, 1700, 90),
    "Cables": (300, 285, 500), "Speaker": (3500, 2600, 80),
}

def generate(seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for month in pd.date_range("2024-01-01", periods=24, freq="MS"):
        season = 1 + 0.15 * np.sin(month.month / 12 * 2 * np.pi)
        for name, (price, cost, units) in PRODUCTS.items():
            qty = max(1, int(rng.normal(units * season, units * 0.15)))
            sell = round(price * rng.normal(1, 0.03), 2)
            unit_cost = round(cost * rng.normal(1, 0.02), 2)
            discount = float(rng.choice([0, 0.05, 0.10], p=[0.5, 0.3, 0.2]))
            profit = round((sell * (1 - discount) - unit_cost) * qty, 2)
            rows.append([month.date(), name, qty, sell, unit_cost, discount, profit])
    return pd.DataFrame(rows, columns=["Date", "Product", "Quantity", "Selling Price",
                                       "Cost Price", "Discount", "Profit"])

def generate_plan(seed=7):
    """SAMPLE next-month plan: planned units near each product's usual volume."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"Product": list(PRODUCTS),
                         "Planned Quantity": [int(u * rng.uniform(0.9, 1.25)) for _, _, u in PRODUCTS.values()]})


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/sample_sales_data.csv", index=False)
    print(f"Saved {len(df)} rows to data/sample_sales_data.csv")
    generate_plan().to_csv("data/sample_future_plan.csv", index=False)
    print("Saved data/sample_future_plan.csv")
