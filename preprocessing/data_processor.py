"""Load, validate and clean uploaded sales data."""
import pandas as pd

REQUIRED = ["Product", "Quantity", "Selling Price", "Cost Price", "Date"]
OPTIONAL = ["Discount", "Profit"]

# Column names we recognise automatically (lowercase, no spaces/underscores)
ALIASES = {
    "Product": ["product", "productname", "item", "itemname"],
    "Quantity": ["quantity", "qty", "unitssold", "units", "quantitysold"],
    "Selling Price": ["sellingprice", "price", "unitprice", "saleprice", "sp"],
    "Cost Price": ["costprice", "cost", "unitcost", "costperunit", "cp"],
    "Date": ["date", "orderdate", "saledate", "month"],
    "Discount": ["discount", "discountpercent", "disc"],
    "Profit": ["profit", "netprofit"],
}


class DataError(Exception):
    """Raised with a user-friendly message when the dataset can't be used."""


def load_file(file) -> pd.DataFrame:
    """Read a CSV or Excel file (path or Streamlit upload)."""
    name = getattr(file, "name", str(file)).lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(file)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            raise DataError("Unsupported file type. Upload a .csv, .xlsx or .xls file.")
    except DataError:
        raise
    except Exception:
        raise DataError("The file couldn't be read. Check that it isn't empty or corrupted.")
    if df.empty:
        raise DataError("The file has no rows.")
    return df


def auto_map_columns(df: pd.DataFrame) -> dict:
    """Guess which uploaded column matches each standard column."""
    simple = {c: c.lower().replace(" ", "").replace("_", "") for c in df.columns}
    mapping = {}
    for std, names in ALIASES.items():
        for col, key in simple.items():
            if key in names and col not in mapping.values():
                mapping[std] = col
                break
    return mapping


def missing_required(mapping: dict) -> list:
    return [c for c in REQUIRED if c not in mapping]


def clean_data(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Return a clean DataFrame with standard column names plus Revenue, Cost, Profit, Month."""
    missing = missing_required(mapping)
    if missing:
        raise DataError("Missing required columns: " + ", ".join(missing) +
                        ". Map them to your file's columns.")

    out = pd.DataFrame({std: df[col] for std, col in mapping.items()})
    out["Product"] = out["Product"].astype(str).str.strip()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for col in ["Quantity", "Selling Price", "Cost Price", "Discount", "Profit"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Missing optional values
    if "Discount" not in out:
        out["Discount"] = 0.0
    out["Discount"] = out["Discount"].fillna(0)
    if out["Discount"].max() > 1:          # values like 10 mean 10%
        out["Discount"] = out["Discount"] / 100

    # Drop rows we can't use
    before = len(out)
    out = out.dropna(subset=["Product", "Date", "Quantity", "Selling Price", "Cost Price"])
    out = out[(out["Quantity"] > 0) & (out["Selling Price"] > 0) & (out["Cost Price"] > 0)]
    out = out[out["Product"].ne("") & out["Product"].ne("nan")]
    if out.empty:
        raise DataError("No valid rows left after cleaning. Check dates and numeric columns.")

    # Revenue / Cost / Profit
    out["Revenue"] = out["Selling Price"] * (1 - out["Discount"]) * out["Quantity"]
    out["Cost"] = out["Cost Price"] * out["Quantity"]
    if "Profit" not in out:
        out["Profit"] = out["Revenue"] - out["Cost"]
    else:
        out["Profit"] = out["Profit"].fillna(out["Revenue"] - out["Cost"])

    out["Month"] = out["Date"].dt.to_period("M").dt.to_timestamp()
    out = out.reset_index(drop=True)
    out.attrs["rows_dropped"] = before - len(out)
    return out


def prepare(file) -> pd.DataFrame:
    """Convenience: load + auto-map + clean in one call."""
    df = load_file(file)
    return clean_data(df, auto_map_columns(df))


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """One row per product per month, so models see monthly totals even if
    the uploaded file has one row per transaction."""
    d = df.assign(Gross=df["Selling Price"] * df["Quantity"])
    g = d.groupby(["Product", "Month"], as_index=False).agg(
        Quantity=("Quantity", "sum"), Gross=("Gross", "sum"), Revenue=("Revenue", "sum"),
        Cost=("Cost", "sum"), Profit=("Profit", "sum"))
    g["Selling Price"] = g["Gross"] / g["Quantity"]
    g["Cost Price"] = g["Cost"] / g["Quantity"]
    g["Discount"] = (1 - g["Revenue"] / g["Gross"]).clip(lower=0, upper=0.95)
    return g.drop(columns="Gross")


PLAN_ALIASES = {
    "Product": ["product", "productname", "item", "itemname"],
    "Planned Quantity": ["plannedquantity", "plannedqty", "plannedunits", "quantity", "qty", "units", "planned"],
}


def load_plan(file) -> pd.DataFrame:
    """Read a next-month sales plan: Product + Planned Quantity."""
    raw = load_file(file)
    keys = {c: c.lower().replace(" ", "").replace("_", "") for c in raw.columns}
    found = {}
    for std, names in PLAN_ALIASES.items():
        for col, key in keys.items():
            if key in names and col not in found.values():
                found[std] = col
                break
    if len(found) < 2:
        raise DataError("The sales plan needs two columns: Product and Planned Quantity.")
    plan = pd.DataFrame({"Product": raw[found["Product"]].astype(str).str.strip(),
                         "Planned Quantity": pd.to_numeric(raw[found["Planned Quantity"]], errors="coerce")})
    plan = plan.dropna()
    plan = plan[(plan["Planned Quantity"] > 0) & plan["Product"].ne("") & plan["Product"].ne("nan")]
    if plan.empty:
        raise DataError("The sales plan has no valid rows. Check the product names and quantities.")
    return plan.groupby("Product", as_index=False)["Planned Quantity"].sum()
