from pathlib import Path

import pandas as pd
import streamlit as st

from models.classification_model import classify_products, train_classifier
from models.regression_model import train_regression
from preprocessing.data_processor import (DataError, auto_map_columns, clean_data,
                                          load_file, missing_required)
from views import overview, product_analysis, prediction

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_sales_data.csv"
PAGES = {"Overview": overview, "Product Analysis": product_analysis,
         "Profit Prediction": prediction}

st.set_page_config(page_title="AI Profit Predictor", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def train_all(df: pd.DataFrame):
    """Train both models once per dataset."""
    reg = train_regression(df)
    clf = train_classifier(df)
    return reg, clf, classify_products(clf, df)


def get_data():
    """Return (clean dataframe, is_sample) or stop with a helpful message."""
    st.sidebar.subheader("Data")
    upload = st.sidebar.file_uploader("Upload sales data", type=["csv", "xlsx", "xls"],
                                      help="Needs: Product, Quantity, Selling Price, Cost Price, Date.")
    path = upload if upload is not None else SAMPLE_PATH
    try:
        raw = load_file(path)
        mapping = auto_map_columns(raw)
        missing = missing_required(mapping)
        if missing:
            st.sidebar.warning("Some columns weren't recognised. Match them below.")
            options = ["-"] + list(raw.columns)
            for std in missing:
                chosen = st.sidebar.selectbox(std, options, key=f"map_{std}")
                if chosen != "-":
                    mapping[std] = chosen
        return clean_data(raw, mapping), upload is None
    except DataError as e:
        st.error(str(e))
        st.stop()


def main():
    st.sidebar.title("Profit Predictor")
    page = st.sidebar.radio("Navigation", list(PAGES), label_visibility="collapsed")
    df, is_sample = get_data()

    if is_sample:
        st.sidebar.info("Showing sample data. Upload your own file to replace it.")
        st.warning("You're viewing SAMPLE data (synthetic, for demonstration).")
    dropped = df.attrs.get("rows_dropped", 0)
    if dropped:
        st.sidebar.caption(f"{dropped} unusable rows were removed.")

    try:
        reg, clf, product_table = train_all(df)
    except DataError as e:
        st.error(str(e))
        st.stop()

    ctx = {"df": df, "reg": reg, "clf": clf, "product_table": product_table, "is_sample": is_sample}
    PAGES[page].render(ctx)


main()
