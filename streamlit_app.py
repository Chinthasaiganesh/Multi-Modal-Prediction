from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

BERT_DIM = 768
VISUAL_DIM = 2048
EXPECTED_NUMERIC_FEATURES = [
    "pack_size",
    "ounce_size",
    "total_ounce",
    "count_size",
    "fl_oz_size",
    "total_fl_oz",
    "is_gluten_free",
    "is_organic",
]


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"[^\w\s.,;:!?%\-\(\)]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def extract_numeric_features(item_text: str) -> np.ndarray:
    if not isinstance(item_text, str):
        item_text = ""

    pack_pattern = r"(?:Pack of|pack of|pack)\s+(\d+)|(\d+)(?:\s*-\s*|\s+)(?:pack|pk|count per order)"
    pack_match = re.search(pack_pattern, item_text, re.IGNORECASE)
    if not pack_match:
        pack_match = re.search(r"(\d+)\s+(?:per case|per pack|count\b|ct\b)", item_text, re.IGNORECASE)

    if pack_match:
        matched_value = next((group for group in pack_match.groups() if group), None)
        pack_size = int(matched_value) if matched_value else 1
    else:
        pack_size = 1

    ounce_match = re.search(r"(\d+\.?\d*)\s*(?:Ounce|Oz\b|oz\b)", item_text, re.IGNORECASE)
    ounce_size = float(ounce_match.group(1)) if ounce_match else 0.0

    fl_oz_match = re.search(r"(\d+\.?\d*)\s*(?:Fl Oz|Fluid Ounce|fl oz|fl\.oz)", item_text, re.IGNORECASE)
    fl_oz_size = float(fl_oz_match.group(1)) if fl_oz_match else 0.0

    count_match = re.search(r"(\d+\.?\d*)\s*(?:count|piece|tablet|capsule)s?", item_text, re.IGNORECASE)
    count_size = float(count_match.group(1)) if count_match else 0.0

    total_ounce = ounce_size * pack_size if ounce_size and pack_size > 1 else ounce_size
    total_fl_oz = fl_oz_size * pack_size if fl_oz_size and pack_size > 1 else fl_oz_size

    return np.array(
        [
            pack_size,
            ounce_size,
            total_ounce,
            count_size,
            fl_oz_size,
            total_fl_oz,
            1 if re.search(r"gluten[ -]free", item_text, re.IGNORECASE) else 0,
            1 if re.search(r"\borganic\b", item_text, re.IGNORECASE) else 0,
        ],
        dtype=np.float32,
    ).reshape(1, -1)


@st.cache_resource(show_spinner=False)
def load_artifacts() -> tuple:
    required = {
        "lgb_model": OUTPUT_DIR / "lgb_model.joblib",
        "tfidf_vectorizer": OUTPUT_DIR / "tfidf_vectorizer.joblib",
        "numeric_scaler": OUTPUT_DIR / "numeric_scaler.joblib",
    }

    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        pretty = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing trained artifacts in output/: {pretty}. Please run training first (src/Final.py)."
        )

    model_lgb = joblib.load(required["lgb_model"])
    tfidf = joblib.load(required["tfidf_vectorizer"])
    scaler = joblib.load(required["numeric_scaler"])

    xgb_model = None
    xgb_error = ""
    xgb_path = OUTPUT_DIR / "xgb_model.joblib"
    if xgb_path.exists():
        try:
            xgb_model = joblib.load(xgb_path)
        except Exception as exc:
            xgb_error = str(exc)

    return model_lgb, xgb_model, tfidf, scaler, xgb_error


def predict_price(catalog_content: str) -> float:
    model_lgb, model_xgb, tfidf, scaler, _ = load_artifacts()

    cleaned = clean_text(catalog_content)
    bert = np.zeros((1, BERT_DIM), dtype=np.float32)
    tfidf_features = tfidf.transform([cleaned]).toarray().astype(np.float32)
    visual = np.zeros((1, VISUAL_DIM), dtype=np.float32)
    numeric = scaler.transform(extract_numeric_features(catalog_content)).astype(np.float32)

    features = np.hstack([bert, tfidf_features, visual, numeric])
    pred_lgb = model_lgb.predict(features)
    if model_xgb is not None:
        pred_xgb = model_xgb.predict(features)
        blended = (pred_lgb + pred_xgb) / 2.0
    else:
        blended = pred_lgb

    price = float(np.expm1(blended)[0])
    return max(price, 0.01)


def prepare_catalog_text(
    product_name: str,
    product_description: str,
    product_details: str,
    catalog_content: str,
) -> str:
    if catalog_content.strip():
        return catalog_content.strip()

    lines = [
        f"Item Name: {product_name.strip()}" if product_name.strip() else "",
        f"Product Description: {product_description.strip()}" if product_description.strip() else "",
        product_details.strip(),
    ]
    combined = "\n".join(line for line in lines if line)
    return combined


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    if "catalog_content" not in df.columns:
        raise ValueError("Input CSV must contain a 'catalog_content' column.")

    output = df.copy()
    output["price"] = output["catalog_content"].fillna("").astype(str).apply(predict_price)

    if "sample_id" in output.columns:
        return output[["sample_id", "price"]]
    return output[["price"]]


def render_single_prediction() -> None:
    st.subheader("Single Product Prediction")

    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        product_name = st.text_input(
            "Product Name",
            value="Rani 14-Spice Mango Chutney, 10.5oz Glass Jar",
        )
        image_link = st.text_input(
            "Image Link (optional)",
            value="https://m.media-amazon.com/images/I/71hoAn78AWL.jpg",
        )
        product_description = st.text_area(
            "Product Description",
            value=(
                "Ready to eat Indian mango chutney made with natural ingredients. "
                "Vegan, gluten free, non-GMO, packed in a glass jar for pantry storage."
            ),
            height=110,
        )
        product_details = st.text_area(
            "Product Details",
            value=(
                "Bullet Point 1: Premium gourmet food grade chutney\n"
                "Bullet Point 2: Vegan, gluten free, all natural\n"
                "Value: 10.5\n"
                "Unit: Ounce\n"
                "Pack Size: 1"
            ),
            height=150,
        )
        manual_catalog = st.text_area(
            "Catalog Content (advanced, optional)",
            help="If provided, this value is used directly by the model.",
            height=130,
        )

    with col_right:
        st.markdown("### Preview")
        if image_link.strip():
            st.image(image_link.strip(), use_container_width=True)
        else:
            st.info("Add an image URL for preview.")

    if st.button("Predict Price", type="primary", use_container_width=True):
        model_input = prepare_catalog_text(product_name, product_description, product_details, manual_catalog)
        if not model_input.strip():
            st.error("Please provide product details or catalog content.")
            return

        try:
            with st.spinner("Predicting with prepared model artifacts..."):
                price = predict_price(model_input)
            st.success(f"Predicted Price: ${price:,.2f}")
            with st.expander("Model Input Used"):
                st.code(model_input)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def render_batch_prediction() -> None:
    st.subheader("Batch CSV Prediction")
    st.caption("Upload a CSV with a 'catalog_content' column. If 'sample_id' exists, output format will be sample_id,price.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")
    if uploaded is None:
        return

    try:
        df = pd.read_csv(uploaded)
        st.write("Input Preview")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("Run Batch Prediction", use_container_width=True):
            with st.spinner("Running predictions..."):
                out_df = predict_batch(df)
            st.success(f"Completed {len(out_df)} predictions.")
            st.dataframe(out_df.head(20), use_container_width=True)
            st.download_button(
                "Download Predictions CSV",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="submission.csv",
                mime="text/csv",
                use_container_width=True,
            )
    except Exception as exc:
        st.error(f"Batch prediction failed: {exc}")


def main() -> None:
    st.set_page_config(
        page_title="Multi-Modal Product Price Predictor",
        page_icon="💲",
        layout="wide",
    )

    st.title("Multi-Modal Product Price Prediction")
    st.caption("Streamlit frontend integrated with saved model artifacts from output/.")

    st.markdown(
        """
        This app uses:
        - lgb_model.joblib
        - xgb_model.joblib
        - tfidf_vectorizer.joblib
        - numeric_scaler.joblib

        Inference pipeline mirrors the prepared app logic using fused features:
        zero-BERT + TF-IDF + zero-visual + scaled numeric features.
        """
    )

    try:
        _, model_xgb, _, _, xgb_error = load_artifacts()
        if model_xgb is None and xgb_error:
            st.warning(
                "Loaded LightGBM model. XGBoost model is unavailable in this environment, "
                "so predictions run in LightGBM-only mode."
            )
        else:
            st.success("Model artifacts loaded successfully.")
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    with tab1:
        render_single_prediction()
    with tab2:
        render_batch_prediction()


if __name__ == "__main__":
    main()
