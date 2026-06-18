from __future__ import annotations

import pandas as pd
import streamlit as st

from app_modules.config import EXPECTED_NUMERIC_FEATURES
from app_modules.inference import predict_batch, predict_price_with_confidence
from app_modules.preprocess import extract_numeric_features, prepare_catalog_text


def render_numeric_preview(catalog_content: str) -> None:
    numeric = extract_numeric_features(catalog_content).flatten().tolist()
    preview_df = pd.DataFrame(
        {
            "feature": EXPECTED_NUMERIC_FEATURES,
            "value": numeric,
        }
    )
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("#### Feature Extraction Preview")
    st.caption("Live view of engineered numeric signals parsed from your current input.")
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_single_prediction() -> None:
    st.subheader("Single Product Prediction")
    st.caption("Fill core product information to generate a polished client-facing price estimate.")

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
                result = predict_price_with_confidence(model_input)
            st.markdown(
                f"""
                <div class="prediction-output">
                    <p class="prediction-title">Prediction Output</p>
                    <p class="prediction-price">${result['price']:,.2f}</p>
                    <!--<p class="prediction-band">Confidence Band: ${result['low']:,.2f} - ${result['high']:,.2f} ({result['model_mode']})</p>-->
                </div>
                """,
                unsafe_allow_html=True,
            )
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
