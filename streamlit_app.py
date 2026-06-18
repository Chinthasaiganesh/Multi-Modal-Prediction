from __future__ import annotations

import streamlit as st

from app_modules.inference import load_artifacts
from app_modules.ui_landing import render_landing_page
from app_modules.ui_prediction import render_batch_prediction, render_single_prediction
from app_modules.ui_theme import inject_custom_theme, render_header, render_theme_switcher


def main() -> None:
    st.set_page_config(
        page_title="Multi Model Price Prediction",
        page_icon="💲",
        layout="wide",
    )

    if "page" not in st.session_state:
        st.session_state.page = "landing"
    if "theme_name" not in st.session_state:
        st.session_state.theme_name = "dark"

    selected_theme = render_theme_switcher()
    inject_custom_theme(selected_theme)

    if st.session_state.page == "landing":
        render_landing_page()
        return

    render_header()

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
