from __future__ import annotations

import base64
import mimetypes

import streamlit as st

from app_modules.config import BASE_DIR


@st.cache_data(show_spinner=False)
def get_landing_image_data_uri() -> str | None:
    image_path = BASE_DIR / "IMG_1243.JPG"
    if not image_path.exists():
        return None

    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "image/jpeg"

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def render_landing_page() -> None:
    landing_image = (
        get_landing_image_data_uri()
        or "https://images.unsplash.com/photo-1556740749-887f6717d7e4?auto=format&fit=crop&w=1600&q=80"
    )
    landing_style = """
        <style>
        .landing-hero {
            position: relative;
            width: 100vw;
            height: 100vh;
            margin-left: calc(-50vw + 50%);
            margin-top: -5rem;
            background: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)), url('__LANDING_IMAGE__');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }

        .landing-overlay {
            text-align: center;
            max-width: 700px;
            z-index: 10;
            animation: fadeInUp 0.8s ease-out;
        }

        .landing-overlay h1 {
            font-family: 'Fraunces', serif;
            font-size: clamp(2.5rem, 6vw, 4.5rem);
            margin-bottom: 1.5rem;
            line-height: 1.1;
            color: #f7fafc;
            text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.5);
            font-weight: 700;
        }

        .landing-overlay p {
            font-size: clamp(1rem, 2vw, 1.3rem);
            color: #e8eef2;
            margin-bottom: 2.5rem;
            line-height: 1.7;
            text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.5);
            font-weight: 500;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (max-width: 900px) {
            .landing-hero {
                height: auto;
                min-height: 100vh;
                padding: 3rem 1.5rem;
            }

            .landing-overlay h1 {
                font-size: clamp(1.8rem, 5vw, 3rem);
                margin-bottom: 1rem;
            }

            .landing-overlay p {
                font-size: clamp(0.9rem, 1.8vw, 1.1rem);
                margin-bottom: 2rem;
            }
        }
        </style>
    """
    st.markdown(landing_style.replace("__LANDING_IMAGE__", landing_image), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="landing-hero">
            <div class="landing-overlay">
                <h1><span style="color: #ffffff;">Product</span><br>Price Prediction</h1>
                <p>
                    Leverage advanced machine learning to predict accurate product prices instantly.
                    Our multi-modal approach combines semantic understanding with engineered features
                    for client-ready pricing estimates.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        .stButton > button:nth-of-type(1) {
            background: linear-gradient(125deg, #0f766e, #115e59) !important;
            color: #f7fafc !important;
            border: none !important;
            border-radius: 999px !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
            padding: 1rem 3rem !important;
            box-shadow: 0 10px 30px rgba(15, 118, 110, 0.5) !important;
            animation: fadeInUp 0.8s ease-out 0.3s both !important;
        }

        .stButton > button:nth-of-type(1):hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 15px 40px rgba(15, 118, 110, 0.7) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        if st.button("Start Predicting →", key="cta_button", use_container_width=True):
            st.session_state.page = "prediction"
            st.rerun()
