from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from app_modules.config import BERT_DIM, OUTPUT_DIR, VISUAL_DIM
from app_modules.preprocess import clean_text, extract_numeric_features


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


def predict_price_with_confidence(catalog_content: str) -> dict[str, float | str]:
    model_lgb, model_xgb, tfidf, scaler, _ = load_artifacts()

    cleaned = clean_text(catalog_content)
    bert = np.zeros((1, BERT_DIM), dtype=np.float32)
    tfidf_features = tfidf.transform([cleaned]).toarray().astype(np.float32)
    visual = np.zeros((1, VISUAL_DIM), dtype=np.float32)
    numeric = scaler.transform(extract_numeric_features(catalog_content)).astype(np.float32)

    features = np.hstack([bert, tfidf_features, visual, numeric])
    pred_lgb_log = float(model_lgb.predict(features)[0])
    if model_xgb is not None:
        pred_xgb_log = float(model_xgb.predict(features)[0])
        blended_log = (pred_lgb_log + pred_xgb_log) / 2.0
        disagreement = abs(pred_lgb_log - pred_xgb_log)
        sigma = max(0.08, disagreement / 2.0)
        model_mode = "blended"
    else:
        blended_log = pred_lgb_log
        sigma = 0.20
        model_mode = "lgb-only"

    z_score = 1.28
    point_price = max(float(np.expm1(blended_log)), 0.01)
    lower_price = max(float(np.expm1(blended_log - z_score * sigma)), 0.01)
    upper_price = max(float(np.expm1(blended_log + z_score * sigma)), lower_price)

    return {
        "price": point_price,
        "low": lower_price,
        "high": upper_price,
        "model_mode": model_mode,
    }


def predict_price(catalog_content: str) -> float:
    result = predict_price_with_confidence(catalog_content)
    return float(result["price"])


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    if "catalog_content" not in df.columns:
        raise ValueError("Input CSV must contain a 'catalog_content' column.")

    output = df.copy()
    output["price"] = output["catalog_content"].fillna("").astype(str).apply(predict_price)

    if "sample_id" in output.columns:
        return output[["sample_id", "price"]]
    return output[["price"]]
