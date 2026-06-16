import json
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
STATIC_DIR = BASE_DIR / "web"

BERT_DIM = 768
VISUAL_DIM = 2048


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"[^\w\s.,;:!?%\-\(\)]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def extract_numeric_features(item_text):
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


class PricePredictor:
    def __init__(self):
        missing = [
            path
            for path in (
                OUTPUT_DIR / "lgb_model.joblib",
                OUTPUT_DIR / "xgb_model.joblib",
                OUTPUT_DIR / "tfidf_vectorizer.joblib",
                OUTPUT_DIR / "numeric_scaler.joblib",
            )
            if not path.exists()
        ]
        if missing:
            names = ", ".join(str(path.relative_to(BASE_DIR)) for path in missing)
            raise FileNotFoundError(f"Missing trained artifacts: {names}. Run src/Final.py first.")

        self.model_a = joblib.load(OUTPUT_DIR / "lgb_model.joblib")
        self.model_b = joblib.load(OUTPUT_DIR / "xgb_model.joblib")
        self.tfidf = joblib.load(OUTPUT_DIR / "tfidf_vectorizer.joblib")
        self.scaler = joblib.load(OUTPUT_DIR / "numeric_scaler.joblib")

    def predict(self, catalog_content):
        cleaned = clean_text(catalog_content)
        bert = np.zeros((1, BERT_DIM), dtype=np.float32)
        tfidf = self.tfidf.transform([cleaned]).toarray().astype(np.float32)
        visual = np.zeros((1, VISUAL_DIM), dtype=np.float32)
        numeric = self.scaler.transform(extract_numeric_features(catalog_content)).astype(np.float32)
        features = np.hstack([bert, tfidf, visual, numeric])

        pred_a = self.model_a.predict(features)
        pred_b = self.model_b.predict(features)
        price = float(np.expm1((pred_a + pred_b) / 2.0)[0])
        return max(price, 0.01)


PREDICTOR = PricePredictor()


class AppHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.path = "/web/index.html"
        elif parsed.path.startswith("/static/"):
            self.path = "/web/" + parsed.path.removeprefix("/static/")
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/predict":
            self.send_error(404, "Not found")
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        content_type = self.headers.get("Content-Type", "")

        try:
            if "application/json" in content_type:
                payload = json.loads(body or "{}")
            else:
                payload = {key: values[0] for key, values in parse_qs(body).items()}

            image_link = payload.get("image_link", "").strip()
            description = payload.get("description", "").strip()
            catalog_content = payload.get("catalog_content", "").strip()

            if not image_link:
                self._json({"ok": False, "error": "Product image link is required."}, status=400)
                return
            if not description:
                self._json({"ok": False, "error": "Product description is required."}, status=400)
                return
            if not catalog_content:
                self._json({"ok": False, "error": "Catalog content is required."}, status=400)
                return

            price = PREDICTOR.predict(catalog_content)
            self._json({"ok": True, "price": round(price, 2)})
        except Exception as exc:
            self._json({"ok": False, "error": str(exc)}, status=500)

    def _json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), AppHandler)
    print(f"Price predictor running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
