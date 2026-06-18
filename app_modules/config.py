from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
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

THEME_TOKENS = {
    "light": {
        "bg_1": "#f4efe6",
        "bg_2": "#f9f6f0",
        "bg_3": "#f7f1e8",
        "ink": "#202224",
        "ink_soft": "#5b6168",
        "heading": "#1f2933",
        "accent": "#0f766e",
        "accent_deep": "#115e59",
        "accent_warm": "#b45309",
        "glass": "rgba(255, 255, 255, 0.62)",
        "glass_border": "rgba(255, 255, 255, 0.8)",
        "surface": "rgba(255, 255, 255, 0.78)",
        "field_bg": "rgba(255, 255, 255, 0.84)",
        "field_border": "#d7dde3",
        "shadow": "0 14px 36px rgba(30, 35, 40, 0.12)",
    },
    "dark": {
        "bg_1": "#111417",
        "bg_2": "#171c20",
        "bg_3": "#1f262b",
        "ink": "#e8eef2",
        "ink_soft": "#a2adb8",
        "heading": "#f6fbff",
        "accent": "#14b8a6",
        "accent_deep": "#0f766e",
        "accent_warm": "#f59e0b",
        "glass": "rgba(16, 19, 22, 0.58)",
        "glass_border": "rgba(89, 100, 112, 0.5)",
        "surface": "rgba(24, 30, 35, 0.74)",
        "field_bg": "rgba(20, 25, 29, 0.92)",
        "field_border": "#3a454f",
        "shadow": "0 14px 34px rgba(0, 0, 0, 0.36)",
    },
}
