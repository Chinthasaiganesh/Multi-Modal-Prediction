from __future__ import annotations

import re

import numpy as np


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
    return "\n".join(line for line in lines if line)
