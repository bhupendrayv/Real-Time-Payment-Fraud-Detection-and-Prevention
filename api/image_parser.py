"""Payment image OCR parser for FraudShield API.

Extracts transaction fields (Amount, Time, card details) from uploaded payment
screenshots or receipts using pytesseract + PIL image preprocessing. Gracefully
degrades when Tesseract binary is unavailable.
"""
from __future__ import annotations

import io
import logging
import re
from typing import Any

logger = logging.getLogger("fraudshield.image_parser")

# Try to import OCR dependencies; set flag so endpoint can report capability
try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter

    # Common Tesseract Windows install path
    import os
    _WIN_TESS_PATHS = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for _p in _WIN_TESS_PATHS:
        if os.path.exists(_p):
            pytesseract.pytesseract.tesseract_cmd = _p
            break

    # Quick capability check
    pytesseract.get_tesseract_version()
    OCR_AVAILABLE = True
    logger.info("Tesseract OCR is available — image parsing enabled.")
except Exception:
    OCR_AVAILABLE = False
    logger.warning(
        "Tesseract OCR not available. /predict/from-image will use field extraction "
        "from filenames/metadata only. Install Tesseract to enable full OCR."
    )


def _preprocess_image(image_bytes: bytes):
    """Enhance image contrast and sharpness for better OCR accuracy."""
    from PIL import Image, ImageEnhance, ImageFilter
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # Grayscale
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    return img


def _extract_amount(text: str) -> float | None:
    """Parse currency amount from raw OCR text."""
    # Match common formats: ₹1,234.56 / $1234.56 / USD 500 / Amount: 99.00
    patterns = [
        r"(?:amount|total|paid|payment|price|charge)[:\s]*[₹$€£]?\s*([\d,]+\.?\d*)",
        r"[₹$€£]\s*([\d,]+\.?\d*)",
        r"\b([\d,]+\.\d{2})\b",  # Any decimal number with 2 dp
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val_str = m.group(1).replace(",", "")
            try:
                val = float(val_str)
                if 0.01 <= val <= 1_000_000:
                    return val
            except ValueError:
                continue
    return None


def _extract_time_seconds(text: str) -> float | None:
    """Convert extracted time/date to seconds-since-epoch offset (simplified)."""
    # Look for HH:MM or HH:MM:SS patterns and convert to seconds from midnight
    m = re.search(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b", text)
    if m:
        h = int(m.group(1))
        mn = int(m.group(2))
        s = int(m.group(3)) if m.group(3) else 0
        if 0 <= h < 24 and 0 <= mn < 60:
            return float(h * 3600 + mn * 60 + s)
    return None


def _extract_card_last4(text: str) -> str | None:
    """Extract last 4 digits of card number if visible."""
    m = re.search(r"\*{4}[\s-]?(\d{4})\b|\bx{4}[\s-]?(\d{4})\b|ending[\s]+(\d{4})", text, re.IGNORECASE)
    if m:
        return m.group(1) or m.group(2) or m.group(3)
    return None


def _extract_merchant(text: str) -> str | None:
    """Heuristically extract merchant name."""
    m = re.search(r"(?:merchant|payee|to|paid to)[:\s]+([A-Za-z0-9 &.'-]{3,40})", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def parse_payment_image(image_bytes: bytes) -> dict[str, Any]:
    """
    Parse a payment receipt/screenshot image and extract transaction fields.

    Returns a dict with:
      - ocr_text: raw extracted text (or None if OCR unavailable)
      - extracted: dict of parsed fields (amount, time_seconds, card_last4, merchant)
      - ocr_available: bool indicating if Tesseract was used
    """
    result: dict[str, Any] = {
        "ocr_available": OCR_AVAILABLE,
        "ocr_text": None,
        "extracted": {
            "amount": None,
            "time_seconds": None,
            "card_last4": None,
            "merchant": None,
        },
    }

    if not OCR_AVAILABLE:
        return result

    try:
        img = _preprocess_image(image_bytes)
        raw_text = pytesseract.image_to_string(img, config="--psm 6")
        result["ocr_text"] = raw_text.strip()

        result["extracted"]["amount"] = _extract_amount(raw_text)
        result["extracted"]["time_seconds"] = _extract_time_seconds(raw_text)
        result["extracted"]["card_last4"] = _extract_card_last4(raw_text)
        result["extracted"]["merchant"] = _extract_merchant(raw_text)

        logger.info(
            f"OCR extracted — Amount: {result['extracted']['amount']}, "
            f"Time: {result['extracted']['time_seconds']}, "
            f"Merchant: {result['extracted']['merchant']}"
        )
    except Exception as e:
        logger.error(f"OCR parsing failed: {e}")
        result["error"] = str(e)

    return result
