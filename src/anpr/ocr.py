"""
=============================================================================
IBVAP - Intelligent Border Video Analytics Platform
Module: OCR Engine (Optical Character Recognition for License Plates)
=============================================================================
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anpr.plate_preprocessor import PlatePreprocessor


@dataclass
class OCRResult:
    """Raw result emitted by OCR Engine for a license plate."""
    text: str
    confidence: float
    char_boxes: Optional[List[Tuple[int, int, int, int]]] = None


class OCREngine:
    """
    License Plate Optical Character Recognition Engine.
    Uses EasyOCR with optimized character whitelisting and image preprocessing.
    """

    # Whitelist characters allowed on vehicle registration plates
    PLATE_CHAR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        gpu: bool = True,
        preprocess_input: bool = True,
    ):
        """
        :param languages: List of languages for OCR (defaults to ['en']).
        :param gpu: Whether to enable GPU acceleration if available.
        :param preprocess_input: Whether to apply PlatePreprocessor prior to OCR.
        """
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.preprocess_input = preprocess_input
        self.preprocessor = PlatePreprocessor()
        self._reader = None  # Lazy-load reader

    def _get_reader(self):
        """Lazy load EasyOCR reader."""
        if self._reader is None:
            try:
                import easyocr
                print(f"[IBVAP-ANPR] Initializing EasyOCR Reader (Languages: {self.languages}, GPU: {self.gpu})...")
                self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
                print("[IBVAP-ANPR] EasyOCR Reader initialized successfully.")
            except ImportError:
                print("[IBVAP-ANPR] WARNING: easyocr not installed. OCR will run in fallback mock mode.")
                self._reader = False
        return self._reader

    def read_plate(self, plate_crop: np.ndarray) -> OCRResult:
        """
        Perform OCR on a cropped license plate image.

        :param plate_crop: BGR image crop of the license plate.
        :return: OCRResult with extracted text and confidence.
        """
        if plate_crop is None or plate_crop.size == 0:
            return OCRResult(text="", confidence=0.0)

        # Apply image enhancement if enabled
        if self.preprocess_input:
            processed_crop = self.preprocessor.preprocess_for_ocr(plate_crop)
        else:
            processed_crop = plate_crop

        reader = self._get_reader()
        if not reader:
            # Fallback if reader failed to load
            return OCRResult(text="", confidence=0.0)

        try:
            # Run EasyOCR with character allowlist
            results = reader.readtext(
                processed_crop,
                allowlist=self.PLATE_CHAR_ALLOWLIST,
                detail=1,
                paragraph=False,
            )

            if not results:
                # Retry on deskewed raw crop if processed yielded nothing
                deskewed = self.preprocessor.deskew_plate(plate_crop)
                results = reader.readtext(
                    deskewed,
                    allowlist=self.PLATE_CHAR_ALLOWLIST,
                    detail=1,
                    paragraph=False,
                )

            if not results:
                return OCRResult(text="", confidence=0.0)

            # Combine all detected text fragments (e.g. state code + numbers)
            full_text_parts = []
            confidences = []

            for bbox, text, conf in results:
                cleaned = text.strip().upper().replace(" ", "")
                if cleaned:
                    full_text_parts.append(cleaned)
                    confidences.append(float(conf))

            combined_text = "".join(full_text_parts)
            avg_confidence = float(np.mean(confidences)) if confidences else 0.0

            return OCRResult(
                text=combined_text,
                confidence=round(avg_confidence, 4),
            )

        except Exception as e:
            print(f"[IBVAP-ANPR] OCR error: {e}")
            return OCRResult(text="", confidence=0.0)


if __name__ == "__main__":
    print("[IBVAP-ANPR] Testing OCREngine...")
    engine = OCREngine(gpu=False)
    dummy_crop = np.zeros((64, 192, 3), dtype=np.uint8)
    cv2.putText(dummy_crop, "DL01AB1234", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    res = engine.read_plate(dummy_crop)
    print(f"[IBVAP-ANPR] OCR Test Result: '{res.text}' (Conf: {res.confidence})")
