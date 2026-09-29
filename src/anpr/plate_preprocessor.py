"""
=============================================================================
IBVAP - Intelligent Border Video Analytics Platform
Module: Plate Preprocessing & Image Enhancement for OCR
=============================================================================
"""

from typing import Optional, Tuple
import cv2
import numpy as np


class PlatePreprocessor:
    """
    Image enhancement pipeline to maximize OCR legibility under adverse surveillance
    conditions (glare, night-vision, low resolution, motion blur, and tilt).
    """

    def __init__(
        self,
        target_height: int = 64,
        target_width: int = 192,
        apply_clahe: bool = True,
        apply_bilateral: bool = True,
    ):
        """
        :param target_height: Normalized height for OCR input.
        :param target_width: Normalized width for OCR input.
        :param apply_clahe: Enhance contrast using CLAHE.
        :param apply_bilateral: Denoise while preserving sharp character edges.
        """
        self.target_height = target_height
        self.target_width = target_width
        self.apply_clahe = apply_clahe
        self.apply_bilateral = apply_bilateral
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    def preprocess_for_ocr(
        self,
        plate_crop: np.ndarray,
        return_grayscale: bool = False,
    ) -> np.ndarray:
        """
        Preprocess and enhance a cropped license plate image for OCR ingestion.

        :param plate_crop: BGR or Grayscale crop of license plate.
        :param return_grayscale: If True, returns single channel image, else 3-channel BGR.
        :return: Enhanced image array.
        """
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        h, w = plate_crop.shape[:2]
        if h < 5 or w < 10:
            return plate_crop

        # 1. Convert to grayscale if BGR
        if len(plate_crop.shape) == 3 and plate_crop.shape[2] == 3:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_crop.copy()

        # 2. Resize to optimal standardized dimensions for OCR
        resized = cv2.resize(gray, (self.target_width, self.target_height), interpolation=cv2.INTER_CUBIC)

        # 3. Apply Bilateral Filter (Smooths background while keeping character borders crisp)
        if self.apply_bilateral:
            filtered = cv2.bilateralFilter(resized, d=9, sigmaColor=75, sigmaSpace=75)
        else:
            filtered = cv2.GaussianBlur(resized, (3, 3), 0)

        # 4. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if self.apply_clahe:
            enhanced = self.clahe.apply(filtered)
        else:
            enhanced = filtered

        # 5. Mild unsharp masking to sharpen characters
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

        if return_grayscale:
            return sharpened

        # Convert back to 3-channel for OCR engines that expect RGB/BGR
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    def deskew_plate(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Correct minor perspective skew/rotation angle of the license plate.
        """
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=10)

        if lines is None or len(lines) == 0:
            return plate_crop

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 30.0:  # Only consider slight tilt
                angles.append(angle)

        if not angles:
            return plate_crop

        median_angle = np.median(angles)
        if abs(median_angle) < 1.0:
            return plate_crop

        (h, w) = plate_crop.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        deskewed = cv2.warpAffine(
            plate_crop, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return deskewed


if __name__ == "__main__":
    print("[IBVAP-ANPR] Testing PlatePreprocessor...")
    preprocessor = PlatePreprocessor()
    dummy_crop = np.random.randint(0, 255, (40, 120, 3), dtype=np.uint8)
    processed = preprocessor.preprocess_for_ocr(dummy_crop)
    print(f"[IBVAP-ANPR] Preprocessor output shape: {processed.shape}")
