"""
=============================================================================
IBVAP - Intelligent Border Video Analytics Platform
Module: Custom YOLO License Plate Detector Engine
File: src/training/train_plate_yolo.py
=============================================================================
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from ultralytics import YOLO

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class PlateDetection:
    """Detection result for a localized license plate."""
    bbox: List[float]        # [x1, y1, x2, y2] in full-frame coordinates
    confidence: float
    plate_crop: np.ndarray   # Cropped BGR image of the license plate
    vehicle_track_id: Optional[int] = None


class PlateDetector:
    """
    License Plate Localization Engine using custom trained YOLO plate model.
    Supports full-frame detection and vehicle-ROI conditioned detection.
    """

    def __init__(
        self,
        model_path: str = "models/anpr/plate_detector.pt",
        fallback_model: str = "models/detection/yolo11n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        """
        :param model_path: Path to custom trained plate detector model (.pt).
        :param fallback_model: Fallback YOLO weights if custom model is not yet trained.
        :param conf_threshold: Detection confidence threshold.
        :param iou_threshold: NMS IoU threshold.
        :param device: Hardware device ('0', 'cuda', 'cpu', etc.).
        """
        # Resolve model path across common project locations
        resolved_path = model_path
        if not os.path.exists(resolved_path):
            candidates = [
                os.path.join(PROJECT_ROOT, model_path),
                os.path.join(PROJECT_ROOT, "models", "anpr", os.path.basename(model_path)),
                os.path.join(PROJECT_ROOT, fallback_model),
                os.path.join(PROJECT_ROOT, "models", "detection", os.path.basename(fallback_model)),
            ]
            for cand in candidates:
                cand_norm = os.path.normpath(cand)
                if os.path.exists(cand_norm):
                    resolved_path = cand_norm
                    break

        self.model_path = resolved_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        print(f"[IBVAP-ANPR] Loading Plate Detector model: '{self.model_path}'...")
        self.model = YOLO(self.model_path)
        print("[IBVAP-ANPR] Plate Detector model loaded successfully.")

    def detect_plates(
        self,
        frame: np.ndarray,
        vehicle_bbox: Optional[List[float]] = None,
        vehicle_track_id: Optional[int] = None,
    ) -> List[PlateDetection]:
        """
        Detect license plates in full frame or within a vehicle ROI crop.

        :param frame: Full BGR video frame.
        :param vehicle_bbox: Optional [vx1, vy1, vx2, vy2] bounding box of detected vehicle.
        :param vehicle_track_id: Persistent tracking ID of the vehicle.
        :return: List of PlateDetection instances.
        """
        if frame is None or frame.size == 0:
            return []

        frame_h, frame_w = frame.shape[:2]

        # Case 1: Vehicle-Conditioned Crop
        if vehicle_bbox is not None:
            vx1, vy1, vx2, vy2 = map(int, vehicle_bbox)
            vx1 = max(0, min(frame_w - 1, vx1))
            vy1 = max(0, min(frame_h - 1, vy1))
            vx2 = max(0, min(frame_w, vx2))
            vy2 = max(0, min(frame_h, vy2))

            crop_w = vx2 - vx1
            crop_h = vy2 - vy1
            if crop_w < 20 or crop_h < 20:
                return []

            vehicle_crop = frame[vy1:vy2, vx1:vx2]
            results = self.model.predict(
                source=vehicle_crop,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )

            plate_detections: List[PlateDetection] = []
            if not results or results[0].boxes is None:
                return plate_detections

            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                bx1, by1, bx2, by2 = map(int, box)
                # Map coordinates back to full frame
                global_x1 = vx1 + bx1
                global_y1 = vy1 + by1
                global_x2 = vx1 + bx2
                global_y2 = vy1 + by2

                plate_crop = frame[global_y1:global_y2, global_x1:global_x2]
                if plate_crop.size == 0:
                    continue

                plate_detections.append(
                    PlateDetection(
                        bbox=[float(global_x1), float(global_y1), float(global_x2), float(global_y2)],
                        confidence=round(float(conf), 4),
                        plate_crop=plate_crop,
                        vehicle_track_id=vehicle_track_id,
                    )
                )

            return plate_detections

        # Case 2: Full-frame Scan
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        plate_detections: List[PlateDetection] = []
        if not results or results[0].boxes is None:
            return plate_detections

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):
            bx1, by1, bx2, by2 = map(int, box)
            bx1 = max(0, min(frame_w - 1, bx1))
            by1 = max(0, min(frame_h - 1, by1))
            bx2 = max(0, min(frame_w, bx2))
            by2 = max(0, min(frame_h, by2))

            plate_crop = frame[by1:by2, bx1:bx2]
            if plate_crop.size == 0:
                continue

            plate_detections.append(
                PlateDetection(
                    bbox=[float(bx1), float(by1), float(bx2), float(by2)],
                    confidence=round(float(conf), 4),
                    plate_crop=plate_crop,
                    vehicle_track_id=vehicle_track_id,
                )
            )

        return plate_detections


if __name__ == "__main__":
    print("[IBVAP-ANPR] Running PlateDetector standalone self-test...")
    detector = PlateDetector()
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    res = detector.detect_plates(dummy_frame)
    print(f"[IBVAP-ANPR] Self-test complete. Detected plates: {len(res)}")