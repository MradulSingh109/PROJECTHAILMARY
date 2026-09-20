"""
=============================================================================
IBVAP - Intelligent Border Video Analytics Platform
Module: End-to-End ANPR Pipeline Coordinator
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

from src.anpr.plate_detector import PlateDetector, PlateDetection
from src.anpr.plate_preprocessor import PlatePreprocessor
from src.anpr.ocr import OCREngine, OCRResult
from src.anpr.plate_parser import PlateParser, ParsedPlate
from src.anpr.plate_tracker import PlateTracker, VehiclePlateRecord
from src.tracking.object_identity import TrackedObject


@dataclass
class ANPRResult:
    """Full end-to-end ANPR outcome for a single vehicle."""
    vehicle_track_id: Optional[int]
    plate_text: str
    formatted_plate: str
    is_valid: bool
    confidence: float
    is_locked: bool
    plate_bbox: Optional[List[float]] = None


class ANPRPipeline:
    """
    Coordinates plate localization, image enhancement, optical character recognition,
    regex parsing, and multi-frame temporal voting across tracked vehicles.
    """

    # Target vehicle classes eligible for ANPR
    VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck

    def __init__(
        self,
        plate_model_path: str = "models/anpr/plate_detector.pt",
        conf_threshold: float = 0.35,
        ocr_gpu: bool = True,
        min_votes_to_lock: int = 3,
    ):
        """
        :param plate_model_path: Path to custom trained license plate YOLO detector.
        :param conf_threshold: Plate detection confidence threshold.
        :param ocr_gpu: Whether to run EasyOCR on GPU.
        :param min_votes_to_lock: Number of consistent frames required to lock plate ID.
        """
        self.detector = PlateDetector(model_path=plate_model_path, conf_threshold=conf_threshold)
        self.preprocessor = PlatePreprocessor()
        self.ocr_engine = OCREngine(gpu=ocr_gpu, preprocess_input=True)
        self.parser = PlateParser()
        self.tracker = PlateTracker(min_votes_to_lock=min_votes_to_lock)

    def process_vehicles(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        frame_id: int,
    ) -> List[ANPRResult]:
        """
        Run ANPR on all detected/tracked vehicles in the current frame.

        :param frame: Full BGR video frame.
        :param tracked_objects: List of TrackedObject instances from ByteTrack.
        :param frame_id: Current frame sequence index.
        :return: List of ANPRResult objects.
        """
        results: List[ANPRResult] = []
        if frame is None or not tracked_objects:
            return results

        active_vehicle_ids = []

        for obj in tracked_objects:
            if obj.class_id not in self.VEHICLE_CLASS_IDS:
                continue

            active_vehicle_ids.append(obj.track_id)
            existing_record = self.tracker.get_plate(obj.track_id)

            # If already locked with high confidence, reuse locked result without expensive re-OCR
            if existing_record and existing_record.is_locked:
                results.append(
                    ANPRResult(
                        vehicle_track_id=obj.track_id,
                        plate_text=existing_record.best_plate_text,
                        formatted_plate=existing_record.best_plate_formatted,
                        is_valid=True,
                        confidence=existing_record.confidence,
                        is_locked=True,
                    )
                )
                continue

            # Step 1: Detect plate within vehicle bounding box
            plate_dets = self.detector.detect_plates(
                frame=frame,
                vehicle_bbox=obj.bbox,
                vehicle_track_id=obj.track_id,
            )

            if not plate_dets:
                continue

            # Take the highest confidence plate detection for this vehicle
            best_plate_det = max(plate_dets, key=lambda d: d.confidence)

            # Step 2: OCR on plate crop
            ocr_out = self.ocr_engine.read_plate(best_plate_det.plate_crop)
            if not ocr_out.text:
                continue

            # Step 3: Parse and Validate
            parsed = self.parser.parse(ocr_out.text, ocr_confidence=ocr_out.confidence)

            # Step 4: Update Temporal Track
            plate_rec = self.tracker.update(obj.track_id, parsed, frame_id=frame_id)

            if plate_rec:
                results.append(
                    ANPRResult(
                        vehicle_track_id=obj.track_id,
                        plate_text=plate_rec.best_plate_text,
                        formatted_plate=plate_rec.best_plate_formatted,
                        is_valid=parsed.is_valid,
                        confidence=plate_rec.confidence,
                        is_locked=plate_rec.is_locked,
                        plate_bbox=best_plate_det.bbox,
                    )
                )

        # Cleanup stale tracks
        self.tracker.cleanup_old_tracks(active_vehicle_ids)
        return results

    def draw_anpr_annotations(
        self,
        frame: np.ndarray,
        anpr_results: List[ANPRResult],
        tracked_objects: Optional[List[TrackedObject]] = None,
    ) -> np.ndarray:
        """
        Draw high-visibility ANPR license plate badges above tracked vehicles.
        """
        annotated = frame.copy()
        track_map = {obj.track_id: obj for obj in tracked_objects} if tracked_objects else {}

        for res in anpr_results:
            # 1. Draw small box around plate if coordinates available
            if res.plate_bbox is not None:
                px1, py1, px2, py2 = map(int, res.plate_bbox)
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 255), 2)

            # 2. Draw prominent Plate Badge above vehicle box
            if res.vehicle_track_id in track_map:
                v_obj = track_map[res.vehicle_track_id]
                vx1, vy1, vx2, vy2 = map(int, v_obj.bbox)

                badge_text = f"PLATE: {res.formatted_plate}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                font_thick = 2
                (tw, th), _ = cv2.getTextSize(badge_text, font, font_scale, font_thick)

                # Badge position: immediately below top bounding box or above
                bx1 = vx1
                by1 = max(0, vy1 - th - 12)
                bx2 = vx1 + tw + 10
                by2 = max(th + 10, vy1)

                # Background badge (Green if locked/valid, Amber if provisional)
                badge_bg = (0, 180, 0) if res.is_locked else (0, 140, 255)
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), badge_bg, -1)
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (255, 255, 255), 1)

                cv2.putText(
                    annotated,
                    badge_text,
                    (bx1 + 5, by2 - 5),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thick - 1,
                    lineType=cv2.LINE_AA,
                )

        return annotated


if __name__ == "__main__":
    print("[IBVAP-ANPR] Testing ANPRPipeline coordinator...")
    pipeline = ANPRPipeline()
    print("[IBVAP-ANPR] ANPRPipeline initialized successfully.")
