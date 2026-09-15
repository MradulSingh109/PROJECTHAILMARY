"""
=============================================================================
IBVAP - Intelligent Border Video Analytics Platform
Module: Object Detection (Stage 1)
Model: YOLO11 (yolo11n.pt)
=============================================================================
"""

import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import cv2
from ultralytics import YOLO


@dataclass
class DetectionResult:
    """Structured representation of a single detected object."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] in pixel coordinates

    def to_dict(self) -> Dict:
        """Convert detection result to standard dictionary."""
        return asdict(self)


class YOLO11Detector:
    """
    YOLO11 Object Detector for Border Surveillance.
    Supports detecting humans, vehicles, and other relevant surveillance objects.
    """

    # Relevant surveillance class subsets from standard 80 COCO classes
    DEFAULT_SURVEILLANCE_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        21: "bear",
    }

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[int]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the YOLO11 detector.

        :param model_path: Path to YOLO weights (.pt) or standard model identifier (e.g. 'yolo11n.pt').
        :param conf_threshold: Minimum confidence score to filter detections (0.0 to 1.0).
        :param iou_threshold: NMS IoU threshold (0.0 to 1.0).
        :param target_classes: List of class IDs to filter detections. If None, detects all classes.
        :param device: Inference device ('0', 'cuda', 'cpu', etc.). If None, auto-selects.
        """
        # Resolve model path across common project locations
        resolved_path = model_path
        if not os.path.exists(resolved_path):
            candidates = [
                os.path.join(os.path.dirname(__file__), os.path.basename(model_path)),
                os.path.join(os.path.dirname(__file__), "..", "detection", os.path.basename(model_path)),
                os.path.join(os.path.dirname(__file__), "..", model_path),
                os.path.join(os.path.dirname(__file__), "..", "..", "models", "detection", os.path.basename(model_path)),
            ]
            for cand in candidates:
                cand_norm = os.path.normpath(cand)
                if os.path.exists(cand_norm):
                    resolved_path = cand_norm
                    break

        self.model_path = resolved_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes
        self.device = device

        print(f"[IBVAP-Detector] Loading YOLO11 model from '{self.model_path}'...")
        self.model = YOLO(self.model_path)
        
        # Load class names dictionary from model
        self.class_names: Dict[int, str] = self.model.names
        print(f"[IBVAP-Detector] Model loaded successfully. Total classes available: {len(self.class_names)}")

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Run object detection on a single image/video frame.

        :param frame: Input image/frame as a BGR numpy array (OpenCV format).
        :return: List of DetectionResult objects containing class_id, class_name, confidence, bbox.
        """
        if frame is None or frame.size == 0:
            return []

        # Run inference through Ultralytics YOLO11
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.target_classes,
            device=self.device,
            verbose=False,
        )

        detections: List[DetectionResult] = []
        if not results:
            return detections

        first_result = results[0]
        if first_result.boxes is None:
            return detections

        boxes = first_result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confs = first_result.boxes.conf.cpu().numpy()  # Confidence scores
        cls_ids = first_result.boxes.cls.cpu().numpy().astype(int)  # Class IDs

        for bbox, conf, cls_id in zip(boxes, confs, cls_ids):
            class_id_int = int(cls_id)
            class_name_str = self.class_names.get(class_id_int, f"class_{class_id_int}")
            
            x1, y1, x2, y2 = [round(float(coord), 2) for coord in bbox]
            conf_score = round(float(conf), 4)

            detection = DetectionResult(
                class_id=class_id_int,
                class_name=class_name_str,
                confidence=conf_score,
                bbox=[x1, y1, x2, y2],
            )
            detections.append(detection)

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        bbox_color: Tuple[int, int, int] = (0, 255, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """
        Draw bounding boxes and class labels with confidence on the frame.

        :param frame: BGR image frame.
        :param detections: List of DetectionResult objects.
        :param bbox_color: BGR tuple for bounding box borders.
        :param text_color: BGR tuple for label text.
        :return: Annotated frame as a BGR numpy array.
        """
        annotated_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            label = f"{det.class_name} {det.confidence:.2f}"

            # Dynamic color coding: Yellow/Orange for humans, Cyan/Green for vehicles
            if det.class_id == 0:  # Person
                current_bbox_color = (0, 165, 255)  # Orange
            elif det.class_id in [2, 3, 5, 7]:  # Car, Motorcycle, Bus, Truck
                current_bbox_color = (0, 255, 255)  # Yellow / Cyan
            else:
                current_bbox_color = bbox_color

            # Draw bounding box rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), current_bbox_color, 2)

            # Draw label banner background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            font_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )

            banner_y1 = max(0, y1 - text_height - 6)
            banner_y2 = y1
            banner_x1 = x1
            banner_x2 = x1 + text_width + 6

            cv2.rectangle(
                annotated_frame,
                (banner_x1, banner_y1),
                (banner_x2, banner_y2),
                current_bbox_color,
                -1,
            )

            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (banner_x1 + 3, banner_y2 - 3),
                font,
                font_scale,
                (0, 0, 0),  # Dark text inside bright banner
                font_thickness,
                lineType=cv2.LINE_AA,
            )

        return annotated_frame


if __name__ == "__main__":
    # Self-test using a dummy frame
    print("[IBVAP-Detector] Running standalone self-test...")
    detector = YOLO11Detector(model_path="yolo11n.pt", conf_threshold=0.35)

    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    results = detector.detect(dummy_frame)
    print(f"[IBVAP-Detector] Test inference completed. Detections count: {len(results)}")
