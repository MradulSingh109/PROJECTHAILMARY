"""
Import bridge to expose YOLO11Detector under src.detection.detector
"""
from models.detection.detector import YOLO11Detector, DetectionResult

__all__ = ["YOLO11Detector", "DetectionResult"]
