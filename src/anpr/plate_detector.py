"""
Import bridge to expose PlateDetector and PlateDetection under src.anpr.plate_detector
"""
from src.training.train_plate_yolo import PlateDetector, PlateDetection

__all__ = ["PlateDetector", "PlateDetection"]

if __name__ == "__main__":
    import numpy as np
    print("[IBVAP-ANPR] Running PlateDetector standalone self-test...")
    detector = PlateDetector()
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    res = detector.detect_plates(dummy_frame)
    print(f"[IBVAP-ANPR] Self-test complete. Detected plates: {len(res)}")
