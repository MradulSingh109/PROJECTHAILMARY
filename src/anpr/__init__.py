"""
IBVAP ANPR Module Package
"""
from src.anpr.plate_preprocessor import PlatePreprocessor
from src.anpr.plate_parser import PlateParser, ParsedPlate
from src.anpr.ocr import OCREngine, OCRResult
from src.anpr.plate_tracker import PlateTracker, VehiclePlateRecord
from src.anpr.plate_detector import PlateDetector, PlateDetection
from src.anpr.anpr_pipeline import ANPRPipeline, ANPRResult

__all__ = [
    "PlatePreprocessor",
    "PlateParser",
    "ParsedPlate",
    "OCREngine",
    "OCRResult",
    "PlateTracker",
    "VehiclePlateRecord",
    "PlateDetector",
    "PlateDetection",
    "ANPRPipeline",
    "ANPRResult",
]
