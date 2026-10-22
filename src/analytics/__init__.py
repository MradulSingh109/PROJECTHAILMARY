"""
=============================================================================
IBVAP - Intelligent Border Video Analytics Platform
Module: Real-time Analytics & Perimeter Security
=============================================================================
"""

from src.analytics.fence_zone import (
    VirtualFence,
    FenceType,
    CrossingDirection,
    IntrusionEvent,
)
from src.analytics.intrusion_detector import IntrusionDetector

__all__ = [
    "VirtualFence",
    "FenceType",
    "CrossingDirection",
    "IntrusionEvent",
    "IntrusionDetector",
]
