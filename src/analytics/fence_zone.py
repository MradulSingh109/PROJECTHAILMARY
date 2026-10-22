"""
=============================================================================
IBVAP - Intelligent Border Video Analytics Platform
Module: Virtual Fence & Perimeter Zone Geometry
File: src/analytics/fence_zone.py
=============================================================================
"""

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class FenceType(str, Enum):
    TRIPWIRE = "tripwire"          # Linear 2-point virtual fence line
    POLYGON_ZONE = "polygon_zone"  # Closed polygonal restricted area


class CrossingDirection(str, Enum):
    ANY = "any"                    # Triggers on crossing in either direction
    LEFT_TO_RIGHT = "left_to_right"# Left-to-right relative to directed line F1 -> F2
    RIGHT_TO_LEFT = "right_to_left"# Right-to-left relative to directed line F1 -> F2
    TOP_TO_BOTTOM = "top_to_bottom"# Top-to-bottom crossing (entering from top)
    BOTTOM_TO_TOP = "bottom_to_top"# Bottom-to-top crossing (returning to top)
    ENTRY_ONLY = "entry_only"      # Outside -> Inside for Polygon Zones
    EXIT_ONLY = "exit_only"        # Inside -> Outside for Polygon Zones


@dataclass
class IntrusionEvent:
    """
    Data contract representing a validated perimeter breach / fence intrusion.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    track_id: int = 0
    class_id: int = 0
    class_name: str = "person"
    fence_id: str = "default_fence"
    fence_type: str = "tripwire"
    breach_type: str = "LINE_CROSSING"  # "LINE_CROSSING" or "ZONE_INTRUSION"
    direction: str = "left_to_right"
    crossing_point: Tuple[int, int] = (0, 0)
    bbox: List[float] = field(default_factory=list)  # [x1, y1, x2, y2]
    frame_id: int = 0
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.0
    snapshot_path: Optional[str] = None


class VirtualFence:
    """
    Represents an active virtual perimeter boundary (Tripwire Line or Polygonal Zone).
    Provides spatial geometric intersection and direction calculations.
    """

    def __init__(
        self,
        fence_id: str,
        coordinates: List[Tuple[int, int]],
        fence_type: FenceType = FenceType.TRIPWIRE,
        direction: CrossingDirection = CrossingDirection.ANY,
        name: str = "Virtual Perimeter",
        target_classes: Optional[List[int]] = None,
    ):
        """
        :param fence_id: Unique identifier for this fence instance.
        :param coordinates: For TRIPWIRE: [(x1, y1), (x2, y2)].
                            For POLYGON_ZONE: [(x1, y1), (x2, y2), (x3, y3), ...].
        :param fence_type: FenceType.TRIPWIRE or FenceType.POLYGON_ZONE.
        :param direction: CrossingDirection enum.
        :param name: Human-readable display label.
        :param target_classes: Target YOLO class IDs to alert on (default: [0] for persons).
        """
        self.fence_id = fence_id
        self.name = name
        self.fence_type = fence_type
        self.direction = direction
        self.coordinates = coordinates
        self.target_classes = set(target_classes) if target_classes is not None else {0}  # 0: person

        if self.fence_type == FenceType.TRIPWIRE and len(self.coordinates) < 2:
            raise ValueError(f"Tripwire fence '{fence_id}' must have at least 2 coordinate points.")
        if self.fence_type == FenceType.POLYGON_ZONE and len(self.coordinates) < 3:
            raise ValueError(f"Polygon zone fence '{fence_id}' must have at least 3 coordinate points.")

        self._np_polygon = np.array(self.coordinates, dtype=np.int32) if len(coordinates) >= 3 else None

    @property
    def start_point(self) -> Tuple[int, int]:
        """Start coordinate for tripwire line."""
        return self.coordinates[0]

    @property
    def end_point(self) -> Tuple[int, int]:
        """End coordinate for tripwire line."""
        return self.coordinates[1]

    def get_side_of_line(self, point: Tuple[float, float]) -> float:
        """
        Calculates the 2D cross-product / signed determinant of a point relative to directed line F1 -> F2.
        
        :return: > 0 for Left side, < 0 for Right side, == 0 on the line.
        """
        x1, y1 = self.coordinates[0]
        x2, y2 = self.coordinates[1]
        px, py = point
        # Cross product: (x2 - x1)*(py - y1) - (y2 - y1)*(px - x1)
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def check_line_crossing(
        self,
        p_prev: Tuple[float, float],
        p_curr: Tuple[float, float],
    ) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]]]:
        """
        Tests whether the movement vector from p_prev to p_curr intersects this tripwire line.

        :param p_prev: Object position in previous frame (x, y).
        :param p_curr: Object position in current frame (x, y).
        :return: (is_crossed, direction_str, intersection_point)
        """
        if self.fence_type != FenceType.TRIPWIRE:
            return False, None, None

        f1 = self.coordinates[0]
        f2 = self.coordinates[1]

        # 1. Orientation test for line segment intersection
        side_prev = self.get_side_of_line(p_prev)
        side_curr = self.get_side_of_line(p_curr)

        # If both points are strictly on the same side, no crossing occurred
        if (side_prev > 0 and side_curr > 0) or (side_prev < 0 and side_curr < 0):
            return False, None, None

        # Check if tripwire endpoints are separated by the movement line
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        A = p_prev
        B = p_curr
        C = f1
        D = f2

        intersects = (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))
        if not intersects:
            return False, None, None

        # 2. Determine Crossing Direction
        dx_fence = f2[0] - f1[0]
        dy_fence = f2[1] - f1[1]

        # For predominantly horizontal fences (|dx| >= |dy|):
        if abs(dx_fence) >= abs(dy_fence):
            if p_curr[1] > p_prev[1]:
                crossing_dir = "top_to_bottom"
            else:
                crossing_dir = "bottom_to_top"
        # For predominantly vertical fences (|dy| > |dx|):
        else:
            if p_curr[0] > p_prev[0]:
                crossing_dir = "left_to_right"
            else:
                crossing_dir = "right_to_left"

        # Check direction filter
        if self.direction == CrossingDirection.TOP_TO_BOTTOM and crossing_dir != "top_to_bottom":
            return False, None, None
        if self.direction == CrossingDirection.BOTTOM_TO_TOP and crossing_dir != "bottom_to_top":
            return False, None, None
        if self.direction == CrossingDirection.LEFT_TO_RIGHT and crossing_dir not in ["left_to_right", "top_to_bottom"]:
            return False, None, None
        if self.direction == CrossingDirection.RIGHT_TO_LEFT and crossing_dir not in ["right_to_left", "bottom_to_top"]:
            return False, None, None

        # 3. Calculate exact intersection point
        denom = (f2[1] - f1[1]) * (p_curr[0] - p_prev[0]) - (f2[0] - f1[0]) * (p_curr[1] - p_prev[1])
        if abs(denom) > 1e-6:
            ua = ((f2[0] - f1[0]) * (p_prev[1] - f1[1]) - (f2[1] - f1[1]) * (p_prev[0] - f1[0])) / denom
            ix = int(p_prev[0] + ua * (p_curr[0] - p_prev[0]))
            iy = int(p_prev[1] + ua * (p_curr[1] - p_prev[1]))
            inter_pt = (ix, iy)
        else:
            inter_pt = (int(p_curr[0]), int(p_curr[1]))

        return True, crossing_dir, inter_pt

    def is_inside_polygon(self, point: Tuple[float, float]) -> bool:
        """
        Tests if a point is inside the restricted polygon zone.
        """
        if self._np_polygon is None:
            return False
        # pointPolygonTest returns > 0 for inside, == 0 on edge, < 0 for outside
        res = cv2.pointPolygonTest(self._np_polygon, (float(point[0]), float(point[1])), False)
        return res >= 0