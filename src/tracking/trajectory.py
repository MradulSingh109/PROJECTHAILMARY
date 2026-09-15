"""
=============================================================================
IBVAP - Intelligent Border Video Analytics Platform
Module: Trajectory Management & Motion Path Analytics
=============================================================================
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class TrajectoryPoint:
    """A single spatial-temporal coordinate in an object's trajectory."""
    x: float
    y: float
    frame_id: int
    timestamp: float


class TrajectoryManager:
    """
    Maintains and analyzes historical movement paths for all tracked objects.
    Enables threat analysis, virtual fence crossing verification, and path visualization.
    """

    def __init__(self, max_points: int = 60, point_mode: str = "bottom_center"):
        """
        :param max_points: Maximum historical trajectory points to keep per track ID.
        :param point_mode: 'center' (centroid) or 'bottom_center' (ground-contact point).
        """
        self.max_points = max_points
        self.point_mode = point_mode
        self._trajectories: Dict[int, deque] = {}

    def add_point(
        self,
        track_id: int,
        point: Tuple[float, float],
        frame_id: int,
        timestamp: float,
    ) -> None:
        """
        Append a new coordinate point to the object's trajectory history.
        """
        if track_id not in self._trajectories:
            self._trajectories[track_id] = deque(maxlen=self.max_points)

        self._trajectories[track_id].append(
            TrajectoryPoint(
                x=float(point[0]),
                y=float(point[1]),
                frame_id=frame_id,
                timestamp=timestamp,
            )
        )

    def get_trajectory(self, track_id: int) -> List[TrajectoryPoint]:
        """Get the full list of recorded trajectory points for a track ID."""
        if track_id in self._trajectories:
            return list(self._trajectories[track_id])
        return []

    def get_trail_points(self, track_id: int) -> List[Tuple[int, int]]:
        """
        Get integer pixel coordinates list for rendering path polylines.
        """
        if track_id not in self._trajectories:
            return []
        return [(int(p.x), int(p.y)) for p in self._trajectories[track_id]]

    def compute_velocity(self, track_id: int, window: int = 5) -> Tuple[float, float, float]:
        """
        Computes the instantaneous velocity vector and speed (in pixels per frame).
        
        :param track_id: Track ID.
        :param window: Number of recent points to average velocity over.
        :return: (vx, vy, speed)
        """
        trail = self.get_trajectory(track_id)
        if len(trail) < 2:
            return 0.0, 0.0, 0.0

        recent = trail[-min(window, len(trail)):]
        p_start = recent[0]
        p_end = recent[-1]

        frame_diff = max(1, p_end.frame_id - p_start.frame_id)
        dx = (p_end.x - p_start.x) / frame_diff
        dy = (p_end.y - p_start.y) / frame_diff
        speed = math.hypot(dx, dy)

        return dx, dy, speed

    def compute_displacement(self, track_id: int) -> float:
        """
        Computes total Euclidean displacement from start of track to current position.
        """
        trail = self.get_trajectory(track_id)
        if len(trail) < 2:
            return 0.0
        p_first = trail[0]
        p_last = trail[-1]
        return math.hypot(p_last.x - p_first.x, p_last.y - p_first.y)

    def is_stationary(self, track_id: int, max_movement_px: float = 15.0, min_frames: int = 30) -> bool:
        """
        Checks whether an object has been stationary / loitering in place.
        """
        trail = self.get_trajectory(track_id)
        if len(trail) < min_frames:
            return False
        recent = trail[-min_frames:]
        max_dist = max(math.hypot(p.x - recent[0].x, p.y - recent[0].y) for p in recent)
        return max_dist < max_movement_px

    def cleanup_old_tracks(self, active_track_ids: List[int]) -> None:
        """
        Remove trajectory buffers for tracks that are no longer active.
        """
        active_set = set(active_track_ids)
        to_delete = [t_id for t_id in self._trajectories if t_id not in active_set]
        for t_id in to_delete:
            del self._trajectories[t_id]

    def draw_trails(
        self,
        frame: np.ndarray,
        active_track_ids: Optional[List[int]] = None,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
        fade_effect: bool = True,
    ) -> np.ndarray:
        """
        Draw sleek motion trails for tracked entities on the frame.
        """
        targets = active_track_ids if active_track_ids is not None else list(self._trajectories.keys())

        for t_id in targets:
            pts = self.get_trail_points(t_id)
            if len(pts) < 2:
                continue

            for i in range(1, len(pts)):
                if fade_effect:
                    # Gradually fade trail alpha from tail to head
                    alpha = float(i) / len(pts)
                    line_thickness = max(1, int(thickness * alpha))
                    pt_color = (
                        int(color[0] * alpha),
                        int(color[1] * alpha),
                        int(color[2] * alpha),
                    )
                else:
                    line_thickness = thickness
                    pt_color = color

                cv2.line(frame, pts[i - 1], pts[i], pt_color, line_thickness, cv2.LINE_AA)

            # Draw glowing dot at current position
            if pts:
                cv2.circle(frame, pts[-1], 3, (0, 255, 0), -1, cv2.LINE_AA)

        return frame
