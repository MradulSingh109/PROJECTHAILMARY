"""
=============================================================================
IBVAP - Intelligent Border Video Analytics Platform
Module: Real-Time Perimeter Intrusion Detector & Event Engine
File: src/analytics/intrusion_detector.py
=============================================================================
"""

import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.fence_zone import (
    VirtualFence,
    FenceType,
    CrossingDirection,
    IntrusionEvent,
)
from src.tracking.object_identity import TrackedObject
from src.tracking.trajectory import TrajectoryManager


class IntrusionDetector:
    """
    Tracks perimeter boundaries, monitors entity trajectory state transitions,
    and dispatches high-priority security intrusion events.
    """

    def __init__(
        self,
        fences: Optional[List[VirtualFence]] = None,
        save_snapshots: bool = False,
        snapshot_dir: str = "runs/intrusions",
        alert_cooldown_frames: int = 45,  # Avoid duplicate alerts for the same track within 45 frames
    ):
        """
        :param fences: List of VirtualFence instances to monitor.
        :param save_snapshots: Whether to save cropped intruder images on breach.
        :param snapshot_dir: Destination folder for breach snapshots.
        :param alert_cooldown_frames: Minimum frame interval before re-alerting on same track.
        """
        self.save_snapshots = save_snapshots
        self.snapshot_dir = Path(snapshot_dir)
        if self.save_snapshots:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.alert_cooldown_frames = alert_cooldown_frames

        # State tracking: {fence_id: {track_id: last_alerted_frame_id}}
        self._alert_history: Dict[str, Dict[int, int]] = {}

        # State tracking for polygon zones: {fence_id: {track_id: is_currently_inside}}
        self._zone_states: Dict[str, Dict[int, bool]] = {}

        # Accumulated intrusion events
        self._all_events: List[IntrusionEvent] = []
        self._active_breaches: List[IntrusionEvent] = []

        # Currently active intruder track IDs (stay active until crossing back)
        self.active_intruders: Set[int] = set()

        self.fences: Dict[str, VirtualFence] = {}
        if fences:
            for f in fences:
                self.add_fence(f)

    def add_fence(self, fence: VirtualFence) -> None:
        """Register a new virtual fence for monitoring."""
        self.fences[fence.fence_id] = fence
        self._alert_history[fence.fence_id] = {}
        self._zone_states[fence.fence_id] = {}

    @property
    def total_breaches(self) -> int:
        """Total number of validated perimeter breaches recorded."""
        return len(self._all_events)

    def process_frame(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        trajectory_manager: TrajectoryManager,
        frame_id: int,
    ) -> List[IntrusionEvent]:
        """
        Analyzes tracked entities against all virtual fences in the current frame.

        :param frame: Current BGR video frame.
        :param tracked_objects: List of TrackedObject instances from ByteTrack.
        :param trajectory_manager: TrajectoryManager instance containing historical path points.
        :param frame_id: Current sequence index of the video frame.
        :return: List of new IntrusionEvent objects triggered in this frame.
        """
        frame_events: List[IntrusionEvent] = []
        if not self.fences or not tracked_objects:
            self._active_breaches = []
            return frame_events

        active_track_ids = {obj.track_id for obj in tracked_objects}
        # Keep active_intruders in sync with currently tracked objects
        self.active_intruders = self.active_intruders.intersection(active_track_ids)

        for fence in self.fences.values():
            for obj in tracked_objects:
                # Check if this object class is targeted by this fence (default: person)
                if obj.class_id not in fence.target_classes:
                    continue

                t_id = obj.track_id
                trail = trajectory_manager.get_trajectory(t_id)

                # Need at least 2 points to compute movement vector
                if len(trail) < 2:
                    continue

                p_prev = (trail[-2].x, trail[-2].y)
                p_curr = (trail[-1].x, trail[-1].y)

                # 1. TRIPWIRE LINE CROSSING DETECTION
                if fence.fence_type == FenceType.TRIPWIRE:
                    is_crossed, direction, inter_pt = fence.check_line_crossing(p_prev, p_curr)

                    if is_crossed:
                        # Top-to-Bottom (or Left-to-Right) entry -> Mark as ACTIVE INTRUDER
                        if direction in ["top_to_bottom", "left_to_right", "crossing"]:
                            self.active_intruders.add(t_id)

                            # Check alert cooldown to debounce repetitive triggers
                            last_alert_frame = self._alert_history[fence.fence_id].get(t_id, -9999)
                            if frame_id - last_alert_frame >= self.alert_cooldown_frames:
                                self._alert_history[fence.fence_id][t_id] = frame_id

                                event = IntrusionEvent(
                                    track_id=t_id,
                                    class_id=obj.class_id,
                                    class_name=obj.class_name,
                                    fence_id=fence.fence_id,
                                    fence_type="tripwire",
                                    breach_type="LINE_CROSSING",
                                    direction=direction or "top_to_bottom",
                                    crossing_point=inter_pt or (int(p_curr[0]), int(p_curr[1])),
                                    bbox=list(obj.bbox),
                                    frame_id=frame_id,
                                    confidence=obj.confidence,
                                )

                                if self.save_snapshots and frame is not None:
                                    self._save_intruder_snapshot(frame, obj, event)

                                frame_events.append(event)
                                self._all_events.append(event)

                        # Bottom-to-Top (or Right-to-Left) exit -> CLEAR INTRUDER STATUS back to normal
                        elif direction in ["bottom_to_top", "right_to_left"]:
                            if t_id in self.active_intruders:
                                self.active_intruders.discard(t_id)
                                print(f"[IBVAP-Intrusion] Track #{t_id} returned back through fence (status cleared to normal).")

                # 2. RESTRICTED POLYGON ZONE INTRUSION DETECTION
                elif fence.fence_type == FenceType.POLYGON_ZONE:
                    is_now_inside = fence.is_inside_polygon(p_curr)
                    was_inside = self._zone_states[fence.fence_id].get(t_id, False)

                    if is_now_inside:
                        self.active_intruders.add(t_id)
                    else:
                        self.active_intruders.discard(t_id)

                    # State transition: Outside -> Inside
                    if is_now_inside and not was_inside:
                        last_alert_frame = self._alert_history[fence.fence_id].get(t_id, -9999)
                        if frame_id - last_alert_frame >= self.alert_cooldown_frames:
                            self._alert_history[fence.fence_id][t_id] = frame_id

                            event = IntrusionEvent(
                                track_id=t_id,
                                class_id=obj.class_id,
                                class_name=obj.class_name,
                                fence_id=fence.fence_id,
                                fence_type="polygon_zone",
                                breach_type="ZONE_INTRUSION",
                                direction="entry",
                                crossing_point=(int(p_curr[0]), int(p_curr[1])),
                                bbox=list(obj.bbox),
                                frame_id=frame_id,
                                confidence=obj.confidence,
                            )

                            if self.save_snapshots and frame is not None:
                                self._save_intruder_snapshot(frame, obj, event)

                            frame_events.append(event)
                            self._all_events.append(event)

                    # Update zone state
                    self._zone_states[fence.fence_id][t_id] = is_now_inside

        # Keep track of active breaches for visual HUD persistence (50-frame glow)
        self._active_breaches = [
            e for e in self._all_events if (frame_id - e.frame_id) < 50
        ]

        return frame_events

    def _save_intruder_snapshot(
        self,
        frame: np.ndarray,
        obj: TrackedObject,
        event: IntrusionEvent,
    ) -> None:
        """Save a cropped image of the intruder when a breach occurs."""
        try:
            h, w = frame.shape[:2]
            x1 = max(0, int(obj.bbox[0]))
            y1 = max(0, int(obj.bbox[1]))
            x2 = min(w, int(obj.bbox[2]))
            y2 = min(h, int(obj.bbox[3]))

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                filename = f"breach_f{event.frame_id}_track{event.track_id}_{event.event_id}.jpg"
                path = self.snapshot_dir / filename
                cv2.imwrite(str(path), crop)
                event.snapshot_path = str(path)
        except Exception as e:
            print(f"[IBVAP-Intrusion] Snapshot error: {e}")

    def render_overlay(
        self,
        frame: np.ndarray,
        frame_id: int,
    ) -> np.ndarray:
        """
        Renders aesthetic high-tech virtual fences, glowing perimeter lines, and breach alerts.
        """
        is_alarm_active = len(self.active_intruders) > 0 or len(self._active_breaches) > 0

        # Pulsing alarm alpha based on frame sine wave
        pulse_alpha = 0.5 + 0.5 * math.sin(frame_id * 0.3) if is_alarm_active else 0.0

        for fence in self.fences.values():
            if fence.fence_type == FenceType.TRIPWIRE:
                p1 = fence.coordinates[0]
                p2 = fence.coordinates[1]

                # Base fence color (Cyan/Green if normal, Bright Red/Orange if breached)
                if is_alarm_active:
                    base_color = (0, 0, 255)       # Red
                    glow_color = (50, 50, 255)
                else:
                    base_color = (0, 255, 128)     # Glowing Emerald
                    glow_color = (0, 180, 80)

                # # 1. Outer Glow
                # cv2.line(frame, p1, p2, glow_color, 6, cv2.LINE_AA)
                # 2. Core Line
                cv2.line(frame, p1, p2, base_color, 2, cv2.LINE_AA)

                # 3. Draw Endpoint Sentinel Nodes
                # cv2.circle(frame, p1, 6, base_color, -1, cv2.LINE_AA)
                # cv2.circle(frame, p2, 6, base_color, -1, cv2.LINE_AA)
                # cv2.circle(frame, p1, 9, (255, 255, 255), 1, cv2.LINE_AA)
                # cv2.circle(frame, p2, 9, (255, 255, 255), 1, cv2.LINE_AA)

                # 4. Draw Directional Chevron / Arrow at midpoint
                # mid_x = (p1[0] + p2[0]) // 2
                # mid_y = (p1[1] + p2[1]) // 2
                # dx = p2[0] - p1[0]
                # dy = p2[1] - p1[1]
                # length = math.hypot(dx, dy)
                # if length > 0:
                #     # Normal vector pointing to the "right" side of the line
                #     nx = -dy / length
                #     ny = dx / length
                #     arrow_len = 16
                #     tip_x = int(mid_x + nx * arrow_len)
                #     tip_y = int(mid_y + ny * arrow_len)
                #     cv2.arrowedLine(frame, (mid_x, mid_y), (tip_x, tip_y), base_color, 2, cv2.LINE_AA, tipLength=0.4)

                # # 5. Fence Label Badge
                # label = f" FENCE: {fence.name.upper()} "
                # (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                # bx = mid_x - tw // 2
                # by = max(20, mid_y - 12)
                # cv2.rectangle(frame, (bx - 4, by - th - 4), (bx + tw + 4, by + 4), (20, 20, 20), -1)
                # cv2.rectangle(frame, (bx - 4, by - th - 4), (bx + tw + 4, by + 4), base_color, 1)
                # cv2.putText(frame, label, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            elif fence.fence_type == FenceType.POLYGON_ZONE and fence._np_polygon is not None:
                # Semi-transparent restricted zone overlay
                overlay = frame.copy()
                zone_color = (0, 0, 255) if is_alarm_active else (0, 165, 255)  # Red vs Orange
                cv2.fillPoly(overlay, [fence._np_polygon], zone_color)
                cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                cv2.polylines(frame, [fence._np_polygon], isClosed=True, color=zone_color, thickness=2, lineType=cv2.LINE_AA)

        # Draw Crossing Points from recent breaches
        for breach in self._active_breaches:
            cx, cy = breach.crossing_point
            cv2.drawMarker(frame, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=18, thickness=3, line_type=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), int(10 + 8 * pulse_alpha), (0, 0, 255), 2, cv2.LINE_AA)

        # Top Alarm HUD Banner if active breach
        if is_alarm_active:
            latest = self._active_breaches[-1] if self._active_breaches else None
            if self.active_intruders:
                intruders_str = ", ".join(f"#{t}" for t in sorted(self.active_intruders))
                banner_text = f"PERIMETER BREACH DETECTED: INTRUDER {intruders_str} "
            elif latest:
                banner_text = f"PERIMETER BREACH DETECTED: TRACK #{latest.track_id} ({latest.class_name.upper()}) "
            else:
                banner_text = "PERIMETER BREACH DETECTED "

            (bw, bh), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            cv2.rectangle(frame, (20, 20), (20 + bw + 20, 20 + bh + 16), (0, 0, 255), -1)
            cv2.rectangle(frame, (20, 20), (20 + bw + 20, 20 + bh + 16), (255, 255, 255), 2)
            cv2.putText(frame, banner_text, (30, 20 + bh + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

        return frame
