"""
=============================================================================
IBVAP - Intelligent Border Video Analytics Platform
Module: ByteTrack Multi-Object Tracker & Surveillance Tracking Engine
=============================================================================
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from ultralytics import YOLO

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.object_identity import TrackedObject, ObjectIdentityManager
from src.tracking.trajectory import TrajectoryManager


class ByteTrackTracker:
    """
    Surveillance multi-object tracker powered by YOLO11 and ByteTrack algorithm.
    Provides persistent track IDs, trajectory management, and analytics metrics.
    """

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
        model_path: str = "models/detection/yolo11n.pt",
        tracker_type: str = "bytetrack.yaml",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.50,
        target_classes: Optional[List[int]] = None,
        device: Optional[str] = None,
        max_trajectory_points: int = 60,
        max_lost_frames: int = 30,
    ):
        """
        Initialize ByteTrack multi-object tracker with YOLO11.

        :param model_path: Path to YOLO11 model weights (.pt).
        :param tracker_type: Tracker configuration ('bytetrack.yaml' or 'botsort.yaml').
        :param conf_threshold: Detection confidence threshold (0.0 - 1.0).
        :param iou_threshold: NMS IoU threshold (0.0 - 1.0).
        :param target_classes: Class IDs to track (e.g., [0] for human, [2, 3, 5, 7] for vehicles).
        :param device: Hardware device ('0', 'cuda', 'cpu', etc.).
        :param max_trajectory_points: Max historical coordinate points per track.
        :param max_lost_frames: Max frames before purging an inactive track ID.
        """
        # Resolve model path across common project directories
        resolved_path = model_path
        if not os.path.exists(resolved_path):
            candidates = [
                os.path.join(PROJECT_ROOT, model_path),
                os.path.join(PROJECT_ROOT, "models", "detection", os.path.basename(model_path)),
                os.path.join(PROJECT_ROOT, "models", model_path),
            ]
            for cand in candidates:
                cand_norm = os.path.normpath(cand)
                if os.path.exists(cand_norm):
                    resolved_path = cand_norm
                    break

        self.model_path = resolved_path
        self.tracker_type = tracker_type
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes
        self.device = device

        print(f"[IBVAP-Tracker] Loading YOLO11 model for ByteTrack tracking: '{self.model_path}'...")
        self.model = YOLO(self.model_path)
        self.class_names: Dict[int, str] = self.model.names

        # Sub-managers for tracking identity and trajectory history
        self.identity_manager = ObjectIdentityManager(max_lost_frames=max_lost_frames)
        self.trajectory_manager = TrajectoryManager(max_points=max_trajectory_points)

        self._frame_counter: int = 0
        print(f"[IBVAP-Tracker] Tracker successfully initialized with '{self.tracker_type}'.")

    def track(
        self,
        frame: np.ndarray,
        frame_id: Optional[int] = None,
    ) -> List[TrackedObject]:
        """
        Execute YOLO11 + ByteTrack inference on a frame and update object trajectories.

        :param frame: BGR image frame (NumPy array).
        :param frame_id: Optional frame index. If None, auto-increments internal counter.
        :return: List of TrackedObject instances with persistent track IDs.
        """
        if frame is None or frame.size == 0:
            return []

        if frame_id is not None:
            self._frame_counter = frame_id
        else:
            self._frame_counter += 1

        curr_frame_id = self._frame_counter
        timestamp = time.time()

        # Run ByteTrack tracking with persist=True
        results = self.model.track(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.target_classes,
            device=self.device,
            tracker=self.tracker_type,
            persist=True,
            verbose=False,
        )

        tracked_objects: List[TrackedObject] = []
        if not results:
            return tracked_objects

        first_res = results[0]
        if first_res.boxes is None or first_res.boxes.id is None:
            return tracked_objects

        boxes = first_res.boxes.xyxy.cpu().numpy()
        confs = first_res.boxes.conf.cpu().numpy()
        clses = first_res.boxes.cls.cpu().numpy().astype(int)
        track_ids = first_res.boxes.id.cpu().numpy().astype(int)

        for bbox, conf, cls_id, t_id in zip(boxes, confs, clses, track_ids):
            class_id_int = int(cls_id)
            track_id_int = int(t_id)
            class_name_str = self.class_names.get(class_id_int, f"class_{class_id_int}")

            x1, y1, x2, y2 = [round(float(c), 2) for c in bbox]
            center_x = round((x1 + x2) / 2.0, 2)
            center_y = round((y1 + y2) / 2.0, 2)

            tracked_obj = TrackedObject(
                track_id=track_id_int,
                class_id=class_id_int,
                class_name=class_name_str,
                confidence=round(float(conf), 4),
                bbox=[x1, y1, x2, y2],
                center=[center_x, center_y],
                frame_id=curr_frame_id,
                timestamp=timestamp,
            )
            tracked_objects.append(tracked_obj)

            # Record trajectory (ground-contact bottom-center for surveillance precision)
            self.trajectory_manager.add_point(
                track_id=track_id_int,
                point=(center_x, y2),
                frame_id=curr_frame_id,
                timestamp=timestamp,
            )

        # Update identity manager
        self.identity_manager.update(tracked_objects, curr_frame_id)

        return tracked_objects

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        draw_trajectories: bool = True,
    ) -> np.ndarray:
        """
        Render persistent track IDs, bounding boxes, target labels, and trajectory trails.

        :param frame: BGR input frame.
        :param tracked_objects: List of TrackedObject instances.
        :param draw_trajectories: Whether to render trailing motion paths.
        :return: Annotated BGR frame.
        """
        annotated = frame.copy()
        active_ids = [obj.track_id for obj in tracked_objects]

        # Step 1: Draw motion trajectory trails
        if draw_trajectories:
            annotated = self.trajectory_manager.draw_trails(
                annotated,
                active_track_ids=active_ids,
                color=(0, 255, 255),
                thickness=2,
                fade_effect=True,
            )

        # Step 2: Draw Tracked Bounding Boxes and Surveillance HUD Labels
        for obj in tracked_objects:
            x1, y1, x2, y2 = map(int, obj.bbox)
            t_id = obj.track_id

            # Dynamic color coding: Human (Orange), Vehicles (Cyan/Yellow), Default (Teal)
            if obj.class_id == 0:  # Person
                color = (0, 165, 255)
            elif obj.class_id in [1, 2, 3, 5, 7]:  # Vehicles
                color = (255, 200, 0)
            else:
                color = (0, 255, 128)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label banner: e.g. "ID: #17 | person 0.91"
            label = f"ID: #{t_id} | {obj.class_name} {obj.confidence:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.50
            font_thickness = 1
            (w_txt, h_txt), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

            banner_y1 = max(0, y1 - h_txt - 6)
            banner_y2 = y1
            banner_x1 = x1
            banner_x2 = x1 + w_txt + 8

            cv2.rectangle(annotated, (banner_x1, banner_y1), (banner_x2, banner_y2), color, -1)
            cv2.putText(
                annotated,
                label,
                (banner_x1 + 4, banner_y2 - 3),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness,
                lineType=cv2.LINE_AA,
            )

        return annotated


if __name__ == "__main__":
    # Standalone verification test
    print("[IBVAP-Tracker] Running standalone Tracker verification test...")
    tracker = ByteTrackTracker(conf_threshold=0.35)

    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    tracks = tracker.track(dummy_frame, frame_id=1)
    print(f"[IBVAP-Tracker] Tracker test completed. Active tracks count: {len(tracks)}")
