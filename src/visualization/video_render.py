"""
=============================================================================
IBVAP - Intelligent Border Video Analytics Platform
Module: Video Rendering & Processing Pipeline (Stage 3: Detection + Tracking + ANPR)
=============================================================================
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Union
import cv2
import numpy as np
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.detection.detector import YOLO11Detector, DetectionResult
from src.tracking.tracker import ByteTrackTracker
from src.tracking.object_identity import TrackedObject
from src.anpr.anpr_pipeline import ANPRPipeline, ANPRResult


class VideoRenderer:
    """
    Renders and processes raw video streams frame-by-frame using YOLO11 + ByteTrack + ANPR,
    displaying real-time surveillance HUD telemetry, trajectory trails, and writing processed outputs.
    """

    def __init__(
        self,
        tracker: Optional[ByteTrackTracker] = None,
        detector: Optional[YOLO11Detector] = None,
        anpr_pipeline: Optional[ANPRPipeline] = None,
        model_path: str = "models/detection/yolo11n.pt",
        plate_model_path: str = "models/anpr/plate_detector.pt",
        enable_tracking: bool = True,
        enable_anpr: bool = False,
        conf_threshold: float = 0.35,
        target_classes: Optional[List[int]] = None,
        show_hud: bool = True,
        draw_trajectories: bool = True,
        max_trajectory_points: int = 120,
    ):
        """
        :param tracker: Optional ByteTrackTracker instance.
        :param detector: Optional YOLO11Detector instance.
        :param anpr_pipeline: Optional ANPRPipeline instance.
        :param model_path: Base detection model weights.
        :param plate_model_path: Custom license plate detector weights.
        :param enable_tracking: If True, executes ByteTrack tracking.
        :param enable_anpr: If True, executes Automatic Number Plate Recognition on vehicles.
        :param conf_threshold: Detection confidence threshold.
        :param target_classes: Class IDs to track/detect.
        :param show_hud: Whether to render surveillance telemetry header on the output.
        :param draw_trajectories: Whether to render trailing motion paths.
        :param max_trajectory_points: Maximum historical trailing points to render.
        """
        self.enable_tracking = enable_tracking
        self.enable_anpr = enable_anpr
        self.show_hud = show_hud
        self.draw_trajectories = draw_trajectories
        self.max_trajectory_points = max_trajectory_points

        if self.enable_tracking:
            if tracker is not None:
                self.tracker = tracker
            else:
                self.tracker = ByteTrackTracker(
                    model_path=model_path,
                    conf_threshold=conf_threshold,
                    target_classes=target_classes,
                    max_trajectory_points=max_trajectory_points,
                )
            self.detector = None
        else:
            self.tracker = None
            if detector is not None:
                self.detector = detector
            else:
                self.detector = YOLO11Detector(
                    model_path=model_path,
                    conf_threshold=conf_threshold,
                    target_classes=target_classes,
                )

        if self.enable_anpr:
            if anpr_pipeline is not None:
                self.anpr_pipeline = anpr_pipeline
            else:
                self.anpr_pipeline = ANPRPipeline(
                    plate_model_path=plate_model_path,
                    conf_threshold=conf_threshold,
                )
        else:
            self.anpr_pipeline = None

    def _draw_surveillance_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        total_frames: int,
        fps: float,
        objects: Union[List[TrackedObject], List[DetectionResult]],
        anpr_results: Optional[List[ANPRResult]] = None,
    ) -> np.ndarray:
        """
        Renders an advanced, dark-themed top HUD bar with surveillance telemetry.
        """
        h, w = frame.shape[:2]
        hud_height = 40
        overlay = frame.copy()

        # Semi-transparent dark banner at top
        cv2.rectangle(overlay, (0, 0), (w, hud_height), (20, 20, 20), -1)
        alpha = 0.75
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Count humans and vehicles
        human_count = sum(1 for obj in objects if obj.class_id == 0)
        vehicle_count = sum(1 for obj in objects if obj.class_id in [1, 2, 3, 5, 7])
        plate_count = len(anpr_results) if anpr_results else 0

        # Telemetry text elements
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        mode_tags = []
        if self.enable_tracking:
            mode_tags.append("TRACKING")
        else:
            mode_tags.append("DETECTION")
        if self.enable_anpr:
            mode_tags.append("ANPR")

        mode_str = " + ".join(mode_tags)
        hud_left = f"IBVAP | {mode_str} | Frame: {frame_idx}/{total_frames} | FPS: {fps:.1f} | {timestamp}"
        hud_right = f"Humans: {human_count} | Vehicles: {vehicle_count}"
        if self.enable_anpr:
            hud_right += f" | Plates: {plate_count}"
        hud_right += f" | Active: {len(objects)}"

        # Render HUD text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, hud_left, (15, 26), font, 0.50, (0, 255, 200), 1, cv2.LINE_AA)

        (right_w, _), _ = cv2.getTextSize(hud_right, font, 0.50, 1)
        cv2.putText(frame, hud_right, (w - right_w - 15, 26), font, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def process_video(
        self,
        input_video_path: str,
        output_video_path: Optional[str] = None,
        display_live: bool = False,
    ) -> str:
        """
        Process a video file frame-by-frame with tracking/detection and optional ANPR.

        :param input_video_path: Path to source raw video.
        :param output_video_path: Destination path for annotated video.
        :param display_live: If True, opens an OpenCV window to display the stream live.
        :return: Path to the saved processed video.
        """
        input_path = Path(input_video_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found at: {input_path}")

        # Default output path if not specified
        if output_video_path is None:
            processed_dir = PROJECT_ROOT / "data" / "processed" / "videos"
            processed_dir.mkdir(parents=True, exist_ok=True)
            tag = "anpr_" if self.enable_anpr else "tracked_"
            output_video_path = str(processed_dir / f"{tag}{input_path.name}")

        output_path = Path(output_video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        modes = []
        if self.enable_tracking:
            modes.append("ByteTrack")
        if self.enable_anpr:
            modes.append("ANPR")
        mode_desc = " + ".join(modes) if modes else "Detection"

        print(f"\n[IBVAP-Renderer] Processing video: '{input_path.name}' ({mode_desc})")
        print(f"  • Resolution: {width}x{height}")
        print(f"  • FPS: {fps:.2f}")
        print(f"  • Total Frames: {total_frames}")
        print(f"  • Output Destination: '{output_path}'\n")

        pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame")
        frame_idx = 0
        t_start = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                t_frame_start = time.time()
                anpr_results: Optional[List[ANPRResult]] = None

                if self.enable_tracking:
                    # Step 1: Run ByteTrack tracking
                    tracked_objects = self.tracker.track(frame, frame_id=frame_idx)
                    # Step 2: Draw persistent IDs and motion trajectories
                    annotated_frame = self.tracker.draw_tracks(
                        frame, tracked_objects, draw_trajectories=self.draw_trajectories
                    )
                    current_items = tracked_objects

                    # Step 3: Run ANPR if enabled
                    if self.enable_anpr and self.anpr_pipeline:
                        anpr_results = self.anpr_pipeline.process_vehicles(
                            frame, tracked_objects, frame_id=frame_idx
                        )
                        annotated_frame = self.anpr_pipeline.draw_anpr_annotations(
                            annotated_frame, anpr_results, tracked_objects
                        )
                else:
                    # Pure detection fallback
                    detections = self.detector.detect(frame)
                    annotated_frame = self.detector.draw_detections(frame, detections)
                    current_items = detections

                # Step 4: Draw Surveillance HUD
                if self.show_hud:
                    elapsed = time.time() - t_frame_start
                    instant_fps = 1.0 / elapsed if elapsed > 0 else fps
                    annotated_frame = self._draw_surveillance_hud(
                        annotated_frame, frame_idx, total_frames, instant_fps, current_items, anpr_results
                    )

                # Step 5: Write to output video
                out.write(annotated_frame)
                pbar.update(1)

                # Optional live preview
                if display_live:
                    cv2.imshow("IBVAP - Live Surveillance Analytics", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\n[IBVAP-Renderer] Live playback interrupted by user.")
                        break

        finally:
            cap.release()
            out.release()
            pbar.close()
            if display_live:
                cv2.destroyAllWindows()

        total_time = time.time() - t_start
        avg_fps = frame_idx / total_time if total_time > 0 else 0
        print(f"\n[IBVAP-Renderer] Completed successfully in {total_time:.2f}s (Avg FPS: {avg_fps:.1f})")
        print(f"[IBVAP-Renderer] Saved processed video to: {output_path}")

        return str(output_path)


if __name__ == "__main__":
    default_input = PROJECT_ROOT / "data" / "raw" / "videos" / "college_campus_raw.mp4"
    default_output = PROJECT_ROOT / "data" / "processed" / "videos" / "college_campus_anpr.mp4"

    if default_input.exists():
        renderer = VideoRenderer(
            enable_tracking=True,
            enable_anpr=True,
            conf_threshold=0.35,
            draw_trajectories=True,
        )
        renderer.process_video(
            input_video_path=str(default_input),
            output_video_path=str(default_output),
            display_live=False,
        )
    else:
        print(f"[IBVAP-Renderer] Input file not found at {default_input}")
