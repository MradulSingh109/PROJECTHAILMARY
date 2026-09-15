"""
=============================================================================
IBVAP - Intelligent Border Video Analytics Platform
Module: Video Rendering & Processing Pipeline (Stage 1)
=============================================================================
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict
import cv2
import numpy as np
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.detection.detector import YOLO11Detector, DetectionResult


class VideoRenderer:
    """
    Renders and processes raw video streams frame-by-frame using YOLO11Detector,
    displaying real-time surveillance HUD telemetry and writing processed outputs.
    """

    def __init__(
        self,
        detector: Optional[YOLO11Detector] = None,
        model_path: str = "models/detection/yolo11n.pt",
        conf_threshold: float = 0.35,
        target_classes: Optional[List[int]] = None,
        show_hud: bool = True,
    ):
        """
        :param detector: Existing YOLO11Detector instance. If None, instantiates a new one.
        :param model_path: Model weights path if instantiating detector.
        :param conf_threshold: Detection confidence threshold.
        :param target_classes: Class IDs to detect (e.g. 0 for human, [2, 3, 5, 7] for vehicles).
        :param show_hud: Whether to render surveillance telemetry header on the output.
        """
        if detector is not None:
            self.detector = detector
        else:
            self.detector = YOLO11Detector(
                model_path=model_path,
                conf_threshold=conf_threshold,
                target_classes=target_classes,
            )
        self.show_hud = show_hud

    def _draw_surveillance_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        total_frames: int,
        fps: float,
        detections: List[DetectionResult],
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
        human_count = sum(1 for d in detections if d.class_id == 0)
        vehicle_count = sum(1 for d in detections if d.class_id in [1, 2, 3, 5, 7])
        other_count = len(detections) - human_count - vehicle_count

        # Telemetry text elements
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        hud_left = f"IBVAP | Frame: {frame_idx}/{total_frames} | FPS: {fps:.1f} | {timestamp}"
        hud_right = f"Humans: {human_count} | Vehicles: {vehicle_count} | Total Targets: {len(detections)}"

        # Render HUD text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, hud_left, (15, 26), font, 0.55, (0, 255, 200), 1, cv2.LINE_AA)

        # Calculate right-aligned position
        (right_w, _), _ = cv2.getTextSize(hud_right, font, 0.55, 1)
        cv2.putText(frame, hud_right, (w - right_w - 15, 26), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def process_video(
        self,
        input_video_path: str,
        output_video_path: Optional[str] = None,
        display_live: bool = False,
    ) -> str:
        """
        Process a video file frame-by-frame, detect objects, annotate, and save output.

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
            output_video_path = str(processed_dir / f"processed_{input_path.name}")

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

        print(f"\n[IBVAP-Renderer] Processing video: '{input_path.name}'")
        print(f"  • Resolution: {width}x{height}")
        print(f"  • FPS: {fps:.2f}")
        print(f"  • Total Frames: {total_frames}")
        print(f"  • Output Destination: '{output_path}'\n")

        pbar = tqdm(total=total_frames, desc="Rendering Frames", unit="frame")
        frame_idx = 0
        t_start = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                t_frame_start = time.time()

                # Step 1: Detect objects using YOLO11
                detections = self.detector.detect(frame)

                # Step 2: Draw detection bounding boxes & class tags
                annotated_frame = self.detector.draw_detections(frame, detections)

                # Step 3: Draw Surveillance HUD
                if self.show_hud:
                    elapsed = time.time() - t_frame_start
                    instant_fps = 1.0 / elapsed if elapsed > 0 else fps
                    annotated_frame = self._draw_surveillance_hud(
                        annotated_frame, frame_idx, total_frames, instant_fps, detections
                    )

                # Step 4: Write to output video
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
    # Test execution using the uploaded raw video
    default_input = PROJECT_ROOT / "data" / "raw" / "videos" / "college_campus_raw.mp4"
    default_output = PROJECT_ROOT / "data" / "processed" / "videos" / "college_campus_processed.mp4"

    if default_input.exists():
        renderer = VideoRenderer(model_path="yolo11n.pt", conf_threshold=0.35)
        renderer.process_video(
            input_video_path=str(default_input),
            output_video_path=str(default_output),
            display_live=False,
        )
    else:
        print(f"[IBVAP-Renderer] Input file not found at {default_input}")
