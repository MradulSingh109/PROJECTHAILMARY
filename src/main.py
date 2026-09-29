"""
=============================================================================
IBVAP - Intelligent Border Video Analytics Platform
Main Entry Point (Stage 3: Detection + ByteTrack Tracking + ANPR)
=============================================================================
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.video_render import VideoRenderer


def main():
    parser = argparse.ArgumentParser(
        description="IBVAP - Intelligent Border Video Analytics Platform"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "videos" / "college_campus_raw.mp4"),
        help="Path to input raw video file or camera stream index/RTSP URL",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output processed video file (defaults to data/processed/videos/)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="models/detection/yolo11n.pt",
        help="Path to YOLO11 base detection model weights (.pt)",
    )
    parser.add_argument(
        "--plate-model",
        type=str,
        default="models/anpr/plate_detector.pt",
        help="Path to custom trained license plate YOLO model (.pt)",
    )
    parser.add_argument(
        "--conf",
        "-c",
        type=float,
        default=0.35,
        help="Confidence threshold for object detection (0.0 - 1.0)",
    )
    parser.add_argument(
        "--anpr",
        action="store_true",
        help="Enable Automatic Number Plate Recognition (ANPR) on tracked vehicles",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable ByteTrack multi-object tracking (fall back to pure frame detection)",
    )
    parser.add_argument(
        "--no-trail",
        action="store_true",
        help="Disable motion trajectory polyline trails",
    )
    parser.add_argument(
        "--trail-length",
        "-t",
        type=int,
        default=120,
        help="Maximum historical trajectory points for motion trails (higher = longer trail distance)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable real-time OpenCV window playback preview",
    )

    args = parser.parse_args()

    enable_tracking = not args.no_track
    draw_trajectories = not args.no_trail
    enable_anpr = args.anpr

    modes = []
    if enable_tracking:
        modes.append("ByteTrack Tracking")
    else:
        modes.append("Detection Only")
    if enable_anpr:
        modes.append("ANPR License Plate Recognition")
    if draw_trajectories:
        modes.append(f"Trajectories ({args.trail_length} pts)")

    mode_str = " + ".join(modes)

    print("=" * 75)
    print(" 🛡️  IBVAP - Intelligent Border Video Analytics Platform")
    print("=" * 75)
    print(f" • Active Pipeline: {mode_str}")
    print(f" • Input Video:     {args.input}")
    print(f" • Detection Model: {args.model}")
    print(f" • ANPR Enabled:    {enable_anpr}")
    if enable_anpr:
        print(f" • Plate Model:     {args.plate_model}")
    print(f" • Confidence:      {args.conf}")
    print(f" • Live Preview:    {args.live}")
    print("=" * 75)

    # Initialize Video Renderer with ByteTrack + ANPR
    renderer = VideoRenderer(
        model_path=args.model,
        plate_model_path=args.plate_model,
        enable_tracking=enable_tracking,
        enable_anpr=enable_anpr,
        conf_threshold=args.conf,
        show_hud=True,
        draw_trajectories=draw_trajectories,
        max_trajectory_points=args.trail_length,
    )

    # Execute Video Processing
    output_file = renderer.process_video(
        input_video_path=args.input,
        output_video_path=args.output,
        display_live=args.live,
    )

    print("\n✅ Processing complete!")
    print(f"📁 Processed Video: {output_file}\n")


if __name__ == "__main__":
    main()
