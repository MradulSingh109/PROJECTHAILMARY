"""
=============================================================================
IBVAP - Intelligent Border Video Analytics Platform
Main Entry Point (Stage 3: Detection + ByteTrack Tracking + ANPR)
=============================================================================
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    # Load the computer-vision stack only for the CLI workflow. Vercel imports
    # this module to serve the lightweight project page.
    from src.visualization.video_render import VideoRenderer

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


class handler(BaseHTTPRequestHandler):
    """Small standard-library Vercel handler for the project showcase page."""

    def do_GET(self):
        path = urlsplit(self.path).path.rstrip("/") or "/"
        query = urlsplit(self.path).query
        if path == "/health" or query == "health=1":
            body = json.dumps({"status": "ok", "service": "IBVAP showcase"}).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        else:
            body = _showcase_page().encode("utf-8")
            content_type = "text/html; charset=utf-8"

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=60")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Keep routine requests out of deployment logs.
        return


def _showcase_page():
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#07131b">
  <meta name="description" content="IBVAP is an AI-assisted video analytics prototype for object detection, multi-object tracking, motion trails, and Indian vehicle plate recognition.">
  <title>IBVAP | Intelligent Border Video Analytics</title>
  <style>
    :root{color-scheme:dark;--bg:#07131b;--panel:#0e2029;--line:#1c3943;--text:#edf7f5;--muted:#a2b8b7;--teal:#5ee3c1;--amber:#ffbd69}
    *{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 76% 0%,#12363a 0,transparent 42%),var(--bg);color:var(--text);font:16px/1.6 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
    a{color:inherit}.wrap{max-width:1120px;margin:auto;padding:28px 24px 60px}.nav{display:flex;justify-content:space-between;align-items:center;padding:8px 0 54px}.brand{font-size:14px;font-weight:800;letter-spacing:.12em}.brand span{color:var(--teal)}.status{color:var(--teal);font:12px ui-monospace,monospace;border:1px solid #28554f;border-radius:99px;padding:7px 12px}
    .hero{max-width:780px;padding:15px 0 65px}.eyebrow{color:var(--teal);font:12px ui-monospace,monospace;letter-spacing:.16em;text-transform:uppercase}.hero h1{font-size:clamp(42px,7vw,76px);line-height:1.02;letter-spacing:-.055em;margin:18px 0 20px}.hero p{max-width:680px;color:var(--muted);font-size:18px}.actions{display:flex;gap:12px;margin-top:28px;flex-wrap:wrap}.button{display:inline-block;padding:11px 16px;border-radius:8px;text-decoration:none;font-weight:700}.primary{background:var(--teal);color:#06201d}.secondary{border:1px solid var(--line);color:var(--text)}
    .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.card{background:linear-gradient(145deg,#10242d,#0b1a22);border:1px solid var(--line);border-radius:14px;padding:22px}.card .num{color:var(--teal);font:12px ui-monospace,monospace}.card h2{font-size:18px;margin:14px 0 8px}.card p{margin:0;color:var(--muted);font-size:14px}.section{padding:30px 0}.section h2{font-size:24px;letter-spacing:-.03em}.pipeline{display:flex;align-items:stretch;gap:9px;flex-wrap:wrap}.step{flex:1;min-width:135px;background:#0d2027;border:1px solid var(--line);border-radius:10px;padding:15px;font-size:13px}.step b{display:block;color:var(--teal);font:11px ui-monospace,monospace;margin-bottom:7px}.arrow{align-self:center;color:var(--teal)}.note{margin-top:30px;border-left:2px solid var(--amber);padding:12px 16px;background:#201d18;color:#d5c8b3;font-size:13px}.footer{border-top:1px solid var(--line);margin-top:48px;padding-top:20px;color:var(--muted);font-size:12px;display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}
    @media(max-width:760px){.grid{grid-template-columns:1fr}.nav{padding-bottom:35px}.pipeline{display:grid;grid-template-columns:1fr 1fr}.arrow{display:none}.hero{padding-bottom:38px}}
  </style>
</head>
<body><main class="wrap">
  <nav class="nav"><div class="brand">IBVAP <span>/</span> FIELD SYSTEMS</div><div class="status">● SHOWCASE ONLINE</div></nav>
  <header class="hero"><div class="eyebrow">Smart India Hackathon · AI video analytics</div><h1>See the scene.<br>Track what moves.</h1><p>IBVAP turns surveillance footage into structured visual context with object detection, persistent tracking, motion trails, and optional Indian number plate recognition.</p><div class="actions"><a class="button primary" href="#pipeline">Explore the pipeline ↓</a><a class="button secondary" href="/health">Service status ↗</a></div></header>
  <section class="grid" aria-label="Platform capabilities">
    <article class="card"><div class="num">01 / DETECT</div><h2>Objects in context</h2><p>YOLO11 identifies people, vehicles, and other supported object classes, with confidence scores and bounding boxes.</p></article>
    <article class="card"><div class="num">02 / TRACK</div><h2>Persistent identities</h2><p>ByteTrack associates detections across frames. Track history supports motion trails, displacement, velocity, and stationary checks.</p></article>
    <article class="card"><div class="num">03 / RECOGNIZE</div><h2>Optional ANPR</h2><p>Vehicle plate localization, image enhancement, OCR, Indian plate-format parsing, and confidence-weighted readings over time.</p></article>
  </section>
  <section class="section" id="pipeline"><h2>From video to visual signal</h2><div class="pipeline"><div class="step"><b>INPUT</b>Recorded video</div><div class="arrow">→</div><div class="step"><b>VISION</b>YOLO11 detection</div><div class="arrow">→</div><div class="step"><b>IDENTITY</b>ByteTrack IDs</div><div class="arrow">→</div><div class="step"><b>OPTIONAL</b>Plate OCR + voting</div><div class="arrow">→</div><div class="step"><b>OUTPUT</b>Annotated video</div></div></section>
  <aside class="note"><strong>Prototype scope:</strong> This Vercel deployment is the project showcase and service-status page. Video inference remains in the repository's local Python CLI; this page does not accept uploads or run models. Recognition results require human review and are not identity verification.</aside>
  <footer class="footer"><span>IBVAP · Intelligent Border Video Analytics Platform</span><span>AI-assisted video review prototype</span></footer>
</main></body></html>"""
