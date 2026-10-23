# IBVAP — Intelligent Border Video Analytics Platform

**An AI-powered video analytics platform engineered for border security, perimeter defense, and critical infrastructure surveillance.** IBVAP provides real-time and offline multi-stage computer vision pipelines integrating YOLO11 object detection, ByteTrack persistent multi-object tracking, directional Virtual Fence Intrusion Detection, live RTSP/HTTP/webcam ingestion, and custom Automatic Number Plate Recognition (ANPR) for Indian registration plates.

> **SIH Prototype**: Developed for high-stakes perimeter monitoring, tactical boundary intrusion alerts, vehicle tracking, and automated surveillance analytics.

---

## 🌟 Key Capabilities

- **Real-Time Live Feed Ingestion**: Ingests direct hardware webcams (`--cam`), RTSP IP camera streams (`rtsp://`), and mobile HTTP camera feeds (`http://<ip>:8080/video`) with ultra-low latency frame buffering (`CAP_PROP_BUFFERSIZE = 1`).
- **YOLO11 Object Detection**: Ultra-fast object detection for persons, vehicles, and surveillance-critical entities using YOLO11n.
- **ByteTrack Multi-Object Tracking**: Retains identity persistence across frame occlusions, records bottom-center ground contact trajectories, and renders fading motion trails.
- **Virtual Fence Intrusion Detection (Tripwires & Polygons)**:
  - Supports linear tripwires and multi-point polygonal restricted zones.
  - Directional crossing logic: triggers on `top_to_bottom`, `bottom_to_top`, `left_to_right`, `right_to_left`, or bidirectional (`any`).
  - **Dynamic Auto-Centering**: Automatically computes video resolution (e.g., 1080p, 720p, 480p) and positions the virtual perimeter cleanly across the center of the frame.
  - **State-Aware Intruder Highlighting**: Perimeter breaches turn the target's bounding box **RED** with a prominent `INTRUDER #ID` badge until they cross back across the boundary.
  - **Intrusion Snapshot Logging**: Automatically saves high-resolution cropped intruder evidence to `runs/intrusions/`.
- **Custom ANPR (Automatic Number Plate Recognition)**:
  - Custom YOLO license plate localizer trained on Indian vehicle datasets.
  - Multi-pass CLAHE contrast enhancement, unsharp masking, and bilateral filtering.
  - EasyOCR with alphanumeric character confusion matrix post-processing.
  - Validates against Indian standard (RTO state-district-alpha-numeric) and High-Security/BH-series formats.
  - Temporal multi-frame consensus voting for plate read stability.
- **Surveillance Telemetry HUD**: Dark glassmorphic heads-up display reporting real-time FPS, frame index, live timestamp, person/vehicle counts, active breach tallies, and stream health.
- **MLOps & Pipeline Reproducibility**: Integrated DVC data pipeline (`dvc.yaml`) and MLflow SQLite experiment tracker (`sqlite:///mlflow.db`) tracking model loss, precision, recall, and mAP50.

---

## 🏗️ Architecture & Pipeline Flow

```mermaid
flowchart TD
    A[Video Source\nWebcam / RTSP / HTTP / MP4] --> B[YOLO11 Object Detection]
    B --> C{Tracking Mode?}
    C -->|ByteTrack| D[Persistent IDs & Trajectory Engine]
    C -->|Raw| E[Frame-by-Frame Detections]
    
    D --> F{Virtual Fence Enabled?}
    F -->|Yes| G[Geometry Intersection & Direction Logic]
    G --> H{Perimeter Breached?}
    H -->|Yes| I[Flag Active Intruder\nTurn BBox Red & Badge\nSave Crop Snapshot]
    H -->|No / Return| J[Normal Status Tracking]
    F -->|No| J

    D --> K{ANPR Enabled?}
    K -->|Yes| L[Crop Vehicle Regions\nYOLO Plate Localizer]
    L --> M[CLAHE Preprocessing & EasyOCR]
    M --> N[Indian Syntax Validation & Temporal Voting]
    K -->|No| O[Object HUD Overlays]

    I --> P[Surveillance Telemetry HUD]
    J --> P
    N --> P
    O --> P
    E --> P
    P --> Q[Real-Time OpenCV Preview & Processed MP4 Export]
```

---

## 📁 Repository Structure

```text
├── configs/
│   └── anpr_data.yaml         # YOLO training configuration for license plates
├── data/
│   ├── dataset/anpr/          # Train, val, and test image/label splits
│   ├── processed/videos/      # Output annotated videos (gitignored)
│   └── raw/videos/            # Sample surveillance test videos
├── models/
│   ├── anpr/
│   │   ├── plate_detector.pt  # Fine-tuned YOLO license plate weights
│   │   └── plate_detector.py  # Model fine-tuning & MLflow logging script
│   └── detection/
│       ├── detector.py        # YOLO11 detection abstraction
│       └── yolo11n.pt         # Pretrained base YOLO11 weights
├── runs/
│   └── intrusions/            # Auto-saved cropped intruder snapshot images
├── src/
│   ├── analytics/
│   │   ├── fence_zone.py      # Virtual fence geometry, vector math, & directions
│   │   └── intrusion_detector.py # Intrusion state engine, snapshot capture & overlay
│   ├── anpr/
│   │   ├── anpr_pipeline.py   # End-to-end ANPR orchestration
│   │   ├── ocr.py             # EasyOCR execution & character heuristics
│   │   ├── plate_parser.py    # Indian RTO / BH regex & syntax validation
│   │   ├── plate_preprocessor.py # Image enhancement (CLAHE, sharpen, denoise)
│   │   └── plate_tracker.py   # Multi-frame consensus voting per vehicle
│   ├── detection/             # Object detection wrappers
│   ├── tracking/
│   │   ├── object_identity.py # TrackedObject data contract
│   │   ├── tracker.py         # ByteTrack integration & track visualization
│   │   └── trajectory.py      # Ground-point motion history & trails
│   ├── visualization/
│   │   └── video_render.py    # Master stream ingester, HUD renderer & recorder
│   └── main.py                # Unified CLI entry point & Vercel handler
├── dvc.yaml                   # DVC pipeline stages
├── params.yaml                # MLOps hyperparameter declarations
├── requirements.txt           # Python dependencies
└── README.md
```

---

## ⚡ Quickstart & Installation

### 1. Environment Setup

Python **3.10+** and CUDA-enabled PyTorch (for GPU acceleration) are recommended:

```bash
# Clone the repository
git clone https://github.com/MradulSingh109/PROJECTHAILMARY.git
cd PROJECTHAILMARY

# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate

# Install requirements
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **CUDA Verification**: Ensure GPU acceleration is available:
> ```bash
> python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
> ```

---

## 🚀 Running the Analytics Pipeline

### 1. Live Camera Streaming

#### A. Built-in Laptop / USB Webcam
Launch the pipeline on the local webcam with an auto-centered horizontal virtual fence:
```powershell
python src/main.py --cam --fence
```

#### B. Phone Camera / IP Stream (HTTP or RTSP)
Connect your phone via *IP Webcam* or an RTSP CCTV camera:
```powershell
# HTTP stream from mobile camera app
python src/main.py --input "http://192.168.137.193:8080/video" --fence

# RTSP IP Camera stream
python src/main.py --input "rtsp://admin:password@192.168.1.100:554/stream1" --fence
```

---

### 2. Video File Processing with Virtual Fence

Run the border surveillance video with the virtual perimeter fence:
```powershell
python src/main.py --input "data/raw/videos/border_video.mp4" --fence
```

#### Directional Crossing Alerts
Trigger alarms only when intruders cross from the top down into the perimeter:
```powershell
python src/main.py --input "data/raw/videos/border_video.mp4" --fence --fence-dir "top_to_bottom"
```

#### Custom Fence Coordinates
Specify custom pixel coordinates (`x1,y1,x2,y2`):
```powershell
# Center fence for a 1080p stream (Y=540)
python src/main.py --input "http://192.168.137.193:8080/video" --fence --fence-coords "100,540,1820,540"

# Specific tripwire for test video
python src/main.py --input "data/raw/videos/border_video.mp4" --fence --fence-coords "50,150,800,150"
```

---

### 3. Automatic Number Plate Recognition (ANPR)

Run multi-object tracking and vehicle license plate recognition:
```powershell
python src/main.py --input "data/raw/videos/anpr/demo.mp4" --anpr --live
```

---

### 4. Combined Multi-Mission Surveillance

Execute object tracking, Indian ANPR, and Perimeter Virtual Fence simultaneously:
```powershell
python src/main.py --input "data/raw/videos/border_video.mp4" --fence --anpr --live
```

---

## 🎛️ CLI Parameters Reference

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--input`, `-i` | String | `data/raw/videos/college_campus_raw.mp4` | Video file path, RTSP stream URL, or HTTP video URL |
| `--cam`, `--webcam` | Flag | `False` | Uses local connected camera (index `0`) |
| `--fence` | Flag | `False` | Activates Virtual Fence Perimeter Intrusion Detection |
| `--fence-coords` | String | `None` (Auto) | `x1,y1,x2,y2`. If omitted, automatically centers across the video frame |
| `--fence-dir` | Choice | `any` | Direction filter: `any`, `top_to_bottom`, `bottom_to_top`, `left_to_right`, `right_to_left` |
| `--fence-type` | Choice | `tripwire` | Geometry type: `tripwire` (linear line) or `polygon_zone` |
| `--anpr` | Flag | `False` | Activates Automatic Number Plate Recognition on tracked vehicles |
| `--plate-model` | Path | `models/anpr/plate_detector.pt` | Custom trained YOLO license plate model weights |
| `--model`, `-m` | Path | `models/detection/yolo11n.pt` | Base YOLO11 detector weights |
| `--conf`, `-c` | Float | `0.35` | Confidence threshold for object detection |
| `--no-track` | Flag | `False` | Disables ByteTrack (falls back to pure frame detection) |
| `--no-trail` | Flag | `False` | Disables motion trajectory trailing lines |
| `--trail-length`, `-t`| Int | `120` | Maximum historical points for motion path trails |
| `--live` | Flag | `False` | Enables real-time OpenCV desktop display window |
| `--output`, `-o` | Path | Auto | Output annotated MP4 destination (`data/processed/videos/`) |

---

## 🧠 MLOps: License Plate Model Training

Fine-tune the custom YOLO license plate detector and track metrics with MLflow and DVC:

### 1. Execute Training with DVC
```powershell
dvc repro
```

### 2. Manual Training Script
```powershell
python models/anpr/plate_detector.py --epochs 15 --batch 6 --imgsz 640 --lr 0.01
```

### 3. Launch MLflow Dashboard
View training loss curves, precision, recall, and mAP50:
```powershell
mlflow ui --backend-store-uri "sqlite:///mlflow.db" --host 127.0.0.1 --port 5000
```
Open **`http://127.0.0.1:5000`** in your browser.

---

## 🔒 Security & Operational Notice

This project is an advanced computer vision prototype developed for the Smart India Hackathon (SIH). Performance, latency, and OCR precision are subject to physical camera optics, illumination, and GPU specifications. Always comply with regional data governance and privacy policies when capturing surveillance video feeds.

---

**IBVAP | Intelligent Border Video Analytics Platform** — Engineered for Real-Time Threat Detection & Border Perimeter Defense.
