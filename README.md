**Player-Tracking-and-Player-Stats-Analysis-in-Tennis**

A cost-effective, vision-based system for tracking tennis players and generating performance analytics from standard broadcast video — no specialized hardware required.

## Overview

Professional player-tracking systems like Hawk-Eye rely on multiple calibrated high-speed cameras, making them financially inaccessible for most players, coaches, and academies. This project builds an automated alternative that extracts detailed performance metrics speed, distance covered, movement heatmaps, and fatigue trends from a single standard broadcast video, using only open-source tools.

## Key Results

|Model|Metric|Value|
|-|-|:-:|
|Player Detection (YOLOv8)|Precision / Recall / F1|96.9% / 97.9% / 97.4%|
|Court Keypoint Detection (YOLOv8-Pose)|mAP50 / mAP75|97.48% / 97.34%|
|Court Keypoint Detection|Mean Pixel Error|3 px|

Full metrics breakdown: [`results/metrics.md`](results/metrics.md)



## How It Works

1. **Player Detection** — YOLOv8 (transfer-learned from COCO) detects players in each frame
2. **Player Tracking** — An adaptive Kalman filter predicts and smooths player positions across frames, handling occlusion and fast movement
3. **Identity Association** — The Hungarian algorithm matches detections to existing tracks frame-to-frame
4. **Court Keypoint Detection** — A fine-tuned YOLOv8-Pose model detects 14 court landmarks per frame
5. **Homography Mapping** — RANSAC + Direct Linear Transform maps image coordinates to a real-world top-down court view
6. **Stats Analysis** — Speed, distance, max speed, fatigue index, and positional heatmaps are computed from the tracked, mapped positions

```
Input Video → Player Detection → Kalman Tracking → Hungarian Association ─┐
           → Court Keypoint Detection → Homography Conversion ────────────┴→ Player Movement Analysis → Output Video
```

## Features

* Upload/clip a tennis match video (YouTube URL supported) via a simple desktop GUI
* Player tracking with occlusion handling
* Real-world speed and distance estimation via homography-corrected coordinates
* Movement heatmaps per player
* Fatigue index tracking within a clip
* No paid software or specialized cameras required

## Tech Stack

* **Detection/Pose:** YOLOv8, YOLOv8-Pose (Ultralytics)
* **Tracking:** Custom adaptive Kalman filter, Hungarian algorithm
* **Geometry:** OpenCV (homography, RANSAC, DLT)
* **Core:** Python, PyTorch, NumPy, SciPy
* **Video handling:** yt-dlp, ffmpeg
* **UI:** Tkinter desktop app



## Datasets

* **Player detection:** 4,166 images (Roboflow, racket-sports domain)
* **Court keypoints:** 8,841 images with 14-point court annotations (Sergey Kosolapov's dataset)
* Datasets are not redistributed in this repo due to size/licensing  see [`docs/`](docs/) for sourcing details.



## Usage

```bash
python src/courtvision_ui.py
```

Paste a YouTube URL or local video path, select a time range, and run analysis. Output is a stats-annotated video with an embedded minimap.

## Project Structure

```
├── src/            # Core pipeline (detection, tracking, homography, analysis)
├── scripts/        # One-off data prep / training scripts
├── docs/           # Project report, documentation
├── results/        # Metrics, evaluation curves
├── samples/        # Sample output videos/images
└── requirements.txt
```

## Limitations

* Assumes a standard broadcast camera angle; extreme zoom or unusual angles reduce homography accuracy
* Single-camera setup (no 3D triangulation)
* Fatigue index is relative to the clip analyzed, not an absolute measure
* 

## Related Publication

Core techniques from this project (YOLOv8-based tracking, adaptive Kalman filtering, homography-based court mapping) were extended and published as a peer-reviewed research paper:

**"Enhancing Tennis Player Tracking Accuracy Using a Vision-Based Framework with YOLOv8, Adaptive Kalman Filtering and Homography-Based Court Mapping"**
Published in *InJET*, Vol. 3, No. 2, 2026 (KEC Conference 2026 Special Issue)
DOI: [10.3126/injet.v3i2.95782](https://doi.org/10.3126/injet.v3i2.95782)

If you use this work, please cite the paper above.



## License

This project is licensed under the MIT License see [LICENSE](LICENSE) for details.

