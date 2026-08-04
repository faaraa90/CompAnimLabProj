## Pose Estimation & Biomechanical Assessment
> Summer 2026 - Computer Animation Lab Project

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-007ACC?style=for-the-badge&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

<p align="center">
  <img src="assets/demo.gif" alt="Pose Estimation Demo" width="600"/>
</p>

## Overview & Key Features

This repository implements a **real-time pose estimation and biomechanical assessment system** developed for the Computer Animation Lab (Summer 2026). It leverages MediaPipe and OpenCV to process video streams, track body movement, and calculate precise joint kinematics dynamically for computer animation and motion analysis.

- **Real-time Pose Estimation:** Tracks 33 3D body landmark keypoints using MediaPipe Pose.
- **Biomechanical Kinematics:** Computes real-time joint angles and angular variations during movement.
- **Visual Feedback & Telemetry:** Overlays visual skeletal tracking and telemetry directly onto the video feed.

## Pipeline Architecture

```text
[ Input Video / Webcam Feed ]
              │
              ▼
    ┌──────────────────┐
    │  OpenCV Frame    │ ───► Captures frame 
    │   Preprocessing  │
    └─────────┬────────┘
              │
              ▼
    ┌──────────────────┐
    │  MediaPipe Pose  │ ───► Detects 33 3D body landmark keypoints
    │   Landmark Engine│
    └─────────┬────────┘
              │
              ▼
    ┌──────────────────┐
    │  Biomechanical   │ ───► Calculates joint kinematics & angles
    │   Angle Engine   │      
    └─────────┬────────┘
              │
              ▼
    ┌──────────────────┐
    │ Visualizer &     │ ───► Overlays skeletal joints, angle values,
    │   Telemetry GUI  │      and metrics on frame
    └─────────┬────────┘
              │
              ▼
[ Rendered Output Display / Video File ]
```

## Project Structure
```text
CompAnimLabProj/
├── assets/             # Demo GIFs and images
├── data/               # Sample videos or test inputs
├── src/                # Core processing scripts and modules
│   ├── __init__.py
│   └── pose_detector.py
├── main.py             # Main execution entry point
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
└── .gitignore          # Git exclusion file

## Dataset
The video files are not included in this repository because of their size.
you can download the videos from the following [Google Drive folder] (https://drive.google.com/drive/folders/1H4ockyZ8WZsETMHapkkToH6sEciXVscJ?usp=drive_link)
please download the dataset and place the video files in the 'data/videos/' directory before running the code. 

## Getting Started

Follow these steps to get a local copy up and running on your machine.

### Prerequisites

* **Python 3.10+** installed on your system.
* **Git** installed on your system.

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/faaraa90/CompAnimLabProj.git](https://github.com/faaraa90/CompAnimLabProj.git)
   cd CompAnimLabProj
   ```

2. **Create and activate a virtual environment (Recommended):**
   * **Windows (PowerShell / Command Prompt):**
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```
   * **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

### Usage

To run the main pose estimation and biomechanical tracker, execute:

```bash
python main.py
```

> **Tip:** Ensure your webcam is connected or specify a sample video path inside `data/` if configured in `main.py`.



##Results
...

##Authors
...
