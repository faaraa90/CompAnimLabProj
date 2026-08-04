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

##Installation
...

##Usage
...

##Results
...

##Authors
...
