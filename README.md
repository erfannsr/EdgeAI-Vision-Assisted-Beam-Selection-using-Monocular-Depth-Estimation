# EdgeAI-Vision-Assisted-Beam-Selection-using-Monocular-Depth-Estimation

> Real-time spatial tracking for **Intelligent Reflecting Surfaces (IRS)** using computer vision and edge-deployable deep learning.

[![Demo Video](https://img.youtube.com/vi/miAZP431W4k/maxresdefault.jpg)](https://youtu.be/miAZP431W4k?si=SHefM4XC3ern5umR)


*Click the thumbnail to watch the demo on YouTube.*

---

## 📖 Overview

This project builds a pipeline that estimates the **3D coordinates** (Azimuth θ and Distance d) of a mobile device using monocular depth estimation and 2D object detection. These spatial estimates are fed into an IRS controller to dynamically optimize beamforming phase shifts and maximize the signal-to-noise ratio (SNR) in real time.

The system is designed to run on edge hardware (Raspberry Pi 5) using compiled neural network models and hardware-accelerated inference.

---

## 🏗️ System Architecture

```
Camera Feed
    │
    |
    |       ┌─────────────┐     ┌──────────────────┐
    │ ────> │  YOLOv8m    │────>│  Bounding Box    │
    |       │  Detection  │     │  [x1, y1, x2, y2]│
    |       └─────────────┘     └────────┬─────────┘
    |                                    │
    |                           ┌────────▼─────────┐
    |                           │  Cookie Cutter   │
    |                           │  ROI (center 15%)│
    |                           └────────┬─────────┘
    |                                    │
    |       ┌─────────────┐     ┌────────▼─────────┐
    | ────> │LightDepthNet│────>│    Depth Map     │
            │             │     │Box corresp.center│
            └─────────────┘     └────────┬─────────┘
                                         │
                                ┌────────▼─────────┐
                                │   EMA Smoothing  │
                                │ + CALIB_FACTOR   │
                                └────────┬─────────┘
                                         │
                                ┌────────▼─────────┐
                                │  (θ, d) Output   │──▶ IRS 
                                └──────────────────┘
```

---
## ✨ Key Features

- **2D Object Detection** — YOLOv8m trained on a custom mobile phone dataset 
- **Monocular Depth Estimation** — Custom-trained LightDepthNet on the NYU Depth V2 dataset 
- **Cookie Cutter ROI Sampling** — Samples only the center 15% of the detected bounding box to suppress hand/background noise
- **EMA Signal Smoothing** — Exponential Moving Average filter for stable, jitter-free distance readings
- **Scale Calibration** — `CALIB_FACTOR` maps model outputs to real-world floating-point meters
- **Edge-Ready** — Models compiled to ONNX and .pt for Raspberry Pi 5 deployment

--- 


## 🛠️ Hardware & Environment

| Component | Details |
|---|---|
| **Training GPU** | NVIDIA RTX A4000 (16 GB) |
| **Target Deployment** | Raspberry Pi 5 |
| **Python** | 3.10  |
| **Frameworks** | PyTorch, Ultralytics |


## 🤖 Models

### YOLOv8m — 2D Object Detection

| Parameter | Value |
|---|---|
| Architecture | YOLOv8m (Medium) |
| Input Resolution | 416 × 416 |
| Training Epochs | 50 |
| mAP50 | 0.982 – 0.987 |
| Box Loss | 0.34 – 0.38 |

Trained on a custom Roboflow dataset of **238 annotated mobile phone images** across varied orientations, lighting conditions, and hand-held scenarios.

### LightDepthNet — Monocular Depth Estimation

| Parameter | Value |
|---|---|
| Architecture | LightDepthNet |
| Training Dataset | NYU Depth V2 (1,449 RGB-D pairs) |
| RMSE | 0.704 |
| δ1 | 0.693 |

*Please refer to the report.md for more details about the model architecture, and more details about NYU2 dataset.*

---



## 🚀 Getting Started
<!-- To use the model, either on a local PC or  -->
### Prerequisites

```bash
python3.10 -m venv edgeAI_beamSelection
source edgeAI_beamSelection/bin/activate
pip install torch ultralytics opencv-python numpy
```

### To run the full pipeline on a local PC:

```bash
python full_pipeline_PC.py
```

### To run the full pipeline on a RaspberryPi:
```bash
cd RPi
python full_pipeline.py
```

### To run the real-time depth estimation model on a local PC:
```bash
python inference_live.py
```

### To run the real-time depth estimation model on a RaspberryPi:
```bash
cd RPi
python inference_live_rpi.py
```

---



## 📚 References

1. Q. Liu and S. Zhou, "LightDepthNet: Lightweight CNN Architecture for Monocular Depth Estimation on Edge Devices," *IEEE Transactions on Circuits and Systems II*, vol. 71, no. 4, pp. 2389–2393, April 2024. [doi:10.1109/TCSII.2023.3337369](https://doi.org/10.1109/TCSII.2023.3337369)

2. Roboflow, "Mobile Phone Detection Dataset." [Online]. Available: [https://universe.roboflow.com/sahana-qam5q/mobile-phone-ewfpu](https://universe.roboflow.com/sahana-qam5q/mobile-phone-ewfpu)

3. N. Silberman, D. Hoiem, P. Kohli and R. Fergus, "Indoor Segmentation and Support Estimation from RGBD Images," *ECCV*, 2012. [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)

---

## 📄 License

This project is made in an effort for the course  `CP 330: Edge AI` at Indian Institute of Science.
Link to course website: https://www.samy101.com/edge-ai-26/ 
