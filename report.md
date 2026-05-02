# Project Report: Vision-Aided IRS Beamforming & Spatial Tracking

## 1. Project Objective
The core goal of this research is to develop a real-time spatial tracking system for **Intelligent Reflecting Surfaces (IRS)** and wireless communication security. The system utilizes computer vision to estimate the 3D coordinates (Azimuth $\theta$ and Distance $d$) of a mobile device, particularly monocular depth estimation, to optimize beamforming parameters and signal-to-noise ratio (SNR).

---

## 2. Hardware & Environment
*   **Development Platform:** 
    * Transitioned from Google Colab with T4 GPU to MacBook Pro M3 Pro utilizing Metal Performance Shaders (MPS) for hardware-accelerated local inference.
    * Monocular depth estimation model is trained on NVIDIA RTX A4000 (16GB) GPU.
*   **Target Deployment:** Raspberry Pi 5.
*   **Software Stack:**
    *   **Python 3.10** with Virtual Environment isolation.
    *   **PyTorch/Ultralytics** for model training and local execution.

---

## 3. Model Development
### A. 2D Object Detection (YOLOv8)
*   **Model:** **YOLOv8m** (Medium) selected for a balance of feature extraction depth and edge performance.
*   **Training Specs:** 50 epochs at a native **$416 \times 416$ resolution**.
*   **Performance Metrics:**
    *   **mAP50:** 0.982 – 0.987.
    *   **Box Loss:** 0.34 – 0.38.

### B. 3D Depth Estimation
*   **Initial Approach:** MiDaS (Small) used for relative disparity mapping.
*   **Advanced Approach:** Transitioned to **LightDepthNet**, a custom-trained monocular depth estimation model for improved domain-specific accuracy.
* **Performance Metrics:**
    * **RMSE** = 0.704
    * **δ1** =  0.693

![model1](images/model1.jpg)
![model2](images/model2.jpg)


### C. Datasets Used
* For YOLO - Object Detection: Custom mobile phone dataset sourced via Roboflow containing 238 annotated images of mobile devices in various orientations, lighting conditions, and hand-held scenarios. It is primarily used to train the 2D object detection component (YOLOv8n) to localize the target for beamforming.

* Monocular Depth Dataset: NYU2 - A large-scale indoor dataset containing 1,449 pairs of aligned RGB images and depth maps. It provides the ground truth depth data required to train the LightDepthNet model to understand indoor spatial geometry and provide floating-point distance estimations - real distance, not relative.

![nyu2_sample](images/nyu2_sample.png)

---

## 4. Technical Achievements & Algorithms
### Spatial Fusion Pipeline
The system implements a "Cookie Cutter" ROI sampling strategy to extract distance data:
1.  **Localization:** YOLOv8 identifies the bounding box $[x_1, y_1, x_2, y_2]$.
2.  **ROI Sampling:** The center **15%** of the bounding box is sampled from the depth map to avoid edge noise from hands or background.
3.  **Median Filtering:** A median value of the ROI is calculated to provide a stable depth prior.

### Signal Smoothing & Calibration
*   **Jitter Reduction:** Implementation of an **Exponential Moving Average (EMA)** filter to stabilize distance readings across frames.
*   **Scale Ambiguity Correction:** Calibration via a `CALIB_FACTOR` to map abstract model outputs to floating-point meters.
*   **Linear Mapping:** Fixed "reversed distance" issues by switching to linear depth scaling optimized for M3 Pro MPS tensors.

---

## 5. Critical Troubleshooting & Lessons Learned
*   **ONNX Architecture Integrity:** Attempting to "hack" ONNX input shapes (e.g., forcing 640x640 to 416x416) causes internal `Concat` layer failures. **Solution:** Models must be exported natively at the target resolution from PyTorch.
*   **Domain Gap:** High mAP in training does not always guarantee detection in blurred video frames. **Solution:** Lowered confidence thresholds and implemented "Active Learning" by including video-extracted frames in the training set.
*   **Hardware Acceleration:** Transitioned from Colab (Cloud) to MBP M3 Pro (MPS) and Hailo-8 (NPU) to achieve low-latency processing required for real-time IRS phase-shift updates.

---

## 6. Next Steps
1.  **Absolute Calibration:** Finalize the `CALIB_FACTOR` using physical tape-measure benchmarks in the lab.
2.  **IRS Integration:** Feed the smoothed distance ($d$) and azimuth ($\theta$) values into the **MDP/Reinforcement Learning** agent. Implement the algorithm into actual IRS hardware.
3.  **HEF Compilation:** Re-compile the natively exported `LightDepthNet.onnx` and `yolov8n_detect.onnx` into Hailo Executable Format for final Raspberry Pi deployment.

---

## 7. References

[1] Q. Liu and S. Zhou, "LightDepthNet: Lightweight CNN Architecture for Monocular Depth Estimation on Edge Devices," in IEEE Transactions on Circuits and Systems II: Express Briefs, vol. 71, no. 4, pp. 2389-2393, April 2024, doi: 10.1109/TCSII.2023.3337369. keywords: {Decoding;Estimation;Merging;Kernel;Computational modeling;Channel estimation;Computational efficiency;Internet of Things;Depth measurement;IoT;neural network;channel pruning;monocular depth estimation},

[2] Roboflow, "Mobile Phone Detection Dataset," [Online]. Available: https://universe.roboflow.com/sahana-qam5q/mobile-phone-ewfpu.

[3] N. Silberman, D. Hoiem, P. Kohli and R. Fergus, "Indoor Segmentation and Support Estimation from RGBD Images," in Proceedings of the European Conference on Computer Vision (ECCV), 2012. https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
