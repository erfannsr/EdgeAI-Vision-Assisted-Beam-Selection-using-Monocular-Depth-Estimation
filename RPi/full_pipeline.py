import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
import sys

from model import LightDepthNet


# --- 1. THE ABI BYPASS HACK ---
# Prevents the "NumPy 96 vs 88" crash when importing Picamera2
try:
    import simplejpeg
except ValueError:
    class MockSimpleJpeg:
        def decode_jpeg(*args, **kwargs): return None
        def encode_jpeg(*args, **kwargs): return None
    sys.modules['simplejpeg'] = MockSimpleJpeg()

from picamera2 import Picamera2


# --- 1. SYSTEM CONFIGURATION ---
RES = 416            # YOLO Model input resolution
HFOV = 62.2          # Camera Horizontal Field of View (Degrees)
NUM_BEAMS = 16       # Size of your IRS Codebook
CALIB_FACTOR = 0.5   # Depth calibration factor
device = 'cpu'

# --- 2. INITIALIZE INFERENCE ENGINES ---
print("Loading Depth model...", flush=True)
depth_model = LightDepthNet()
depth_model.load_state_dict(torch.load("checkpoints/abs_depth_best.pth", map_location=device))
depth_model = depth_model.to(device)
depth_model.eval()

print("Loading YOLO model...", flush=True)
yolo_model = YOLO("checkpoints/yolo_best.pt").to(device)

smoothed_dist = None

def depth_inference(model, img, device):
    """Calculates depth map from the input image."""
    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized / 255.0

    tensor = torch.tensor(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        depth = model(tensor)

    return depth.squeeze().cpu().numpy()

# --- 4. INITIALIZE CAMERA HARDWARE ---
print("Waking up Picamera2...", flush=True)
picam2 = Picamera2()
# We force the ISP to do the resizing, saving valuable CPU cycles
config = picam2.create_video_configuration(main={"format": "RGB888", "size": (RES, RES)})
picam2.configure(config)
picam2.start()

print("\n--- PIPELINE LOOP STARTED ---")
print("Press 'q' in the video window to exit.")

try:
    while True:
        t_start = time.perf_counter()
        
        
        frame_rgb = picam2.capture_array() 
        img_rgb = frame_rgb
        frame = frame_rgb
        h, w, _ = img_rgb.shape

        # Step B: Depth Inference
        depth = depth_inference(depth_model, img_rgb, device)
        depth_map = cv2.resize(depth, (w, h))

        # Step C: YOLO Inference
        results = yolo_model.predict(img_rgb, conf=0.3, imgsz=416, device=device, verbose=False)

        # Step D: Fusion & Beam Calculation
        for box in results[0].boxes:
            # Get Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Clamp coordinates to image bounds
            cx_clamped = max(0, min(w - 1, cx))
            cy_clamped = max(0, min(h - 1, cy))

            # Get Depth of the BBox Center
            center_val = depth_map[cy_clamped, cx_clamped]
            current_dist = center_val * CALIB_FACTOR

            # Smooth depth using Exponential Moving Average
            if smoothed_dist is None:
                smoothed_dist = current_dist
            else:
                smoothed_dist = 0.15 * current_dist + 0.85 * smoothed_dist

            # Calculate Continuous Azimuth Angle & Beam Index
            # Using actual image width 'w' instead of 416 since Ultralytics maps boxes to original dimensions
            x_norm = (cx / w) - 0.5
            angle = x_norm * HFOV
            
            beam_index = int((x_norm + 0.5) * NUM_BEAMS)
            beam_index = max(0, min(beam_index, NUM_BEAMS - 1)) # Clamp to valid range

            # Step E: Visualization Overlay
            # 1. Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 2. Draw Depth Info
            cv2.putText(frame, f"Depth: {smoothed_dist:.2f}m", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 3. Draw Beam Info
            cv2.putText(frame, f"BEAM {beam_index} ({angle:.1f} deg)", (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Overlay System Metrics
        latency = (time.perf_counter() - t_start) * 1000
        cv2.putText(frame, f"Latency: {latency:.0f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"IRS Codebook: {NUM_BEAMS} Beams", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show Output Window
        cv2.imshow("Vision-Aided Beamforming & Depth Controller", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nPipeline stopped by user.")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera hardware released.")