import cv2
import torch
import numpy as np
import time
import onnxruntime as ort

from model import LightDepthNet

# --- 1. SYSTEM CONFIGURATION ---
MODEL_PATH = r"checkpoints\phone_detector_416.onnx" # ONNX model path
RES = 416            # YOLO Model input resolution
HFOV = 62.2          # Camera Horizontal Field of View (Degrees)
NUM_BEAMS = 16       # Size of your IRS Codebook
CALIB_FACTOR = 0.5   # Depth calibration factor
device = 'cpu'

# --- 2. INITIALIZE INFERENCE ENGINES ---
print("Loading Depth model (PyTorch)...", flush=True)
depth_model = LightDepthNet()
depth_model.load_state_dict(torch.load("checkpoints/abs_depth_best.pth", map_location=device))
depth_model = depth_model.to(device)
depth_model.eval()

print("Loading YOLO model (ONNX)...", flush=True)
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

smoothed_dist = None

def depth_inference(model, img, device):
    """Calculates depth map from the input image."""
    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized / 255.0
    tensor = torch.tensor(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        depth = model(tensor)
    return depth.squeeze().cpu().numpy()

# --- 3. INITIALIZE CAMERA HARDWARE ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open webcam")
    exit(1)

print("\n--- PIPELINE LOOP STARTED ---")
print("Press 'q' in the video window to exit.")

try:
    while cap.isOpened():
        t_start = time.perf_counter()
        
        # Capture Frame
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]
        
        #  Depth Inference (PyTorch)
        depth = depth_inference(depth_model, img_rgb, device)
        depth_map = cv2.resize(depth, (orig_w, orig_h))

        #  YOLO Inference (ONNX)
        frame_resized = cv2.resize(img_rgb, (RES, RES))
        input_data = frame_resized.transpose(2, 0, 1)
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32) / 255.0

        outputs = session.run(None, {input_name: input_data})
        
        # Parse ONNX Results
        preds = np.squeeze(outputs[0])
        scores = preds[4, :]
        best_idx = np.argmax(scores)
        max_score = scores[best_idx]

        # Fusion & Beam Calculation
        if max_score > 0.3:  # Detection Threshold
            # Note: ONNX outputs are relative to the 416x416 resized image
            x_c_416, y_c_416, w_416, h_416 = preds[0:4, best_idx]
            
            # Map coordinates back to original frame dimensions for accurate depth mapping
            x_c = int((x_c_416 / RES) * orig_w)
            y_c = int((y_c_416 / RES) * orig_h)
            w = int((w_416 / RES) * orig_w)
            h = int((h_416 / RES) * orig_h)
            
            # Bounding Box Coordinates
            x1, y1 = int(x_c - w/2), int(y_c - h/2)
            x2, y2 = int(x_c + w/2), int(y_c + h/2)

            # Clamp center coordinates safely to image bounds
            cx_clamped = max(0, min(orig_w - 1, x_c))
            cy_clamped = max(0, min(orig_h - 1, y_c))

            # Get Depth of the BBox Center
            center_val = depth_map[cy_clamped, cx_clamped]
            current_dist = center_val * CALIB_FACTOR

            # Smooth depth using Exponential Moving Average
            if smoothed_dist is None:
                smoothed_dist = current_dist
            else:
                smoothed_dist = 0.15 * current_dist + 0.85 * smoothed_dist

            # Calculate Continuous Azimuth Angle & Beam Index
            x_norm = (x_c_416 / RES) - 0.5
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
    cap.release()
    cv2.destroyAllWindows()
    print("Camera hardware released.")