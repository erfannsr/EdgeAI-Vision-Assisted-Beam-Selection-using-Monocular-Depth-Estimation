import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset
import cv2
import numpy as np
from torch.utils.data import random_split
 
from model import LightDepthNet
 
def inference(model, img, device):
    model.eval()
 
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
 
    with torch.no_grad():
        depth = model(img)
 
    return depth.squeeze().cpu().numpy()
 
 
if __name__ == "__main__":
    
    device = 'cpu'


    # --- Load INT8 Quantized ---
    int8_path = "checkpoints/qat_lightdepthnet_int8.pt"
    print(f"Loading INT8 JIT Model from {int8_path}...")
    model_int8 = torch.jit.load(int8_path, map_location=device)
    
    model = model_int8
    
    model.eval()
    
    
    print("Model loaded.")
 
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam")
        exit(1)
 
    # --- Matplotlib setup ---
    plt.ion()   # interactive mode
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
 
    ax1, ax2 = axes
    im1 = ax1.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    im2 = ax2.imshow(np.zeros((480, 640)), cmap='plasma')
 
    ax1.set_title("RGB")
    ax2.set_title("Depth")

    ax1.axis('off')
    ax2.axis('off')
 
    plt.show()
 
    print("Starting loop... (Ctrl+C to stop)")
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
 
        # --- Inference ---
        t0 = time.perf_counter()
        depth = inference(model, img, device)
        elapsed_ms = (time.perf_counter() - t0) * 1000
 
        print(f"Latency: {elapsed_ms:.2f} ms")
 
        depth = cv2.resize(depth, (orig_w, orig_h))
 
        im1.set_data(img)
        im2.set_data(depth)
 
        im2.set_clim(vmin=np.percentile(depth, 1),
                     vmax=np.percentile(depth, 99))
 
        plt.pause(0.001)   
 
        time.sleep(0.1)
 
    cap.release()
 