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

from picamera2 import Picamera2
 
import matplotlib
matplotlib.use('TkAgg')
            
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
    
    model = LightDepthNet()
    model.load_state_dict(torch.load("checkpoints/abs_depth_best.pth", map_location=device))
    model = model.to(device)
    model.eval()

    
    print("Model loaded.")
 
    picam2 = Picamera2()

    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    time.sleep(2)  # camera warmup
 
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
        # --- Capture frame from Pi camera ---
        frame = picam2.capture_array()   # already RGB888
        # img = frame  # already RGB

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = rgb

        orig_h, orig_w = img.shape[:2]
 
        # --- Inference ---
        t0 = time.perf_counter()
        depth = inference(model, img, device)
        elapsed_ms = (time.perf_counter() - t0) * 1000
 
        print(f"Latency: {elapsed_ms:.2f} ms")
 
        depth = cv2.resize(depth, (orig_w, orig_h))
 
        # --- Update plots (NO plt.show here) ---
        im1.set_data(img)
        im2.set_data(depth)
 
        # Optional: normalize for better visualization
        im2.set_clim(vmin=np.percentile(depth, 1),
                     vmax=np.percentile(depth, 99))
 
        plt.pause(0.001)   # THIS replaces waitKey
 
        # Optional slowdown (since you said latency doesn't matter)
        time.sleep(0.1)
 