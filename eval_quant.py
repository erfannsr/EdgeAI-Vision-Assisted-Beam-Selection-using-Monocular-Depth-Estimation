import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import LightDepthNet
from data_loader import NYUDataset

def compute_metrics(pred, gt):
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()

    # avoid division by zero
    pred = np.clip(pred, a_min=1e-7, a_max=None)
    gt = np.clip(gt, a_min=1e-7, a_max=None)

    # scale alignment
    eps = 1e-6
    scale = np.median(gt) / (np.median(pred) + eps)
    pred = pred * scale

    rmse = np.sqrt(((pred - gt) ** 2).mean())
    ratio = np.maximum(pred / gt, gt / pred)
    delta1 = (ratio < 1.25).mean()

    return rmse, delta1

def get_model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

def main():
    device = torch.device('cpu')
    torch.backends.quantized.engine = 'qnnpack'

    # --- Load Data ---
    root = "/home/erfan/2.EdgeAI/Project/Datasets/archive/nyu_data/data"
    print("Loading NYU Test Set...")
    test_set = NYUDataset(root, split="test")
    # Batch size 1 is best for measuring accurate per-frame latency
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4) 

    # --- Load FP32 Baseline ---
    fp32_path = "checkpoints/best_model_relu.pth"
    print(f"Loading FP32 Model from {fp32_path}...")
    model_fp32 = LightDepthNet()
    model_fp32.load_state_dict(torch.load(fp32_path, map_location=device))
    model_fp32.to(device)
    model_fp32.eval()

    # --- Load INT8 Quantized ---
    int8_path = "checkpoints/qat_lightdepthnet_int8.pt"
    print(f"Loading INT8 JIT Model from {int8_path}...")
    model_int8 = torch.jit.load(int8_path, map_location=device)
    model_int8.eval()

    # --- Evaluation Trackers ---
    metrics = {
        "fp32": {"rmse": [], "d1": [], "times": []},
        "int8": {"rmse": [], "d1": [], "times": []}
    }

    print("\nRunning Inference...")
    with torch.no_grad():
        for imgs, depths in tqdm(test_loader, desc="Evaluating"):
            imgs, depths = imgs.to(device), depths.to(device)

            # FP32 Inference
            start_t = time.time()
            preds_fp32 = model_fp32(imgs)
            metrics["fp32"]["times"].append(time.time() - start_t)
            
            rmse_f, d1_f = compute_metrics(preds_fp32, depths)
            metrics["fp32"]["rmse"].append(rmse_f)
            metrics["fp32"]["d1"].append(d1_f)

            # INT8 Inference
            start_t = time.time()
            preds_int8 = model_int8(imgs)
            metrics["int8"]["times"].append(time.time() - start_t)

            rmse_i, d1_i = compute_metrics(preds_int8, depths)
            metrics["int8"]["rmse"].append(rmse_i)
            metrics["int8"]["d1"].append(d1_i)

    # --- Report ---
    print("\n" + "="*50)
    print(" QUANTIZATION EVALUATION REPORT")
    print("="*50)
    
    # 1. File Size
    size_fp32 = get_model_size_mb(fp32_path)
    size_int8 = get_model_size_mb(int8_path)
    print(f"File Size:")
    print(f"  FP32: {size_fp32:.2f} MB")
    print(f"  INT8: {size_int8:.2f} MB ({(size_int8/size_fp32)*100:.1f}% of original)")
    
    # 2. Performance (dropping first 10 warmup frames for accurate timing)
    fps_fp32 = 1.0 / np.mean(metrics["fp32"]["times"][10:])
    fps_int8 = 1.0 / np.mean(metrics["int8"]["times"][10:])
    print(f"\nLatency (CPU, Batch=1):")
    print(f"  FP32: {np.mean(metrics['fp32']['times'][10:])*1000:.2f} ms/iter ({fps_fp32:.1f} FPS)")
    print(f"  INT8: {np.mean(metrics['int8']['times'][10:])*1000:.2f} ms/iter ({fps_int8:.1f} FPS)")
    print(f"  Speedup: {fps_int8/fps_fp32:.2f}x")

    # 3. Accuracy
    print(f"\nAccuracy Metrics:")
    print(f"  FP32 -> RMSE: {np.mean(metrics['fp32']['rmse']):.4f} | δ1: {np.mean(metrics['fp32']['d1']):.4f}")
    print(f"  INT8 -> RMSE: {np.mean(metrics['int8']['rmse']):.4f} | δ1: {np.mean(metrics['int8']['d1']):.4f}")
    
    # Degradation calculation
    rmse_diff = np.mean(metrics['int8']['rmse']) - np.mean(metrics['fp32']['rmse'])
    d1_drop = np.mean(metrics['fp32']['d1']) - np.mean(metrics['int8']['d1'])
    print(f"\nDegradation:")
    print(f"  RMSE Penalty: +{rmse_diff:.4f}")
    print(f"  δ1 Drop:      -{d1_drop:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()