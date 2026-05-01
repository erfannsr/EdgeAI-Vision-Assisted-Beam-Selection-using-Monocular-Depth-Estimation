import os
import copy
import torch
import numpy as np
from torch.utils.data import random_split
from tqdm import tqdm

import torch.ao.quantization
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

from model import LightDepthNet
from data_loader import NYUDataset

# -- Data Loader
def get_dataloaders(root, batch_size=16):
    full_train = NYUDataset(root, split="train")
    val_size = int(0.1 * len(full_train))
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size])
    test_set = NYUDataset(root, split="test")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

# -- Metrics
def depth_loss(pred, gt):
    mask = gt > 0
    return torch.mean(torch.abs(pred[mask] - gt[mask]))

def compute_metrics(pred, gt):
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    pred = np.clip(pred, a_min=1e-7, a_max=None)
    gt = np.clip(gt, a_min=1e-7, a_max=None)

    eps = 1e-6
    scale = np.median(gt) / (np.median(pred) + eps)
    pred = pred * scale

    rmse = np.sqrt(((pred - gt) ** 2).mean())
    ratio = np.maximum(pred / gt, gt / pred)
    delta1 = (ratio < 1.25).mean()

    return rmse, delta1

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running QAT on {device}...")

    # Load Data
    root = "/home/erfan/2.EdgeAI/Project/Datasets/archive/nyu_data/data"
    train_loader, val_loader, _ = get_dataloaders(root, batch_size=32)

    # Load FP32 Baseline
    model = LightDepthNet()
    model.load_state_dict(torch.load("checkpoints/best_model_relu.pth", map_location=device))
    model.train() # Must be in train mode for QAT prep
    
    # Setup QAT Engine and Config
    torch.backends.quantized.engine = 'qnnpack'
    qconfig_mapping = get_default_qat_qconfig_mapping("qnnpack")
    example_inputs = (torch.randn(1, 3, 224, 224),)

    # Prepare model for QAT (fuses layers and inserts fake-quant nodes)
    print("Tracing and preparing FX Graph for QAT...")
    qat_model = prepare_qat_fx(model, qconfig_mapping, example_inputs)
    qat_model.to(device)

    # Fine-Tuning Setup (Low Learning Rate is critical)
    EPOCHS = 10
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=1e-5)
    criterion = depth_loss

    best_val_rmse = float('inf')

    for epoch in range(EPOCHS):
        # --- TRAIN LOOP ---
        qat_model.train()
        total_loss = 0

        for imgs, depths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            imgs, depths = imgs.to(device), depths.to(device)

            preds = qat_model(imgs)
            loss = criterion(preds, depths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f}")

        # --- VALIDATION LOOP ---
        # To get accurate INT8 metrics, we must convert a copy of the model to true INT8
        qat_model.eval()
        
        print("Converting copy to INT8 for validation...")
        val_model_int8 = convert_fx(copy.deepcopy(qat_model).cpu())
        val_model_int8.eval()

        rmse_list, d1_list = [], []
        
        with torch.no_grad():
            for imgs, depths in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val INT8]"):
                # INT8 model must run on CPU
                imgs, depths = imgs.cpu(), depths.cpu()

                preds = val_model_int8(imgs)
                rmse, d1 = compute_metrics(preds, depths)
                rmse_list.append(rmse)
                d1_list.append(d1)

        avg_rmse = np.mean(rmse_list)
        print(f"Val INT8 RMSE: {avg_rmse:.4f}, δ1: {np.mean(d1_list):.4f}")

        # --- SAVE BEST MODEL ---
        if avg_rmse < best_val_rmse:
            best_val_rmse = avg_rmse
            os.makedirs("checkpoints", exist_ok=True)
            
            # Save the traced INT8 JIT model ready for edge deployment
            traced_int8 = torch.jit.trace(val_model_int8, example_inputs[0].cpu())
            traced_int8.save("checkpoints/qat_lightdepthnet_int8.pt")
            
            print(f"New best INT8 model saved! RMSE = {best_val_rmse:.4f}.")
            
        # Switch back to train mode for the next epoch
        qat_model.train()

    print("\nQAT complete. Best INT8 model saved to checkpoints/qat_lightdepthnet_int8.pt")

if __name__ == "__main__":
    main()