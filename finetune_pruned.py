import os
import torch
import numpy as np
from torch.utils.data import random_split
from tqdm import tqdm

# Note: We do NOT import LightDepthNet from model.py here because 
# the saved model object already contains its modified architecture definition.
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

# -- Loss and helper functions
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

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load dataset
    root = "/home/erfan/2.EdgeAI/Project/Datasets/archive/nyu_data/data"
    train_loader, val_loader, test_loader = get_dataloaders(root, batch_size=64)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # 1. LOAD THE ENTIRE PRUNED MODEL (Architecture + Weights)
    print("Loading physically pruned model...")
    model = torch.load("checkpoints/pruned_lightdepthnet.pth", weights_only=False)
    model = model.to(device)

    # 2. FINE-TUNING HYPERPARAMETERS
    # Using a much lower learning rate
    EPOCHS = 15 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = depth_loss

    best_val_rmse = float('inf')
    best_epoch = 0

    for epoch in range(EPOCHS):
        # TRAIN LOOP
        model.train()
        total_loss = 0

        for imgs, depths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, depths = imgs.to(device), depths.to(device)

            preds = model(imgs)
            loss = criterion(preds, depths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f}")

        # VALIDATION LOOP
        model.eval()
        rmse_list, d1_list = [], []
        val_loss = 0

        with torch.no_grad():
            for imgs, depths in val_loader:
                imgs, depths = imgs.to(device), depths.to(device)

                preds = model(imgs)
                loss = criterion(preds, depths)

                rmse, d1 = compute_metrics(preds, depths)
                rmse_list.append(rmse)
                d1_list.append(d1)

                val_loss += loss.item()

        avg_rmse = np.mean(rmse_list)
        print(f"Val RMSE: {avg_rmse:.4f}, δ1: {np.mean(d1_list):.4f}")

        # 3. SAVE THE WHOLE MODEL AGAIN
        if avg_rmse < best_val_rmse:
            best_val_rmse = avg_rmse
            # We save the entire model object again so it can be exported to JIT/ONNX later
            torch.save(model, "checkpoints/best_pruned_finetuned.pth")
            print(f"New best model saved! RMSE = {best_val_rmse:.4f}.")
            best_epoch = epoch
    
    print(f"Training complete. Best model saved at epoch {best_epoch+1}.")