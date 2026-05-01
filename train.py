import os
import glob
import random
import torch
import cv2
import numpy as np
from torch.utils.data import random_split


import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from model import LightDepthNet
from data_loader import NYUDataset

# -- Data Loader
def get_dataloaders(mat_filepath, batch_size=16):
    full_dataset = NYUDataset(mat_filepath)

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    # Use a fixed generator seed 
    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(
        full_dataset, 
        [train_size, val_size, test_size], 
        generator=generator
    )

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

    rmse = np.sqrt(((pred - gt) ** 2).mean())

    ratio = np.maximum(pred / gt, gt / pred)
    delta1 = (ratio < 1.25).mean()

    return rmse, delta1


def inference(model, img, device):
    model.eval()

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        depth = model(img)

    return depth.squeeze().cpu().numpy()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load dataset
    # Point directly to the .mat file instead of the directory
    mat_filepath = "/home/erfan/2.EdgeAI/Project/Datasets/NYU2/nyu_depth_v2_labeled.mat" # <-- Update this if your .mat is saved elsewhere
    
    # Generate dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(mat_filepath, batch_size=64)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}")

    # initialize model
    model = LightDepthNet()
    model = model.to(device)

    # Train the model: 
    EPOCHS = 500
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # criterion = nn.L1Loss()
    criterion = depth_loss

    best_val_loss = float('inf')
    best_epoch=0

    for epoch in range(EPOCHS):
        # TRAIN LOOP
        model.train()
        total_loss = 0

        for imgs, depths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} | best_val_loss: {best_val_loss:.4f} (epoch {best_epoch+1})"):
            imgs, depths = imgs.to(device), depths.to(device)


            preds = model(imgs)


            loss = criterion(preds, depths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f}")

        # print(f"imgs.shape: {imgs.shape}, imgs.min: {imgs.min()}, imgs.max: {imgs.max()}")
        # print(f"depths.shape: {depths.shape}, depths.min: {depths.min()}, depths.max: {depths.max()}")
        # print(f"preds.shape: {preds.shape}, preds.min: {preds.min()}, preds.max: {preds.max()}")
        # exit()

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

        val_loss = val_loss / len(val_loader)

        print(f"Val RMSE: {np.mean(rmse_list):.4f}, δ1: {np.mean(d1_list):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/abs_depth_best.pth")
            print(f"new best model saved as checkpoints/abs_depth_best.pth, val_loss = {best_val_loss:.4f}.")
            best_epoch = epoch
    
    print("Training complete. Best model saved as 'checkpoints/abs_depth_best.pth'.")
