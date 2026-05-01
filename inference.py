import os
import glob
import random
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import time


import torch.nn as nn
import torch.nn.functional as F


from tqdm import tqdm

from model import LightDepthNet
from data_loader import NYUDataset


    
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


def depth_loss(pred, gt):
    mask = gt > 0
    return torch.mean(torch.abs(pred[mask] - gt[mask]))

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
    # device = 'cpu'
    # -- Data Loaders --
    root = "/home/erfan/2.EdgeAI/Project/Datasets/archive/nyu_data/data"
    _, val_loader, test_loader = get_dataloaders(root)

    print(f"test_loader: {len(test_loader.dataset)}")

    model = LightDepthNet()

    model.load_state_dict(torch.load("checkpoints/abs_depth_best.pth"))
    model = model.to(device)
    print(f"model loaded successfully.")    


    criterion = depth_loss
    rmse_list, d1_list = [], []
    test_loss = 0

    with torch.no_grad():
        for imgs, depths in test_loader:
            imgs, depths = imgs.to(device), depths.to(device)

            preds = model(imgs)
            loss = criterion(preds, depths)

            rmse, d1 = compute_metrics(preds, depths)
            rmse_list.append(rmse)
            d1_list.append(d1)

            test_loss += loss.item()

    test_loss = test_loss / len(test_loader)

    print(f"test RMSE: {np.mean(rmse_list):.4f}, δ1: {np.mean(d1_list):.4f}")