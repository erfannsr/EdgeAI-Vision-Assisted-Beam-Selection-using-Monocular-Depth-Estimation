import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import h5py

class NYUDataset(Dataset):
    def __init__(self, mat_filepath):
        # We only need the path to the .mat file now.
        self.mat_filepath = mat_filepath
        self.length = 1449 # The exact number of images in the .mat file

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Open lazily in the worker to avoid Multiprocessing pickling errors
        with h5py.File(self.mat_filepath, 'r') as f:
            # Transpose MATLAB arrays to standard Python (H, W, C)
            img = np.array(f['images'][idx]).transpose((2, 1, 0))
            
            # Absolute metric depth in meters (Float32)
            depth = np.array(f['depths'][idx]).transpose((1, 0)).astype(np.float32)

        # Standard preprocessing
        img = cv2.resize(img, (224, 224))
        # Use INTER_NEAREST for depth to avoid interpolating new fake depth values
        depth = cv2.resize(depth, (224, 224), interpolation=cv2.INTER_NEAREST)

        img = img / 255.0

        # Convert to tensors
        img = torch.tensor(img).permute(2, 0, 1).float()
        depth = torch.tensor(depth).unsqueeze(0).float()

        return img, depth
