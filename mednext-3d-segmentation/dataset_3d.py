"""
PyTorch 3D Volumetric Dataset with 3D Spatial & Intensity Augmentations
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class VolumetricAugmentations3D:
    """
    3D Medical Augmentation pipeline:
    - Random 3D spatial flips (D, H, W)
    - Random 90-degree rotations in the axial (H-W) plane
    - Random intensity jitter & contrast scaling
    """
    def __init__(self, p_flip: float = 0.5, p_rot: float = 0.5, p_intensity: float = 0.3):
        self.p_flip = p_flip
        self.p_rot = p_rot
        self.p_intensity = p_intensity

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        # image and mask shape: (D, H, W)

        # 1. Random 3D Flips
        if random.random() < self.p_flip:
            axis = random.choice([0, 1, 2])  # flip depth, height, or width
            image = np.flip(image, axis=axis)
            mask = np.flip(mask, axis=axis)

        # 2. Random 90-degree rotations in axial plane (H, W)
        if random.random() < self.p_rot:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k=k, axes=(1, 2))
            mask = np.rot90(mask, k=k, axes=(1, 2))

        # 3. Random Intensity Scaling (Image only)
        if random.random() < self.p_intensity:
            scale = random.uniform(0.85, 1.15)
            shift = random.uniform(-0.05, 0.05)
            image = np.clip(image * scale + shift, 0.0, 1.0)

        return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class LIDC3DDataset(Dataset):
    """
    PyTorch Dataset for 3D nodule volumes stored as .npz cubes.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        cubes_dir: str,
        is_train: bool = True,
        augment: bool = True
    ):
        self.df = dataframe.reset_index(drop=True)
        self.cubes_dir = cubes_dir
        self.is_train = is_train
        self.augment = VolumetricAugmentations3D() if (is_train and augment) else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.cubes_dir, row["filepath"])

        data = np.load(file_path)
        image = data["image"].astype(np.float32)  # (D, H, W)
        mask = data["mask"].astype(np.float32)    # (D, H, W)

        # Apply 3D data augmentation if enabled
        if self.augment is not None:
            image, mask = self.augment(image, mask)

        # Add channel dimension: (1, D, H, W)
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor

