import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
import os
import nibabel as nib

def zScoreNormalize(image):
        mean = image.mean()
        std = image.std()
        if std > 0:
            image = (image - mean) / std
        else:
            image = image - mean
        return image

def RandomFlip(image, mask):
    axes = [0, 1, 2]  # D, H, W axes
    for axis in axes:
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=axis)
            mask = np.flip(mask, axis=axis)
    return image, mask

def RandomRotate_90(image, mask):
    k = np.random.randint(0, 4)  # 0, 90, 180, 270 degrees
    axes = (1, 2)  # rotate in-plane (H, W)
    image = np.rot90(image, k, axes)
    mask = np.rot90(mask, k, axes)
    return image, mask

def TrainingTransform(image, mask):
    image, mask = RandomFlip(image, mask)
    image, mask = RandomRotate_90(image, mask)
    image = zScoreNormalize(image)

    return image, mask

def TestTransform(image, mask):
    image = zScoreNormalize(image)

    return image, mask

def Resize3dTensor(img_tensor, target_shape=(128,128,128), mode_type='trilinear'):
    """
    img_tensor: torch tensor of shape (C, D, H, W)
    """
    img_tensor = img_tensor.unsqueeze(0)  # add batch dim
    img_resized = F.interpolate(img_tensor, size=target_shape, mode=mode_type)
    return img_resized.squeeze(0)
    
def to_channels(label_slice, dtype=np.float32):
    """Convert 2D label slice to one-hot channels."""
    num_classes = int(label_slice.max()) + 1
    out = np.zeros((num_classes,) + label_slice.shape, dtype=dtype)
    for c in range(num_classes):
        out[c] = (label_slice == c)
    return out

class HipMriDataset2D(Dataset):
    """Dataset for pre-saved 2D slices."""

    def __init__(self, image_path, mask_path, transform=None):
        self.image_paths = sorted([os.path.join(image_path, f) for f in os.listdir(image_path)])
        self.mask_paths = sorted([os.path.join(mask_path, f) for f in os.listdir(mask_path)])
        self.transform = transform

        assert len(self.image_paths) == len(self.mask_paths), "Number of images and masks must match"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load 2D slice
        image = nib.load(self.image_paths[idx]).get_fdata().astype(np.float32)
        mask = nib.load(self.mask_paths[idx]).get_fdata().astype(np.uint8)

        # Apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)

        # Convert mask to one-hot channels
        mask = to_channels(mask, dtype=np.uint8)

        # Convert to tensors
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()  # [1, H, W]
        mask_tensor = torch.from_numpy(mask).float()                  # [C, H, W]

        return image_tensor, mask_tensor