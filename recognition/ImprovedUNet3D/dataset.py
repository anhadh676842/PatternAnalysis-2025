import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
import os
import nibabel as nib

def to_channels(label_volume, dtype=np.float32):
    """Convert label map (3D) to one-hot channels (4D)."""
    num_classes = int(label_volume.max()) + 1
    out = np.zeros((num_classes,) + label_volume.shape, dtype=dtype)
    for c in range(num_classes):
        out[c] = (label_volume == c)
    return out

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

    image[mask == 0] = 0
    image = np.clip(image, -5, 5)
    image = (image + 5) / 10.0
    return image, mask

def Resize3dTensor(img_tensor, target_shape=(128,128,128), mode_type='trilinear'):
    """
    img_tensor: torch tensor of shape (C, D, H, W)
    """
    img_tensor = img_tensor.unsqueeze(0)  # add batch dim
    img_resized = F.interpolate(img_tensor, size=target_shape, mode=mode_type)
    return img_resized.squeeze(0)
    
class HipMriDataset3D(Dataset):
    """Dataset for prostate cancer 3D Images."""

    def __init__(self, image_path, mask_path, transform=None, train=True):
        # Download and load the dataset
        self.image_dataset_path = image_path
        self.mask_dataset_path = mask_path
        self.transform = transform
        self.dataset = []

        image_paths = [os.path.join(self.image_dataset_path, img_name)
                        for img_name in sorted(os.listdir(self.image_dataset_path))]
        mask_paths = [os.path.join(self.mask_dataset_path, mask_name)
                       for mask_name in sorted(os.listdir(self.mask_dataset_path))]
        
        if train:
            image_paths = image_paths[:int(0.8*len(image_paths))]
            mask_paths = mask_paths[:int(0.8*len(mask_paths))]
        else:
            image_paths = image_paths[int(0.8*len(image_paths)):]
            mask_paths = mask_paths[int(0.8*len(mask_paths)):]
        
        for case in range(len(image_paths)):
            self.dataset.append((image_paths[case], mask_paths[case]))

    def __len__(self):
        return len(self.dataset) 

    def __getitem__(self, idx):
        # Get image and mask
        image = nib.load(self.dataset[idx][0])
        mask = nib.load(self.dataset[idx][1])

        image_np = image.get_fdata().astype(np.float32)
        mask_np = mask.get_fdata().astype(np.uint8)  # convert
        
        # Apply transforms to image
        if self.transform:
            image_np, mask_np = self.transform(image_np, mask_np)

        mask_np = to_channels(mask_np, dtype=np.uint8)  # convert to one-hot channels

        # Convert to tensor
        image_np = torch.from_numpy(image_np).unsqueeze(0).float() 
        mask_np = torch.from_numpy(mask_np).float() 

        mask_np = Resize3dTensor(mask_np, target_shape=(128,128,128), mode_type='nearest')
        image_np = Resize3dTensor(image_np, target_shape=(128,128,128))

        return image_np, mask_np