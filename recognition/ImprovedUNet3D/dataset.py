import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms.functional as TF

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm
import random
import nibabel as nib
import utils

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

def load_data_3D(imageNames, normImage=False, categorical=False, dtype=np.float32,
                 getAffines=False, orient=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.

    normImage: bool (normalise the image 0.0-1.0)
    orient: Apply orientation and resample image? Good for images with large slice
            thickness or anisotropic resolution
    dtype: Type of the data. If dtype=np.uint8, it is assumed that the data is labels
    early_stop: Stop loading pre-maturely? Leaves arrays mostly empty, for quick
                loading and testing scripts.
    '''
    affines = []

    # interp = 'continuous'
    interp = 'linear'
    if dtype == np.uint8:  # assume labels
        interp = 'nearest'

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    if orient:
        niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)
        # testResultName = "oriented.nii.gz"
        # niftiImage.to_filename(testResultName)
    first_case = niftiImage.get_fdata(caching='unchanged')

    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # sometimes extra dims, remove

    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        if orient:
            niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]  # sometimes extra dims in HipMRI_study data
        inImage = inImage[:, :, :depth]  # clip slices
        inImage = inImage.astype(dtype)

        if normImage:
            # inImage = inImage / np.linalg.norm(inImage)
            # inImage = 255. * inImage / inImage.max()
            inImage = (inImage - inImage.mean()) / inImage.std()

        if categorical:
            inImage = utils.to_channels(inImage, dtype=dtype)
            # images[i, :, :, :, :] = inImage
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2], :inImage.shape[3]] = inImage  # with pad
        else:
            # images[i, :, :, :] = inImage
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2]] = inImage  # with pad

        affines.append(affine)

        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images
    
class HipMriDataset3D(Dataset):
    """Dataset for prostate cancer 3D Images."""

    def __init__(self, image_path, mask_path, transform=None):
        # Download and load the dataset
        self.image_dataset_path = image_path
        self.mask_dataset_path = mask_path
        self.transform = transform
        self.dataset = []

        image_paths = [os.path.join(self.image_dataset_path, img_name)
                        for img_name in sorted(os.listdir(self.image_dataset_path))]
        mask_paths = [os.path.join(self.mask_dataset_path, mask_name)
                       for mask_name in sorted(os.listdir(self.image_dataset_path))]
        
        for case in range(len(image_paths)):
            case_image = nib.load(image_paths[case])
            case_mask = nib.load(mask_paths[case])
            self.dataset.append((case_image, case_mask))

    def __len__(self):
        return len(self.dataset) 

    def __getitem__(self, idx):
        # Get image and mask
        image, mask = self.dataset[idx]

        image_np = image.get_fdata().astype(np.float32)

        mask_np = mask.get_fdata().as_type(np.uint8)  # convert
        binary_mask = np.zeros_like(mask_np, dtype=np.uint8)

        affine = mask.affine

        # Apply transforms to image
        if self.transform:
            image_np = self.transform(image_np)

        binary_mask[mask_np != 5] = 0  # prostate_voxels
        binary_mask[mask_np == 5] = 1  # prostate voxels

        # Convert to tensor
        binary_mask = torch.from_numpy(binary_mask).unsqueeze(0).int() # add channel dim
        image_np = torch.from_numpy(image_np).unsqueeze(0).float() # add channel dim

        return image_np, binary_mask, affine