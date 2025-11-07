import dataset
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch.nn.functional as F
import random
import train

from module import UNet2D, BinaryDiceLoss  # your 2D U-Net implementation
from dataset import HipMriDataset2D  # 2D dataset + DiceLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

if __name__ == "__main__":
    test_dataset = HipMriDataset2D(
        image_path='/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test',
        mask_path='/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_test',
        transform=dataset.TrainingTransform2D,
        train=True
    )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = UNet2D(in_channels=1, out_channels=1, base_filters=16, dropout=0.1)

    losses = []
    model.eval()
    total_dice = 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            pred_mask = (probs > 0.5).float()

            # Dice coefficient for the prostate class (assumes class=1)
            intersection = (pred_mask * masks).sum(dim=(1,2,3))
            union = pred_mask.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3))
            dice_score = ((2*intersection + 1e-6) / (union + 1e-6)).mean().item()
            total_dice += dice_score

        avg_dice = total_dice / len(test_loader)       
        

    print("Training complete!")
    train.plot_loss(losses)