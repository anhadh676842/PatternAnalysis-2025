import dataset
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch.nn.functional as F
import random

from module import UNet2D, BinaryDiceLoss  # your 2D U-Net implementation
from dataset import HipMriDataset2D  # 2D dataset + DiceLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# -----------------------------
# Quick visualization of loss
# -----------------------------
def plot_loss(losses):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, 'bo-', linewidth=2, markersize=8)
    plt.title('🔥 Training Loss (Dice)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_slice(image, mask_pred, mask_gt):
    """Visualize a single 2D slice with prediction and ground truth"""
    _, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image[0], cmap='gray')
    axes[0].set_title("Input Image")
    axes[1].imshow(mask_gt[0], cmap='Reds')
    axes[1].set_title("Ground Truth")
    axes[2].imshow(mask_pred[0], cmap='Reds')
    axes[2].set_title("Predicted Mask")
    for ax in axes:
        ax.axis('off')
    plt.show()

# -----------------------------
# Training function
# -----------------------------
def train(model, train_loader, test_loader, epochs=50, lr=1e-3, dice_threshold=0.75):
    model.to(device)
    criterion = BinaryDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    losses = []

    print("Starting 2D slice training...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # logits
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

        # -----------------------------
        # Validation & early stopping
        # -----------------------------
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
            print(f"Validation Dice (Prostate) after Epoch {epoch+1}: {avg_dice:.4f}")

        # Early stopping if prostate Dice exceeds threshold
        if avg_dice >= dice_threshold:
            print(f"Early stopping: Prostate Dice {avg_dice:.4f} >= {dice_threshold}")
            # visualize a random slice
            sample_img, sample_mask = next(iter(test_loader))
            sample_img = sample_img.to(device)
            sample_mask = sample_mask.to(device)
            sample_output = model(sample_img)
            pred_mask = (torch.sigmoid(sample_output) > 0.5).float()
            visualize_slice(sample_img[0].cpu().numpy(), pred_mask[0].cpu().numpy(), sample_mask[0].cpu().numpy())
            break

    print("🎯 Training complete!")
    plot_loss(losses)
    return losses

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    train_dataset = HipMriDataset2D(
        image_path='/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train',
        mask_path='/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train',
        transform=dataset.TrainingTransform2D,
        train=True
    )

    test_dataset = HipMriDataset2D(
        image_path='/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate',
        mask_path='/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate',
        transform=dataset.TestTransform2D,
        train=False
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = UNet2D(in_channels=1, out_channels=1, base_filters=16, dropout=0.1)

    train(model, train_loader, test_loader, epochs=50, lr=1e-3, dice_threshold=0.75)