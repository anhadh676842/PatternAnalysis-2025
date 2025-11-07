import dataset
import matplotlib.pyplot  as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch.nn.functional as F
from module import ImprovedUNet3D, DiceLoss
import dataset
import random

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Quick visualization of loss
def plot_loss(losses, loss_type='dice'):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, 'bo-', linewidth=2, markersize=8)

    title_map = {
        'bce': '🔥 Training Loss (BCE)',
        'dice': '🔥 Training Loss (Dice)',
        'combined': '🔥 Training Loss (Combined BCE + Dice)'
    }
    plt.title(title_map.get(loss_type, '🔥 Training Loss'), fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def visualize_cnn_slices(cnn_output, slice_indices=None):
    """
    Visualize CNN segmentation slice-by-slice using matplotlib.
    
    cnn_output: C x H x W x D (numpy array)
    slice_indices: list of slice indices along depth (D)
    """
    # Convert to label indices
    label_volume = np.argmax(cnn_output, axis=0)
    
    D = label_volume.shape[2]
    if slice_indices is None:
        slice_indices = [D // 4, D // 2, 3 * D // 4]
    
    fig, axes = plt.subplots(1, len(slice_indices), figsize=(15, 5))
    
    for i, idx in enumerate(slice_indices):
        axes[i].imshow(label_volume[:, :, idx], cmap='tab20')
        axes[i].set_title(f'Slice {idx}')
        axes[i].axis('off')
    plt.show()

def train(model, train_loader, test_dataset, epochs=1, lr=0.001):
    model.to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    losses = []

    print(" Starting training with Instance Norm, LeakyReLU, and Softmax activation...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training loop with progress
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            print(f"pred_pet shape: {outputs.shape}, masks shape: {masks.shape}")

            loss = criterion(outputs, masks)

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step(loss.detach().item())

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"📈 Epoch {epoch+1}/{epochs} Complete: Avg Loss = {avg_loss:.4f}")

        model.eval()
        atThreshold = True
        with torch.no_grad():
            total_dice = 0
            for image, mask in test_loader:
                image = image.to(device)
                mask = mask.to(device)

                output = model(image)
                total_dice += 1 - criterion(output, mask).item()
                if (atThreshold and 1 - criterion(output, mask).item() < 0.7):
                    atThreshold = False

            avg_dice = total_dice / len(test_dataset)
            print(f"🧪 Validation Dice Score after Epoch {epoch+1}: {avg_dice:.4f}")

        if (atThreshold): 
            visualize_cnn_slices(cnn_output=output)
            break

    print(" Training complete with enhanced U-Net!")
    plot_loss(losses, loss_type='dice')
    return losses

if __name__ == "__main__":
    # Example usage
    train_dataset = dataset.HipMriDataset3D(image_path='semantic_MRs_anon',
                                            mask_path='semantic_labels_anon',
                                            transform=dataset.TrainingTransform,
                                            train=True)
    
    test_dataset = dataset.HipMriDataset3D(image_path='semantic_MRs_anon',
                                            mask_path='semantic_labels_anon',
                                            train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = ImprovedUNet3D(in_channels=1, base_filters=16, dropout=0.3)

    train(model, train_loader, test_dataset, epochs=50, lr=0.001)
