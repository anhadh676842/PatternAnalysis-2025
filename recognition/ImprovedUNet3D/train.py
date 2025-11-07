import dataset
import matplotlib.pyplot  as plt
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from module import ImprovedUNet3D, DiceLoss
import dataset
import random

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

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

def train(model, train_loader, test_dataset, epochs=100, lr=0.001):
    model.to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    print(" Starting training with Instance Norm, LeakyReLU, and Softmax activation...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training loop with progress
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            print(f"pred_pet shape: {outputs.shape}, masks shape: {masks.shape}")
            loss = criterion(outputs, masks)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"📈 Epoch {epoch+1}/{epochs} Complete: Avg Loss = {avg_loss:.4f}")

        # Visualize predictions after each epoch (or every few epochs)
        #if (epoch) % visualize_every == 0:
        #    show_epoch_predictions(model, test_dataset, epoch + 1, n=3)

        model.eval()
        with torch.no_grad():
            total_dice = 0
            for i in range(len(test_dataset)):
                image, mask = test_dataset[i]
                image = image.unsqueeze(0).to(device)
                mask = mask.to(device)

                output = model(image)
                pred_mask = (output > 0.5).float()

                intersection = (pred_mask * mask).sum()
                dice_score = (2.0 * intersection) / (pred_mask.sum() + mask.sum() + 1e-6)
                total_dice += dice_score.item()

            avg_dice = total_dice / len(test_dataset)
            print(f"🧪 Validation Dice Score after Epoch {epoch+1}: {avg_dice:.4f}")

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

    train(model, train_loader, test_dataset, epochs=50, lr=0.001, visualize_every=5)
