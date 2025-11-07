import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 2D Convolutional Block
# -----------------------------
class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.act(self.bn2(self.conv2(x)))
        return x

# -----------------------------
# 2D U-Net
# -----------------------------
class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=64, dropout=0.0):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock2D(in_channels, base_filters, dropout)
        self.enc2 = ConvBlock2D(base_filters, base_filters*2, dropout)
        self.enc3 = ConvBlock2D(base_filters*2, base_filters*4, dropout)
        self.enc4 = ConvBlock2D(base_filters*4, base_filters*8, dropout)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock2D(base_filters*8, base_filters*16, dropout)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_filters*16, base_filters*8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock2D(base_filters*16, base_filters*8, dropout)
        
        self.up3 = nn.ConvTranspose2d(base_filters*8, base_filters*4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock2D(base_filters*8, base_filters*4, dropout)
        
        self.up2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock2D(base_filters*4, base_filters*2, dropout)
        
        self.up1 = nn.ConvTranspose2d(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ConvBlock2D(base_filters*2, base_filters, dropout)
        
        # Output
        self.segmentation_head = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        out = self.segmentation_head(d1)  # logits
        return out
    
class BinaryDiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.

    Dice Loss = 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)

    Works for 2D or 3D tensors:
        predictions: [B, 1, H, W] or [B, 1, D, H, W] (logits)
        targets: [B, 1, H, W] or [B, 1, D, H, W] (binary 0/1)
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Args:
            predictions (torch.Tensor): logits from the model [B, 1, ...]
            targets (torch.Tensor): binary ground truth [B, 1, ...]
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(predictions)
        targets = targets.float()

        # Flatten spatial dimensions per batch
        dims = tuple(range(1, predictions.ndim))  # flatten everything except batch

        intersection = torch.sum(probs * targets, dims)
        pred_sum = torch.sum(probs, dims)
        target_sum = torch.sum(targets, dims)

        dice_score = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        dice_loss = 1.0 - dice_score.mean()  # average over batch

        return dice_loss
