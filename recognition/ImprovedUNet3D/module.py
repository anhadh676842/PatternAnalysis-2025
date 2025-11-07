import torch
import torch.nn as nn
import torch.nn.functional as F

class Upsample3D(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        return x

# -----------------------------
# Pre-activation Residual Block (Context Module)
# -----------------------------
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3, stride=1):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.act1 = nn.LeakyReLU(0.01)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act2 = nn.LeakyReLU(0.01)
        
        self.dropout = nn.Dropout3d(dropout)
        
        if in_channels != out_channels or stride > 1:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        return self.act2(x)

# -----------------------------
# Localization Module
# -----------------------------x
class LocalizationModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3x3 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.01)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.act(self.conv3x3(x))
        x = self.conv1x1(x)
        return x

# -----------------------------
# Full 3D U-Net with deep supervision
# -----------------------------
class ImprovedUNet3D(nn.Module):
    def __init__(self, in_channels=6, base_filters=16, dropout=0.3):
        super().__init__()
        self.upsample1 = Upsample3D(base_filters*16)
        self.upsample2 = Upsample3D(base_filters*8)
        self.upsample3 = Upsample3D(base_filters*4)
        self.upsample4 = Upsample3D(base_filters*2)

        self.convInput = nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1)
        self.convOutput = nn.Conv3d(base_filters*2, base_filters*2, kernel_size=3, padding=1)

        self.StrideConv1 = nn.Conv3d(base_filters, base_filters*2, kernel_size=3, padding=1, stride=2)
        self.StrideConv2 = nn.Conv3d(base_filters*2, base_filters*4, kernel_size=3, padding=1, stride=2)
        self.StrideConv3 = nn.Conv3d(base_filters*4, base_filters*8, kernel_size=3, padding=1, stride=2)
        self.StrideConv4 = nn.Conv3d(base_filters*8, base_filters*16, kernel_size=3, padding=1, stride=2)

        #self.maxPool1 = nn.MaxPool3d(2)
        #self.maxPool2 = nn.MaxPool3d(2)
        #self.maxPool3 = nn.MaxPool3d(2)
        #self.maxPool4 = nn.MaxPool3d(2)

        # Encoder / context pathway
        self.enc1 = ResidualBlock3D(base_filters, base_filters, dropout) 
        self.enc2 = ResidualBlock3D(base_filters*2, base_filters*2, dropout) 
        self.enc3 = ResidualBlock3D(base_filters*4, base_filters*4, dropout)
        self.enc4 = ResidualBlock3D(base_filters*8, base_filters*8, dropout)
        self.enc5 = ResidualBlock3D(base_filters*16, base_filters*16, dropout)
        
        # Decoder / localization pathway    
        self.loc3 = LocalizationModule(base_filters*8 + base_filters*8, base_filters*8)
        self.loc2 = LocalizationModule(base_filters*4 + base_filters*4, base_filters*4)
        self.loc1 = LocalizationModule(base_filters*2 + base_filters*2, base_filters*2)

        # Segmentation layers
        self.seg1 = nn.Conv3d(base_filters*4, 6, kernel_size=1)
        self.seg2= nn.Conv3d(base_filters*2, 6, kernel_size=1)
        self.seg3 = nn.Conv3d(base_filters*2, 6, kernel_size=1)
    
    def forward(self, x):
        x = self.convInput(x)
        # Encoder
        e1 = self.enc1(x)

        e2 = self.StrideConv1(e1)
        e2 = self.enc2(e2)

        e3 = self.StrideConv2(e2)
        e3 = self.enc3(e3)

        e4 = self.StrideConv3(e3)
        e4 = self.enc4(e4)

        e5 = self.StrideConv4(e4)
        e5 = self.enc5(e5)

        u1 = self.upsample1(e5)
        u1 = torch.cat((u1, e4), dim=1)
        u1 = self.loc3(u1)

        u2 = self.upsample2(u1)
        u2 = torch.cat((u2, e3), dim=1)
        u2 = self.loc2(u2)

        res1 = self.seg1(u2)
        res1 = F.interpolate(res1, size=e2.shape[2:], mode='nearest')

        u3 = self.upsample3(u2)
        u3 = torch.cat((u3, e2), dim=1)
        u3 = self.loc1(u3)

        res2 = self.seg2(u3)
        res2 = res1 + res2
        res2 = F.interpolate(res2, size=e1.shape[2:], mode='nearest')

        u4 = self.upsample4(u3)     
        u4 = torch.cat((u4, e1), dim=1)

        out = self.convOutput(u4)
        out = self.seg3(out)
        out = res2 + out
        
        return out
    
class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss supporting one-hot encoded targets.

    Dice Loss = 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)

    Works for both 2D and 3D tensors:
        predictions: [B, C, H, W] or [B, C, D, H, W]
        targets: same shape (one-hot encoded)
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Args:
            predictions (torch.Tensor): Model outputs after sigmoid or softmax [B, C, ...]
            targets (torch.Tensor): One-hot encoded ground truth [B, C, ...]
        """
        # Ensure floating point
        predictions = F.softmax(predictions.float())
        targets = targets.float()

        # Flatten across spatial dimensions but keep class and batch
        dims = tuple(range(2, predictions.ndim))  # e.g. (2,3,4) for 3D data

        # Compute intersection and union per class
        intersection = torch.sum(predictions * targets, dims)
        pred_sum = torch.sum(predictions, dims)
        target_sum = torch.sum(targets, dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)

        # Average over classes and batch
        dice_loss = 1.0 - dice_per_class.mean()

        return dice_loss
