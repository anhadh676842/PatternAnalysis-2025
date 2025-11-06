import torch
import torch.nn as nn
import torch.nn.functional as F

class Upsample3D(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv3d(in_channels, in_channels/2, kernel_size=3, padding=1)

    def forward(self, x):
        x.repeat_interleave(self.scale_factor, dim=2).repeat_interleave(self.scale_factor, dim=3).repeat_interleave(self.scale_factor, dim=4)
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
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x + residual

# -----------------------------
# Localization Module
# -----------------------------x
class LocalizationModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3x3 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.01)
        self.conv1x1 = nn.Conv3d(out_channels, out_channels // 2, kernel_size=1)
    
    def forward(self, x):
        x = self.act(self.conv3x3(x))
        x = self.conv1x1(x)
        return x

# -----------------------------
# Upsample by voxel repetition
# -----------------------------
def upsample_repeat(x):
    # Double each spatial dimension by repeating voxels
    return x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)

# -----------------------------
# Full 3D U-Net with deep supervision
# -----------------------------
class ImprovedUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base_filters=16, dropout=0.3):
        super().__init__()
        self.upsample1 = Upsample3D(base_filters*16)
        self.upsample2 = Upsample3D(base_filters*8)
        self.upsample3 = Upsample3D(base_filters*4)
        self.upsample4 = Upsample3D(base_filters*2)

        self.convInput = nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1)
        self.convOutput = nn.Conv3d(base_filters*2, base_filters*2, kernel_size=3, padding=1)

        self.StrideConv1 = nn.Conv3d(base_filters*2, base_filters*2, kernel_size=3, padding=1, stride=2)
        self.StrideConv2 = nn.Conv3d(base_filters*4, base_filters*4, kernel_size=3, padding=1, stride=2)
        self.StrideConv3 = nn.Conv3d(base_filters*8, base_filters*8, kernel_size=3, padding=1, stride=2)
        self.StrideConv4 = nn.Conv3d(base_filters*16, base_filters*16, kernel_size=3, padding=1, stride=2)

        # Encoder / context pathway
        self.enc1 = ResidualBlock3D(base_filters, base_filters, dropout) 
        self.enc2 = ResidualBlock3D(base_filters*2, base_filters*2, dropout, stride=2) 
        self.enc3 = ResidualBlock3D(base_filters*4, base_filters*4, dropout, stride=2)
        self.enc4 = ResidualBlock3D(base_filters*8, base_filters*8, dropout, stride=2)
        self.enc5 = ResidualBlock3D(base_filters*16, base_filters*16, dropout, stride=2)
        
        # Decoder / localization pathway    
        self.loc3 = LocalizationModule(base_filters*8 + base_filters*8, base_filters*4)
        self.loc2 = LocalizationModule(base_filters*4 + base_filters*4, base_filters*2)
        self.loc1 = LocalizationModule(base_filters*2 + base_filters*2, base_filters)

        # Segmentation layers
        self.seg1 = nn.Conv3d(base_filters*4, 1, kernel_size=1)
        self.seg2= nn.Conv3d(base_filters*2, 1, kernel_size=1)
        self.seg3 = nn.Conv3d(base_filters*2, 1, kernel_size=1)
    
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
        u1 = torch.concat((u1, e4))
        u1 = self.loc3(u1)

        u2 = self.upsample2(u1)
        u2 = torch.concat(u2, e3)
        u2 = self.loc2(u2)

        res1 = self.seg1(u1)
        res1 = F.interpolate(res1, size=e1.shape[2:], mode='nearest')

        u3 = self.upsample3(u2)
        u3 = torch.concat(u3, e2)
        u3 = self.loc1(u3)

        res2 = self.seg2(u3)
        res2 = res1 + res2
        res2 = F.interpolate(res2, size=e1.shape[2:], mode='nearest')

        u4 = self.upsample4(u3)     
        u4 = torch.concat(u4, e1)

        out = self.convOutput(u4)
        out = self.seg3(out)
        out = res2 + out
        out = F.softmax(out, dim=1)
        
        return out
    
class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation.

    Dice Loss = 1 - Dice Coefficient
    Dice Coefficient = (2 * |X ∩ Y|) / (|X| + |Y|)

    Args:
        smooth (float): Smoothing factor to avoid division by zero (default: 1e-6)
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Sigmoid output from model [B, H, W] (values between 0-1)
            targets: Binary ground truth [B, H, W] (values 0 or 1)
        """
        # Flatten tensors using reshape to handle non-contiguous memory layout
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1).float()

        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

        # Return Dice Loss (1 - Dice Coefficient)
        return 1 - dice_coeff