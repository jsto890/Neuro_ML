# scripts/models_smri.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    """
    A straightforward 3DxCNN for binary classification (e.g. AD vs PD vs CN).
    Input:  [B, 1, D, H, W]  single‐channel sMRI
    Output: [B, num_classes] logits
    """
    def __init__(self, num_classes=2, base_channels=32):
        super().__init__()
        
        # Initial conv block
        self.init_conv = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU()
        )
        
        # Residual blocks
        self.block1 = ResidualBlock(base_channels, base_channels*2)
        self.block2 = ResidualBlock(base_channels*2, base_channels*4)
        self.block3 = ResidualBlock(base_channels*4, base_channels*8)
        
        # Final pooling and classifier
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(base_channels*8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        self._initialized = False

    def _initialize_classifier(self, x):
        with torch.no_grad():
            x = self.init_conv(x)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.pool(x)
            n_features = x.view(x.size(0), -1).size(1)
        
        self.classifier[0] = nn.Linear(n_features, 512)
        self._initialized = True

    def forward(self, x):
        if not self._initialized:
            self._initialize_classifier(x)
            
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels)
            )
        
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout3d(0.2)
        
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.relu(out)
        
        out = self.pool(out)
        out = self.dropout(out)
        
        return out

class SMRI_GradCAM_3DCNN(nn.Module):
    """
    3D CNN architecture with the final conv block exposed for Grad-CAM.
    Input:  [B, 1, D, H, W]  single‐channel sMRI
    Output: (logits, fmap) where fmap is [B, C, d, h, w] from the last conv block.
    """
    def __init__(self, in_channels=1, base_channels=16, num_classes=3):
        super().__init__()
        # Block 1: [1→16]
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)  # → [B, base_channels, D/2, H/2, W/2]
        )
        # Block 2: [16→32]
        self.block2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)  # → [B, base_channels*2, D/4, H/4, W/4]
        )
        # Final convolutional block for Grad-CAM: [32→64], no pooling here
        self.features = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(inplace=True)
            # Output shape: [B, base_channels*4, D/4, H/4, W/4]
        )
        # Global pool + classification
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=1)  # → [B, 64, 1, 1, 1]
        self.classifier  = nn.Linear(base_channels*4, num_classes)

    def forward(self, x):
        # x: [B, 1, D, H, W]
        x = self.block1(x)           # [B, 16, D/2, H/2, W/2]
        x = self.block2(x)           # [B, 32, D/4, H/4, W/4]
        fmap = self.features(x)      # [B, 64, D/4, H/4, W/4]
        out  = self.global_pool(fmap)  # [B, 64, 1,1,1]
        out  = out.view(out.size(0), -1)  # [B, 64]
        logits = self.classifier(out)     # [B, num_classes]
        return logits, fmap
