# scripts/models_smri.py

import torch
import torch.nn as nn

class Simple3DCNN(nn.Module):
    """
    A straightforward 3DxCNN for binary classification (e.g. AD vs PD vs CN).
    Input:  [B, 1, D, H, W]  single‐channel sMRI
    Output: [B, num_classes] logits
    """
    def __init__(self, in_channels=1, base_channels=16, num_classes=2):
        super().__init__()
        # Convolutional stem
        self.conv1 = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=2)  # → [B, base_channels, D/2, H/2, W/2]

        self.conv2 = nn.Conv3d(base_channels, base_channels*2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool3d(kernel_size=2)  # → [B, base_channels*2, D/4, H/4, W/4]

        self.conv3 = nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        # AdaptiveAvgPool3d(1) → [B, base_channels*4, 1, 1, 1]
        self.pool3 = nn.AdaptiveAvgPool3d(output_size=1)

        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(base_channels*4, num_classes)

    def forward(self, x):
        # x shape = [B, 1, D, H, W]
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)  # [B, base_channels, D/2, H/2, W/2]

        x = self.relu2(self.conv2(x))
        x = self.pool2(x)  # [B, base_channels*2, D/4, H/4, W/4]

        x = self.relu3(self.conv3(x))
        x = self.pool3(x)  # [B, base_channels*4, 1, 1, 1]

        x = self.flatten(x)       # [B, base_channels*4]
        logits = self.fc(x)       # [B, num_classes]
        return logits


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
