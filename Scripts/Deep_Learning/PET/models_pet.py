# scripts/models_pet.py

import torch
import torch.nn as nn

# --- Additional imports for model variants ---
try:
    from monai.networks.nets import resnet, DenseNet121
    from monai.networks.nets import resnet18, resnet34, resnet50, resnet101, resnet152
except ImportError:
    resnet = None
    DenseNet121 = None
    resnet18 = None
    resnet34 = None
    resnet50 = None
    resnet101 = None
    resnet152 = None

try:
    from efficientnet_pytorch_3d import EfficientNet3D
except ImportError:
    EfficientNet3D = None

# Import transformer models
try:
    from transformer_models import get_transformer_model
except ImportError:
    get_transformer_model = None

class Simple3DCNN(nn.Module):
    """
    A straightforward 3DxCNN for binary classification (e.g. AD vs PD vs CN).
    Input:  [B, 1, D, H, W]  single‐channel PET
    Output: [B, num_classes] logits
    """
    def __init__(self, num_classes=2, base_channels=16, dropout_p=0.0):  # Reduced base channels
        super().__init__()
        self.features = nn.Sequential(
            # First conv block
            nn.Conv3d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            # Second conv block
            nn.Conv3d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            # Third conv block
            nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        
        # Initialize classifier with a placeholder
        self.classifier = nn.Sequential(
            nn.Linear(1, 256),  # Reduced intermediate size
            nn.ReLU(),
            nn.Dropout(dropout_p if dropout_p > 0 else 0.5),  # Use provided dropout or default 0.5
            nn.Linear(256, num_classes)
        )
        
        self._initialized = False

    def _initialize_classifier(self, x):
        # Get the size of the flattened features
        with torch.no_grad():
            x = self.features(x)
            n_features = x.view(x.size(0), -1).size(1)
        
        # Get the device of the input tensor
        device = x.device
        
        # Replace the first linear layer with the correct size and move to the same device
        self.classifier[0] = nn.Linear(n_features, 256).to(device)
        
        # Ensure all classifier layers are on the same device
        for layer in self.classifier:
            layer.to(device)
            
        self._initialized = True

    def forward(self, x):
        # x shape = [B, 1, D, H, W]
        # Ensure classifier matches current feature size (handles dynamic input shapes)
        if not self._initialized:
            self._initialize_classifier(x)
        
        features = self.features(x)
        flat = features.view(features.size(0), -1)
        
        # If feature size changed (e.g., different input dims), reinitialize first FC layer on-the-fly
        expected_in_features = self.classifier[0].in_features if isinstance(self.classifier[0], nn.Linear) else None
        current_in_features = flat.size(1)
        if expected_in_features is None or expected_in_features != current_in_features:
            device = features.device
            self.classifier[0] = nn.Linear(current_in_features, 256).to(device)
            for layer in self.classifier:
                layer.to(device)
            self._initialized = True
        
        logits = self.classifier(flat)
        return logits


class PET_GradCAM_3DCNN(nn.Module):
    """
    3D CNN architecture with the final conv block exposed for Grad-CAM.
    Input:  [B, 1, D, H, W]  single‐channel PET
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

# --- Model Factory Function ---
def get_3d_model(model_name, num_classes=2, in_channels=1, base_channels=16, use_pretrained=False, dropout_p: float = 0.0,
                 vit_drop_rate: float = 0.0, vit_attn_drop_rate: float = 0.0, vit_drop_path_rate: float = 0.0):
    """
    Returns a 3D CNN model instance by name.
    Supported: 'Simple3DCNN', 'ResNet18_3D', 'DenseNet121_3D', 'EfficientNetB0_3D',
               'VisionTransformer3D', 'SwinUNETRClassifier'
    
    Args:
        model_name: Name of the model to create
        num_classes: Number of output classes
        in_channels: Number of input channels (1 for PET)
        base_channels: Base number of channels for Simple3DCNN
        use_pretrained: Whether to use pretrained weights (for ResNet, DenseNet, EfficientNet)
    """
    model_name = model_name.lower()
    
    if model_name == "simple3dcnn":
        return Simple3DCNN(num_classes=num_classes, base_channels=base_channels, dropout_p=dropout_p)
    
    elif model_name == "resnet18_3d":
        if resnet is None:
            raise ImportError("MONAI is required for 3D ResNet. Install with 'pip install monai'.")
        
        if use_pretrained:
            print("Loading pretrained ResNet18_3D...")
            # MONAI MedicalNet pretrained weights require these flags
            model = resnet18(
                pretrained=True,
                spatial_dims=3,
                n_input_channels=in_channels,
                num_classes=num_classes,
                feed_forward=False,
                shortcut_type='A',
                bias_downsample=True
            )
        else:
            print("Creating ResNet18_3D from scratch...")
            model = resnet18(pretrained=False, spatial_dims=3, n_input_channels=in_channels, num_classes=num_classes)
        # Defensive: ensure classifier matches requested num_classes
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear) and model.fc.out_features != num_classes:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    elif model_name == "resnet50_3d":
        if resnet is None:
            raise ImportError("MONAI is required for 3D ResNet. Install with 'pip install monai'.")
        
        if use_pretrained:
            print("Loading pretrained ResNet50_3D...")
            # MONAI MedicalNet pretrained weights require these flags
            model = resnet50(
                pretrained=True,
                spatial_dims=3,
                n_input_channels=in_channels,
                num_classes=num_classes,
                feed_forward=False,
                shortcut_type='B',
                bias_downsample=False
            )
        else:
            print("Creating ResNet50_3D from scratch...")
            model = resnet50(pretrained=False, spatial_dims=3, n_input_channels=in_channels, num_classes=num_classes)
        # Defensive: ensure classifier matches requested num_classes
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear) and model.fc.out_features != num_classes:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    elif model_name == "densenet121_3d":
        if DenseNet121 is None:
            raise ImportError("MONAI is required for 3D DenseNet. Install with 'pip install monai'.")
        
        if use_pretrained:
            print("Warning: DenseNet121_3D does not support pretrained weights for 3D spatial dimensions.")
            print("Creating DenseNet121_3D from scratch...")
            model = DenseNet121(pretrained=False, spatial_dims=3, in_channels=in_channels, out_channels=num_classes)
        else:
            print("Creating DenseNet121_3D from scratch...")
            model = DenseNet121(pretrained=False, spatial_dims=3, in_channels=in_channels, out_channels=num_classes)
        return model
    
    elif model_name == "efficientnetb0_3d":
        if EfficientNet3D is None:
            raise ImportError("efficientnet_pytorch_3d is required for EfficientNet3D. Install with 'pip install git+https://github.com/shijianjian/EfficientNet-PyTorch-3D'.")
        
        if use_pretrained:
            print("Loading pretrained EfficientNetB0_3D...")
            try:
                # Try to load pretrained weights
                model = EfficientNet3D.from_pretrained("efficientnet-b0", advprop=True)
                # Modify the final layer for our task
                model._fc = nn.Linear(model._fc.in_features, num_classes)
                
                # Modify the first conv layer to accept in_channels if different from default (3)
                if in_channels != 3:
                    model._conv_stem = nn.Conv3d(
                        in_channels, 
                        model._conv_stem.out_channels, 
                        kernel_size=model._conv_stem.kernel_size, 
                        stride=model._conv_stem.stride, 
                        padding=model._conv_stem.padding, 
                        bias=False
                    )
            except Exception as e:
                print(f"Warning: Could not load pretrained EfficientNet weights: {e}")
                print("Falling back to from-scratch initialization...")
                model = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': num_classes})
                
                # Modify the first conv layer to accept in_channels if different from default (3)
                if in_channels != 3:
                    model._conv_stem = nn.Conv3d(
                        in_channels, 
                        model._conv_stem.out_channels, 
                        kernel_size=model._conv_stem.kernel_size, 
                        stride=model._conv_stem.stride, 
                        padding=model._conv_stem.padding, 
                        bias=False
                    )
        else:
            print("Creating EfficientNetB0_3D from scratch...")
        # Create EfficientNet3D with only num_classes in override_params
        model = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': num_classes})
        
        # Modify the first conv layer to accept in_channels if different from default (3)
        if in_channels != 3:
            model._conv_stem = nn.Conv3d(
                in_channels, 
                model._conv_stem.out_channels, 
                kernel_size=model._conv_stem.kernel_size, 
                stride=model._conv_stem.stride, 
                padding=model._conv_stem.padding, 
                bias=False
            )
        
        return model
    
    # Transformer models
    elif model_name in ["visiontransformer3d", "swinunetrclassifier", "fullswinunetrclassifier"]:
        if get_transformer_model is None:
            raise ImportError("Transformer models are not available. Install required dependencies.")
        if model_name == "visiontransformer3d":
            return get_transformer_model(
                model_name,
                num_classes=num_classes,
                in_channels=in_channels,
                drop_rate=vit_drop_rate,
                attn_drop_rate=vit_attn_drop_rate,
                drop_path_rate=vit_drop_path_rate,
            )
        else:
            # SwinUNETR models
            return get_transformer_model(model_name, num_classes=num_classes, in_channels=in_channels)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# --- Pretrained Model Information ---
def get_pretrained_model_info():
    """
    Returns information about available pretrained models.
    """
    info = {
        "ResNet18_3D": {
            "source": "MONAI",
            "pretrained_on": "Medical imaging datasets (various)",
            "input_size": "Flexible (3D)",
            "pretrained_available": True,
            "notes": "Good baseline for medical imaging tasks"
        },
        "ResNet50_3D": {
            "source": "MONAI", 
            "pretrained_on": "Medical imaging datasets (various)",
            "input_size": "Flexible (3D)",
            "pretrained_available": True,
            "notes": "Deeper model, potentially better performance"
        },
        "DenseNet121_3D": {
            "source": "MONAI",
            "pretrained_on": "Medical imaging datasets (various)", 
            "input_size": "Flexible (3D)",
            "pretrained_available": False,
            "notes": "Dense connections, good feature reuse. No pretrained weights for 3D."
        },
        "EfficientNetB0_3D": {
            "source": "Community (GitHub)",
            "pretrained_on": "Natural images (ImageNet)",
            "input_size": "224x224x224 (3D)",
            "pretrained_available": True,
            "notes": "May need adaptation for medical imaging"
        },
        "Simple3DCNN": {
            "source": "Custom",
            "pretrained_on": "None",
            "input_size": "Flexible (3D)",
            "pretrained_available": False,
            "notes": "Baseline model, trained from scratch"
        }
    }
    return info
