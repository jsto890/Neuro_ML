# 3D Transformer Models for Medical Image Classification

This directory contains implementations of 3D Vision Transformers and Swin UNETR models for structural MRI (sMRI) classification tasks, specifically designed for Alzheimer's Disease (AD) vs Control (CN) classification.

## Overview

The transformer models provide state-of-the-art performance for 3D medical image classification by leveraging self-attention mechanisms and hierarchical feature learning. This implementation includes:

- **Vision Transformer 3D**: Adapted from the original ViT architecture for 3D volumes
- **Swin UNETR Classifier**: Modified Swin UNETR architecture for classification tasks
- **Advanced Training Features**: Mixed precision training, gradient accumulation, warmup scheduling

## Models

### 1. Vision Transformer 3D (`VisionTransformer3D`)

A 3D adaptation of the Vision Transformer architecture for medical image classification.

**Key Features:**
- 3D patch embedding
- Multi-head self-attention
- Position embeddings
- Classification head

**Architecture:**
```
Input: [B, 1, D, H, W] → Patch Embedding → Transformer Blocks → Classification Head → [B, num_classes]
```

**Parameters:**
- `img_size`: Input volume size (default: [97, 115, 97])
- `patch_size`: Patch size for embedding (default: 16)
- `embed_dim`: Embedding dimension (default: 1024 - Large)
- `depth`: Number of transformer blocks (default: 16 - Large)
- `num_heads`: Number of attention heads (default: 16 - Large)

### 2. Swin UNETR Classifier (`SwinUNETRClassifier`)

A modified Swin UNETR architecture adapted for classification tasks.

**Key Features:**
- Swin Transformer encoder
- Hierarchical feature learning
- Global average pooling
- Classification head

**Architecture:**
```
Input: [B, 1, D, H, W] → Swin Encoder → Global Pooling → Classification Head → [B, num_classes]
```

**Parameters:**
- `img_size`: Input volume size (default: [97, 115, 97])
- `feature_size`: Feature size for Swin Transformer (default: 96 - Large)
- `drop_rate`: Dropout rate (default: 0.1)

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- GPU with at least 8GB VRAM (16GB recommended, 24GB optimal)
- 32GB RAM recommended (64GB optimal)
- Multi-core CPU for data loading (64+ cores optimal)

### Quick Installation

```bash
# Make the installation script executable
chmod +x install_transformers.sh

# Run the installation script
./install_transformers.sh
```

### Manual Installation

```bash
# Create conda environment
conda create -n transformers_3d python=3.9 -y
conda activate transformers_3d

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install MONAI
pip install monai

# Install other dependencies
pip install timm einops transformers accelerate pyyaml
pip install numpy scipy scikit-learn matplotlib seaborn pandas nibabel
```

## Usage

### 1. Basic Training

```bash
# Train Vision Transformer (Hardware Optimized)
python train_transformers.py \
    --train_csv train.csv \
    --val_csv val.csv \
    --data_root /path/to/data \
    --labels 0 1 \
    --model VisionTransformer3D \
    --config config_hardware_optimized.yaml \
    --epochs 30 \
    --batch_size 4 \
    --num_workers 32

# Train Swin UNETR (Hardware Optimized)
python train_transformers.py \
    --train_csv train.csv \
    --val_csv val.csv \
    --data_root /path/to/data \
    --labels 0 1 \
    --model SwinUNETRClassifier \
    --config config_hardware_optimized.yaml \
    --epochs 30 \
    --batch_size 4 \
    --num_workers 32
```

### 2. Integration with Main Training Pipeline

The transformer models are integrated into the main training pipeline:

```bash
# Run all models including transformers
python train_smri.py \
    --train_csv train.csv \
    --val_csv val.csv \
    --data_root /path/to/data \
    --labels 0 1 \
    --run_all

# Run specific transformer models
python train_smri.py \
    --train_csv train.csv \
    --val_csv val.csv \
    --data_root /path/to/data \
    --labels 0 1 \
    --models VisionTransformer3D SwinUNETRClassifier
```

### 3. Model Evaluation

```bash
# Evaluate trained transformer model
python evaluate_model.py \
    --model_path checkpoints/run_20241201_120000/VisionTransformer3D/best_transformer_model.pth \
    --test_csv test.csv \
    --data_root /path/to/data \
    --output_dir ./transformer_evaluation \
    --num_classes 2
```

### 4. Grad-CAM Visualization

```bash
# Generate Grad-CAM visualizations for transformer models
python visualise_gradcam.py \
    --val_csv val.csv \
    --data_root /path/to/data \
    --checkpoint checkpoints/run_20241201_120000/SwinUNETRClassifier_GradCAM/best_transformer_model.pth \
    --output_dir ./gradcam_transformer
```

## Hardware Optimization

### Threadripper 3990X + RTX 6000 Setup

For your specific hardware configuration:
- **CPU**: AMD Ryzen Threadripper 3990X (64 cores, 128 threads)
- **GPU**: Quadro RTX 6000 (24GB VRAM)
- **Image Dimensions**: 97×115×97

Use the hardware-optimized configuration:

```bash
# Use hardware-optimized config
--config config_hardware_optimized.yaml
--batch_size 4
--num_workers 32
```

**Key Optimizations:**
- **Batch Size**: 4 (instead of 1) - RTX 6000 can handle larger batches
- **Data Workers**: 32 (instead of 2) - Threadripper can handle many parallel workers
- **Gradient Accumulation**: Disabled - not needed with 24GB VRAM
- **Mixed Precision**: Enabled for faster training

**Expected Performance:**
- Vision Transformer (Large): ~3-4 hours for 30 epochs
- Swin UNETR (Large): ~4-5 hours for 30 epochs
- Memory usage: ~20-22GB VRAM
- CPU usage: ~60-80% across all cores

## Configuration

### Transformer Configuration File (`config_transformers.yaml`)

The configuration file contains model-specific hyperparameters and training settings:

```yaml
# Vision Transformer 3D Configuration (Large)
vision_transformer_3d:
  img_size: [97, 115, 97]
  patch_size: 16
  embed_dim: 1024
  depth: 16
  num_heads: 16
  mlp_ratio: 4.0
  drop_rate: 0.1

# Training Configuration
training:
  learning_rate: 1e-4
  weight_decay: 1e-5
  warmup_epochs: 5
  cosine_schedule: true
  optimizer: "adamw"
  mixed_precision: true
  gradient_accumulation_steps: 2
```

### Model Variants

The configuration supports different model sizes:

- **Small**: `vit_small` - 384 embed_dim, 6 depth, 6 heads
- **Medium**: `vit_medium` - 512 embed_dim, 8 depth, 8 heads  
- **Large**: `vit_large` - 1024 embed_dim, 16 depth, 16 heads (Default)
- **Extra Large**: `vit_xlarge` - 1280 embed_dim, 20 depth, 20 heads

## Advanced Features

### 1. Mixed Precision Training

Automatically enabled for faster training and reduced memory usage:

```python
# Automatically handled in train_transformers.py
scaler = GradScaler()
with autocast():
    logits = model(smri)
    loss = criterion(logits, labels)
```

### 2. Gradient Accumulation

For large models that don't fit in GPU memory:

```yaml
training:
  gradient_accumulation_steps: 2  # Effective batch size = batch_size * accumulation_steps
```

### 3. Learning Rate Warmup

Essential for transformer training stability:

```yaml
training:
  warmup_epochs: 5
  cosine_schedule: true
```

### 4. Label Smoothing

Improves generalization:

```yaml
training:
  label_smoothing: 0.1
```

## Performance Optimization

### Memory Management

1. **Batch Size**: Start with batch_size=1 for large models
2. **Gradient Accumulation**: Use to simulate larger batch sizes
3. **Mixed Precision**: Automatically reduces memory usage by ~50%
4. **Gradient Checkpointing**: Available for Swin UNETR models

### Training Speed

1. **Mixed Precision**: 1.5-2x speedup
2. **Data Loading**: Use appropriate num_workers
3. **GPU Memory**: Ensure sufficient VRAM for your model size

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch_size to 1
   - Enable gradient accumulation
   - Use mixed precision training
   - Try smaller model variants

2. **Slow Training**
   - Enable mixed precision
   - Reduce num_workers if CPU bottleneck
   - Use gradient accumulation for larger effective batch sizes

3. **Poor Performance**
   - Increase warmup_epochs
   - Adjust learning rate
   - Try different model variants
   - Check data preprocessing

### Debugging

```bash
# Test model loading
python -c "from transformer_models import *; model = VisionTransformer3D(); print('Model loaded successfully')"

# Test with dummy data
python -c "
import torch
from transformer_models import VisionTransformer3D
model = VisionTransformer3D()
x = torch.randn(1, 1, 97, 115, 97)
output = model(x)
print(f'Input shape: {x.shape}')
print(f'Output shape: {output.shape}')
"
```
## Contributing

To add new transformer models or improve existing ones:

1. Add model implementation to `transformer_models.py`
2. Update `get_transformer_model()` function
3. Add configuration to `config_transformers.yaml`
4. Update documentation
5. Test with dummy data and real data

## License

This implementation is part of the P4P project and follows the same licensing terms. 