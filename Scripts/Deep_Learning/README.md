# Deep Learning Directory

This directory contains deep learning approaches for neurodegenerative disease detection using raw medical imaging data. The deep learning pipeline includes 3D CNN architectures, Vision Transformers, and comprehensive evaluation tools.

## 📁 Directory Structure

```
Deep_Learning/
├── README.md                     # This file
├── MRI/                          # MRI deep learning
│   ├── config_hardware_optimised.yaml
│   ├── config_transformers.yaml
│   ├── dataset.py
│   ├── models_smri.py
│   ├── transformer_models.py
│   ├── train_smri.py
│   ├── evaluate_model.py
│   ├── gradcam.py
│   ├── visualise_gradcam.py
│   ├── split_labels.py
│   ├── regenerate_plots.py
│   ├── regenerate_fold_csv.py
│   └── show_pretrained_info.py
├── PET/                          # PET deep learning
│   ├── config_hardware_optimised.yaml
│   ├── config_transformers.yaml
│   ├── dataset.py
│   ├── models_pet.py
│   ├── transformer_models.py
│   ├── train_pet.py
│   ├── evaluate_model.py
│   ├── evaluate_model_backup.py
│   ├── gradcam.py
│   ├── visualise_gradcam.py
│   ├── split_labels.py
│   ├── regenerate_plots.py
│   └── show_pretrained_info.py
└── DSPECT/                       # SPECT deep learning
    ├── config_hardware_optimised.yaml
    ├── config_transformers.yaml
    ├── dataset.py
    ├── models_spect.py
    ├── transformer_models.py
    ├── train_spect.py
    ├── evaluate_model.py
    ├── split_labels.py
    └── regenerate_plots.py
```

## 🧠 Model Architectures

### Custom Models (Created by P4P Team)

#### Simple3DCNN
- **Purpose**: Custom 3D convolutional neural network designed for medical imaging
- **Architecture**: 
  - 3D convolutions with batch normalization and ReLU activation
  - Progressive feature extraction through multiple conv blocks
  - Global average pooling and fully connected layers
  - Dropout for regularization
- **Use case**: Baseline model for comparison and lightweight inference
- **Input**: 3D medical images (e.g., 91×109×91 voxels)
- **Output**: Multi-class predictions (CN/AD/PD)

#### VisionTransformer3D
- **Purpose**: Custom Vision Transformer adapted for 3D medical images
- **Architecture**:
  - 3D patch embedding to convert image patches to tokens
  - Multi-head self-attention mechanism for spatial relationships
  - Feed-forward networks with residual connections
  - Positional encoding for 3D spatial awareness
  - Classification head for final predictions
- **Use case**: State-of-the-art performance with attention-based feature learning
- **Input**: 3D medical images with 3D patch extraction
- **Output**: Multi-class predictions with attention maps

### External Pre-trained Models

#### ResNet18/ResNet50 (3D Adaptation)
- **Source**: Torchvision models adapted for 3D
- **Architecture**: Residual networks with skip connections
- **Use case**: Transfer learning from ImageNet pre-trained weights
- **Benefits**: Proven architecture with pre-trained features

#### EfficientNet (3D Adaptation)
- **Source**: EfficientNet adapted for 3D medical imaging
- **Architecture**: Compound scaling with efficient building blocks
- **Use case**: High accuracy with reduced computational cost
- **Benefits**: Optimal balance between accuracy and efficiency

#### DenseNet (3D Adaptation)
- **Source**: Torchvision models adapted for 3D
- **Architecture**: Densely connected convolutional networks
- **Use case**: Feature reuse and parameter efficiency
- **Benefits**: Reduced overfitting and improved gradient flow

#### SwinUNETR
- **Source**: MONAI framework
- **Architecture**: Swin Transformer with U-Net architecture
- **Use case**: Medical image segmentation and classification
- **Benefits**: Hierarchical feature extraction with transformer attention

### Model Selection
```python
# Available models
models = {
    # Custom P4P models
    'simple3d': Simple3DCNN,           # Our custom 3D CNN
    'transformer': VisionTransformer3D, # Our custom 3D Vision Transformer
    
    # External pre-trained models
    'resnet18': ResNet18_3D,           # 3D ResNet18
    'resnet50': ResNet50_3D,           # 3D ResNet50
    'densenet': DenseNet_3D,           # 3D DenseNet
    'efficientnet': EfficientNet_3D,   # 3D EfficientNet
    'swinunetr': SwinUNETR             # SwinUNETR from MONAI
}
```

## 🚀 Training Pipeline

### Training Script (`train_*.py`)

#### Purpose
Complete training pipeline with data loading, model training, and evaluation.

#### Usage
```bash
# MRI training
cd MRI
python train_smri.py --config config_hardware_optimised.yaml

# PET training
cd PET
python train_pet.py --config config_hardware_optimised.yaml

# SPECT training
cd DSPECT
python train_spect.py --config config_hardware_optimised.yaml
```

#### Features
- **Automatic hardware detection**: GPU/CPU optimisation
- **Memory optimisation**: Efficient memory usage
- **Cross-validation**: Stratified k-fold cross-validation
- **Early stopping**: Prevent overfitting
- **Model checkpointing**: Save best models
- **Comprehensive logging**: Detailed training logs

### Data Loading (`dataset.py`)

#### Purpose
Efficient data loading and preprocessing for medical images.

#### Features
- **NIfTI loading**: Load 3D medical images
- **Data augmentation**: Random rotations, flips, and intensity changes
- **Memory mapping**: Efficient memory usage for large datasets
- **Error handling**: Robust error handling for corrupted files

#### Usage
```python
from dataset import SMRIDataset

# Create dataset
dataset = SMRIDataset(
    data_dir='~/path/to/data',
    labels_file='~/path/to/labels.csv',
    transform=train_transform
)

# Create data loader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)
```

## 📊 Model Evaluation

### Evaluation Script (`evaluate_model.py`)

#### Purpose
Comprehensive model evaluation with multiple metrics and visualisations.

#### Usage
```bash
python evaluate_model.py \
    --model ~/path/to/model.pth \
    --data ~/path/to/test_data \
    --output ~/path/to/evaluation_results
```

#### Features
- **Multiple metrics**: Accuracy, precision, recall, F1-score, AUC
- **Confusion matrix**: Detailed classification analysis
- **ROC curves**: Receiver operating characteristic curves
- **Calibration analysis**: Model calibration assessment
- **Statistical testing**: Significance testing

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve
- **Matthews Correlation Coefficient**: Balanced accuracy measure

## 🔍 Model Interpretability

### Grad-CAM (`gradcam.py`)

#### Purpose
Generate gradient-weighted class activation maps for model interpretability.

#### Usage
```bash
python visualise_gradcam.py \
    --model ~/path/to/model.pth \
    --input ~/path/to/image.nii.gz \
    --output ~/path/to/gradcam.nii.gz
```

#### Features
- **3D Grad-CAM**: Generate 3D activation maps
- **Multiple classes**: Support for multiclass predictions
- **Visualization**: Overlay activation maps on original images
- **Export**: Save activation maps as NIfTI files

### Saliency Maps
- **Gradient-based**: Compute input gradients
- **Integrated gradients**: More stable saliency maps
- **Occlusion sensitivity**: Systematic occlusion analysis

## 🎨 Visualization Tools

### Plot Regeneration (`regenerate_plots.py`)

#### Purpose
Regenerate plots from saved model results.

#### Usage
```bash
python regenerate_plots.py \
    --results ~/path/to/results \
    --output ~/path/to/plots
```

#### Features
- **Performance plots**: ROC curves, confusion matrices
- **Training curves**: Loss and accuracy curves
- **Feature maps**: Activation visualisations
- **Comparison plots**: Model comparison visualisations

### Interactive Visualization
- **3D rendering**: Interactive 3D image visualisation
- **Overlay display**: Show activation maps overlaid on images
- **Animation**: Animate through image slices

## 🔧 Configuration

### Hardware Optimization (`config_hardware_optimised.yaml`)

#### GPU Configuration
```yaml
hardware:
  use_gpu: true
  gpu_memory_fraction: 0.8
  mixed_precision: true
  dataloader_workers: 4
  pin_memory: true
```

#### Memory Optimization
```yaml
memory:
  batch_size: 8
  gradient_accumulation: 2
  checkpoint_gradients: true
  clear_cache_frequency: 100
```

### Transformer Configuration (`config_transformers.yaml`)

#### Vision Transformer Settings
```yaml
transformer:
  patch_size: [8, 8, 8]
  embed_dim: 768
  depth: 12
  num_heads: 12
  mlp_ratio: 4.0
  dropout: 0.1
  attention_dropout: 0.1
```

## 📈 Training Monitoring

### Logging
- **TensorBoard**: Real-time training monitoring
- **WandB**: Weights & Biases integration
- **Custom logging**: Detailed training logs

### Metrics Tracking
- **Loss curves**: Training and validation loss
- **Accuracy curves**: Training and validation accuracy
- **Learning rate**: Learning rate scheduling
- **Memory usage**: GPU and CPU memory usage

### Checkpointing
- **Model checkpoints**: Save model weights
- **Optimizer state**: Save optimizer state
- **Best model**: Save best performing model
- **Resume training**: Resume from checkpoints

## 🚨 Common Issues

### Training Issues
1. **Memory errors**: Reduce batch size or use gradient accumulation
2. **Convergence problems**: Adjust learning rate or architecture
3. **Overfitting**: Use regularization or data augmentation
4. **Slow training**: Optimize data loading or use mixed precision

### Data Issues
1. **File format**: Ensure NIfTI format compatibility
2. **Image dimensions**: Check image size consistency
3. **Label format**: Verify label file format
4. **Data loading**: Check data loader configuration

### Hardware Issues
1. **GPU memory**: Monitor GPU memory usage
2. **CUDA compatibility**: Check CUDA version compatibility
3. **Driver issues**: Update GPU drivers
4. **Performance**: Optimize hardware configuration

## 🔍 Debugging

### Training Debugging
- **Check data loading**: Verify data loader output
- **Monitor gradients**: Check gradient flow
- **Validate loss**: Ensure loss is decreasing
- **Check predictions**: Validate model predictions

### Model Debugging
- **Architecture validation**: Test model forward pass
- **Parameter counting**: Count model parameters
- **Memory profiling**: Profile memory usage
- **Performance profiling**: Profile training speed

## 📚 Dependencies

### Core Libraries
- **PyTorch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **nibabel**: Medical image I/O
- **numpy**: Numerical operations
- **pandas**: Data manipulation

### Optional Libraries
- **transformers**: Hugging Face transformers
- **timm**: PyTorch image models
- **wandb**: Weights & Biases
- **tensorboard**: TensorBoard logging

## 🚀 Performance Optimization

### Training Optimization
- **Mixed precision**: Use FP16 for faster training
- **Gradient accumulation**: Simulate larger batch sizes
- **Data loading**: Optimize data loader performance
- **Model compilation**: Use PyTorch 2.0 compilation

### Memory Optimization
- **Gradient checkpointing**: Trade compute for memory
- **Model sharding**: Distribute model across GPUs
- **Data streaming**: Stream data from disk
- **Memory mapping**: Use memory-mapped files

## 📞 Support

For deep learning issues:
- Check GPU memory and CUDA compatibility
- Validate data format and loading
- Review model architecture and configuration
- Monitor training logs and metrics
- Test with smaller datasets first
