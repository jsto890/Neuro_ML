# Model Plot Regeneration

This document explains how to regenerate evaluation plots for previously trained models using the new box plot format.

## Overview

The new plotting system includes:
- **Box plots** instead of bar charts for better statistical visualization
- **Model name and image type** in plot titles
- **Standard deviations** and statistical annotations
- **Publication-quality formatting** optimized for engineering journals

## Quick Start

### Regenerate All Models
```bash
# Regenerate all models in your checkpoint directory
python regenerate_all_plots.py --checkpoint_dir ~/data/checkpoints_multi

# Dry run to see what would be processed
python regenerate_all_plots.py --checkpoint_dir ~/data/checkpoints_multi --dry_run

# Only regenerate specific image type
python regenerate_all_plots.py --checkpoint_dir ~/data/checkpoints_multi --image_type sMRI

# Only regenerate specific model
python regenerate_all_plots.py --checkpoint_dir ~/data/checkpoints_multi --model_name Simple3DCNN
```

### Regenerate Individual Models
```bash
# MRI models
python Scripts/Deep_Learning/MRI/regenerate_plots.py \
  --model_dir ~/data/checkpoints_multi/run_20250918_091023/Simple3DCNN \
  --labels 0 1 2

# PET models  
python Scripts/Deep_Learning/PET/regenerate_plots.py \
  --model_dir ~/data/checkpoints_multi/run_20250918_091023/Simple3DCNN \
  --labels 0 1 2

# SPECT models
python Scripts/Deep_Learning/DSPECT/regenerate_plots.py \
  --model_dir ~/data/checkpoints_multi/run_20250918_091023/Simple3DCNN \
  --labels 0 1 2
```

## What Gets Regenerated

For each model, the script will:

1. **Load saved predictions and probabilities** from `test_evaluation_plots_fold_*/` directories
2. **Generate new box plot format plots** with:
   - Model name and image type in title
   - Box plots with mean/std annotations
   - Publication-quality formatting
   - Standard deviations displayed
3. **Save to** `evaluation_plots/model_evaluation_analysis.png`

## Requirements

- Models must have been trained with the updated training scripts that save predictions/probabilities
- The following files must exist in each fold directory:
  - `predictions.npy`
  - `probabilities.npy` 
  - `labels.npy`

## File Structure

```
checkpoints_multi/
├── run_20250918_091023/
│   ├── Simple3DCNN/
│   │   ├── test_evaluation_plots_fold_1/
│   │   │   ├── predictions.npy
│   │   │   ├── probabilities.npy
│   │   │   └── labels.npy
│   │   ├── test_evaluation_plots_fold_2/
│   │   │   └── ...
│   │   └── evaluation_plots/          # ← New plots generated here
│   │       └── model_evaluation_analysis.png
│   └── VisionTransformer3D/
│       └── ...
└── run_20250917_180728/
    └── ...
```

## New Plot Features

### Box Plots
- **Metrics Plot**: Box plot showing distribution of performance metrics
- **Prediction Distribution**: Box plot of predicted vs actual class counts
- **Probability Distribution**: Box plot of confidence scores
- **Per-Class Performance**: Box plot of precision/recall/F1 per class
- **Class Balance Analysis**: Box plot of class distribution
- **Confidence Analysis**: Box plot of prediction confidence by class

### Enhanced Information
- **Model Name**: Clearly displayed in plot title
- **Image Type**: sMRI, PET, or SPECT specified
- **Statistical Annotations**: Mean (μ) and standard deviation (σ) displayed
- **Publication Format**: 300 DPI, Times New Roman font, professional styling

## Troubleshooting

### "No predictions/probabilities files found"
- Ensure models were trained with updated training scripts
- Check that `.npy` files exist in fold directories

### "Could not import plotting helpers"
- Run from the correct environment (sMRI3d, PET3d, etc.)
- Ensure all dependencies are installed

### "No trained models found"
- Verify checkpoint directory path is correct
- Check that model directories contain `.pth` files

## Examples

### Regenerate Only Recent Models
```bash
# Find recent runs
ls -la ~/data/checkpoints_multi/run_*

# Regenerate specific recent run
python regenerate_all_plots.py \
  --checkpoint_dir ~/data/checkpoints_multi/run_20250918_091023
```

### Regenerate with Custom Labels
```bash
# For binary classification (AD vs rest)
python regenerate_all_plots.py \
  --checkpoint_dir ~/data/checkpoints_multi \
  --labels 0 1

# For custom class mapping
python regenerate_all_plots.py \
  --checkpoint_dir ~/data/checkpoints_multi \
  --labels 1 2 3
```

## Output

The regenerated plots will be saved as:
- **Main plot**: `evaluation_plots/model_evaluation_analysis.png`
- **Metrics**: `evaluation_plots/evaluation_metrics.json`

These plots are optimized for publication and include all the statistical information needed for engineering journal submissions.
