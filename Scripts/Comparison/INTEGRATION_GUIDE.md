# Bayesian Analysis Integration Guide

## Overview

The Bayesian model comparison system integrates seamlessly with your existing P4P deep learning workflow. It works directly with the model outputs your training scripts already generate.

## Your Current Workflow

Your models already save exactly what we need:

```
checkpoints_multi_mri/run_20250918_143555/Simple3DCNN/
├── test_evaluation_plots_fold_1/
│   ├── predictions.npy      # ✅ Already saved
│   ├── probabilities.npy    # ✅ Already saved  
│   ├── labels.npy          # ✅ Already saved
│   └── evaluation_metrics.json
├── test_evaluation_plots_fold_2/
├── test_evaluation_plots_fold_3/
├── test_evaluation_plots_fold_4/
└── test_evaluation_plots_fold_5/
```

## Integration Points

### 1. **Data Compatibility** ✅
- **No changes needed** to your existing training scripts
- Works with current `.npy` file outputs
- Handles multiclass predictions (CN/AD/PD = 0/1/2)
- Uses existing fold structure for hierarchical modeling

### 2. **Complementary Analysis** 
- **Existing**: `compare_models.py` - Traditional statistical tests
- **New**: `bayesian_model_comparison.py` - Bayesian uncertainty quantification

### 3. **Enhanced Insights**
Your existing analysis provides:
- Per-fold metrics with confidence intervals
- Wilcoxon signed-rank tests
- McNemar tests for predictions
- DeLong tests for AUC

Bayesian analysis adds:
- **Hierarchical modeling** across sites/folds
- **Uncertainty quantification** with credible intervals
- **Model comparison probabilities** (P(model A > model B))
- **Calibration analysis** with Bayesian ECE
- **Stacking ensemble** with LOO weights

## Quick Start

### 1. Install Dependencies
```bash
cd Scripts/Comparison
pip install -r requirements_bayesian.txt
```

### 2. Test Setup
```bash
python test_bayesian_setup.py
```

### 3. Run Analysis
```bash
# Analyze your specific run
python run_bayesian_analysis.py --base-dir ~/reseng202500013-ndd-ml/data

# Or analyze specific models
python run_bayesian_analysis.py \
    --base-dir ~/reseng202500013-ndd-ml/data \
    --models Simple3DCNN ResNet3D
```

## Example Results

### Traditional Analysis (compare_models.py)
```
Model Comparison Results:
Simple3DCNN: AUC = 0.823 ± 0.045 (95% CI)
ResNet3D: AUC = 0.845 ± 0.038 (95% CI)
DeLong test p-value: 0.023 (significant)
```

### Bayesian Analysis (bayesian_model_comparison.py)
```
Hierarchical Accuracy Estimates:
Simple3DCNN: 0.823 (0.789, 0.857) [95% credible interval]
ResNet3D: 0.845 (0.812, 0.879) [95% credible interval]

Model Comparison Probabilities:
P(ResNet3D > Simple3DCNN) = 0.78 [78% probability ResNet3D is better]
P(Simple3DCNN > ResNet3D) = 0.22

Calibration Analysis:
Simple3DCNN: intercept=0.12, slope=0.95 [well calibrated]
ResNet3D: intercept=-0.05, slope=1.02 [slightly overconfident]
```

## Key Advantages for Your Research

### 1. **Proper Uncertainty Quantification**
- Credible intervals vs. confidence intervals
- Accounts for hierarchical structure (sites/folds)
- Quantifies uncertainty in model rankings

### 2. **Model Comparison with Probabilities**
- P(model A > model B) instead of just p-values
- Bayesian model selection with stacking weights
- Ensemble methods with theoretical guarantees

### 3. **Calibration Assessment**
- Bayesian calibration curves
- ECE with uncertainty
- Per-class calibration analysis

### 4. **Multiclass Handling**
- One-vs-rest calibration for each class
- Macro-averaged AUC with credible intervals
- Per-class performance analysis

## Workflow Integration

### Option 1: Standalone Analysis
```bash
# After training your models
python Scripts/Comparison/compare_models.py --run-dirs /path/to/checkpoints
python Scripts/Comparison/run_bayesian_analysis.py --base-dir /path/to/data
```

### Option 2: Integrated Pipeline
```python
# In your training script
from Scripts.Comparison.bayesian_model_comparison import BayesianModelComparison

# After training
comparator = BayesianModelComparison("~/results/bayesian")
results = comparator.run_complete_analysis([checkpoint_dir])
```

### Option 3: Research Publication
```python
# Generate publication-ready results
results = run_bayesian_analysis(checkpoint_dirs)

# Extract key findings
model_ranking = results.accuracy_results['model_comparisons']
calibration_quality = results.calibration_results
ensemble_weights = results.stacking_results['weights']
```

## Performance Considerations

### Computational Requirements
- **Memory**: ~2-4GB for typical datasets
- **Time**: 5-15 minutes per model (depending on data size)
- **CPU**: Uses multiple cores automatically
- **GPU**: Not required (CPU sampling is sufficient)

### Scaling Tips
- Start with 2-3 models to test setup
- Use JAX backend for faster sampling (optional)
- Reduce samples for quick exploratory analysis

## Publication Benefits

### 1. **Rigorous Statistical Analysis**
- Hierarchical Bayesian modeling
- Proper uncertainty quantification
- Model comparison with probabilities

### 2. **Calibration Assessment**
- Bayesian calibration curves
- ECE with credible intervals
- Per-class calibration analysis

### 3. **Ensemble Methods**
- Stacking with LOO weights
- Uncertainty-aware model selection
- Theoretical guarantees

### 4. **Reproducible Results**
- Fixed random seeds
- Comprehensive documentation
- Standardized output format

## Troubleshooting

### Common Issues
1. **Missing dependencies**: Run `pip install -r requirements_bayesian.txt`
2. **Memory issues**: Reduce number of models or samples
3. **Convergence issues**: Increase tuning samples
4. **Data format**: Ensure `.npy` files are correct shape

### Getting Help
1. Run `python test_bayesian_setup.py` to check setup
2. Check the README_BAYESIAN.md for detailed documentation
3. Use the example scripts as templates

## Next Steps

1. **Install dependencies** and test setup
2. **Run on small dataset** first (2-3 models)
3. **Compare with existing results** to validate
4. **Integrate into research workflow** for publications
5. **Extend with additional models** as needed

The Bayesian analysis provides a significant upgrade to your model comparison capabilities while working seamlessly with your existing infrastructure!
