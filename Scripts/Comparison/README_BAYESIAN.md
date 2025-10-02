# Bayesian Model Comparison for P4P Deep Learning Models

This module provides comprehensive Bayesian analysis for comparing deep learning models trained on P4P data (sMRI, PET, SPECT). It implements state-of-the-art Bayesian methods for model comparison with proper uncertainty quantification.

## Features

### 🔬 **Hierarchical Accuracy Estimation**
- Beta-binomial models with partial pooling across sites/folds
- Proper uncertainty quantification for model performance
- Model ranking with probabilities (P(model A > model B))

### 🎯 **Trial-Level Skill Analysis**
- Mixed-effects logistic regression using Bambi
- Random effects for sites and subjects
- PSIS-LOO model comparison
- Stacking weights for principled ensemble methods

### 📊 **Bayesian Calibration Analysis**
- Platt-style calibration with site pooling
- Per-class calibration intercepts and slopes
- Bayesian ECE (Expected Calibration Error) estimation
- Multiclass calibration comparison

### 📈 **Bayesian AUC Estimation**
- Binormal model for AUC computation
- Per-class AUC with credible intervals
- Multiclass macro-averaged AUC comparison

### 🤝 **Stacking Ensemble Methods**
- Bayesian stacking with LOO weights
- Uncertainty-aware model selection
- Ensemble performance estimation

## Installation

```bash
# Install Bayesian dependencies
cd Scripts/Comparison
pip install -r requirements_bayesian.txt
```

## Quick Start

### Basic Usage

```python
from bayesian_model_comparison import BayesianModelComparison

# Initialize comparator
comparator = BayesianModelComparison("~/P4P_results/bayesian_analysis")

# Run analysis on your model outputs
results = comparator.run_complete_analysis(
    run_dirs=["/path/to/your/checkpoints_multi_mri"],
    models=["Simple3DCNN", "ResNet3D"]  # Optional: specify models
)
```

### Command Line Usage

```bash
python bayesian_model_comparison.py \
    --run-dirs /path/to/checkpoints_multi_mri \
    --models Simple3DCNN ResNet3D \
    --output-dir ~/P4P_results/bayesian_analysis
```

## Data Requirements

The analysis expects model outputs in this structure:

```
run_directory/
├── ModelName1/
│   ├── test_evaluation_plots_fold_1/
│   │   ├── predictions.npy      # Model predictions (0, 1, 2 for CN/AD/PD)
│   │   ├── probabilities.npy    # Prediction probabilities (3-class softmax)
│   │   ├── labels.npy           # True labels
│   │   └── evaluation_metrics.json
│   ├── test_evaluation_plots_fold_2/
│   └── ... (folds 3-5)
└── ModelName2/
    └── ...
```

## Analysis Outputs

### 📁 **Directory Structure**
```
output_dir/
├── plots/
│   ├── hierarchical_accuracy_analysis.png
│   ├── calibration_analysis.png
│   ├── auc_comparison.png
│   └── model_comparison_matrix.png
├── results/
│   ├── accuracy_results.json
│   ├── model_comparisons.csv
│   ├── skill_results.json
│   └── ensemble_results.json
└── data/
    ├── accuracy_data.csv
    ├── skill_data.csv
    ├── calibration_data.csv
    └── auc_data.csv
```

### 📊 **Key Results**

#### **Hierarchical Accuracy Results**
- Posterior accuracy estimates with 95% credible intervals
- Model comparison probabilities
- Site/fold random effects

#### **Model Comparison Matrix**
- P(model A > model B) for all model pairs
- Uncertainty-aware ranking

#### **Calibration Analysis**
- Per-class calibration intercepts and slopes
- Bayesian ECE estimates
- Calibration quality comparison

#### **AUC Analysis**
- Per-class AUC with credible intervals
- Multiclass macro-averaged AUC
- Model performance ranking

## Advanced Usage

### Custom Analysis

```python
# Load data manually
model_data = comparator.load_model_data(run_dirs, models)
data_dict = comparator.prepare_data_for_analysis(model_data)

# Run specific analyses
accuracy_results = comparator.hierarchical_accuracy_analysis(data_dict['accuracy'])
calibration_results = comparator.bayesian_calibration_analysis(data_dict['calibration'])

# Create custom visualizations
comparator.create_visualizations(results, data_dict)
```

### Multiclass Considerations

The analysis handles your 3-class problem (CN/AD/PD) by:

1. **One-vs-Rest Calibration**: Each class is calibrated separately against the others
2. **Macro-Averaged AUC**: AUC computed per class, then macro-averaged
3. **Multiclass Accuracy**: Uses overall accuracy across all classes
4. **Per-Class Analysis**: Separate calibration and AUC for each class

### Site/Fold Handling

- **Sites**: Currently uses fold as site identifier (can be extended to use actual scanner/site info)
- **Partial Pooling**: Hierarchical models pool information across sites/folds
- **Random Effects**: Accounts for site-specific variations

## Example Results Interpretation

### Accuracy Comparison
```
Model Accuracy Estimates (95% CI):
  Simple3DCNN: 0.8234 (0.7891, 0.8577)
  ResNet3D: 0.8456 (0.8123, 0.8789)
  DenseNet3D: 0.8345 (0.8012, 0.8678)

Model Comparison Probabilities:
  P(ResNet3D > Simple3DCNN) = 0.78
  P(DenseNet3D > Simple3DCNN) = 0.65
  P(ResNet3D > DenseNet3D) = 0.71
```

### Calibration Analysis
```
Class 0 (CN) Calibration:
  Simple3DCNN: intercept=0.12, slope=0.95 (well calibrated)
  ResNet3D: intercept=-0.05, slope=1.02 (slightly overconfident)
```

## Integration with Existing Workflow

This Bayesian analysis complements your existing `compare_models.py` by providing:

1. **Uncertainty Quantification**: Proper credible intervals vs. simple confidence intervals
2. **Hierarchical Modeling**: Accounts for site/fold structure
3. **Bayesian Model Comparison**: PSIS-LOO and stacking weights
4. **Calibration Analysis**: Bayesian calibration assessment
5. **Ensemble Methods**: Principled model combination

## Performance Notes

- **Sampling**: Uses NUTS sampler with 2000 samples + 2000 tuning
- **Convergence**: Monitor trace plots for convergence (saved in ArviZ format)
- **Memory**: Handles multiclass data efficiently
- **Parallelization**: PyMC can use multiple cores automatically

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install with `pip install -r requirements_bayesian.txt`
2. **Memory Issues**: Reduce number of samples or models
3. **Convergence Issues**: Increase tuning samples or adjust priors
4. **Data Format**: Ensure `.npy` files are in correct format

### Performance Tips

1. **Start Small**: Test with 2-3 models first
2. **Check Convergence**: Use ArviZ diagnostics
3. **Use JAX Backend**: Install `jax` and `jaxlib` for faster sampling
4. **Monitor Memory**: Large datasets may require chunking

## Citation

If you use this Bayesian analysis in your research, please cite:

```bibtex
@software{p4p_bayesian_analysis,
  title={Bayesian Model Comparison for P4P Deep Learning Models},
  author={P4P Team},
  year={2025},
  url={https://github.com/your-repo/P4P}
}
```

## References

- [PyMC Documentation](https://www.pymc.io/)
- [ArviZ Documentation](https://python.arviz.org/)
- [Bambi Documentation](https://bambinos.github.io/bambi/)
- [Bayesian Data Analysis (Gelman et al.)](https://www.stat.columbia.edu/~gelman/book/)
- [Model Selection and Multi-Model Inference (Burnham & Anderson)](https://link.springer.com/book/10.1007/b97636)
