# Classical Learning Pipeline for Neurodegenerative Disease Detection

This directory contains advanced classical machine learning pipelines for early detection of neurodegenerative diseases using radiomics features from multimodal imaging data (MRI, PET, SPECT).

## Pipeline Overview

### 1. **Basic Classical Pipeline** (`radiomics_classifier.py`)
- Simple Random Forest classifier
- Basic preprocessing and feature selection
- Good starting point for baseline performance

### 2. **Enhanced Pipeline** (`enhanced_classifier.py`)
- Multiple algorithms (Random Forest, SVM, Logistic Regression, Gradient Boosting)
- Advanced feature selection (mutual information, F-statistic)
- Ensemble voting classifier
- Improved performance and model diversity

### 3. **Optimized Pipeline** (`optimized_classifier.py`) ⭐ **RECOMMENDED**
- **Bayesian optimization** for hyperparameter tuning
- **Diverse base models**: SVM, Random Forest, Logistic Regression, XGBoost, LightGBM
- Advanced feature engineering (polynomial features, statistical summaries)
- Stacking ensemble with cross-validation
- Outlier removal and robust preprocessing
- Best performance and clinical applicability

## Key Features

### Bayesian Optimization
- **SVM Hyperparameter Tuning**: Uses Bayesian optimization to find optimal C, gamma, kernel, and regularization parameters
- **XGBoost Optimization**: Optimizes learning rate, depth, subsample, regularization parameters
- **LightGBM Optimization**: Optimizes similar parameters with LightGBM-specific settings
- **Efficient Search**: More efficient than grid search, finds better parameters in fewer iterations

### Advanced Models
- **XGBoost**: Gradient boosting with regularization, handles missing values well
- **LightGBM**: Fast gradient boosting, good for large datasets
- **SVM**: Robust classifier with different kernels (linear, RBF)
- **Random Forest**: Ensemble of decision trees, good for feature importance
- **Logistic Regression**: Interpretable linear model

### Feature Engineering
- **Polynomial Features**: Captures non-linear relationships
- **Statistical Summaries**: Mean, std, skew, kurtosis of feature groups
- **Feature Selection**: Multiple methods (variance, mutual info, F-statistic, RFE)
- **Outlier Removal**: IQR or Z-score based outlier detection

### Ensemble Methods
- **Stacking Ensemble**: Uses cross-validation predictions as meta-features
- **Meta-learner**: Logistic regression to combine base model predictions
- **Diversity**: Different algorithms capture different patterns in data

## Installation

### Required Dependencies
```bash
pip install scikit-learn pandas numpy matplotlib seaborn scipy
```

### Optional Advanced Dependencies
```bash
# For Bayesian optimization
pip install scikit-optimize

# For XGBoost
pip install xgboost

# For LightGBM
pip install lightgbm
```

## Usage

### Quick Start (Optimized Pipeline)
```bash
cd Scripts/Classic_Learning
python run_optimized.py
```

### With Custom Configuration
```bash
python run_optimized.py --config config_optimized.yaml
```

### Binary Classification Only
```bash
python run_optimized.py --binary_only
```

## Configuration

### Key Configuration Options (`config_optimized.yaml`)

```yaml
# Bayesian optimization settings
models:
  svm:
    optimization_method: "bayesian"  # or "grid"
    bayesian_iterations: 50
  
  advanced_models:
    xgboost:
      enabled: true
      bayesian_iterations: 30
    lightgbm:
      enabled: true
      bayesian_iterations: 30

# Feature engineering
feature_engineering:
  polynomial_degree: 2
  statistical_features: true
  outlier_removal: true
  outlier_method: "iqr"
```

## Output Structure

```
optimized_classical_results/
├── optimized_svm_model.pkl          # Optimized SVM model
├── optimized_ensemble_model.pkl     # Stacking ensemble model
├── scaler.pkl                       # Fitted scaler
├── feature_importance.csv           # Feature importance rankings
├── evaluation_plots.png             # Performance visualization
├── results_summary.json             # Detailed results
├── pipeline.log                     # Execution log
└── config_used.yaml                 # Configuration used
```

## Performance Analysis

### Recent Results (Binary Classification)
- **Test Accuracy**: ~70%
- **Low Overfitting**: Good generalization
- **Feature Importance**: Polynomial features dominate
- **Model Diversity**: Multiple algorithms contribute to ensemble

### Key Insights
1. **Polynomial Features**: Most important features are polynomial combinations
2. **Model Stability**: Ensemble reduces variance and improves robustness
3. **Clinical Relevance**: SVM provides good interpretability for clinical use
4. **Scalability**: Pipeline handles missing values and outliers robustly

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install scikit-optimize xgboost lightgbm
   ```

2. **Convergence Warnings**
   - Increase `max_iter` in configuration
   - Use `StandardScaler` instead of `RobustScaler`
   - Reduce polynomial degree

3. **Memory Issues**
   - Reduce `bayesian_iterations`
   - Use fewer features in selection
   - Reduce ensemble size

4. **Poor Performance**
   - Check data quality and missing values
   - Try different feature selection methods
   - Adjust outlier removal threshold

### Performance Optimization

1. **Faster Training**
   - Reduce Bayesian optimization iterations
   - Use fewer CV folds
   - Disable advanced models if not needed

2. **Better Results**
   - Increase polynomial degree (if computational resources allow)
   - Add more statistical features
   - Try different outlier removal methods

## Advanced Usage

### Custom Feature Engineering
```python
# Add custom features in optimized_classifier.py
def add_custom_features(self, X):
    # Your custom feature engineering
    return X_enhanced
```

### Custom Models
```python
# Add custom models to ensemble
from your_custom_model import CustomClassifier
base_models['custom'] = CustomClassifier()
```

### Hyperparameter Tuning
```python
# Modify search spaces in optimize_advanced_models()
search_spaces = {
    'your_param': Real(0.1, 10.0, prior='log-uniform'),
    # Add more parameters
}
```

## Research Applications

This pipeline is designed for:
- **Early Disease Detection**: Binary classification of healthy vs. diseased
- **Feature Discovery**: Understanding important radiomics features
- **Clinical Translation**: Interpretable models for medical use
- **Multi-modal Analysis**: MRI, PET, SPECT data integration

## Contributing

1. Test changes with different datasets
2. Update configuration files for new features
3. Document new algorithms and methods
4. Maintain backward compatibility

## Citation

If you use this pipeline in your research, please cite:
```
Storey, J. (2025). P4P: Early Detection of Neurodegenerative Diseases 
using Multimodal Imaging and Machine Learning. 
[University of Auckland]
``` 