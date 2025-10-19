# Classic Learning Directory

This directory contains classical machine learning approaches for neurodegenerative disease detection using radiomics features extracted from medical images. The classical learning pipeline includes feature engineering, model training, evaluation, and deployment.

## 📁 Directory Structure

```
Classic_Learning/
├── README.md                     # This file
├── complete_workflow.py          # Complete radiomics workflow
├── run_best_model.py             # Run the best performing model
├── run_fdr_comparison.py         # False discovery rate comparison
└── Enhanced/                     # Enhanced classifiers
    ├── config_enhanced.yaml      # Enhanced configuration
    ├── enhanced_classifier.py    # Advanced ML classifiers
    └── run_enhanced.py           # Enhanced classifier pipeline
```

## 🔄 Complete Workflow

### Main Workflow Script (`complete_workflow.py`)

#### Purpose
Provides a complete end-to-end workflow from radiomics features to trained models using the Improved Optimized Pipeline.

#### Usage
```bash
# Basic usage
python complete_workflow.py \
    --input ~/path/to/radiomics_features.csv \
    --output ~/path/to/results/

# With custom configuration
python complete_workflow.py \
    --input ~/path/to/radiomics_features.csv \
    --output ~/path/to/results/ \
    --config ~/path/to/config.yaml

# With custom random seed
python complete_workflow.py \
    --input ~/path/to/radiomics_features.csv \
    --output ~/path/to/results/ \
    --random-state 123
```

#### Features
- **Complete pipeline**: End-to-end radiomics classification
- **Feature engineering**: Advanced feature selection and engineering
- **Model optimisation**: Bayesian optimisation for hyperparameters
- **Ensemble methods**: Stacking and voting ensembles
- **Comprehensive evaluation**: Multiple metrics and visualisations
- **Clinical interpretability**: SHAP analysis and feature importance

#### Output Structure
```
results/
├── models/                       # Trained models
│   ├── optimized_svm_model.pkl  # Primary SVM model
│   ├── optimized_ensemble_model.pkl  # Ensemble model
│   └── feature_scaler.pkl       # Feature scaler
├── plots/                        # Visualization plots
│   ├── model_comparison.png     # Model performance comparison
│   ├── roc_curves.png          # ROC curves
│   ├── feature_importance.png  # Feature importance
│   └── confusion_matrix.png    # Confusion matrix
├── metrics/                      # Performance metrics
│   ├── svm_results.json        # SVM model results
│   ├── ensemble_results.json   # Ensemble model results
│   └── feature_engineering.json # Feature engineering results
└── logs/                         # Training logs
    └── complete_workflow_*.log  # Workflow execution log
```

### Best Model Runner (`run_best_model.py`)

#### Purpose
Run the best performing model from previous training sessions.

#### Usage
```bash
python run_best_model.py \
    --input ~/path/to/radiomics_features.csv \
    --output ~/path/to/results/ \
    --model ~/path/to/best_model.pkl
```

#### Features
- **Model loading**: Load pre-trained models
- **Prediction**: Generate predictions on new data
- **Evaluation**: Comprehensive model evaluation
- **Reporting**: Generate detailed reports

### FDR Comparison (`run_fdr_comparison.py`)

#### Purpose
Compare models using False Discovery Rate (FDR) correction for multiple comparisons.

#### Usage
```bash
python run_fdr_comparison.py \
    --input ~/path/to/results/ \
    --output ~/path/to/fdr_comparison/
```

#### Features
- **FDR correction**: Multiple comparison correction
- **Statistical testing**: Hypothesis testing between models
- **Effect size**: Cohen's d and other effect size measures
- **Visualization**: Statistical comparison plots

## 🚀 Enhanced Classifiers (`Enhanced/`)

### Enhanced Classifier (`enhanced_classifier.py`)

#### Purpose
Advanced machine learning classifiers with multiple algorithms and ensemble methods.

#### Supported Algorithms
- **Random Forest**: Ensemble of decision trees
- **SVM**: Support Vector Machine with RBF kernel
- **Logistic Regression**: Linear and polynomial features
- **Gradient Boosting**: XGBoost and LightGBM
- **Extra Trees**: Extremely Randomized Trees
- **K-Nearest Neighbors**: Distance-based classification

#### Features
- **Multiple algorithms**: Support for various ML algorithms
- **Feature engineering**: Advanced feature selection and engineering
- **Hyperparameter optimisation**: Grid search and random search
- **Ensemble methods**: Voting and stacking ensembles
- **Cross-validation**: Stratified k-fold cross-validation
- **Calibration**: Probability calibration for better predictions

#### Usage
```bash
cd Enhanced

# Run enhanced classifier
python run_enhanced.py \
    --input ~/path/to/radiomics_features.csv \
    --output ~/path/to/enhanced_results/ \
    --config config_enhanced.yaml
```

#### Configuration (`config_enhanced.yaml`)
```yaml
data:
  binary_only: true
  random_state: 42

feature_engineering:
  variance_threshold: 0.01
  polynomial_degrees: [2]
  outlier_method: 'iqr_3x'
  normalization:
    enabled: true
    scaler: 'robust'

feature_selection:
  method: 'mutual_info_rfecv'
  mutual_info_k: 50
  rfecv_cv: 5

models:
  random_forest:
    n_estimators: [100, 200, 300]
    max_depth: [10, 20, None]
  svm:
    C: [0.1, 1, 10, 100]
    gamma: ['scale', 'auto', 0.001, 0.01]
  xgboost:
    n_estimators: [100, 200, 300]
    learning_rate: [0.01, 0.1, 0.2]
```

## 📊 Model Performance

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve
- **Matthews Correlation Coefficient**: Balanced accuracy measure

### Cross-Validation
- **Stratified K-Fold**: Maintains class distribution
- **Nested Cross-Validation**: Prevents data leakage
- **Time Series Split**: For temporal data (if applicable)

### Model Selection
- **Grid Search**: Exhaustive hyperparameter search
- **Random Search**: Randomized hyperparameter search
- **Bayesian Optimization**: Efficient hyperparameter optimisation

## 🔍 Feature Engineering

### Feature Selection Methods
- **Variance Threshold**: Remove low-variance features
- **Mutual Information**: Select features with high mutual information
- **Recursive Feature Elimination**: Iterative feature selection
- **Model-based Selection**: Use model feature importance

### Feature Engineering Techniques
- **Polynomial Features**: Generate polynomial interactions
- **Statistical Features**: Mean, std, skewness, kurtosis
- **Outlier Detection**: IQR-based outlier removal
- **Normalization**: Robust scaling and standardisation

### Feature Validation
- **Data Leakage Prevention**: Proper train/test splitting
- **Feature Stability**: Cross-validation feature selection
- **Statistical Significance**: Feature importance testing

## 🎯 Model Deployment

### Clinical Deployment
- **Model Serialization**: Save models in pickle format
- **Feature Scaler**: Save preprocessing scalers
- **Configuration**: Save model configuration
- **Documentation**: Generate model documentation

### Prediction Interface
- **Batch Prediction**: Process multiple samples
- **Single Prediction**: Process individual samples
- **Confidence Intervals**: Uncertainty quantification
- **Feature Importance**: Explain individual predictions

## 📈 Visualization

### Performance Plots
- **ROC Curves**: Receiver Operating Characteristic curves
- **Precision-Recall Curves**: Precision-recall trade-offs
- **Confusion Matrix**: Classification performance matrix
- **Learning Curves**: Training vs validation performance

### Feature Analysis
- **Feature Importance**: Bar plots of feature importance
- **Feature Correlations**: Correlation heatmaps
- **Feature Distributions**: Histograms and box plots
- **Feature Selection**: Feature selection progress

### Model Comparison
- **Performance Comparison**: Side-by-side model comparison
- **Statistical Testing**: Significance testing between models
- **Effect Size**: Effect size visualisation
- **Bias Analysis**: Bias detection and visualisation

## 🔧 Configuration

### Default Configuration
The pipeline uses sensible defaults for most parameters:

```python
default_config = {
    'data': {
        'binary_only': True,
        'random_state': 42
    },
    'feature_engineering': {
        'variance_threshold': 0.01,
        'polynomial_degrees': [2],
        'outlier_method': 'iqr_3x',
        'normalization': {
            'enabled': True,
            'scaler': 'robust'
        }
    },
    'feature_selection': {
        'method': 'mutual_info_rfecv',
        'mutual_info_k': 50,
        'rfecv_cv': 5
    },
    'evaluation': {
        'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
        'cv_folds': 5
    }
}
```

### Custom Configuration
Create custom configuration files for specific use cases:

```yaml
# custom_config.yaml
data:
  binary_only: false  # Enable multiclass
  random_state: 123

feature_engineering:
  variance_threshold: 0.05
  polynomial_degrees: [2, 3]
  outlier_method: 'zscore'
  outlier_threshold: 3.0

models:
  svm:
    C: [0.1, 1, 10]
    gamma: ['scale', 'auto']
```

## 🚨 Common Issues

### Data Issues
1. **Missing values**: Handle missing values in radiomics features
2. **Infinite values**: Check for infinite values in features
3. **Data leakage**: Ensure proper train/test splitting
4. **Class imbalance**: Handle imbalanced datasets

### Model Issues
1. **Overfitting**: Use regularization and cross-validation
2. **Underfitting**: Increase model complexity
3. **Convergence**: Check optimisation convergence
4. **Memory issues**: Use batch processing for large datasets

### Feature Issues
1. **High dimensionality**: Use dimensionality reduction
2. **Correlated features**: Remove highly correlated features
3. **Feature scaling**: Ensure proper feature scaling
4. **Feature selection**: Use appropriate feature selection methods

## 🔍 Debugging

### Logging
- **Workflow logs**: Detailed execution logs
- **Model logs**: Training and evaluation logs
- **Error logs**: Error and warning messages
- **Performance logs**: Timing and memory usage

### Validation
- **Data validation**: Check input data quality
- **Feature validation**: Validate feature engineering
- **Model validation**: Cross-validation results
- **Output validation**: Check prediction quality

## 📚 Dependencies

### Required Libraries
- **scikit-learn**: Machine learning algorithms
- **numpy**: Numerical operations
- **pandas**: Data manipulation
- **matplotlib**: Plotting
- **seaborn**: Statistical plotting
- **xgboost**: Gradient boosting
- **lightgbm**: Light gradient boosting

### Optional Libraries
- **shap**: Model interpretability
- **optuna**: Hyperparameter optimisation
- **imbalanced-learn**: Imbalanced data handling

## 🚀 Performance Optimization

### Memory Management
- **Batch processing**: Process data in batches
- **Memory profiling**: Monitor memory usage
- **Garbage collection**: Clean up unused objects

### Computational Optimization
- **Parallel processing**: Use multiple CPU cores
- **Vectorization**: Use vectorized operations
- **Caching**: Cache intermediate results

## 📞 Support

For classical learning issues:
- Check configuration files and parameters
- Validate input data format and quality
- Review log files for detailed error messages
- Test with sample data first
- Check library versions and compatibility
