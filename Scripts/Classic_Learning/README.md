# Classical Learning Pipeline for Neurodegenerative Disease Detection

This directory contains advanced classical machine learning pipelines for early detection of neurodegenerative diseases using radiomics features from multimodal imaging data (MRI, PET, SPECT).

## 🎯 **Current Status: OPTIMIZATION COMPLETE**

### **Chosen Models for Clinical Deployment** ⭐

#### **Primary Model: Optimized SVM**
- **File**: `optimized_svm_model.pkl`
- **Performance**: 78.0% test accuracy, 0.839 AUC
- **Status**: **Selected for clinical deployment**
- **Reason**: Best performance with good interpretability

#### **Backup Model: Stacking Ensemble**
- **File**: `optimized_ensemble_model.pkl`
- **Performance**: 70.7% test accuracy, 0.790 AUC
- **Status**: **Selected as robust backup**
- **Reason**: Ensemble diversity provides reliability

#### **Required Components**
- **Scaler**: `optimized_scaler.pkl` - Required for feature scaling
- **Feature Importance**: `optimized_feature_importance.csv` - For clinical interpretation

### **Other Models: Research/Comparison Only**
- `Base_svm_linear`, `Base_svm_rbf` - Comparison models
- `Base_random_forest`, `Base_logistic_regression` - Baseline models  
- `Base_xgboost`, `Base_lightgbm` - Advanced comparison models

**These are for research analysis only - not needed for clinical use.**

## 🚀 **Clinical Deployment**

### **Clinical Scripts Location**
Clinical deployment scripts have been moved to:
```
Scripts/Clinical_Deploy/Classic/
├── predict_clinical.py    # Clinical prediction system
├── validate_model.py      # Model validation
└── README.md             # Clinical deployment guide
```

### **For Clinical Use: Use These Files Only**
```
optimized_classical_results/
├── optimized_svm_model.pkl          # ← PRIMARY MODEL (Use this!)
├── optimized_ensemble_model.pkl     # ← BACKUP MODEL
├── optimized_scaler.pkl             # ← REQUIRED for scaling
└── optimized_feature_importance.csv # ← For interpretation
```

## 📋 **Pipeline Overview**

### 1. **Basic Classical Pipeline** (`radiomics_classifier.py`)
- Simple Random Forest classifier
- Basic preprocessing and feature selection
- Good starting point for baseline performance

### 2. **Enhanced Pipeline** (`enhanced_classifier.py`)
- Multiple algorithms (Random Forest, SVM, Logistic Regression, Gradient Boosting)
- Advanced feature selection (mutual information, F-statistic)
- Ensemble voting classifier
- Improved performance and model diversity

### 3. **Optimized Pipeline** (`optimized_classifier.py`) ⭐ **COMPLETE**
- **Bayesian optimization** for hyperparameter tuning
- **Diverse base models**: SVM, Random Forest, Logistic Regression, XGBoost, LightGBM
- Advanced feature engineering (polynomial features, statistical summaries)
- Stacking ensemble with cross-validation
- Outlier removal and robust preprocessing
- **Best performance and clinical applicability**

## 🎯 **Key Features**

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

## 📊 **Performance Analysis**

### Recent Results (Binary Classification)
- **Test Accuracy**: 78.0% (vs 71.7% baseline)
- **Low Overfitting**: Good generalization (81.3% → 78.0%)
- **Feature Importance**: Polynomial features dominate
- **Model Diversity**: Multiple algorithms contribute to ensemble

### Key Insights
1. **Polynomial Features**: Most important features are polynomial combinations
2. **Model Stability**: Ensemble reduces variance and improves robustness
3. **Clinical Relevance**: SVM provides good interpretability for clinical use
4. **Scalability**: Pipeline handles missing values and outliers robustly

## 🚀 **Next Steps After Optimization**

### **Immediate Actions (Do These Now)**

#### **1. Clinical Validation**
```bash
cd Scripts/Clinical_Deploy/Classic
python validate_model.py
```

#### **2. Test Clinical Predictions**
```bash
# Test on sample data
python predict_clinical.py ~/reseng202500013-ndd-ml/data/radiomics_MRI_mri_labels.csv test_predictions.csv
```

#### **3. Clinical Integration**
```python
# Load your optimized SVM model
import pickle
import pandas as pd

# Load the model and scaler
with open('optimized_svm_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('optimized_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Make predictions on new data
def predict_new_patient(new_features):
    scaled_features = scaler.transform(new_features)
    prediction = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)
    return prediction, probability
```

### **Research Actions (Optional)**

#### **1. Feature Analysis**
- Review `optimized_feature_importance.csv` for biomarker discovery
- Analyze polynomial feature meanings
- Correlate features with biological mechanisms

#### **2. Model Comparison**
- Compare all base models for research publication
- Analyze ensemble diversity
- Study feature importance across algorithms

#### **3. Advanced Research**
- Multi-modal integration (MRI + PET + SPECT)
- Temporal analysis for disease progression
- Clinical risk score development

## 📁 **Output Structure**

```
optimized_classical_results/
├── optimized_svm_model.pkl          # ← PRIMARY CLINICAL MODEL
├── optimized_ensemble_model.pkl     # ← BACKUP MODEL
├── optimized_scaler.pkl             # ← REQUIRED FOR PREDICTIONS
├── optimized_feature_importance.csv # ← FEATURE INTERPRETATION
├── optimized_evaluation_plots.png   # ← PERFORMANCE VISUALIZATION
├── optimized_results_summary.json   # ← DETAILED RESULTS
├── feature_engineering_results.json # ← ENGINEERING DETAILS
└── optimized_pipeline.log          # ← EXECUTION LOG
```

## 🏥 **Clinical Usage Guidelines**

### **High Confidence Predictions (≥80%)**
- Use for clinical decision making
- High reliability for diagnosis

### **Medium Confidence Predictions (60-80%)**
- Use with caution
- Consider additional clinical context

### **Low Confidence Predictions (<60%)**
- Do not use for clinical decisions
- Recommend additional testing

### **Model Limitations**
- Trained on binary classification (0/1)
- Requires same feature preprocessing
- Performance may vary on different populations

## 🔧 **Installation**

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

## 🚨 **Troubleshooting**

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

## 🎯 **Clinical Recommendations**

1. **Primary Model**: Use `optimized_svm_model.pkl` for all clinical predictions
2. **Backup Model**: Use `optimized_ensemble_model.pkl` for critical decisions
3. **Feature Analysis**: Review `optimized_feature_importance.csv` for key biomarkers
4. **Validation**: Test on independent dataset before clinical deployment
5. **Monitoring**: Track performance over time and update model periodically

## 📚 **Research Applications**

This pipeline is designed for:
- **Early Disease Detection**: Binary classification of healthy vs. diseased
- **Feature Discovery**: Understanding important radiomics features
- **Clinical Translation**: Interpretable models for medical use
- **Multi-modal Analysis**: MRI, PET, SPECT data integration

## 🤝 **Contributing**

1. Test changes with different datasets
2. Update configuration files for new features
3. Document new algorithms and methods
4. Maintain backward compatibility

## 📄 **Citation**

If you use this pipeline in your research, please cite:
```
Storey, J. (2025). P4P: Early Detection of Neurodegenerative Diseases 
using Multimodal Imaging and Machine Learning. 
[University of Auckland]
``` 