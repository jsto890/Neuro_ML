# SHAP Interpretability for Classical Models

## 🎯 Overview

This directory includes comprehensive SHAP (SHapley Additive exPlanations) interpretability tools for classical machine learning models used in neurodegenerative disease detection. 

**Why SHAP?**
- Provides mathematically rigorous feature attribution based on game theory
- Shows which features contribute to each prediction (local explanations)
- Reveals overall feature importance across the dataset (global explanations)
- Works with a wide variety of model types (tree-based, linear, neural networks)
- More reliable than simple feature importance methods

**Note:** SHAP interpretability is specifically designed for classical models. For deep learning models (CNNs), we use GradCAM and other visualization techniques instead.

---

## 📋 What's Included

### Files
- **`shap_interpretability.py`** - Core SHAP module with `SHAPInterpreter` class
- **`run_shap_analysis.py`** - Command-line script to generate SHAP reports
- **`requirements_enhanced.txt`** - All dependencies including SHAP

### Supported Models
✅ **Tree-based** (fast, exact explanations):
- RandomForest
- GradientBoosting
- ExtraTrees
- XGBoost
- LightGBM

✅ **Linear models** (fast, exact explanations):
- LogisticRegression
- LinearSVM

✅ **Other models** (slower, approximate explanations):
- SVM (with KernelExplainer)
- KNN (with KernelExplainer)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd Scripts/Classic_Learning/Enhanced/
pip install -r requirements_enhanced.txt
```

Or just install SHAP:
```bash
pip install shap
```

### 2. Generate SHAP Reports

**Analyze a single trained model:**
```bash
python run_shap_analysis.py \
    --model /path/to/trained_model.pkl \
    --data /path/to/radiomics_features.csv \
    --output shap_results
```

**Analyze all models in a directory:**
```bash
python run_shap_analysis.py \
    --model_dir /path/to/models/ \
    --data /path/to/radiomics_features.csv \
    --output shap_results \
    --all
```

**With custom class names:**
```bash
python run_shap_analysis.py \
    --model /path/to/model.pkl \
    --data /path/to/data.csv \
    --output shap_results \
    --class_names CN AD PD
```

---

## 📊 What Gets Generated

For each model, SHAP analysis creates:

### 1. **Summary Plot** (`shap_summary_*.png`)
- Shows the most important features
- Each point represents one prediction
- Color indicates feature value (red=high, blue=low)
- Position shows SHAP value (impact on prediction)

**How to interpret:**
- Features at top = most important
- Wide spread = high variability in impact
- Red points on right = high feature values increase prediction
- Blue points on left = low feature values decrease prediction

### 2. **Bar Plot** (`shap_bar_*.png`)
- Mean absolute SHAP values for each feature
- Simple ranking of feature importance
- Good for quick comparison

### 3. **Dependence Plots** (`shap_dependence_*.png`)
- Shows relationship between feature value and SHAP value
- One plot per top feature
- Color indicates interaction with another feature
- Reveals non-linear relationships

**How to interpret:**
- X-axis = feature value
- Y-axis = SHAP value (impact on prediction)
- Trend shows how feature affects predictions
- Color shows how another feature modulates this effect

### 4. **Waterfall Plot** (`shap_waterfall_*.png`)
- Explains a single prediction
- Shows how each feature pushes prediction from base value to final prediction
- Red = pushes toward positive class
- Blue = pushes toward negative class

### 5. **SHAP Values CSV** (`shap_values_*.csv`)
- Raw SHAP values for all samples and features
- Can be used for custom analysis
- Each row = one sample
- Each column = SHAP value for one feature

### 6. **Summary Statistics** (`shap_summary_stats_*.json`)
- Metadata about the analysis
- Top features with mean absolute SHAP values
- Sample and feature counts

---

## 💻 Programmatic Usage

### Basic Usage

```python
from shap_interpretability import SHAPInterpreter
import pickle

# Load your trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create SHAP interpreter
interpreter = SHAPInterpreter(
    model=model,
    X_train=X_train,  # Training data for background
    feature_names=feature_names,
    output_dir='shap_results',
    model_name='RandomForest',
    class_names=['CN', 'AD']
)

# Generate comprehensive report
interpreter.generate_comprehensive_report(X_test, y_test)
```

### Custom Analysis

```python
# Compute SHAP values
shap_values = interpreter.compute_shap_values(X_test)

# Create specific plots
interpreter.plot_summary(X_test, max_display=15, plot_type='dot')
interpreter.plot_bar(X_test, max_display=20)
interpreter.plot_dependence(X_test, feature_idx=5)  # 6th feature
interpreter.plot_waterfall(X_test, sample_idx=0)  # First sample

# Export SHAP values for analysis
shap_df = interpreter.export_shap_values(X_test, y_test)
```

### Quick Load and Analyze

```python
from shap_interpretability import load_model_and_generate_shap

interpreter = load_model_and_generate_shap(
    model_path='model.pkl',
    X_train=X_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=feature_names,
    output_dir='shap_results',
    model_name='SVM',
    class_names=['Healthy', 'Diseased']
)
```

---

## 🔬 Clinical Interpretation Guide

### For Researchers

**Identifying Key Biomarkers:**
1. Look at the **bar plot** for overall feature importance
2. Check **summary plot** to see if effects are consistent across samples
3. Use **dependence plots** to understand relationships:
   - Linear trend = simple relationship
   - Curved trend = non-linear relationship
   - Scattered = complex interactions

**Understanding Model Decisions:**
1. Use **waterfall plots** for individual patients
2. Identify which features drive misclassifications
3. Compare SHAP values between correctly and incorrectly classified samples

### For Clinicians

**Reading SHAP Plots:**
- **Summary Plot**: Shows which brain regions/features matter most
- **Waterfall Plot**: Explains why a specific patient was classified a certain way
- **Red features**: Increase disease probability
- **Blue features**: Decrease disease probability

**Example Interpretation:**
```
If waterfall plot shows:
- Hippocampal atrophy (red, +0.3) → increases AD probability
- Normal cortical thickness (blue, -0.2) → decreases AD probability
- Base value: 0.5 (50% probability)
- Final prediction: 0.6 (60% probability)

→ Model predicts AD with 60% confidence, primarily due to hippocampal atrophy
```

---

## 📈 Best Practices

### 1. **Data Preparation**
- Use the **same preprocessing** for SHAP analysis as used during training
- Ensure feature names match training data
- Use training data as background for SHAP computation

### 2. **Computational Efficiency**
- **Tree models**: Very fast (TreeExplainer)
- **Linear models**: Fast (LinearExplainer)
- **Other models**: Can be slow (KernelExplainer)
  - For large datasets, use `max_samples` parameter to limit computation
  - Consider using a representative subset of data

### 3. **Interpretation**
- SHAP values are **additive**: sum of all SHAP values + base value = model prediction
- Compare SHAP importance with other methods (permutation importance, etc.)
- Look for consistency across different models
- Be cautious with highly correlated features (can split importance)

### 4. **Validation**
- Check if important features make clinical sense
- Verify findings on independent test set
- Compare with domain knowledge and literature

---

## 🔧 Troubleshooting

### Common Issues

**1. "SHAP library not available"**
```bash
pip install shap
```

**2. "Model type not supported"**
- Most sklearn models are supported
- For custom models, may need to implement predict function
- Consider wrapping model in sklearn-compatible interface

**3. "KernelExplainer is too slow"**
- Reduce background data size (use kmeans sampling)
- Limit number of test samples with `max_samples`
- Use tree-based models for faster explanations

**4. "SHAP values don't match feature importance"**
- This is normal! They measure different things:
  - Feature importance: average impact across all trees/coefficients
  - SHAP: per-prediction attribution with interaction effects
- SHAP is generally more reliable

**5. "Memory error on large datasets"**
- Reduce number of samples
- Use batched computation
- Increase system memory or use cloud computing

### Advanced Debugging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check SHAP values manually:
```python
print(f"SHAP values shape: {interpreter.shap_values.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Feature names count: {len(feature_names)}")
```

---

## 📚 Additional Resources

### SHAP Documentation
- Official SHAP docs: https://shap.readthedocs.io/
- SHAP paper: https://arxiv.org/abs/1705.07874
- Tutorial notebooks: https://github.com/slundberg/shap

### Related Interpretability Methods
- **LIME**: Local Interpretable Model-agnostic Explanations
- **Permutation Importance**: Feature importance via shuffling
- **Partial Dependence Plots**: Marginal effect of features
- **ICE Plots**: Individual Conditional Expectation plots

### Clinical ML Interpretability
- "Interpretable Machine Learning" by Christoph Molnar
- "Explainable AI in Healthcare" - review papers

---

## 🎓 Example Workflow

### Complete Analysis Pipeline

```bash
# 1. Train models (if not already done)
cd Scripts/Classic_Learning/Enhanced/
python run_enhanced.py

# 2. Generate SHAP reports for all models
python run_shap_analysis.py \
    --model_dir outputs/models/ \
    --data /path/to/radiomics_data.csv \
    --output shap_analysis/ \
    --class_names CN AD PD \
    --all

# 3. Review results
ls shap_analysis/
# - shap_RandomForest/
# - shap_SVM/
# - shap_LogisticRegression/
# ... (one directory per model)

# 4. Compare models
# - Look at bar plots to see which features each model prioritizes
# - Check if important features are consistent across models
# - Use summary plots to understand feature effects
```

### Integration with Enhanced Pipeline

```python
from enhanced_classifier import EnhancedRadiomicsClassifier
from shap_interpretability import SHAPInterpreter

# 1. Train models
classifier = EnhancedRadiomicsClassifier(
    input_path='radiomics_data.csv',
    output_dir='outputs/'
)
classifier.load_data()
classifier.preprocess()
classifier.select_features()
classifier.train_models()

# 2. Generate SHAP for best model
best_model_name = classifier.get_best_model()
best_model = classifier.best_models[best_model_name]

interpreter = SHAPInterpreter(
    model=best_model,
    X_train=classifier.X_train,
    feature_names=classifier.selected_features,
    output_dir='outputs/shap/',
    model_name=best_model_name
)

interpreter.generate_comprehensive_report(
    classifier.X_test,
    classifier.y_test
)
```

---

## ⚠️ Important Notes

1. **SHAP is for Classical Models Only**
   - For deep learning models (CNNs), use GradCAM instead
   - DeepSHAP exists but is computationally expensive for 3D medical images

2. **Computational Cost**
   - Tree models: Very fast
   - Linear models: Fast
   - Complex models: Can be very slow (minutes to hours)

3. **Interpretation Caveats**
   - SHAP assumes features are independent (may not hold for medical data)
   - High correlation between features can affect interpretation
   - Always validate findings with domain experts

4. **Clinical Use**
   - SHAP explanations are for research and development
   - Not a substitute for clinical judgment
   - Should be validated on independent cohorts

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review SHAP documentation: https://shap.readthedocs.io/
3. See main README in `Scripts/Classic_Learning/`

---

## 🔄 Updates and Maintenance

**Version History:**
- v1.0 (2025-10) - Initial SHAP integration for Enhanced pipeline

**Future Enhancements:**
- Integration with Optimised pipeline
- Batch processing for large model collections
- Interactive SHAP plots (HTML output)
- SHAP-based feature selection
- Comparison of SHAP values across different splits

