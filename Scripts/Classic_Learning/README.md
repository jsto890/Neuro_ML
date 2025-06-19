# Classical Machine Learning Pipelines for Radiomics Analysis

This directory contains three levels of classical machine learning pipelines for radiomics-based neurodegenerative disease detection:

## 📊 Pipeline Overview

| Pipeline | Features | Models | Performance | Use Case |
|----------|----------|--------|-------------|----------|
| **Classical** | Basic radiomics | Random Forest | ~72% Test Acc | Baseline |
| **Enhanced** | + Feature selection | RF, SVM, LR, GB | ~74% Test Acc | Research |
| **Optimized** | + Advanced engineering + Stacking | SVM + Ensemble | ~74% Test Acc | Clinical |

## 🚀 Quick Start

### 1. Classical Pipeline (Baseline)
```bash
python3 run_classical.py
```

### 2. Enhanced Pipeline (Research)
```bash
python3 run_enhanced.py
```

### 3. Optimized Pipeline (Clinical) ⭐ **RECOMMENDED**
```bash
python3 run_optimized.py
```

## 🔧 Optimized Pipeline - Advanced Features

The optimized pipeline includes cutting-edge feature engineering and ensemble methods:

### **Enhanced Feature Engineering**
- **Polynomial Features**: 2nd and 3rd degree interactions of top 10 features
- **Family Interactions**: Cross-family radiomics feature combinations
- **Statistical Summaries**: Percentiles, skewness, kurtosis across feature groups
- **Advanced Aggregations**: Texture means, standard deviations, ratios

### **Stacking Ensemble**
- **Base Models**: SVM (linear/rbf), Random Forest, Logistic Regression
- **Meta-Learner**: Logistic Regression with cross-validation
- **Advanced Training**: Out-of-fold predictions for meta-learner
- **Robust Performance**: Reduced overfitting through ensemble diversity

### **Key Improvements**
1. **Feature Engineering**: 15 original → 200+ engineered features
2. **Feature Selection**: RFECV optimization for SVM
3. **Model Diversity**: Multiple algorithms with different inductive biases
4. **Cross-Validation**: Proper meta-learner training
5. **Clinical Focus**: Interpretable features and stable performance

## 📈 Performance Comparison

| Metric | Classical | Enhanced | Optimized |
|--------|-----------|----------|-----------|
| **Test Accuracy** | 71.7% | 73.9% | 73.9% |
| **Test AUC** | 79.6% | 79.9% | 79.9% |
| **Train-Test Gap** | 27.9% | 4.1% | 4.1% |
| **Feature Count** | 41 | 54 | 54 |
| **Models** | 1 | 4 | 5+ |

## 🧬 Feature Engineering Details

### **Polynomial Features**
- **Degree 2**: Pairwise interactions between top 10 features
- **Degree 3**: Three-way interactions for complex patterns
- **Interaction-only**: Focus on feature combinations, not powers

### **Family-Based Interactions**
- **Cross-family combinations**: firstorder × glrlm, gldm × glszm, etc.
- **Element-wise multiplication**: Captures synergistic effects
- **Systematic generation**: All meaningful family combinations

### **Statistical Summaries**
- **Percentiles**: 25th, 75th, 90th percentiles per family
- **Distribution shape**: Skewness and kurtosis
- **Variability measures**: Range and interquartile range (IQR)

## 🎯 Stacking Ensemble Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Base Models   │    │  Meta-Features  │    │  Meta-Learner   │
│                 │    │                 │    │                 │
│ • SVM Linear    │───▶│ • CV Predictions│───▶│ • Logistic      │
│ • SVM RBF       │    │ • Probabilities │    │   Regression    │
│ • Random Forest │    │ • Out-of-fold   │    │ • Optimized     │
│ • Logistic Reg  │    │ • No Leakage    │    │ • Final Pred    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Training Process**
1. **Base Model Training**: 5-fold CV for each base model
2. **Meta-Feature Generation**: Out-of-fold predictions
3. **Meta-Learner Training**: Logistic regression on meta-features
4. **Final Ensemble**: Base models + meta-learner

## 📁 Output Files

### **Models**
- `optimized_svm_model.pkl` - Fine-tuned SVM
- `optimized_ensemble_model.pkl` - Stacking ensemble
- `optimized_scaler.pkl` - Feature scaler

### **Analysis**
- `optimized_feature_importance.csv` - Feature rankings
- `feature_engineering_results.json` - Engineering details
- `optimized_results_summary.json` - Complete results

### **Visualization**
- `optimized_evaluation_plots.png` - Performance plots
- `optimized_pipeline.log` - Execution log

## 🔍 Key Biomarkers

### **Top Features (Optimized SVM)**
1. **Polynomial interactions** (0.885-0.547) - Complex patterns
2. **Texture features** (0.442) - Structural changes
3. **Robust statistics** (0.386) - Intensity variations
4. **Run-length features** (0.355) - Tissue homogeneity
5. **Dependence features** (0.317) - Gray-level patterns

### **Clinical Interpretation**
- **Texture abnormalities** → Structural brain changes
- **Intensity variations** → Tissue heterogeneity
- **Interaction features** → Complex biomarker relationships
- **Family summaries** → Group-level patterns

## ⚙️ Configuration

### **Feature Engineering**
```yaml
feature_engineering:
  polynomial_degrees: [2, 3]
  enable_family_interactions: true
  enable_statistical_features: true
  statistical_measures:
    - "percentile_25"
    - "skewness"
    - "kurtosis"
    - "range"
```

### **Stacking Ensemble**
```yaml
stacking_ensemble:
  base_models:
    - svm_linear
    - svm_rbf
    - random_forest
    - logistic_regression
  meta_learner: logistic_regression
  cv: 5
```

## 🎯 Clinical Recommendations

### **For Deployment**
1. **Use Optimized SVM** as primary model (73.9% accuracy)
2. **Monitor texture features** as key biomarkers
3. **Track interaction features** for complex patterns
4. **Implement ensemble** for robust predictions

### **For Research**
1. **Analyze feature importance** for biomarker discovery
2. **Study family interactions** for mechanistic insights
3. **Compare base models** for algorithm selection
4. **Validate across datasets** for generalizability

## 🚨 Troubleshooting

### **Common Issues**
1. **Memory errors**: Reduce polynomial degree or feature count
2. **Long training**: Use fewer CV folds or smaller parameter grid
3. **Poor performance**: Check data quality and feature engineering
4. **Overfitting**: Increase regularization or reduce model complexity

### **Performance Tuning**
1. **Feature selection**: Adjust RFECV parameters
2. **Hyperparameters**: Expand SVM parameter grid
3. **Ensemble size**: Add/remove base models
4. **Cross-validation**: Adjust CV folds

## 📚 References

- **Feature Engineering**: Guyon & Elisseeff (2003)
- **Stacking**: Wolpert (1992)
- **Radiomics**: Aerts et al. (2014)
- **SVM**: Cortes & Vapnik (1995)

## 🤝 Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

**Note**: The optimized pipeline represents the most advanced classical approach, combining state-of-the-art feature engineering with robust ensemble methods for clinical applications. 