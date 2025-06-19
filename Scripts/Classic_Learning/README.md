# Classical Learning Pipelines for Neurodegenerative Disease Detection

This directory contains three levels of classical machine learning pipelines for early detection of neurodegenerative diseases using radiomics features from multimodal imaging data (MRI, PET, SPECT).

## 📂 Pipeline Overview & When to Use Each

| Pipeline   | Location         | Description                                                                 | When to Use                                      |
|------------|------------------|-----------------------------------------------------------------------------|--------------------------------------------------|
| Classic    | `Classic/`       | Baseline Random Forest pipeline with simple preprocessing and feature selection. | For quick baselines, sanity checks, or small datasets. |
| Enhanced   | `Enhanced/`      | Multi-model pipeline (RF, SVM, LR, GBM), advanced feature selection, ensemble voting. | For research, model comparison, or improved performance. |
| Optimised  | `Optimised/`     | Advanced pipeline with Bayesian optimization, stacking, polynomial features, outlier removal, and clinical-ready SVM. | For best performance, clinical deployment, or publication. |

**See the README in each subdirectory for full details and usage instructions.**

---

## Directory Structure

```
Classic_Learning/
├── Classic/      # Baseline Random Forest pipeline
├── Enhanced/     # Multi-model, advanced feature selection, ensemble
├── Optimised/    # Bayesian optimization, stacking, clinical-ready
└── README.md     # (this file)
```

## Quick Start

1. **Choose your pipeline:**
   - For a baseline: `Classic/`
   - For research/model comparison: `Enhanced/`
   - For best/clinical: `Optimised/`

2. **See the README in the chosen subdirectory for setup and usage.**

---

## Clinical Deployment

Optimised models and clinical scripts are in `Scripts/Clinical_Deploy/Classic/`.

---

## For More Information
- Each subdirectory contains a detailed README.
- For troubleshooting, see the end of each pipeline README.
- For deep learning, see `../Deep_Learning/MRI/README.md`.
---

This directory contains advanced classical machine learning pipelines for early detection of neurodegenerative diseases using radiomics features from multimodal imaging data (MRI, PET, SPECT).

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