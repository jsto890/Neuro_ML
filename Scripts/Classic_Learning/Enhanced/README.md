# Enhanced Pipeline: Multi-Model & Ensemble Learning

This pipeline builds on the Classic pipeline by adding multiple algorithms, advanced feature selection, and ensemble methods for improved performance and model diversity.

## What Does It Do?
- Loads radiomics features and labels
- Handles missing values and preprocessing
- Performs advanced feature selection (mutual information, F-statistic)
- Trains multiple models: Random Forest, SVM, Logistic Regression, Gradient Boosting
- Combines models with a voting ensemble
- Evaluates and compares all models
- Outputs models, scaler, feature importance, and evaluation plots

## When to Use This Pipeline
- **Research and model comparison**: Test multiple algorithms on your data
- **Improved performance**: Get better results than a single model
- **Feature engineering**: Explore which features matter most
- **Ensemble learning**: Increase robustness and reduce overfitting

## How to Run
```bash
cd Scripts/Classic_Learning/Enhanced/
python3 run_enhanced.py
```
- Edit `config_enhanced.yaml` to set data paths and parameters.

## Outputs
- Trained models for each algorithm (e.g., `rf_model.pkl`, `svm_model.pkl`)
- `scaler.pkl` — Feature scaler
- `feature_importance.csv` — Ranked feature importances
- `evaluation_plots.png` — ROC, confusion matrix, etc.
- `results_summary.json` — Performance metrics for all models

## Interpretation
- Compare model performance to select the best for your data
- Use feature importance to guide further research

## 🔬 SHAP Interpretability (NEW!)

### Quick Start - Single Fold
```bash
# Analyze all models in a single fold
python run_shap_analysis.py \
    --model_dir outputs/outercv_fold_5/ \
    --data /path/to/radiomics_data.csv \
    --output shap_results \
    --class_names CN PD \
    --all
```

### Comprehensive Analysis - All Folds, All Models ⭐ RECOMMENDED
```bash
# Complete analysis: ALL 8 models × all folds + ensemble
python run_shap_comprehensive.py \
    --cv_dir /path/to/run_20251010_171321 \
    --data /path/to/radiomics_data.csv \
    --output shap_comprehensive \
    --class_names CN PD
```

**Analyzes all 8 models by default:**
- RandomForest, ExtraTrees, GradientBoosting
- XGBoost, LightGBM
- SVM, LogisticRegression, KNN

**What you get:**
- ✅ Per-model SHAP averaged across all folds
- ✅ Cross-model comparison (which features do different models prioritize?)
- ✅ Ensemble feature importance (voting + weighted strategies)
- ✅ Consensus biomarkers (stable across models AND folds)
- ✅ Comprehensive visualizations and rankings

**Speed options:**
```bash
# Fast: Tree models only (~10 min)
--model_types randomforest extratrees gradientboosting xgboost lightgbm

# Medium: Tree + Linear (~20 min)
--model_types randomforest xgboost lightgbm logisticregression

# Full: All 8 models (~30-40 min)
# (no --model_types argument, uses default)
```

### Single Model Analysis (compare across folds)
```bash
# Deep dive into one model across all folds
python run_shap_multifold.py \
    --cv_dir /path/to/run_20251010_171321 \
    --data /path/to/radiomics_data.csv \
    --output shap_multifold_rf \
    --model_type randomforest \
    --class_names CN PD
```

📖 **See `README_SHAP_COMPREHENSIVE.md` for complete documentation**

## Troubleshooting
- If convergence warnings occur, increase `max_iter` in config
- For best results, ensure input data is clean and preprocessed
- For maximum performance, try the Optimised pipeline 