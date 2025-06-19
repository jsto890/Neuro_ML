# Optimised Pipeline: Advanced, Clinical-Ready Machine Learning

This pipeline is the most advanced option, designed for maximum performance, robust feature engineering, and clinical deployment. It includes Bayesian optimization, stacking ensembles, polynomial features, and outlier removal.

## What Does It Do?
- Loads and preprocesses radiomics features and labels
- Removes outliers and handles missing data robustly
- Performs advanced feature engineering (polynomial features, statistical summaries)
- Selects top features using multiple methods and recursive elimination
- Trains and tunes multiple models (SVM, Random Forest, Logistic Regression, XGBoost, LightGBM) with Bayesian optimization
- Builds a stacking ensemble for best performance
- Outputs clinical-ready models, scaler, feature importance, and detailed results

## When to Use This Pipeline
- **Clinical deployment**: For highest accuracy and reliability
- **Publication-quality research**: For best results and advanced analysis
- **Large or complex datasets**: Handles high-dimensional data and outliers
- **Model interpretability**: Provides feature importance and model explanations

## How to Run
```bash
cd Scripts/Classic_Learning/Optimised/
python3 install_advanced_deps.py  # (First time only, to install extra dependencies)
python3 run_optimized.py
```
- Edit `config_optimized.yaml` to set data paths and parameters.

## Outputs
- `optimized_svm_model.pkl` — Primary clinical SVM model
- `optimized_ensemble_model.pkl` — Stacking ensemble (backup)
- `optimized_scaler.pkl` — Feature scaler
- `optimized_feature_importance.csv` — Ranked feature importances
- `optimized_evaluation_plots.png` — ROC, confusion matrix, etc.
- `optimized_results_summary.json` — Performance metrics and optimization details
- `feature_engineering_results.json` — Feature engineering summary
- `optimized_pipeline.log` — Execution log

## Interpretation
- Use the SVM model for clinical predictions
- Use the ensemble for backup or research
- Review feature importance for biomarker discovery

## Troubleshooting
- If you see convergence or memory warnings, adjust config parameters (e.g., `max_iter`, feature count)
- For clinical use, always validate on independent data
- For simpler/faster runs, try the Classic or Enhanced pipelines 