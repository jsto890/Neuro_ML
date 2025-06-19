# Classic Pipeline: Baseline Random Forest Classifier

This pipeline provides a simple, robust baseline for radiomics-based classification of neurodegenerative diseases using Random Forests.

## What Does It Do?
- Loads radiomics features and labels
- Handles missing values and basic preprocessing
- Selects informative features
- Trains a Random Forest classifier
- Evaluates performance (accuracy, ROC AUC, etc.)
- Outputs model, scaler, feature importance, and evaluation plots

## When to Use This Pipeline
- **First-time users**: Quick baseline to check data and pipeline
- **Sanity checks**: Validate data quality and feature extraction
- **Small datasets**: Simple, interpretable model
- **Benchmarking**: Compare with more advanced pipelines

## How to Run
```bash
cd Scripts/Classic_Learning/Classic/
python3 run_classical.py
```
- Edit `config_classical.yaml` to set data paths and parameters.

## Outputs
- `random_forest_model.pkl` — Trained Random Forest model
- `scaler.pkl` — Feature scaler
- `feature_importance.csv` — Ranked feature importances
- `evaluation_plots.png` — ROC, confusion matrix, etc.
- `results_summary.json` — Performance metrics

## Interpretation
- Use feature importance to identify key biomarkers
- Use evaluation plots to assess model quality

## Troubleshooting
- If you see many missing values, check your input data and preprocessing
- For more advanced analysis, try the Enhanced or Optimised pipelines 