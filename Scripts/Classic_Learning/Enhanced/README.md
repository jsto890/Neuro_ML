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

**Generate comprehensive SHAP explanations for your models:**

```bash
# Analyze a trained model
python run_shap_analysis.py \
    --model outputs/rf_model.pkl \
    --data /path/to/radiomics_data.csv \
    --output shap_results

# Analyze all models at once
python run_shap_analysis.py \
    --model_dir outputs/ \
    --data /path/to/radiomics_data.csv \
    --output shap_results \
    --all
```

**What you get:**
- Summary plots showing most important features
- Dependence plots revealing feature relationships
- Individual prediction explanations (waterfall plots)
- Exportable SHAP values for further analysis

📖 **See `README_SHAP.md` for complete documentation and usage guide.**

## Troubleshooting
- If convergence warnings occur, increase `max_iter` in config
- For best results, ensure input data is clean and preprocessed
- For maximum performance, try the Optimised pipeline 