# Comprehensive Multi-Fold, Multi-Model SHAP Analysis

## Overview

The `run_shap_comprehensive.py` script performs exhaustive SHAP interpretability analysis across:
- **All CV folds** (e.g., 5-fold cross-validation)
- **All model types** (RandomForest, XGBoost, LightGBM, etc.)
- **Multi-class classification** (handles CN vs PD, or CN vs AD vs PD)

It generates three levels of analysis:

1. **Per-Model Cross-Fold Analysis** - Averaged SHAP values across folds for each model
2. **Cross-Model Comparison** - Which features do different models prioritize?
3. **Ensemble Feature Importance** - Simulates ensemble learning by combining all models

---

## Quick Start

```bash
cd Scripts/Classic_Learning/Enhanced/

python run_shap_comprehensive.py \
    --cv_dir ~/data/classic_results/enhanced_run_SPECT/run_20251010_171321 \
    --data ~/data/radiomics_spect.csv \
    --output ~/data/shap_comprehensive \
    --model_types randomforest gradientboosting xgboost lightgbm \
    --class_names CN PD
```

---

## What It Does

### Stage 1: Data Collection
- Scans all `outercv_fold_*` directories
- For each fold, loads each specified model
- Computes SHAP values on test set
- Handles feature selection automatically

### Stage 2: Per-Model Aggregation
For each model type (e.g., RandomForest):
- Aggregates SHAP values across all folds
- Computes: mean, std, median, CV (coefficient of variation)
- Identifies features consistently in top 10 across folds

### Stage 3: Cross-Model Comparison
- Compares which features different models prioritize
- Identifies consensus features (important across all models)
- Measures cross-model stability

### Stage 4: Ensemble Analysis
Three ensemble strategies:
1. **Equal-Weighted**: Simple average across all models
2. **Stability-Weighted**: Weight models by inverse of CV (more stable = higher weight)
3. **Voting-Based**: Count how many models rank feature in top 20

---

## Generated Outputs

### CSV Files

**`model_comparison_feature_importance.csv`**
- All features with importance from each model
- `consensus_importance`: Average across models
- `cross_model_cv`: Stability across models
- `[model]_mean`: Mean SHAP for each model
- `[model]_cv`: Within-fold CV for each model
- `[model]_top10_freq`: How often in top 10 across folds

**`ensemble_feature_importance.csv`**
- All features with ensemble scores
- `ensemble_equal`: Equal-weighted ensemble
- `ensemble_weighted`: Stability-weighted ensemble
- `vote_frequency`: Proportion of models ranking in top 20
- Individual model importances

### Visualizations

**Model Comparison:**
- `model_comparison_heatmap.png` - Top 30 features × models (heatmap)
- `consensus_features_grouped.png` - Top 25 consensus features (grouped bars)
- `cross_model_stability.png` - Feature stability across models

**Ensemble Analysis:**
- `ensemble_importance_weighted.png` - Top 30 by weighted ensemble
- `ensemble_voting_frequency.png` - Top 30 by voting
- `ensemble_strategy_comparison.png` - Three strategies side-by-side

### JSON Summaries

**`model_comparison_summary.json`**
```json
{
  "n_models": 4,
  "model_types": ["randomforest", "gradientboosting", "xgboost", "lightgbm"],
  "top_10_consensus": [...],
  "top_10_stable": [...]
}
```

**`ensemble_summary.json`**
```json
{
  "ensemble_strategy": "stability_weighted",
  "model_weights": {"randomforest": 1.23, ...},
  "top_10_features": [...],
  "top_20_by_voting": [...]
}
```

---

## Interpreting Results

### Consensus Features
**High consensus + low CV = Most reliable biomarkers**

Look for features with:
- High `consensus_importance` (important across models)
- Low `cross_model_cv` (consistent across models)
- High `vote_frequency` (many models agree)

Example:
```
Feature: original_glcm_JointEnergy
  Consensus Importance: 0.045
  Cross-Model CV: 0.15  (very stable!)
  Vote Frequency: 1.0   (all models agree!)
  ✅ EXCELLENT BIOMARKER
```

### Model Disagreement
Features with **high importance but high CV** may indicate:
- Model-specific effects
- Feature interactions
- Different modeling assumptions

Example:
```
Feature: original_shape_Elongation
  Consensus Importance: 0.032
  Cross-Model CV: 0.85  (variable!)
  Vote Frequency: 0.5   (only half agree)
  ⚠️ USE WITH CAUTION - model-specific
```

### Ensemble Strategies

**When to use each:**

1. **Equal-Weighted** (`ensemble_equal`)
   - When all models perform similarly
   - Most democratic approach
   - Good starting point

2. **Stability-Weighted** (`ensemble_weighted`) ⭐ **RECOMMENDED**
   - Prioritizes reliable models (low CV across folds)
   - Reduces influence of overfitted models
   - Best for clinical use

3. **Voting-Based** (`vote_frequency`)
   - Most conservative
   - Only trusts features most models agree on
   - Good for high-confidence features

---

## Command Line Options

### Required
- `--cv_dir` - Directory with outercv_fold_* subdirectories
- `--data` - CSV file with radiomics features
- `--output` - Output directory

### Optional
- `--model_types` - Models to analyze (default: randomforest gradientboosting xgboost lightgbm)
- `--class_names` - Class names for plots (e.g., CN AD PD)
- `--test_size` - Train/test split ratio (default: 0.2)
- `--random_state` - Random seed (default: 42)

### Examples

**Analyze specific models:**
```bash
python run_shap_comprehensive.py \
    --cv_dir ~/data/results/ \
    --data ~/data/features.csv \
    --output ~/data/shap_rf_xgb \
    --model_types randomforest xgboost
```

**Multi-class (3 classes):**
```bash
python run_shap_comprehensive.py \
    --cv_dir ~/data/results/ \
    --data ~/data/features.csv \
    --output ~/data/shap_multiclass \
    --class_names CN AD PD
```

**All models:**
```bash
python run_shap_comprehensive.py \
    --cv_dir ~/data/results/ \
    --data ~/data/features.csv \
    --output ~/data/shap_all \
    --model_types randomforest extratrees gradientboosting xgboost lightgbm svm logisticregression knn
```

---

## Workflow Integration

### Complete Analysis Pipeline

```bash
# 1. Run comprehensive analysis
python run_shap_comprehensive.py \
    --cv_dir ~/data/results/enhanced_run_SPECT/run_20251010_171321 \
    --data ~/data/radiomics_spect.csv \
    --output ~/data/shap_comprehensive \
    --model_types randomforest gradientboosting xgboost lightgbm \
    --class_names CN PD

# 2. Review top consensus features
cat ~/data/shap_comprehensive/model_comparison_summary.json | jq '.top_10_consensus'

# 3. Check ensemble recommendations
cat ~/data/shap_comprehensive/ensemble_summary.json | jq '.top_10_features'

# 4. Examine visualizations
open ~/data/shap_comprehensive/*.png
```

### Integration with Single-Fold Analysis

For detailed analysis of a specific fold/model:
```bash
# First: Run comprehensive analysis (above)

# Then: Deep dive into best fold for best model
python run_shap_analysis.py \
    --model_dir ~/data/results/.../outercv_fold_5 \
    --data ~/data/radiomics_spect.csv \
    --output ~/data/shap_fold5_detailed \
    --class_names CN PD \
    --all
```

---

## Understanding the Ensemble Process

The ensemble analysis simulates what would happen if you created a **voting classifier** or **stacking ensemble** across all models and folds.

### How It Works

1. **Collect** SHAP values from all models in all folds
2. **Normalize** to account for different scales
3. **Weight** by model stability (optional)
4. **Aggregate** using averaging or voting

### Why This Matters

In practice, ensemble models often outperform individual models. This analysis tells you:
- Which features an ensemble would prioritize
- Which features are robust across different algorithms
- Which features to trust for clinical decisions

### Comparison to Traditional Feature Importance

**Traditional** (e.g., RandomForest `feature_importances_`):
- Single model perspective
- Can be biased by model assumptions
- No cross-fold validation

**This Ensemble Approach**:
- Multi-model consensus
- Cross-validated across folds
- SHAP provides consistent interpretation across model types
- More reliable for decision-making

---

## Clinical Interpretation

### For Biomarker Discovery

**High-Confidence Biomarkers:**
1. High consensus importance
2. Low cross-model CV
3. High vote frequency
4. Stable across folds (from individual model analysis)

**Example workflow:**
```python
# Load results
import pandas as pd
df = pd.read_csv('ensemble_feature_importance.csv')

# Filter for high-confidence features
confident = df[
    (df['ensemble_weighted'] > df['ensemble_weighted'].quantile(0.9)) &  # Top 10% importance
    (df['vote_frequency'] > 0.75)  # 75% of models agree
].sort_values('ensemble_weighted', ascending=False)

print("High-confidence biomarkers:")
print(confident[['feature', 'ensemble_weighted', 'vote_frequency']])
```

### For Model Selection

Use `model_comparison_summary.json` to see which models:
- Find the most consistent features
- Agree with other models
- Have stable performance across folds

---

## Performance Notes

### Computational Cost

- **Time**: ~5-30 minutes depending on:
  - Number of folds (typically 5)
  - Number of models (4-8 typical)
  - Dataset size
  - Model types (tree models faster than KNN/SVM)

- **Memory**: ~2-8 GB depending on:
  - Number of features
  - Number of samples
  - Number of models loaded simultaneously

### Optimization Tips

1. **Start small**: Test with 2-3 models first
2. **Parallel-friendly**: Could parallelize across folds (not implemented yet)
3. **Caching**: Saves time on repeated analyses
4. **Model selection**: Tree models (RF, XGB, LGB) are much faster than SVM/KNN

---

## Troubleshooting

### "No results for [model]"
- Model not present in all folds
- Feature mismatch (should auto-resolve)
- Check individual fold logs

### "Could not resolve feature mismatch"
- `enhanced_results_summary.json` missing
- Different feature sets across folds (unusual)
- Check training logs

### Inconsistent results across models
- **Normal!** Different models learn different patterns
- High cross-model CV is expected for some features
- Focus on consensus features for clinical use

### Memory errors
- Reduce number of models analyzed simultaneously
- Reduce test set size in `load_data`
- Analyze models separately and combine manually

---

## Comparison with Other Scripts

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `run_shap_analysis.py` | Single fold, all models | Quick check, debugging |
| `run_shap_multifold.py` | One model, all folds | Model-specific stability |
| `run_shap_comprehensive.py` | All models, all folds | **Full analysis, ensemble** ⭐ |

**Recommendation**: Start with `run_shap_comprehensive.py` for overview, then drill down with others if needed.

---

## Citation & References

If you use this analysis in publications, cite:
- SHAP: Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
- Ensemble methods: Your ensemble learning citations
- This pipeline: Your lab/project

---

## Example Output

```
===================================================================================
Starting Comprehensive Multi-Fold, Multi-Model SHAP Analysis
Folds: 5 | Models: 4
===================================================================================

Processing Fold 1...
✓ Fold 1, randomforest: SHAP computed successfully
✓ Fold 1, gradientboosting: SHAP computed successfully
✓ Fold 1, xgboost: SHAP computed successfully
✓ Fold 1, lightgbm: SHAP computed successfully
...

===================================================================================
Aggregating Results Across Folds
===================================================================================

Aggregating randomforest: 5 folds
Aggregating gradientboosting: 5 folds
Aggregating xgboost: 5 folds
Aggregating lightgbm: 5 folds

===================================================================================
Comparing Feature Importance Across Models
===================================================================================

Top 10 Consensus Features (averaged across all models):
  original_glcm_JointEnergy                          | Importance: 0.0452 | CV: 0.12
  original_shape_MajorAxisLength                     | Importance: 0.0389 | CV: 0.18
  original_firstorder_Skewness                       | Importance: 0.0356 | CV: 0.15
  ...

===================================================================================
Computing Ensemble Feature Importance
===================================================================================

Model weights (based on stability):
  randomforest        : 1.2456
  xgboost             : 1.1834
  lightgbm            : 1.0923
  gradientboosting    : 0.9876

Top 10 Ensemble Features (stability-weighted):
  original_glcm_JointEnergy                          | Importance: 0.0468 | Votes: 4/4
  original_shape_MajorAxisLength                     | Importance: 0.0402 | Votes: 4/4
  original_firstorder_Skewness                       | Importance: 0.0365 | Votes: 4/4
  ...

===================================================================================
Comprehensive SHAP Analysis Complete!
===================================================================================

Results saved to: /path/to/shap_comprehensive/
```

---

## Future Enhancements

Potential additions:
- Multi-class support (separate analysis per class)
- Interaction analysis (SHAP interaction values)
- Feature clustering based on SHAP patterns
- Automated report generation (HTML/PDF)
- Integration with model performance metrics
- Parallel processing across folds

---

For questions or issues, see the main `README.md` or create an issue.

