# FDR Feature Selection Comparison Pipeline

This pipeline implements and compares three feature selection approaches for radiomics classification:

1. **FDR Selection**: False Discovery Rate correction using Benjamini-Hochberg method
2. **Current Selection**: Mutual Information + RFECV (current best approach)
3. **No Selection**: All features after basic preprocessing

## 🎯 Purpose

Compare the effectiveness of different feature selection methods, particularly focusing on:
- **MCC (Matthews Correlation Coefficient)** as the primary metric
- **FDR correction** for statistical rigor in feature selection
- **Performance comparison** across all approaches
- **Feature count reduction** vs. performance trade-offs

## 📋 Features Implemented

### ✅ **MCC (Matthews Correlation Coefficient)**
- Implemented in all evaluation metrics
- Primary comparison metric for imbalanced datasets
- More robust than accuracy for binary classification

### ✅ **FDR (False Discovery Rate) Feature Selection**
- Benjamini-Hochberg correction for multiple testing
- Statistical rigor in feature selection
- Configurable significance level (default: α = 0.05)

### ✅ **Training-Only Feature Selection**
- All feature selection methods fitted on training data only
- Proper application to validation and test sets
- Prevents data leakage

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_fdr.txt
```

### 2. Run FDR Comparison
```bash
python run_fdr_comparison.py --input radiomics_features.csv --output results/
```

### 3. View Results
```bash
cat results/comparison_report.txt
```

## 📊 What Gets Compared

| Approach | Method | Features | Statistical Rigor |
|----------|--------|----------|-------------------|
| **FDR Selection** | Benjamini-Hochberg FDR correction | Statistically significant features | ✅ High |
| **Current Selection** | MutualInfo + RFECV | Top-ranked features | ⚠️ Medium |
| **No Selection** | All features | All features after preprocessing | ❌ None |

## 📈 Output Files

### Core Results
- `comparison_report.txt` - Detailed comparison report
- `comparison_results.json` - Comparison results (JSON format)
- `detailed_results.json` - Detailed results for each approach
- `feature_engineering_results.json` - Feature engineering details

### Models
- `fdr_selection_svm_model.pkl` - SVM model with FDR selection
- `fdr_selection_ensemble_model.pkl` - Ensemble model with FDR selection
- `current_selection_svm_model.pkl` - SVM model with current selection
- `current_selection_ensemble_model.pkl` - Ensemble model with current selection
- `no_selection_svm_model.pkl` - SVM model with no selection
- `no_selection_ensemble_model.pkl` - Ensemble model with no selection

### Preprocessing
- `scaler.pkl` - Feature scaler
- `enhanced_fdr_pipeline.log` - Execution log

## 🔧 Configuration Options

### FDR Alpha Level
```bash
# More conservative (fewer features)
python run_fdr_comparison.py --input data.csv --output results/ --fdr-alpha 0.01

# Less conservative (more features)
python run_fdr_comparison.py --input data.csv --output results/ --fdr-alpha 0.1
```

### Random Seed
```bash
python run_fdr_comparison.py --input data.csv --output results/ --random-state 123
```

### Binary Classification Only
```bash
python run_fdr_comparison.py --input data.csv --output results/ --binary-only
```

### Verbose Logging
```bash
python run_fdr_comparison.py --input data.csv --output results/ --verbose
```

## 📊 Understanding the Results

### Key Metrics
1. **MCC (Matthews Correlation Coefficient)**: Primary metric for comparison
   - Range: -1 to +1
   - Higher is better
   - Robust to class imbalance

2. **Accuracy**: Overall correct predictions
   - Can be misleading with imbalanced data

3. **AUC**: Area Under ROC Curve
   - Range: 0 to 1
   - Higher is better

### Feature Count Analysis
- **FDR Selection**: Typically selects fewer, statistically significant features
- **Current Selection**: Balances feature count and performance
- **No Selection**: Uses all features (may lead to overfitting)

## 🎯 Recommendations

### For Clinical Use
- **Primary**: Use the approach with highest MCC
- **Secondary**: Consider feature count for interpretability
- **Tertiary**: Choose FDR selection for statistical rigor

### For Research
- **Compare all approaches** to understand feature importance
- **Use FDR selection** for publication-quality statistical analysis
- **Report MCC** as the primary performance metric

### For Model Deployment
- **Test on independent data** before final selection
- **Monitor performance** over time
- **Consider computational cost** of feature selection

## 🔬 Technical Details

### FDR Correction Method
```python
# Benjamini-Hochberg FDR correction
rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
    p_values, 
    alpha=fdr_alpha, 
    method='fdr_bh'
)
```

### Feature Selection Pipeline
1. **Basic Preprocessing**: Variance thresholding + scaling
2. **FDR Selection**: Statistical correction + feature filtering
3. **Model Training**: SVM + Ensemble for each approach
4. **Evaluation**: Comprehensive metrics on all splits
5. **Comparison**: Detailed analysis and reporting

### Data Leakage Prevention
- All feature selection fitted on training data only
- Transformers applied to validation/test sets
- Proper train/validation/test splits maintained

## 🐛 Troubleshooting

### statsmodels Not Available
```bash
pip install statsmodels>=0.13.0
```

### Memory Issues
- Reduce feature count in preprocessing
- Use smaller FDR alpha (more conservative selection)

### Convergence Issues
- Check data quality and preprocessing
- Verify binary classification labels

## 📚 References

1. **FDR Correction**: Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing.
2. **MCC Metric**: Matthews, B. W. (1975). Comparison of the predicted and observed secondary structure of T4 phage lysozyme.
3. **Feature Selection**: Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection.

## 🤝 Contributing

To add new feature selection methods or evaluation metrics:

1. Modify `enhanced_fdr_classifier.py`
2. Add new approach to the comparison pipeline
3. Update this README with new information
4. Test with your data

## 📞 Support

For issues or questions:
1. Check the log file: `enhanced_fdr_pipeline.log`
2. Review the comparison report: `comparison_report.txt`
3. Verify input data format and requirements 