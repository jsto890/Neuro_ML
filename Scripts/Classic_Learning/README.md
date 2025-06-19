# Radiomics Classical Learning Pipeline

A comprehensive machine learning pipeline for training and evaluating Random Forest models on radiomics features extracted from neuroimaging data.

## 📁 Files Overview

- `radiomics_classifier.py` - Main pipeline implementation
- `preprocessing.py` - Data preprocessing utilities
- `run_classical.py` - Simple runner script
- `config_classical.yaml` - Configuration file
- `README.md` - This documentation

## 🚀 Quick Start

### Prerequisites

Install required packages:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn pyyaml
```

### Step 1: Extract Radiomics Features

First, ensure you have radiomics features extracted:
```bash
cd Scripts/Feature_Extraction/pyRadioMics/
python3 simple_radiomics.py --labels ~/reseng202500013-ndd-ml/data/mri_labels.csv --output-dir ~/reseng202500013-ndd-ml/data/ --config ~/reseng202500013-ndd-ml/P4P/config.yaml
```

### Step 2: Run Classical Learning Pipeline

```bash
cd Scripts/Classic_Learning/
python3 run_classical.py
```

This will:
- Load the radiomics features
- Preprocess the data
- Train a Random Forest model
- Evaluate performance
- Generate visualizations
- Save all results

## 📊 Pipeline Stages

### Stage 0: Input Validation
- Load radiomics CSV file
- Validate required columns (`subject_id`, `label`)
- Check data types and structure

### Stage 1: Data Preprocessing
- **Missing Value Handling**: Remove rows with missing values
- **Feature Cleaning**: Remove constant/near-constant features
- **Feature Selection**: Select top 100 most informative features
- **Scaling**: Standardize features using z-score normalization

### Stage 2: Data Splitting
- **Stratified Split**: 60% train, 20% validation, 20% test
- **Subject Independence**: Ensure no subject leakage between splits

### Stage 3: Model Training
- **Random Forest Classifier** with hyperparameter tuning
- **Grid Search** over:
  - `n_estimators`: [100, 200, 500]
  - `max_depth`: [None, 10, 20, 30]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]
  - `max_features`: ['sqrt', 'log2', None]
- **5-fold Cross-validation** with ROC AUC scoring

### Stage 4: Model Evaluation
- **Performance Metrics**:
  - Accuracy, Precision, Recall, F1-score
  - ROC AUC, Precision-Recall curves
  - Confusion matrices
- **Cross-validation** scores
- **Feature importance** rankings

### Stage 5: Results & Visualization
- **Plots**: ROC curves, confusion matrices, feature importance
- **Artifacts**: Trained model, scaler, predictions
- **Reports**: Detailed performance summaries

## 📁 Output Files

After running the pipeline, you'll find these files in the output directory:

### Core Files
- `random_forest_model.pkl` - Trained Random Forest model
- `scaler.pkl` - Feature scaler for preprocessing new data
- `feature_importance.csv` - Feature importance rankings

### Visualizations
- `evaluation_plots.png` - Comprehensive performance plots
  - ROC curves for all splits
  - Confusion matrix (test set)
  - Top 10 feature importances
  - Performance metrics comparison

### Reports
- `results_summary.json` - Detailed results in JSON format
- `pipeline.log` - Complete execution log

## 🔧 Configuration

Edit `config_classical.yaml` to customize the pipeline:

```yaml
# Data preprocessing
preprocessing:
  remove_missing: true
  variance_threshold: 0.01
  feature_selection:
    enabled: true
    method: "k_best"  # k_best, mutual_info, pca
    n_features: 100
  scaling:
    method: "standard"  # standard, robust, minmax

# Model parameters
model:
  param_grid:
    n_estimators: [100, 200, 500]
    max_depth: [None, 10, 20, 30]
    # ... more parameters
```

## 📈 Expected Results

For a typical radiomics dataset with 474 subjects and 3 classes:

- **Processing Time**: 5-15 minutes (depending on system)
- **Feature Count**: ~1000+ radiomics features → 100 selected features
- **Model Performance**: 
  - Test Accuracy: 70-85%
  - ROC AUC: 0.75-0.90
  - Cross-validation consistency

## 🔍 Advanced Usage

### Custom Input/Output Paths
```bash
python3 run_classical.py --input path/to/radiomics.csv --output-dir results/
```

### Different Random Seed
```bash
python3 run_classical.py --random-state 123
```

### Using Custom Configuration
```bash
python3 run_classical.py --config my_config.yaml
```

### Programmatic Usage
```python
from radiomics_classifier import RadiomicsClassifier

classifier = RadiomicsClassifier(
    input_path='path/to/radiomics.csv',
    output_dir='results/',
    random_state=42
)
success = classifier.run_pipeline()
```

## 🧪 Model Interpretation

### Feature Importance
The pipeline automatically identifies the most important radiomics features:
- **Shape features**: Volume, surface area, sphericity
- **Texture features**: GLCM, GLRLM, GLSZM features
- **Intensity features**: First-order statistics

### Performance Analysis
- **ROC curves**: Show model discrimination ability
- **Confusion matrices**: Reveal class-specific performance
- **Cross-validation**: Assess model stability

## 🔄 Extending the Pipeline

### Adding New Models
1. Import your model in `radiomics_classifier.py`
2. Modify the `train_model()` method
3. Update the parameter grid

### Custom Preprocessing
1. Add preprocessing steps in `preprocessing.py`
2. Integrate into the `preprocess_pipeline()` method

### Additional Evaluation Metrics
1. Add metrics to the `evaluate_model()` method
2. Update visualization functions

## 🐛 Troubleshooting

### Common Issues

**"Input file not found"**
- Ensure radiomics extraction completed successfully
- Check file path in configuration

**"Memory error"**
- Reduce `n_features` in configuration
- Use smaller parameter grid

**"Poor performance"**
- Check class balance in labels
- Try different feature selection methods
- Adjust hyperparameter ranges

### Debug Mode
Enable verbose logging by modifying the logging level in `radiomics_classifier.py`.

## 📚 References

- [Scikit-learn Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
- [Radiomics Feature Extraction](https://pyradiomics.readthedocs.io/)
- [Machine Learning Best Practices](https://scikit-learn.org/stable/modules/cross_validation.html)

## 🤝 Contributing

To extend this pipeline:
1. Add new preprocessing methods to `preprocessing.py`
2. Implement additional models in `radiomics_classifier.py`
3. Update configuration options in `config_classical.yaml`
4. Document changes in this README 