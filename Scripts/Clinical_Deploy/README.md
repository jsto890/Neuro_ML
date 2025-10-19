# Clinical Deployment Directory

This directory contains tools for deploying trained models in clinical settings, including prediction interfaces, model validation, and interpretability analysis. The clinical deployment tools support both classical machine learning and deep learning models.

## 📁 Directory Structure

```
Clinical_Deploy/
├── README.md                     # This file
├── Classic/                      # Classical ML deployment
│   ├── predict_clinical.py       # Clinical prediction interface
│   ├── validate_model.py         # Model validation
│   ├── run_shap_comprehensive.py # Comprehensive SHAP analysis
│   └── shap_interpretability.py  # SHAP interpretability module
└── Deep/                         # Deep learning deployment
    ├── predict_clinical_deep.py  # Deep learning prediction
    └── validate_model_deep.py    # Deep model validation
```

## 🔬 Classical ML Deployment (`Classic/`)

### Clinical Prediction (`predict_clinical.py`)

#### Purpose
Deploy classical machine learning models for clinical prediction using radiomics features.

#### Usage
```bash
cd Classic

# Single prediction
python predict_clinical.py \
    --model ~/path/to/model.pkl \
    --input ~/path/to/features.csv \
    --output ~/path/to/prediction.json

# Batch prediction
python predict_clinical.py \
    --model ~/path/to/model.pkl \
    --input ~/path/to/batch_features.csv \
    --output ~/path/to/batch_predictions.json \
    --batch
```

#### Features
- **Single prediction**: Predict individual samples
- **Batch prediction**: Process multiple samples efficiently
- **Confidence intervals**: Uncertainty quantification
- **Feature importance**: Explain individual predictions
- **JSON output**: Structured prediction results
- **Error handling**: Robust error handling and validation

#### Input Format
```csv
subject_id,feature_1,feature_2,feature_3,...
sub-001,0.123,0.456,0.789,...
sub-002,0.234,0.567,0.890,...
```

#### Output Format
```json
{
  "predictions": [
    {
      "subject_id": "sub-001",
      "predicted_class": "CN",
      "predicted_label": 0,
      "probabilities": {
        "CN": 0.85,
        "AD": 0.10,
        "PD": 0.05
      },
      "confidence": 0.85,
      "feature_importance": {
        "feature_1": 0.123,
        "feature_2": 0.456
      }
    }
  ],
  "model_info": {
    "model_type": "SVM",
    "training_date": "2024-01-01",
    "performance": {
      "accuracy": 0.85,
      "auc": 0.92
    }
  }
}
```

### Model Validation (`validate_model.py`)

#### Purpose
Validate trained models on new datasets and assess performance.

#### Usage
```bash
python validate_model.py \
    --model ~/path/to/model.pkl \
    --data ~/path/to/validation_data.csv \
    --output ~/path/to/validation_results/
```

#### Features
- **Performance assessment**: Comprehensive model evaluation
- **Statistical testing**: Significance testing
- **Bias detection**: Detect model bias
- **Calibration analysis**: Model calibration assessment
- **Confidence intervals**: Uncertainty quantification

### SHAP Interpretability (`run_shap_analysis.py`)

#### Purpose
Generate SHAP (SHapley Additive exPlanations) values for model interpretability.

#### Usage
```bash
python run_shap_analysis.py \
    --model ~/path/to/model.pkl \
    --data ~/path/to/features.csv \
    --output ~/path/to/shap_results/
```

#### Features
- **Feature importance**: Global and local feature importance
- **Summary plots**: SHAP summary plots
- **Dependence plots**: Feature interaction analysis
- **Force plots**: Individual prediction explanations
- **Waterfall plots**: Step-by-step prediction breakdown

#### SHAP Output
```
shap_results/
├── shap_summary.png              # Feature importance summary
├── shap_dependence_*.png         # Feature dependence plots
├── shap_force_*.png              # Force plots
├── shap_waterfall_*.png          # Waterfall plots
├── shap_values.csv               # SHAP values
└── shap_summary_stats.json       # Summary statistics
```

### Comprehensive SHAP Analysis (`run_shap_comprehensive.py`)

#### Purpose
Comprehensive SHAP analysis across multiple models and folds.

#### Usage
```bash
python run_shap_comprehensive.py \
    --results_dir ~/path/to/cv_results/ \
    --output ~/path/to/comprehensive_shap/
```

#### Features
- **Multi-model analysis**: Analyze multiple models
- **Cross-validation**: Analyze across CV folds
- **Feature stability**: Assess feature importance stability
- **Model comparison**: Compare model interpretability

### Multi-fold SHAP Analysis (`run_shap_multifold.py`)

#### Purpose
Analyze SHAP values across multiple cross-validation folds.

#### Usage
```bash
python run_shap_multifold.py \
    --data ~/path/to/features.csv \
    --cv_dir ~/path/to/cv_models/ \
    --output ~/path/to/multifold_shap/
```

#### Features
- **Cross-validation analysis**: Analyze across CV folds
- **Feature stability**: Assess feature importance consistency
- **Statistical analysis**: Statistical significance testing
- **Visualization**: Comprehensive visualisation plots

### SHAP Interpretability Module (`shap_interpretability.py`)

#### Purpose
Core SHAP interpretability functionality for classical ML models.

#### Supported Models
- **Tree-based**: RandomForest, XGBoost, LightGBM, GradientBoosting
- **Linear**: LogisticRegression, LinearSVM
- **Other**: SVM (with KernelExplainer), KNN (with KernelExplainer)

#### Usage
```python
from shap_interpretability import SHAPInterpreter

# Create interpreter
interpreter = SHAPInterpreter(
    model=model,
    X_train=X_train,
    feature_names=feature_names,
    output_dir='~/path/to/output'
)

# Generate SHAP values
shap_values = interpreter.compute_shap_values(X_test)

# Create plots
interpreter.create_summary_plot(X_test)
interpreter.create_dependence_plot(X_test, feature_name='feature_1')
interpreter.create_force_plot(X_test, sample_idx=0)
```

## 🧠 Deep Learning Deployment (`Deep/`)

### Deep Learning Prediction (`predict_clinical_deep.py`)

#### Purpose
Deploy deep learning models for clinical prediction using raw medical images.

#### Usage
```bash
cd Deep

# Single image prediction
python predict_clinical_deep.py \
    --model ~/path/to/model.pth \
    --input ~/path/to/image.nii.gz \
    --output ~/path/to/prediction.json

# Batch prediction
python predict_clinical_deep.py \
    --model ~/path/to/model.pth \
    --input ~/path/to/images/ \
    --output ~/path/to/predictions.json \
    --batch
```

#### Features
- **3D image processing**: Process 3D medical images
- **Grad-CAM generation**: Generate activation maps
- **Saliency maps**: Generate saliency maps
- **Occlusion sensitivity**: Occlusion sensitivity analysis
- **Multiple outputs**: Prediction, confidence, and interpretability maps

#### Output Format
```json
{
  "predictions": [
    {
      "subject_id": "sub-001",
      "predicted_class": "CN",
      "predicted_label": 0,
      "probabilities": {
        "CN": 0.85,
        "AD": 0.10,
        "PD": 0.05
      },
      "confidence": 0.85,
      "interpretability": {
        "gradcam_path": "~/path/to/gradcam.nii.gz",
        "saliency_path": "~/path/to/saliency.nii.gz",
        "occlusion_path": "~/path/to/occlusion.nii.gz"
      }
    }
  ],
  "model_info": {
    "architecture": "Simple3DCNN",
    "input_shape": [91, 109, 91],
    "training_date": "2024-01-01"
  }
}
```

### Deep Model Validation (`validate_model_deep.py`)

#### Purpose
Validate deep learning models on new datasets.

#### Usage
```bash
python validate_model_deep.py \
    --model ~/path/to/model.pth \
    --data ~/path/to/validation_data/ \
    --output ~/path/to/validation_results/
```

#### Features
- **3D image validation**: Validate on 3D medical images
- **Performance assessment**: Comprehensive evaluation
- **Interpretability analysis**: Generate interpretability maps
- **Statistical analysis**: Statistical significance testing

## 🔍 Interpretability Analysis

### SHAP Analysis (Classical ML)

#### Global Interpretability
- **Feature importance**: Overall feature importance
- **Feature interactions**: Feature interaction analysis
- **Model behaviour**: Understanding model decision-making

#### Local Interpretability
- **Individual predictions**: Explain individual predictions
- **Feature contributions**: Feature contribution analysis
- **Decision boundaries**: Understand decision boundaries

### Grad-CAM Analysis (Deep Learning)

#### Activation Maps
- **Class activation**: Class-specific activation maps
- **Spatial attention**: Spatial attention visualisation
- **Feature importance**: Visual feature importance

#### Usage
```bash
# Generate Grad-CAM
python visualise_gradcam.py \
    --model ~/path/to/model.pth \
    --input ~/path/to/image.nii.gz \
    --output ~/path/to/gradcam.nii.gz
```

## 📊 Clinical Reports

### Report Generation
- **HTML reports**: Interactive HTML reports
- **PDF reports**: Printable PDF reports
- **JSON reports**: Machine-readable JSON reports

### Report Content
- **Prediction results**: Model predictions and confidence
- **Interpretability**: SHAP values and Grad-CAM maps
- **Model information**: Model performance and metadata
- **Clinical context**: Clinical interpretation and recommendations

## 🔧 Configuration

### Model Configuration
```yaml
model:
  type: "classical"  # or "deep"
  path: "~/path/to/model.pkl"
  scaler_path: "~/path/to/scaler.pkl"
  feature_names: "~/path/to/features.json"

prediction:
  confidence_threshold: 0.8
  uncertainty_quantification: true
  interpretability: true

output:
  format: "json"  # json, csv, html
  include_probabilities: true
  include_confidence: true
  include_interpretability: true
```

### Clinical Settings
```yaml
clinical:
  confidence_threshold: 0.8
  uncertainty_threshold: 0.2
  interpretability_required: true
  report_format: "html"
  include_recommendations: true
```

## 🚨 Quality Control

### Input Validation
- **Data format**: Validate input data format
- **Feature validation**: Check feature completeness
- **Image validation**: Validate image format and dimensions
- **Label validation**: Check label format and values

### Output Validation
- **Prediction validation**: Validate prediction outputs
- **Confidence validation**: Check confidence intervals
- **Interpretability validation**: Validate interpretability maps
- **Report validation**: Check report completeness

### Error Handling
- **Graceful degradation**: Handle errors gracefully
- **Error logging**: Comprehensive error logging
- **User feedback**: Clear error messages
- **Recovery**: Automatic error recovery when possible

## 📈 Performance Monitoring

### Model Performance
- **Prediction accuracy**: Monitor prediction accuracy
- **Confidence calibration**: Monitor confidence calibration
- **Bias detection**: Monitor for model bias
- **Drift detection**: Detect model performance drift

### System Performance
- **Prediction latency**: Monitor prediction speed
- **Memory usage**: Monitor memory consumption
- **CPU/GPU usage**: Monitor computational resources
- **Error rates**: Monitor error rates

## 🔒 Security and Privacy

### Data Privacy
- **Data encryption**: Encrypt sensitive data
- **Access control**: Implement access controls
- **Audit logging**: Log all access and operations
- **Data anonymization**: Anonymize patient data

### Model Security
- **Model encryption**: Encrypt model files
- **Access control**: Control model access
- **Version control**: Track model versions
- **Integrity checking**: Verify model integrity

## 📞 Support

For clinical deployment issues:
- Check model file paths and formats
- Validate input data format and quality
- Review configuration files
- Check log files for detailed error messages
- Test with sample data first
- Verify interpretability outputs
