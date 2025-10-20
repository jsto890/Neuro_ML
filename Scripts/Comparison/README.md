# Comparison Directory

This directory contains tools for comparing different machine learning models and approaches using statistical methods, including Bayesian analysis, performance comparison, and model evaluation.

## 📁 Directory Structure

```
Comparison/
├── README.md                     # This file
├── bayesian_model_comparison.py  # Bayesian model comparison
├── compare_models.py             # Model comparison utilities
└── run_bayesian_analysis.py      # Bayesian analysis pipeline
```

## 🔬 Bayesian Model Comparison

### Bayesian Analysis (`bayesian_model_comparison.py`)

#### Purpose
Comprehensive Bayesian analysis for model comparison including hierarchical accuracy estimation, trial-level skill models, and Bayesian calibration analysis.

#### Key Features
- **Hierarchical accuracy estimation**: Beta-binomial models for accuracy estimation
- **Trial-level skill models**: Bambi for model comparison
- **Bayesian calibration analysis**: Multiclass prediction calibration
- **Bayesian AUC estimation**: AUC estimation with uncertainty
- **Stacking ensemble methods**: Bayesian model averaging
- **Comprehensive reporting**: Detailed statistical reports

#### Usage
```bash
python bayesian_model_comparison.py \
    --input ~/path/to/model_results/ \
    --output ~/path/to/bayesian_analysis/ \
    --config ~/path/to/config.yaml
```

#### Supported Analyses
1. **Hierarchical Accuracy Models**
   - Beta-binomial models for accuracy estimation
   - Hierarchical modeling across folds
   - Uncertainty quantification

2. **Trial-Level Skill Models**
   - Bambi-based model comparison
   - Subject-level performance analysis
   - Site-specific effects

3. **Bayesian Calibration Analysis**
   - Multiclass calibration assessment
   - Temperature scaling analysis
   - Calibration uncertainty

4. **Bayesian AUC Estimation**
   - AUC with credible intervals
   - ROC curve uncertainty
   - Performance comparison

5. **Stacking Ensemble Methods**
   - Bayesian model averaging
   - Ensemble uncertainty
   - Model weight estimation

#### Configuration
```yaml
bayesian:
  accuracy_models:
    hierarchical: true
    beta_binomial: true
    cross_validation: true
  
  skill_models:
    trial_level: true
    subject_level: true
    site_effects: true
  
  calibration:
    multiclass: true
    temperature_scaling: true
    uncertainty: true
  
  auc_estimation:
    credible_intervals: true
    roc_uncertainty: true
    comparison: true
  
  ensemble:
    stacking: true
    bayesian_averaging: true
    uncertainty: true

sampling:
  chains: 4
  draws: 2000
  tune: 1000
  target_accept: 0.95
```

### Model Comparison (`compare_models.py`)

#### Purpose
Utilities for comparing different machine learning models and approaches.

#### Features
- **Performance comparison**: Compare model performance metrics
- **Statistical testing**: Significance testing between models
- **Effect size analysis**: Cohen's d and other effect size measures
- **Visualization**: Model comparison plots
- **Report generation**: Comprehensive comparison reports

#### Usage
```bash
python compare_models.py \
    --models ~/path/to/model1_results/ ~/path/to/model2_results/ \
    --output ~/path/to/comparison_results/
```

#### Comparison Methods
1. **Performance Metrics**
   - Accuracy, precision, recall, F1-score
   - ROC-AUC, PR-AUC
   - Matthews Correlation Coefficient
   - Calibration metrics

2. **Statistical Testing**
   - McNemar's test for paired comparisons
   - Cochran's Q test for multiple comparisons
   - Multiple comparison correction (FDR)
   - Bootstrap confidence intervals

3. **Effect Size Analysis**
   - Cohen's d for effect size
   - Confidence intervals for effect sizes
   - Practical significance assessment

### Bayesian Analysis Pipeline (`run_bayesian_analysis.py`)

#### Purpose
Complete pipeline for running Bayesian analysis on model comparison results.

#### Usage
```bash
python run_bayesian_analysis.py \
    --input ~/path/to/model_results/ \
    --output ~/path/to/bayesian_results/ \
    --config ~/path/to/config.yaml
```

#### Pipeline Steps
1. **Data Loading**: Load model results and metadata
2. **Data Validation**: Validate input data format
3. **Bayesian Analysis**: Run Bayesian models
4. **Statistical Testing**: Perform statistical tests
5. **Visualization**: Generate comparison plots
6. **Report Generation**: Create comprehensive reports

## 📊 Statistical Methods

### Bayesian Statistics

#### Hierarchical Models
- **Beta-binomial models**: For accuracy estimation
- **Hierarchical structure**: Across folds and models
- **Uncertainty quantification**: Credible intervals
- **Model comparison**: Bayes factors and posterior probabilities

#### Trial-Level Analysis
- **Subject-level effects**: Individual subject performance
- **Site effects**: Multi-site study analysis
- **Random effects**: Account for clustering
- **Mixed-effects models**: Combine fixed and random effects

#### Calibration Analysis
- **Temperature scaling**: Calibrate model outputs
- **Multiclass calibration**: For multi-class problems
- **Uncertainty quantification**: Calibration uncertainty
- **Reliability diagrams**: Visualize calibration

### Frequentist Statistics

#### Hypothesis Testing
- **McNemar's test**: Paired binary comparisons
- **Cochran's Q test**: Multiple related samples
- **Chi-square tests**: Independence testing
- **Fisher's exact test**: Small sample sizes

#### Multiple Comparisons
- **False Discovery Rate (FDR)**: Benjamini-Hochberg procedure
- **Family-wise Error Rate (FWER)**: Bonferroni correction
- **Step-down procedures**: Holm and Hochberg methods
- **Bootstrap methods**: Non-parametric testing

#### Effect Size Analysis
- **Cohen's d**: Standardized mean difference
- **Hedges' g**: Bias-corrected effect size
- **Glass's delta**: Alternative effect size measure
- **Confidence intervals**: Effect size uncertainty

## 📈 Visualization

### Performance Comparison Plots
- **Box plots**: Performance distribution comparison
- **Violin plots**: Distribution shape comparison
- **Scatter plots**: Performance correlation analysis
- **Heatmaps**: Performance matrix visualisation

### Statistical Analysis Plots
- **Forest plots**: Effect size comparison
- **Funnel plots**: Publication bias assessment
- **Q-Q plots**: Normality assessment
- **Residual plots**: Model diagnostics

### Bayesian Analysis Plots
- **Trace plots**: MCMC convergence assessment
- **Posterior plots**: Posterior distribution visualisation
- **Credible intervals**: Uncertainty visualisation
- **Bayes factor plots**: Model comparison visualisation

## 🔧 Configuration

### Bayesian Configuration
```yaml
bayesian:
  models:
    accuracy:
      hierarchical: true
      beta_binomial: true
      cross_validation: true
    
    skill:
      trial_level: true
      subject_level: true
      site_effects: true
    
    calibration:
      multiclass: true
      temperature_scaling: true
      uncertainty: true

  sampling:
    chains: 4
    draws: 2000
    tune: 1000
    target_accept: 0.95
    
  diagnostics:
    convergence: true
    effective_sample_size: true
    rhat_threshold: 1.01
```

### Comparison Configuration
```yaml
comparison:
  metrics:
    - accuracy
    - precision
    - recall
    - f1_score
    - roc_auc
    - pr_auc
    - mcc
  
  statistical_tests:
    mcnemar: true
    cochrans_q: true
    multiple_comparison: true
    fdr_correction: true
  
  effect_size:
    cohens_d: true
    confidence_intervals: true
    practical_significance: true
  
  visualisation:
    box_plots: true
    violin_plots: true
    scatter_plots: true
    heatmaps: true
```

## 📊 Output Structure

### Bayesian Analysis Output
```
bayesian_results/
├── models/                       # Bayesian models
│   ├── accuracy_model.pkl       # Accuracy model
│   ├── skill_model.pkl          # Skill model
│   └── calibration_model.pkl    # Calibration model
├── results/                      # Analysis results
│   ├── accuracy_results.json    # Accuracy analysis
│   ├── skill_results.json       # Skill analysis
│   ├── calibration_results.json # Calibration analysis
│   └── auc_results.json         # AUC analysis
├── plots/                        # Visualization plots
│   ├── accuracy_comparison.png  # Accuracy comparison
│   ├── skill_analysis.png       # Skill analysis
│   ├── calibration_plots.png    # Calibration plots
│   └── auc_comparison.png       # AUC comparison
└── reports/                      # Analysis reports
    ├── bayesian_report.html     # HTML report
    ├── bayesian_report.pdf      # PDF report
    └── summary.json             # Summary results
```

### Model Comparison Output
```
comparison_results/
├── statistical_tests/            # Statistical test results
│   ├── mcnemar_results.json     # McNemar test results
│   ├── cochrans_q_results.json  # Cochran's Q test results
│   └── effect_size_results.json # Effect size analysis
├── plots/                        # Comparison plots
│   ├── performance_comparison.png # Performance comparison
│   ├── statistical_tests.png     # Statistical test plots
│   └── effect_size_plots.png     # Effect size plots
└── reports/                      # Comparison reports
    ├── comparison_report.html    # HTML report
    ├── comparison_report.pdf     # PDF report
    └── summary.json              # Summary results
```

## 🚨 Common Issues

### Bayesian Analysis Issues
1. **Convergence problems**: Check MCMC convergence
2. **Sampling issues**: Adjust sampling parameters
3. **Model specification**: Verify model specification
4. **Data format**: Check input data format

### Statistical Testing Issues
1. **Multiple comparisons**: Use appropriate correction
2. **Effect size interpretation**: Consider practical significance
3. **Sample size**: Ensure adequate sample size
4. **Assumptions**: Check statistical assumptions

### Visualization Issues
1. **Plot clarity**: Ensure plots are clear and informative
2. **Color schemes**: Use accessible colour schemes
3. **Figure size**: Optimize figure sizes
4. **Labels**: Include clear labels and legends

## 🔍 Debugging

### Bayesian Analysis Debugging
- **Check convergence**: Use trace plots and R-hat
- **Validate models**: Check model specification
- **Review diagnostics**: Check effective sample size
- **Test with simple data**: Use simple test cases

### Statistical Testing Debugging
- **Check assumptions**: Verify statistical assumptions
- **Review test results**: Check test statistics and p-values
- **Validate effect sizes**: Check effect size calculations
- **Test with known data**: Use datasets with known results

## 📚 Dependencies

### Bayesian Analysis
- **PyMC**: Probabilistic programming
- **ArviZ**: Bayesian analysis and visualisation
- **Bambi**: Bayesian model building
- **xarray**: Multi-dimensional arrays

### Statistical Testing
- **scipy**: Scientific computing
- **statsmodels**: Statistical models
- **scikit-learn**: Machine learning utilities
- **numpy**: Numerical operations

### Visualization
- **matplotlib**: Plotting
- **seaborn**: Statistical plotting
- **plotly**: Interactive plotting
- **bokeh**: Interactive visualisation

## 🚀 Performance Optimization

### Bayesian Analysis Optimization
- **Parallel sampling**: Use multiple chains
- **GPU acceleration**: Use GPU for sampling
- **Memory optimisation**: Optimize memory usage
- **Caching**: Cache intermediate results

### Statistical Testing Optimization
- **Vectorized operations**: Use vectorized computations
- **Parallel processing**: Use multiple cores
- **Memory mapping**: Use memory-mapped files
- **Batch processing**: Process data in batches

## 📞 Support

For comparison analysis issues:
- Check input data format and quality
- Validate statistical assumptions
- Review Bayesian model specification
- Check convergence diagnostics
- Test with sample data first
- Review statistical test results
