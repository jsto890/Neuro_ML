#!/usr/bin/env python3
"""
Bayesian Model Comparison for P4P Deep Learning Models

This module implements comprehensive Bayesian analysis for model comparison including:
1. Hierarchical accuracy estimation with beta-binomial models
2. Trial-level skill models with Bambi for model comparison  
3. Bayesian calibration analysis for multiclass predictions
4. Bayesian AUC estimation and stacking ensemble methods
5. Comprehensive reporting and visualization

Author: P4P Team
Date: 2025
"""

import argparse
import json
import os
import sys
import glob
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Bayesian analysis libraries
import pymc as pm
import arviz as az
import bambi as bmb

# Traditional ML metrics
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, precision_recall_curve, average_precision_score,
    matthews_corrcoef
)
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
from statsmodels.stats.multitest import multipletests

# Suppress PyMC warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pymc')
warnings.filterwarnings('ignore', category=FutureWarning, module='pymc')

plt.style.use("default")
sns.set_palette("husl")


@dataclass
class ModelFoldData:
    """Container for per-fold model data"""
    model: str
    fold: int
    site: Optional[str]  # Can be derived from subject metadata
    predictions: np.ndarray
    probabilities: np.ndarray
    labels: np.ndarray
    subject_ids: List[str]
    n_classes: int


@dataclass
class BayesianResults:
    """Container for Bayesian analysis results"""
    accuracy_results: Dict[str, Any]
    skill_results: Dict[str, Any] 
    calibration_results: Dict[str, Any]
    auc_results: Dict[str, Any]
    stacking_results: Dict[str, Any]


class BayesianModelComparison:
    """
    Comprehensive Bayesian model comparison system for P4P models.
    
    Supports multiclass classification with proper uncertainty quantification,
    hierarchical modeling across sites/folds, and ensemble methods.
    """
    
    def __init__(self, output_dir: str, random_seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        
        # Set random seeds
        np.random.seed(random_seed)
        # pm.set_utilities_random_seed(random_seed)  # Not available in PyMC 5.x
        
        # Create subdirectories
        self.plots_dir = self.output_dir / "plots"
        self.results_dir = self.output_dir / "results" 
        self.data_dir = self.output_dir / "data"
        
        for d in [self.plots_dir, self.results_dir, self.data_dir]:
            d.mkdir(exist_ok=True)
        # Fixed class label mapping: 0→CN, 1→AD, 2→PD
        self.class_labels = ["CN", "AD", "PD"]
    
    def load_model_data(self, run_dirs: List[str], models: Optional[List[str]] = None) -> List[ModelFoldData]:
        """
        Load model data from run directories.
        
        Args:
            run_dirs: List of run directories containing model outputs
            models: Optional list of specific models to include
            
        Returns:
            List of ModelFoldData objects
        """
        print("Loading model data...")
        model_data = []
        
        self.model_source_dirs: Dict[str, str] = {}
        for run_dir in run_dirs:
            run_dir = Path(run_dir)
            if not run_dir.exists():
                print(f"Warning: Run directory {run_dir} does not exist")
                continue
                
            # Find model directories
            model_dirs = self._discover_model_dirs(run_dir)
            
            for model_name, model_dir in model_dirs.items():
                if models and model_name not in models:
                    continue
                    
                print(f"Processing model: {model_name}")
                # Track source directory for this model
                self.model_source_dirs[model_name] = str(model_dir)
                
                # Load fold data
                fold_data = self._load_model_folds(model_dir, model_name)
                model_data.extend(fold_data)
        
        print(f"Loaded data for {len(set(d.model for d in model_data))} models, {len(model_data)} folds total")
        return model_data

    # ---- helpers ----
    def _normal_pdf(self, x: np.ndarray, mu: float, sd: float) -> np.ndarray:
        sd = max(float(sd), 1e-6)
        return (1.0 / (sd * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu)/sd)**2)

    def _fit_beta_from_samples(self, arr: np.ndarray) -> Tuple[float, float]:
        """Fit Beta(alpha,beta) by method of moments from samples in (0,1)."""
        eps = 1e-6
        arr = np.asarray(arr)
        arr = np.clip(arr, eps, 1 - eps)
        m = float(np.mean(arr))
        v = float(np.var(arr))
        # Guard against degenerate variance
        if v <= 0:
            # Highly concentrated at m; pick large concentration
            k = 1000.0
            alpha = max(eps, m * k)
            beta = max(eps, (1 - m) * k)
            return alpha, beta
        # Ensure variance feasible for Beta: v < m(1-m)
        max_v = m * (1 - m) - eps
        if v >= max_v and max_v > eps:
            v = max_v
        k = (m * (1 - m)) / v - 1.0
        if k <= 0:
            # fallback to moderate concentration
            k = 10.0
        alpha = max(eps, m * k)
        beta = max(eps, (1 - m) * k)
        return alpha, beta

    def _beta_pdf(self, x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        eps = 1e-9
        x = np.clip(x, eps, 1 - eps)
        a = max(alpha, eps)
        b = max(beta, eps)
        # Compute in log-space for stability: pdf = x^(a-1) * (1-x)^(b-1) / B(a,b)
        from math import lgamma
        log_B = lgamma(a) + lgamma(b) - lgamma(a + b)
        log_pdf = (a - 1) * np.log(x) + (b - 1) * np.log(1 - x) - log_B
        return np.exp(log_pdf)
    
    def _discover_model_dirs(self, run_dir: Path) -> Dict[str, Path]:
        """Find model subdirectories containing run summaries."""
        model_dirs = {}
        
        for entry in sorted(run_dir.iterdir()):
            if not entry.is_dir():
                continue
                
            # Check for run summary
            summary_files = list(entry.glob("*_run_summary.json"))
            if not summary_files:
                # Check for test evaluation plots
                eval_dirs = list(entry.glob("test_evaluation_plots_fold_*"))
                if not eval_dirs:
                    continue
            
            model_dirs[entry.name] = entry
        
        return model_dirs
    
    def _load_model_folds(self, model_dir: Path, model_name: str) -> List[ModelFoldData]:
        """Load all fold data for a model."""
        fold_data = []
        
        # Find all test evaluation directories
        eval_dirs = sorted(model_dir.glob("test_evaluation_plots_fold_*"))
        
        for eval_dir in eval_dirs:
            # Extract fold number
            fold_match = eval_dir.name.split("_")[-1]  # Should be "fold_X"
            try:
                fold_num = int(fold_match.replace("fold_", ""))
            except ValueError:
                print(f"Warning: Could not parse fold number from {eval_dir.name}")
                continue
            
            # Load data files
            pred_file = eval_dir / "predictions.npy"
            prob_file = eval_dir / "probabilities.npy" 
            label_file = eval_dir / "labels.npy"
            
            if not all(f.exists() for f in [pred_file, prob_file, label_file]):
                print(f"Warning: Missing data files in {eval_dir}")
                continue
            
            try:
                predictions = np.load(pred_file)
                probabilities = np.load(prob_file)
                labels = np.load(label_file)
                
                # Generate subject IDs if not available
                subject_ids = [f"{model_name}_fold{fold_num}_sample{i}" 
                             for i in range(len(labels))]
                
                n_classes = probabilities.shape[1] if len(probabilities.shape) > 1 else len(np.unique(labels))
                
                fold_data.append(ModelFoldData(
                    model=model_name,
                    fold=fold_num,
                    site=f"fold_{fold_num}",  # Using fold as site for now
                    predictions=predictions,
                    probabilities=probabilities,
                    labels=labels,
                    subject_ids=subject_ids,
                    n_classes=n_classes
                ))
                
            except Exception as e:
                print(f"Error loading data from {eval_dir}: {e}")
                continue
        
        return fold_data
    
    def prepare_data_for_analysis(self, model_data: List[ModelFoldData]) -> Dict[str, pd.DataFrame]:
        """
        Prepare dataframes for different Bayesian analyses.
        
        Args:
            model_data: List of ModelFoldData objects
            
        Returns:
            Dictionary with dataframes for different analyses
        """
        print("Preparing data for Bayesian analysis...")
        print(f"Total model data entries: {len(model_data)}")
        
        # 1. Hierarchical accuracy data (model × site/fold level)
        accuracy_rows = []
        
        # 2. Trial-level skill data (individual predictions)
        skill_rows = []
        
        # 3. Calibration data (probabilities and outcomes)
        calib_rows = []
        
        # 4. AUC data (scores for each class)
        auc_rows = []
        
        for data in model_data:
            print(f"Processing {data.model}, fold {data.fold}, n_classes={data.n_classes}")
            print(f"  Predictions shape: {data.predictions.shape}, unique values: {np.unique(data.predictions)}")
            print(f"  Probabilities shape: {data.probabilities.shape}")
            print(f"  Labels shape: {data.labels.shape}, unique values: {np.unique(data.labels)}")
            
            # Hierarchical accuracy data
            correct = (data.predictions == data.labels).sum()
            total = len(data.labels)
            accuracy_rows.append({
                'model': data.model,
                'site': data.site,
                'k': correct,  # number correct
                'n': total,    # number evaluated
                'accuracy': correct / total
            })
            
            # Trial-level skill data
            for i in range(len(data.labels)):
                skill_rows.append({
                    'subject_id': data.subject_ids[i],
                    'model': data.model,
                    'site': data.site,
                    'correct': int(data.predictions[i] == data.labels[i]),
                    'true_label': int(data.labels[i]),
                    'predicted_label': int(data.predictions[i])
                })
            
            # Calibration data (multiclass - one-vs-rest for each class)
            print(f"  Processing calibration for {data.n_classes} classes")
            for class_idx in range(data.n_classes):
                class_mask = data.labels == class_idx
                print(f"    Class {class_idx}: {np.sum(class_mask)} samples")
                if not np.any(class_mask):
                    print(f"    Skipping class {class_idx} - no samples")
                    continue
                    
                # Use logits for calibration (log probabilities)
                if len(data.probabilities.shape) > 1:
                    class_probs = data.probabilities[class_mask, class_idx]
                    print(f"    Class {class_idx} probabilities shape: {class_probs.shape}")
                    # Convert to logits
                    class_logits = np.log(class_probs / (1 - class_probs + 1e-8))
                else:
                    class_probs = data.probabilities[class_mask]
                    class_logits = np.log(class_probs / (1 - class_probs + 1e-8))
                
                for i, (logit, prob, true_label) in enumerate(zip(class_logits, class_probs, data.labels[class_mask])):
                    calib_rows.append({
                        'model': data.model,
                        'site': data.site,
                        'subject_id': data.subject_ids[np.where(data.labels == class_idx)[0][i]],
                        'class': class_idx,
                        'y': 1,  # One-vs-rest: this class vs others
                        'logit': logit,
                        'probability': prob
                    })
            
            # AUC data (for each class)
            for class_idx in range(data.n_classes):
                if len(data.probabilities.shape) > 1:
                    class_probs = data.probabilities[:, class_idx]
                else:
                    class_probs = data.probabilities
                
                for i, (prob, true_label) in enumerate(zip(class_probs, data.labels)):
                    auc_rows.append({
                        'model': data.model,
                        'site': data.site,
                        'subject_id': data.subject_ids[i],
                        'class': class_idx,
                        'score': prob,
                        'y': int(true_label == class_idx)  # Binary for this class
                    })
        
        # Create dataframes and add debugging info
        accuracy_df = pd.DataFrame(accuracy_rows)
        skill_df = pd.DataFrame(skill_rows)
        calib_df = pd.DataFrame(calib_rows)
        auc_df = pd.DataFrame(auc_rows)
        
        print(f"\nData preparation summary:")
        print(f"  Accuracy data: {len(accuracy_df)} rows")
        print(f"  Skill data: {len(skill_df)} rows")
        print(f"  Calibration data: {len(calib_df)} rows")
        print(f"  AUC data: {len(auc_df)} rows")
        
        if not skill_df.empty:
            print(f"  Skill data columns: {skill_df.columns.tolist()}")
            print(f"  Skill data models: {skill_df['model'].unique()}")
            print(f"  Skill data sites: {skill_df['site'].unique()}")
            print(f"  Skill data correct values: {skill_df['correct'].unique()}")
        
        return {
            'accuracy': accuracy_df,
            'skill': skill_df,
            'calibration': calib_df,
            'auc': auc_df
        }
    
    def hierarchical_accuracy_analysis(self, df_accuracy: pd.DataFrame) -> Dict[str, Any]:
        """
        Hierarchical accuracy estimation with beta-binomial models.
        
        Estimates performance with proper uncertainty and partial pooling across sites/folds.
        """
        print("Running hierarchical accuracy analysis...")
        
        if df_accuracy.empty:
            return {}
        
        # Enforce consistent ordering across the pipeline
        models = np.array(sorted(df_accuracy["model"].unique().tolist()))
        sites = np.array(sorted(df_accuracy["site"].unique().tolist()))
        M, S = len(models), len(sites)
        
        # Create indices
        model_cat = pd.Categorical(df_accuracy["model"], categories=models, ordered=True)
        site_cat = pd.Categorical(df_accuracy["site"], categories=sites, ordered=True)
        model_idx = model_cat.codes
        site_idx = site_cat.codes
        k = df_accuracy["k"].values
        n = df_accuracy["n"].values
        
        with pm.Model() as accuracy_model:
            # Partial pooling: logit accuracy per model with site-specific random effects
            # Tighter priors help reduce divergences
            alpha_model = pm.Normal("alpha_model", 0, 1.0, shape=M)
            u_site_raw = pm.Normal("u_site_raw", 0, 1, shape=S)
            sigma_site = pm.HalfNormal("sigma_site", 0.5)
            u_site = pm.Deterministic("u_site", u_site_raw * sigma_site)
            
            logit_theta = alpha_model[model_idx] + u_site[site_idx]
            theta = pm.Deterministic("theta", pm.math.sigmoid(logit_theta))
            
            # Observed data
            k_obs = pm.Binomial("k_obs", n=n, p=theta, observed=k)
            
            # Posterior accuracy per model (marginalised over sites)
            acc_model = pm.Deterministic("acc_model", pm.math.sigmoid(alpha_model))
            
            # Sample with improved parameters to reduce divergences
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                idata = pm.sample(
                    2000,
                    tune=3000,
                    target_accept=0.995,
                    random_seed=self.random_seed,
                    return_inferencedata=True,
                    progressbar=True,
                    cores=4,
                )
                idata.extend(pm.sample_posterior_predictive(idata, random_seed=self.random_seed))
        
        # Extract results
        acc_samples = idata.posterior["acc_model"].values.reshape(-1, M)
        
        results = {
            'idata': idata,
            'models': models,
            'sites': sites,
            'accuracy_samples': acc_samples,
            'accuracy_means': np.mean(acc_samples, axis=0),
            'accuracy_std': np.std(acc_samples, axis=0),
            'accuracy_ci_lower': np.percentile(acc_samples, 5.5, axis=0),
            'accuracy_ci_upper': np.percentile(acc_samples, 94.5, axis=0)
        }
        
        # Model ranking probabilities
        model_pairs = []
        for i in range(M):
            for j in range(i+1, M):
                prob_better = np.mean(acc_samples[:, i] > acc_samples[:, j])
                model_pairs.append({
                    'model_a': models[i],
                    'model_b': models[j], 
                    'prob_a_better': prob_better,
                    'prob_b_better': 1 - prob_better
                })
        
        results['model_comparisons'] = pd.DataFrame(model_pairs)
        
        return results
    
    def trial_level_skill_analysis(self, df_skill: pd.DataFrame) -> Dict[str, Any]:
        """
        Trial-level skill model with random effects using Bambi.
        
        Allows comparison of models via PSIS-LOO and provides stacking weights.
        """
        print("Running trial-level skill analysis...")
        
        if df_skill.empty:
            print("Warning: Skill dataframe is empty, skipping trial-level analysis")
            return {}
        
        print(f"Skill dataframe shape: {df_skill.shape}")
        print(f"Skill dataframe columns: {df_skill.columns.tolist()}")
        print(f"Skill dataframe dtypes:\n{df_skill.dtypes}")
        
        # Check for missing values
        print(f"Missing values:\n{df_skill.isnull().sum()}")
        
        # Check data types and values
        # Enforce consistent ordering
        ordered_models = np.array(sorted(df_skill['model'].unique().tolist()))
        print(f"Model column unique values: {ordered_models}")
        print(f"Site column unique values: {df_skill['site'].unique()}")
        print(f"Correct column unique values: {df_skill['correct'].unique()}")
        print(f"Subject_id column sample: {df_skill['subject_id'].head().tolist()}")
        
        # Check if we have enough variation
        if df_skill['correct'].nunique() < 2:
            print("Warning: No variation in correct column, skipping analysis")
            return {}
        
        try:
            # Mixed-effects logistic: start simpler to reduce divergences
            # First try without subject random effects; these are very high-dimensional
            print("Creating Bambi model (simplified: random intercept by site only)...")
            # Use ordered categorical to ensure stable design matrix
            df_skill = df_skill.copy()
            df_skill['model'] = pd.Categorical(df_skill['model'], categories=ordered_models, ordered=True)
            df_skill['site'] = pd.Categorical(df_skill['site'], categories=sorted(df_skill['site'].unique().tolist()), ordered=True)
            bm = bmb.Model(
                "correct ~ 0 + model + (1|site)",
                data=df_skill,
                family="bernoulli",
            )
            print("Bambi model created successfully")
        except Exception as e:
            print(f"Error creating Bambi model: {e}")
            return {}
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use include_sample=True to include log likelihood; bump target_accept to reduce divergences
            idata = bm.fit(
                target_accept=0.995,
                random_seed=self.random_seed,
                draws=1500,
                tune=2500,
                chains=4,
                cores=4,
                include_sample=True,
            )
        
        # PSIS-LOO 
        try:
            # Try to compute LOO with the inference data
            loo = az.loo(idata)
            print("✅ LOO computation successful!")
        except Exception as e:
            print(f"Info: Skipping LOO/WAIC (log likelihood unavailable for this model): {e}")
            loo = None
        
        # Extract fixed effects (model coefficients)
        try:
            coef_da = idata.posterior["model"]  # xarray DataArray with dims (chain, draw, model)
            coef_vals = coef_da.values
            # Flatten chains and draws into samples: (chains*draws, n_coefs)
            model_effects = coef_vals.reshape((-1, coef_vals.shape[-1]))
        except Exception as e:
            print(f"Warning: Could not extract model effects cleanly: {e}")
            model_effects = np.empty((0,))
        
        results = {
            'idata': idata,
            'loo': loo,
            'model_effects': model_effects,
            'model_means': np.mean(model_effects, axis=0),
            'model_std': np.std(model_effects, axis=0)
        }
        
        return results
    
    def bayesian_calibration_analysis(self, df_calib: pd.DataFrame) -> Dict[str, Any]:
        """
        Bayesian calibration analysis with site pooling.
        
        Calibrates each model and compares ECE posteriors.
        """
        print("Running Bayesian calibration analysis...")
        
        if df_calib.empty:
            return {}
        
        # Run calibration per class and model
        calibration_results = {}
        
        # Skip calibration analysis for now due to dimension issues
        print("Skipping calibration analysis due to data structure complexity...")
        
        return calibration_results
    
    def bayesian_auc_analysis(self, df_auc: pd.DataFrame) -> Dict[str, Any]:
        """
        Bayesian AUC estimation for each class and model.
        
        Uses binormal model for AUC computation.
        """
        print("Running Bayesian AUC analysis...")
        
        if df_auc.empty:
            return {}
        
        # Skip AUC analysis for now due to complexity
        print("Skipping AUC analysis due to data structure complexity...")
        auc_results = {}
        
        return auc_results
    
    def stacking_ensemble_analysis(self, model_data: List[ModelFoldData]) -> Dict[str, Any]:
        """
        Bayesian stacking ensemble with LOO weights.
        
        Creates ensemble with uncertainty-aware model selection.
        """
        print("Running stacking ensemble analysis...")
        
        # For now, implement a simple version
        # In practice, you'd need pointwise log-likelihood from each model
        
        # Aggregate predictions across folds for each model
        model_predictions = {}
        
        for data in model_data:
            if data.model not in model_predictions:
                model_predictions[data.model] = {
                    'predictions': [],
                    'probabilities': [],
                    'labels': []
                }
            
            model_predictions[data.model]['predictions'].extend(data.predictions)
            model_predictions[data.model]['probabilities'].extend(data.probabilities)
            model_predictions[data.model]['labels'].extend(data.labels)
        
        # Convert to numpy arrays
        # Convert to numpy arrays in a stable, sorted model order
        models_sorted = sorted(model_predictions.keys())
        for model in models_sorted:
            model_predictions[model]['predictions'] = np.array(model_predictions[model]['predictions'])
            model_predictions[model]['probabilities'] = np.array(model_predictions[model]['probabilities'])
            model_predictions[model]['labels'] = np.array(model_predictions[model]['labels'])
        
        # Ensemble with multiple weighting strategies
        if len(model_predictions) > 1:
            models = models_sorted
            all_probs = np.array([model_predictions[model]['probabilities'] for model in models])
            
            # Calculate individual accuracies first
            individual_accs = {
                model: accuracy_score(model_predictions[model]['labels'], 
                                    model_predictions[model]['predictions'])
                for model in models
            }
            
            # helper to compute macro AUC given probs and true labels
            def compute_macro_auc(probabilities: np.ndarray, labels: np.ndarray) -> float:
                # one-vs-rest macro AUC
                y_true = labels
                y_score = probabilities
                # handle labels starting at 0
                classes = np.unique(y_true)
                aucs = []
                for c in classes:
                    y_bin = (y_true == c).astype(int)
                    aucs.append(roc_auc_score(y_bin, y_score[:, int(c)]))
                return float(np.mean(aucs))

            true_labels = model_predictions[models[0]]['labels']

            # 1. Equal weight ensemble (original)
            equal_weights = np.ones(len(models)) / len(models)
            equal_ensemble_probs = np.average(all_probs, axis=0, weights=equal_weights)
            equal_ensemble_preds = np.argmax(equal_ensemble_probs, axis=1)
            equal_ensemble_acc = accuracy_score(
                true_labels, 
                equal_ensemble_preds
            )
            equal_ensemble_auc = compute_macro_auc(equal_ensemble_probs, true_labels)
            
            # 2. Performance-weighted ensemble (based on individual accuracies)
            acc_values = np.array([individual_accs[model] for model in models])
            # Softmax weights to make them sum to 1 and emphasize differences
            performance_weights = np.exp(acc_values * 10) / np.sum(np.exp(acc_values * 10))
            weighted_ensemble_probs = np.average(all_probs, axis=0, weights=performance_weights)
            weighted_ensemble_preds = np.argmax(weighted_ensemble_probs, axis=1)
            weighted_ensemble_acc = accuracy_score(
                true_labels, 
                weighted_ensemble_preds
            )
            weighted_ensemble_auc = compute_macro_auc(weighted_ensemble_probs, true_labels)
            
            # 3. Top-2 weighted ensemble (DenseNet + Simple3DCNN only)
            top2_models = ['DenseNet121_3D', 'Simple3DCNN']
            top2_mask = [model in top2_models for model in models]
            if sum(top2_mask) >= 2:
                top2_probs = all_probs[top2_mask]
                top2_weights = np.array([individual_accs[model] for model in models if model in top2_models])
                top2_weights = top2_weights / np.sum(top2_weights)
                top2_ensemble_probs = np.average(top2_probs, axis=0, weights=top2_weights)
                top2_ensemble_preds = np.argmax(top2_ensemble_probs, axis=1)
                top2_ensemble_acc = accuracy_score(
                    true_labels, 
                    top2_ensemble_preds
                )
                top2_ensemble_auc = compute_macro_auc(top2_ensemble_probs, true_labels)
            else:
                top2_ensemble_acc = None
                top2_ensemble_preds = None
                top2_ensemble_probs = None
                top2_ensemble_auc = None
            
            return {
                'models': models,
                'individual_accuracies': individual_accs,
                'model_source_dirs': {m: self.model_source_dirs.get(m, '') for m in models},
                'ensemble_weights': {
                    'equal': equal_weights.tolist(),
                    'performance_weighted': performance_weights.tolist(),
                    'top2_weighted': top2_weights.tolist() if 'top2_weights' in locals() else None
                },
                'ensemble_results': {
                    'equal_weight': {
                        'predictions': equal_ensemble_preds,
                        'probabilities': equal_ensemble_probs,
                        'accuracy': equal_ensemble_acc,
                        'auc_macro_ovr': equal_ensemble_auc
                    },
                    'performance_weighted': {
                        'predictions': weighted_ensemble_preds,
                        'probabilities': weighted_ensemble_probs,
                        'accuracy': weighted_ensemble_acc,
                        'auc_macro_ovr': weighted_ensemble_auc
                    },
                    'top2_weighted': {
                        'predictions': top2_ensemble_preds,
                        'probabilities': top2_ensemble_probs,
                        'accuracy': top2_ensemble_acc,
                        'auc_macro_ovr': top2_ensemble_auc
                    } if top2_ensemble_acc is not None else None
                }
            }
        
        return {}
    
    def create_visualizations(self, results: BayesianResults, data_dict: Dict[str, pd.DataFrame]):
        """Create comprehensive visualizations for all analyses."""
        print("Creating visualizations...")
        
        # 1. Hierarchical accuracy plots
        if results.accuracy_results:
            try:
                self._plot_hierarchical_accuracy(results.accuracy_results)
            except Exception as e:
                print(f"Warning: Could not create accuracy plots: {e}")
        
        # 2. Calibration plots (skipped for now)
        # 3. AUC comparison plots (skipped for now)
        
        # 4. Model comparison heatmaps
        if results.accuracy_results and isinstance(results.accuracy_results.get('model_comparisons'), pd.DataFrame):
            try:
                if not results.accuracy_results['model_comparisons'].empty:
                    self._plot_model_comparison_heatmap(results.accuracy_results)
                else:
                    print("Info: Model comparison dataframe is empty; skipping heatmap")
            except Exception as e:
                print(f"Warning: Could not create comparison heatmap: {e}")

        # 5. Publication-ready plots
        try:
            # a) Confusion matrices per model and ensemble (if skill data available)
            if 'skill' in data_dict and not data_dict['skill'].empty:
                self._plot_confusion_matrices(data_dict['skill'])
            # b) ROC and PR curves per class per model (if auc data available)
            if 'auc' in data_dict and not data_dict['auc'].empty:
                self._plot_roc_pr_curves(data_dict['auc'])
                # Overlaid AUC distributions (per-model across folds)
                self._plot_auc_distributions(data_dict['auc'])
                # Two-panel AUC with bootstrap pairwise probabilities
                self._plot_auc_two_panel(data_dict['auc'])
            # c) Site effects forest (if idata available)
            if results.accuracy_results and 'idata' in results.accuracy_results:
                self._plot_site_effects(results.accuracy_results['idata'])
            # d) Posterior pairwise differences and rank probabilities
            if results.accuracy_results and 'accuracy_samples' in results.accuracy_results:
                self._plot_posterior_differences_and_ranks(results.accuracy_results)
            # e) Ensemble weights visualization
            if results.stacking_results and 'ensemble_weights' in results.stacking_results:
                self._plot_ensemble_weights(results.stacking_results)
            # f) MCC distributions (per-model across folds)
            if 'skill' in data_dict and not data_dict['skill'].empty:
                self._plot_mcc_distributions(data_dict['skill'])
                # Two-panel MCC with bootstrap pairwise probabilities
                self._plot_mcc_two_panel(data_dict['skill'])
            # g) One-vs-rest per-class ACC and AUC (AD/PD/CN vs rest)
            if 'skill' in data_dict and 'auc' in data_dict and not data_dict['skill'].empty and not data_dict['auc'].empty:
                self._plot_ovr_acc_auc(data_dict['skill'], data_dict['auc'])
            # h) Frequentist significance tests and calibration metrics
            if 'auc' in data_dict and not data_dict['auc'].empty and 'skill' in data_dict and not data_dict['skill'].empty:
                self._run_frequentist_comparisons(data_dict['auc'], data_dict['skill'])
        except Exception as e:
            print(f"Warning: Failed to create publication plots: {e}")
    
    def _plot_hierarchical_accuracy(self, results: Dict[str, Any]):
        """Plot hierarchical accuracy results."""
        print("Creating hierarchical accuracy plots...")
        if not results:
            print("Warning: No results to plot")
            return
        
        print(f"Results keys: {results.keys()}")
        print(f"Models: {results.get('models', 'Not found')}")
        print(f"Accuracy means shape: {results.get('accuracy_means', 'Not found')}")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        ax1, ax2 = axes[0], axes[1]
        
        # Overlaid Beta distributions (fitted) on a single shared axis for easier comparison
        models = results['models']
        means = results['accuracy_means']
        ci_lower = results['accuracy_ci_lower']
        ci_upper = results['accuracy_ci_upper']
        print(f"Plotting {len(models)} models with means: {means}")
        try:
            # Build x-range from CI envelope, clipped to [0,1]
            x_min = max(0.0, float(np.min(ci_lower) - 0.05))
            x_max = min(1.0, float(np.max(ci_upper) + 0.05))
            x = np.linspace(x_min, x_max, 1000)
            colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
            for i, model in enumerate(models):
                # Fit Beta from posterior samples for this model
                samples = results.get('accuracy_samples', None)
                if samples is not None and isinstance(samples, np.ndarray) and samples.shape[1] == len(models):
                    arr = samples[:, i]
                    alpha, beta = self._fit_beta_from_samples(arr)
                    pdf = self._beta_pdf(x, alpha, beta)
                    pdf = pdf / np.max(pdf)
                    ax1.plot(x, pdf, color=colors[i], lw=2, label=model)
                    ax1.fill_between(x, 0, pdf, color=colors[i], alpha=0.08)
                    # mean marker
                    mu = float(np.mean(arr))
                    ax1.axvline(mu, color=colors[i], lw=1, alpha=0.6)
            ax1.set_xlabel('Accuracy')
            ax1.set_ylabel('Relative density')
            ax1.set_title('Model accuracy Beta distributions (from posterior)')
            ax1.grid(True, alpha=0.3)
            ax1.legend(title='Models', bbox_to_anchor=(1.04, 1), loc='upper left')
        except Exception as e:
            print(f"Warning: failed to draw overlaid normal densities: {e}")
        
        # Model comparison probabilities
        if 'model_comparisons' in results:
            comp_df = results['model_comparisons']
            if not comp_df.empty:
                # Create heatmap of comparison probabilities
                models = sorted(set(comp_df['model_a'].tolist() + comp_df['model_b'].tolist()))
                n_models = len(models)
                heatmap = np.zeros((n_models, n_models))
                
                for _, row in comp_df.iterrows():
                    i = models.index(row['model_a'])
                    j = models.index(row['model_b'])
                    heatmap[i, j] = row['prob_a_better']
                    heatmap[j, i] = row['prob_b_better']
                
                im = ax2.imshow(heatmap, cmap='RdBu_r', vmin=0, vmax=1)
                ax2.set_xticks(range(n_models))
                ax2.set_yticks(range(n_models))
                ax2.set_xticklabels(models, rotation=45)
                ax2.set_yticklabels(models)
                ax2.set_title('Model Comparison Probabilities\n(P(model A > model B))')
                
                # Add colorbar
                plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'hierarchical_accuracy_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_auc_distributions(self, df_auc: pd.DataFrame):
        """Overlaid normal PDFs of per-model AUC across folds (macro one-vs-rest)."""
        try:
            models = sorted(df_auc['model'].unique().tolist())
            classes = sorted(df_auc['class'].unique().tolist())
            auc_per_model = {m: [] for m in models}
            # compute macro AUC per fold by grouping per subject across classes
            # approximate: aggregate all class rows per model and compute macro AUC on the pooled set
            for m in models:
                dfm = df_auc[df_auc['model']==m]
                # derive per-subject macro AUC by averaging classwise contributions
                # fallback simplification: use overall macro AUC over pooled rows
                try:
                    y_true = dfm['y'].values
                    # We cannot compute macro directly from pooled; skip to per-class auc then average
                    auc_vals = []
                    for c in classes:
                        dmc = dfm[dfm['class']==c]
                        if dmc.empty:
                            continue
                        auc_vals.append(roc_auc_score(dmc['y'].values, dmc['score'].values))
                    if len(auc_vals)>0:
                        auc_per_model[m].append(float(np.mean(auc_vals)))
                except Exception:
                    continue
            # Prepare distributions (fit Beta per model over fold-wise AUCs)
            plt.figure(figsize=(8,5))
            colors = plt.cm.tab10(np.linspace(0,1,len(models)))
            for i, m in enumerate(models):
                arr = np.array(auc_per_model[m])
                if arr.size == 0:
                    continue
                alpha, beta = self._fit_beta_from_samples(arr)
                # dynamic x-range focus but include tails
                x = np.linspace(0.0, 1.0, 800)
                pdf = self._beta_pdf(x, alpha, beta)
                pdf = pdf / (np.max(pdf) if np.max(pdf)>0 else 1.0)
                plt.plot(x, pdf, color=colors[i], lw=2, label=f"{m}")
                plt.fill_between(x, 0, pdf, color=colors[i], alpha=0.08)
                plt.axvline(float(np.mean(arr)), color=colors[i], lw=1, alpha=0.6)
            plt.xlabel('Macro AUC (one-vs-rest)')
            plt.ylabel('Relative density')
            plt.title('AUC distributions across folds (Beta fit)')
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.04,1), loc='upper left')
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'auc_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: failed auc distributions: {e}")

    def _bootstrap_pairwise_prob_matrix(self, values_by_model: Dict[str, np.ndarray], n_boot: int = 2000, random_state: int = 42) -> Tuple[np.ndarray, List[str]]:
        rng = np.random.default_rng(random_state)
        models = sorted(values_by_model.keys())
        M = len(models)
        probs = np.zeros((M, M))
        # for each pair, estimate P(i > j)
        for i in range(M):
            vi = np.asarray(values_by_model[models[i]])
            for j in range(M):
                if i == j:
                    probs[i, j] = 0.5
                    continue
                vj = np.asarray(values_by_model[models[j]])
                if vi.size == 0 or vj.size == 0:
                    continue
                cnt = 0
                for _ in range(n_boot):
                    si = rng.choice(vi, size=vi.size, replace=True)
                    sj = rng.choice(vj, size=vj.size, replace=True)
                    cnt += (np.mean(si) > np.mean(sj))
                probs[i, j] = cnt / n_boot
        return probs, models

    def _plot_auc_two_panel(self, df_auc: pd.DataFrame):
        """Two-panel AUC: overlaid densities + bootstrap pairwise probability heatmap."""
        try:
            models = sorted(df_auc['model'].unique().tolist())
            classes = sorted(df_auc['class'].unique().tolist())
            auc_per_model = {m: [] for m in models}
            for m in models:
                dfm = df_auc[df_auc['model']==m]
                auc_vals = []
                for c in classes:
                    dmc = dfm[dfm['class']==c]
                    if dmc.empty:
                        continue
                    auc_vals.append(roc_auc_score(dmc['y'].values, dmc['score'].values))
                if len(auc_vals)>0:
                    auc_per_model[m] = np.array(auc_vals)
            # Left panel densities
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
            colors = plt.cm.tab10(np.linspace(0,1,len(models)))
            for i, m in enumerate(models):
                arr = np.array(auc_per_model[m])
                if arr.size == 0:
                    continue
                alpha, beta = self._fit_beta_from_samples(arr)
                x = np.linspace(0.0, 1.0, 800)
                pdf = self._beta_pdf(x, alpha, beta)
                pdf = pdf / (np.max(pdf) if np.max(pdf)>0 else 1.0)
                ax1.plot(x, pdf, color=colors[i], lw=2, label=f"{m}")
                ax1.fill_between(x, 0, pdf, color=colors[i], alpha=0.08)
                ax1.axvline(float(np.mean(arr)), color=colors[i], lw=1, alpha=0.6)
            ax1.set_xlabel('Macro AUC (one-vs-rest)')
            ax1.set_ylabel('Relative density')
            ax1.set_title('AUC distributions across folds (Beta fit)')
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.04,1), loc='upper left')
            # Right panel bootstrap pairwise heatmap
            prob_matrix, model_order = self._bootstrap_pairwise_prob_matrix({m: np.array(auc_per_model[m]) for m in models})
            im = ax2.imshow(prob_matrix, cmap='RdBu_r', vmin=0, vmax=1)
            ax2.set_xticks(range(len(model_order)))
            ax2.set_yticks(range(len(model_order)))
            ax2.set_xticklabels(model_order, rotation=45)
            ax2.set_yticklabels(model_order)
            ax2.set_title('P(AUC row > AUC column) (bootstrap)')
            plt.colorbar(im, ax=ax2)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'auc_two_panel.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: failed AUC two-panel: {e}")

    def _plot_mcc_two_panel(self, df_skill: pd.DataFrame):
        """Two-panel MCC: overlaid densities + bootstrap pairwise probability heatmap."""
        try:
            models = sorted(df_skill['model'].unique().tolist())
            sites = sorted(df_skill['site'].unique().tolist())
            mcc_per_model = {m: [] for m in models}
            for m in models:
                vals = []
                for s in sites:
                    dfs = df_skill[(df_skill['model']==m) & (df_skill['site']==s)]
                    if dfs.empty:
                        continue
                    y_true = dfs['true_label'].values
                    y_pred = dfs['predicted_label'].values
                    try:
                        vals.append(matthews_corrcoef(y_true, y_pred))
                    except Exception:
                        continue
                mcc_per_model[m] = np.array(vals)
            # Left panel densities
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
            colors = plt.cm.tab10(np.linspace(0,1,len(models)))
            for i, m in enumerate(models):
                arr = np.array(mcc_per_model[m])
                if arr.size == 0:
                    continue
                # Transform MCC (-1,1) to (0,1) via (x+1)/2 to fit Beta
                arr01 = (arr + 1.0) / 2.0
                alpha, beta = self._fit_beta_from_samples(arr01)
                x = np.linspace(0.0, 1.0, 800)
                pdf = self._beta_pdf(x, alpha, beta)
                pdf = pdf / (np.max(pdf) if np.max(pdf)>0 else 1.0)
                # map back x to MCC axis
                x_mcc = x*2.0 - 1.0
                ax1.plot(x_mcc, pdf, color=colors[i], lw=2, label=f"{m}")
                ax1.fill_between(x_mcc, 0, pdf, color=colors[i], alpha=0.08)
                ax1.axvline(float(np.mean(arr)), color=colors[i], lw=1, alpha=0.6)
            ax1.set_xlabel('MCC')
            ax1.set_ylabel('Relative density')
            ax1.set_title('MCC distributions across folds (Beta fit via transform)')
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.04,1), loc='upper left')
            # Right panel bootstrap heatmap
            prob_matrix, model_order = self._bootstrap_pairwise_prob_matrix(mcc_per_model)
            im = ax2.imshow(prob_matrix, cmap='RdBu_r', vmin=0, vmax=1)
            ax2.set_xticks(range(len(model_order)))
            ax2.set_yticks(range(len(model_order)))
            ax2.set_xticklabels(model_order, rotation=45)
            ax2.set_yticklabels(model_order)
            ax2.set_title('P(MCC row > MCC column) (bootstrap)')
            plt.colorbar(im, ax=ax2)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'mcc_two_panel.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: failed MCC two-panel: {e}")

    def _plot_mcc_distributions(self, df_skill: pd.DataFrame):
        """Overlaid normal PDFs of per-model MCC across folds."""
        try:
            models = sorted(df_skill['model'].unique().tolist())
            sites = sorted(df_skill['site'].unique().tolist())
            mcc_per_model = {m: [] for m in models}
            for m in models:
                for s in sites:
                    dfs = df_skill[(df_skill['model']==m) & (df_skill['site']==s)]
                    if dfs.empty:
                        continue
                    y_true = dfs['true_label'].values
                    y_pred = dfs['predicted_label'].values
                    try:
                        mcc = matthews_corrcoef(y_true, y_pred)
                        mcc_per_model[m].append(float(mcc))
                    except Exception:
                        continue
            plt.figure(figsize=(8,5))
            colors = plt.cm.tab10(np.linspace(0,1,len(models)))
            from scipy.stats import norm
            for i, m in enumerate(models):
                arr = np.array(mcc_per_model[m])
                if arr.size == 0:
                    continue
                mu = float(np.mean(arr))
                sd = float(np.std(arr) + 1e-6)
                x = np.linspace(mu-4*sd, mu+4*sd, 800)
                pdf = norm.pdf(x, loc=mu, scale=sd)
                pdf = pdf / (np.max(pdf) if np.max(pdf)>0 else 1.0)
                plt.plot(x, pdf, color=colors[i], lw=2, label=f"{m}")
                plt.fill_between(x, 0, pdf, color=colors[i], alpha=0.08)
                plt.axvline(mu, color=colors[i], lw=1, alpha=0.6)
            plt.xlabel('Matthews correlation coefficient (MCC)')
            plt.ylabel('Relative density')
            plt.title('MCC distributions across folds (normal approx)')
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.04,1), loc='upper left')
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'mcc_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: failed mcc distributions: {e}")
    
    def _plot_calibration_analysis(self, results: Dict[int, Dict[str, Any]]):
        """Plot calibration analysis results."""
        if not results:
            return
        
        n_classes = len(results)
        fig, axes = plt.subplots(2, n_classes, figsize=(5*n_classes, 10))
        if n_classes == 1:
            axes = axes.reshape(2, 1)
        
        for class_idx, class_results in results.items():
            models = class_results['models']
            intercept_means = class_results['intercept_means']
            slope_means = class_results['slope_means']
            
            # Intercept comparison
            axes[0, class_idx].bar(models, intercept_means)
            axes[0, class_idx].set_title(f'Class {class_idx}: Calibration Intercepts')
            axes[0, class_idx].set_ylabel('Intercept (a)')
            axes[0, class_idx].tick_params(axis='x', rotation=45)
            axes[0, class_idx].grid(True, alpha=0.3)
            
            # Slope comparison  
            axes[1, class_idx].bar(models, slope_means)
            axes[1, class_idx].set_title(f'Class {class_idx}: Calibration Slopes')
            axes[1, class_idx].set_ylabel('Slope (b)')
            axes[1, class_idx].tick_params(axis='x', rotation=45)
            axes[1, class_idx].grid(True, alpha=0.3)
            
            # Add reference lines
            axes[0, class_idx].axhline(0, color='red', linestyle='--', alpha=0.5)
            axes[1, class_idx].axhline(1, color='red', linestyle='--', alpha=0.5, 
                                     label='Perfect calibration')
            if class_idx == 0:
                axes[1, class_idx].legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'calibration_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_auc_comparison(self, results: Dict[int, Dict[str, Dict[str, Any]]]):
        """Plot AUC comparison across classes and models."""
        if not results:
            return
        
        # Collect all AUC data
        auc_data = []
        for class_idx, class_results in results.items():
            for model, model_results in class_results.items():
                auc_data.append({
                    'class': f'Class {class_idx}',
                    'model': model,
                    'auc_mean': model_results['auc_mean'],
                    'auc_std': model_results['auc_std'],
                    'auc_ci_lower': model_results['auc_ci_lower'],
                    'auc_ci_upper': model_results['auc_ci_upper']
                })
        
        if not auc_data:
            return
        
        df_auc = pd.DataFrame(auc_data)
        
        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        classes = df_auc['class'].unique()
        models = df_auc['model'].unique()
        
        x = np.arange(len(classes))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            model_data = df_auc[df_auc['model'] == model]
            means = model_data['auc_mean'].values
            stds = model_data['auc_std'].values
            
            ax.bar(x + i * width, means, width, label=model, 
                  yerr=stds, capsize=5)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('AUC')
        ax.set_title('Bayesian AUC Estimates by Class and Model')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'auc_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison_heatmap(self, results: Dict[str, Any]):
        """Plot model comparison heatmap."""
        if 'model_comparisons' not in results:
            return
        
        comp_df = results['model_comparisons']
        if comp_df.empty:
            return
        
        # Create symmetric matrix
        models = sorted(set(comp_df['model_a'].tolist() + comp_df['model_b'].tolist()))
        n_models = len(models)
        matrix = np.ones((n_models, n_models))
        
        for _, row in comp_df.iterrows():
            i = models.index(row['model_a'])
            j = models.index(row['model_b'])
            matrix[i, j] = row['prob_a_better']
            matrix[j, i] = row['prob_b_better']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdBu_r', 
                   xticklabels=models, yticklabels=models, 
                   center=0.5, vmin=0, vmax=1)
        plt.title('Model Comparison Matrix\n(P(model row > model column))')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'model_comparison_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrices(self, df_skill: pd.DataFrame):
        """Plot normalized confusion matrices per model and the equal/performance-weighted ensembles if available."""
        try:
            models = sorted(df_skill['model'].unique().tolist())
            true = df_skill['true_label'].values
            pred = df_skill['predicted_label'].values
            n_classes = int(max(true.max(), pred.max()) + 1)
            # Per-model confusion
            for model in models:
                dfm = df_skill[df_skill['model'] == model]
                cm = confusion_matrix(dfm['true_label'], dfm['predicted_label'], labels=list(range(n_classes)), normalize='true')
                plt.figure(figsize=(5,4))
                sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=False)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix (normalized) - {model}')
                plt.tight_layout()
                plt.savefig(self.plots_dir / f'confusion_{model}.png', dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"Warning: failed confusion matrices: {e}")

    def _plot_roc_pr_curves(self, df_auc: pd.DataFrame):
        """Plot ROC and PR curves per class and model with macro/micro summaries."""
        try:
            models = sorted(df_auc['model'].unique().tolist())
            classes = sorted(df_auc['class'].unique().tolist())
            for cls in classes:
                plt.figure(figsize=(10,4))
                # ROC subplot
                ax1 = plt.subplot(1,2,1)
                # PR subplot
                ax2 = plt.subplot(1,2,2)
                for model in models:
                    dfm = df_auc[(df_auc['model']==model) & (df_auc['class']==cls)]
                    if dfm.empty:
                        continue
                    y = dfm['y'].values
                    s = dfm['score'].values
                    # ROC
                    try:
                        auc_val = roc_auc_score(y, s)
                        fpr = np.linspace(0,1,101)
                        # approximate ROC curve using thresholds (fallback: skip detailed curve)
                        # Using sklearn to compute ROC points would need roc_curve import; we'll rely on AUC only for now
                        ax1.plot([0,1],[0,1], color='gray', ls='--', lw=0.5) if model==models[0] else None
                        ax1.plot([],[], label=f"{model} AUC={auc_val:.3f}")
                    except Exception:
                        pass
                    # PR
                    try:
                        precision, recall, _ = precision_recall_curve(y, s)
                        ap = average_precision_score(y, s)
                        ax2.plot(recall, precision, lw=1.5, label=f"{model} AP={ap:.3f}")
                    except Exception:
                        pass
                ax1.set_title(f'ROC (Class {cls})')
                ax1.set_xlabel('FPR')
                ax1.set_ylabel('TPR')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax2.set_title(f'PR (Class {cls})')
                ax2.set_xlabel('Recall')
                ax2.set_ylabel('Precision')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.plots_dir / f'roc_pr_class_{cls}.png', dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"Warning: failed roc/pr curves: {e}")

    def _plot_site_effects(self, idata: az.InferenceData):
        """Plot site random effects forest from hierarchical model if available."""
        try:
            if 'u_site' not in idata.posterior:
                return
            u = idata.posterior['u_site'].values.reshape(-1, idata.posterior['u_site'].values.shape[-1])
            means = np.mean(u, axis=0)
            lower = np.percentile(u, 2.5, axis=0)
            upper = np.percentile(u, 97.5, axis=0)
            idx = np.arange(len(means))
            plt.figure(figsize=(8,6))
            plt.errorbar(means, idx, xerr=[means-lower, upper-means], fmt='o', capsize=4)
            plt.yticks(idx, [f'site_{i+1}' for i in idx])
            plt.xlabel('Random intercept (logit scale)')
            plt.title('Site effects (hierarchical model)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'site_effects_forest.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: failed site effects plot: {e}")

    def _plot_posterior_differences_and_ranks(self, acc_results: Dict[str, Any]):
        """Plot posterior differences acc_i - acc_j and rank probabilities."""
        try:
            samples = acc_results.get('accuracy_samples', None)
            models = acc_results.get('models', [])
            if samples is None or len(models) == 0:
                return
            samples = np.asarray(samples)
            M = samples.shape[1]
            # Differences
            plt.figure(figsize=(10,6))
            colors = plt.cm.tab10(np.linspace(0,1,M))
            from scipy.stats import gaussian_kde
            for i in range(M):
                for j in range(i+1, M):
                    diff = samples[:, i] - samples[:, j]
                    kde = gaussian_kde(diff)
                    xs = np.linspace(np.min(diff), np.max(diff), 400)
                    plt.plot(xs, kde(xs), label=f"{models[i]} - {models[j]}")
            plt.axvline(0, color='k', ls='--', lw=1)
            plt.xlabel('Accuracy difference')
            plt.ylabel('Density')
            plt.title('Posterior differences between models')
            plt.legend(bbox_to_anchor=(1.04,1), loc='upper left')
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'posterior_differences.png', dpi=300, bbox_inches='tight')
            plt.close()
            # Ranks
            ranks = np.argsort(-samples, axis=1)  # descending
            rank_counts = np.zeros((M, M), dtype=int)
            for r in ranks:
                for pos, m_idx in enumerate(r):
                    rank_counts[m_idx, pos] += 1
            rank_probs = rank_counts / rank_counts.sum(axis=1, keepdims=True)
            plt.figure(figsize=(8,6))
            bottom = np.zeros(M)
            for pos in range(M):
                plt.bar(np.arange(M), rank_probs[:, pos], bottom=bottom, label=f'Rank {pos+1}')
                bottom += rank_probs[:, pos]
            plt.xticks(np.arange(M), models, rotation=30)
            plt.ylabel('Probability')
            plt.title('Posterior rank probabilities')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'posterior_rank_probs.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: failed posterior diffs/ranks: {e}")

    def _plot_ensemble_weights(self, stacking_results: Dict[str, Any]):
        """Plot ensemble weights for performance-weighted and top-2 ensembles."""
        try:
            if 'ensemble_weights' not in stacking_results or 'models' not in stacking_results:
                return
            models = stacking_results['models']
            weights = stacking_results['ensemble_weights']
            plt.figure(figsize=(8,4))
            if weights.get('performance_weighted') is not None:
                plt.bar(np.arange(len(models))-0.2, weights['performance_weighted'], width=0.4, label='Performance-weighted')
            if weights.get('equal') is not None:
                plt.bar(np.arange(len(models))+0.2, weights['equal'], width=0.4, label='Equal')
            plt.xticks(np.arange(len(models)), models, rotation=20)
            plt.ylabel('Weight')
            plt.title('Ensemble weights')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'ensemble_weights.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: failed ensemble weights plot: {e}")

    def _run_frequentist_comparisons(self, df_auc: pd.DataFrame, df_skill: pd.DataFrame):
        """Run DeLong-like bootstrap for AUC, Cochran's Q + McNemar for ACC, and compute Brier/ECE + reliability curves."""
        try:
            out_dir = self.results_dir
            models = sorted(df_skill['model'].unique().tolist())
            classes = sorted(df_auc['class'].unique().tolist())

            # 1) Accuracy: Cochran's Q (omnibus) + McNemar pairwise
            # Build per-subject correctness matrix per model
            # We align by subject_id across models
            pivot = df_skill.pivot_table(index='subject_id', columns='model', values='correct', aggfunc='first')
            pivot = pivot[models].dropna()
            # Cochran's Q test
            try:
                q_stat, q_p = cochrans_q(pivot.values)
            except Exception as e:
                q_stat, q_p = np.nan, np.nan
            # Pairwise McNemar
            mcnemar_p = pd.DataFrame(index=models, columns=models, data=np.nan)
            for i, a in enumerate(models):
                for j, b in enumerate(models):
                    if i >= j:
                        continue
                    a_correct = pivot[a].astype(int).values
                    b_correct = pivot[b].astype(int).values
                    # 2x2 of disagreements
                    n01 = int(np.sum((a_correct == 0) & (b_correct == 1)))
                    n10 = int(np.sum((a_correct == 1) & (b_correct == 0)))
                    table = np.array([[0, n01],[n10, 0]])
                    try:
                        res = mcnemar(table, exact=False, correction=True)
                        mcnemar_p.loc[a, b] = res.pvalue
                        mcnemar_p.loc[b, a] = res.pvalue
                    except Exception:
                        pass
            # Adjust p-values (Holm) for upper triangle
            pvals = mcnemar_p.values[np.triu_indices(len(models), k=1)]
            mask_valid = ~pd.isna(pvals)
            adj = np.full_like(pvals, np.nan, dtype=float)
            if np.any(mask_valid):
                rej, p_adj, *_ = multipletests(pvals[mask_valid], method='holm')
                adj[mask_valid] = p_adj
            # Put back
            k = 0
            for i in range(len(models)):
                for j in range(i+1, len(models)):
                    if not np.isnan(pvals[k]):
                        mcnemar_p.iloc[i, j] = adj[k]
                        mcnemar_p.iloc[j, i] = adj[k]
                    k += 1
            # Save
            pd.DataFrame({'Q_stat':[q_stat], 'Q_pvalue':[q_p]}).to_csv(out_dir / 'cochrans_q.csv', index=False)
            mcnemar_p.to_csv(out_dir / 'mcnemar_pvalues_holm.csv')

            # 2) AUC: bootstrap paired differences (DeLong alternative) per class and macro, with BH-FDR
            def bootstrap_auc_diff(df_class: pd.DataFrame, a: str, b: str, B: int = 2000) -> float:
                rng = np.random.default_rng(self.random_seed)
                dfa = df_class[df_class['model']==a]
                dfb = df_class[df_class['model']==b]
                # align by subject_id
                merged = dfa.merge(dfb, on=['subject_id'], suffixes=('_a','_b'))
                y = merged['y_a'].values.astype(int)  # same as y_b
                sa = merged['score_a'].values
                sb = merged['score_b'].values
                # observed diff
                try:
                    auc_a = roc_auc_score(y, sa)
                    auc_b = roc_auc_score(y, sb)
                except Exception:
                    return np.nan
                diff_obs = auc_a - auc_b
                n = len(y)
                if n < 10:
                    return np.nan
                diffs = np.empty(B)
                for i in range(B):
                    idx = rng.integers(0, n, n)
                    try:
                        diffs[i] = roc_auc_score(y[idx], sa[idx]) - roc_auc_score(y[idx], sb[idx])
                    except Exception:
                        diffs[i] = np.nan
                diffs = diffs[~np.isnan(diffs)]
                if diffs.size < 100:
                    return np.nan
                # two-sided p-value via percentile
                p = 2 * min(np.mean(diffs >= diff_obs), np.mean(diffs <= diff_obs))
                return p

            # per-class p-value matrices
            auc_p_mats = {}
            for c in classes:
                dfc = df_auc[df_auc['class']==c]
                pmat = pd.DataFrame(index=models, columns=models, data=np.nan)
                for i, a in enumerate(models):
                    for j, b in enumerate(models):
                        if i >= j:
                            continue
                        p = bootstrap_auc_diff(dfc, a, b)
                        pmat.loc[a,b] = p
                        pmat.loc[b,a] = p
                # BH-FDR
                pvals = pmat.values[np.triu_indices(len(models), k=1)]
                mask_valid = ~pd.isna(pvals)
                if np.any(mask_valid):
                    _, p_adj, _, _ = multipletests(pvals[mask_valid], method='fdr_bh')
                    k = 0
                    for i in range(len(models)):
                        for j in range(i+1, len(models)):
                            if mask_valid[k]:
                                pmat.iloc[i, j] = p_adj[np.sum(mask_valid[:k+1]) - 1]
                                pmat.iloc[j, i] = pmat.iloc[i, j]
                            k += 1
                auc_p_mats[c] = pmat
                pmat.to_csv(out_dir / f'delong_boot_pvalues_class_{c}.csv')

            # Macro-AUC: average per subject across classes, then bootstrap
            macro_p = pd.DataFrame(index=models, columns=models, data=np.nan)
            # Build per-subject macro scores per model
            for i, a in enumerate(models):
                for j, b in enumerate(models):
                    if i >= j:
                        continue
                    p_list = []
                    for c in classes:
                        p = bootstrap_auc_diff(df_auc[df_auc['class']==c], a, b)
                        if not np.isnan(p):
                            p_list.append(p)
                    if p_list:
                        macro_p.loc[a,b] = np.mean(p_list)
                        macro_p.loc[b,a] = macro_p.loc[a,b]
            # FDR
            pvals = macro_p.values[np.triu_indices(len(models), k=1)]
            mask_valid = ~pd.isna(pvals)
            if np.any(mask_valid):
                _, p_adj, _, _ = multipletests(pvals[mask_valid], method='fdr_bh')
                k = 0
                for i in range(len(models)):
                    for j in range(i+1, len(models)):
                        if mask_valid[k]:
                            macro_p.iloc[i, j] = p_adj[np.sum(mask_valid[:k+1]) - 1]
                            macro_p.iloc[j, i] = macro_p.iloc[i, j]
                        k += 1
            macro_p.to_csv(out_dir / 'delong_boot_pvalues_macro.csv')

            # 3) Calibration: Brier, ECE, and reliability curves
            # Build probability arrays per subject per model
            # We need per-subject predicted probability of true class
            # Reconstruct from auc df by taking score at each sample for class==true_label
            # Easier: rebuild from model_data in main flow; here approximate using df_auc
            calib_summary = []
            for m in models:
                # ECE with 15 bins
                try:
                    # Aggregate per subject best guess (max prob) and true
                    dfm = df_auc[df_auc['model']==m]
                    # For each subject, collect per-class scores and ground truth
                    grp = dfm.groupby('subject_id')
                    y_true = []
                    p_true = []
                    for sid, g in grp:
                        # infer true label as class with y==1
                        true_rows = g[g['y']==1]
                        if true_rows.empty:
                            continue
                        c_true = int(true_rows['class'].iloc[0])
                        # prob for true class
                        p = float(g[g['class']==c_true]['score'].iloc[0])
                        y_true.append(1)
                        p_true.append(p)
                    y_true = np.array(y_true)
                    p_true = np.array(p_true)
                    # Brier for positive class (one-vs-rest approximation)
                    brier = float(np.mean((p_true - y_true)**2))
                    # ECE
                    bins = np.linspace(0,1,16)
                    inds = np.digitize(p_true, bins) - 1
                    ece = 0.0
                    for b in range(len(bins)-1):
                        mask = inds == b
                        if not np.any(mask):
                            continue
                        conf = np.mean(p_true[mask])
                        acc = np.mean(y_true[mask])
                        ece += (np.sum(mask)/len(p_true)) * abs(acc - conf)
                    calib_summary.append({'model': m, 'brier': brier, 'ece': float(ece)})
                    # Reliability curve
                    plt.figure(figsize=(4,4))
                    # plot points per bin
                    xs = []
                    ys = []
                    for b in range(len(bins)-1):
                        mask = inds == b
                        if not np.any(mask):
                            continue
                        xs.append(np.mean(p_true[mask]))
                        ys.append(np.mean(y_true[mask]))
                    plt.plot([0,1],[0,1], ls='--', c='gray')
                    plt.plot(xs, ys, marker='o')
                    plt.xlabel('Predicted probability')
                    plt.ylabel('Observed frequency')
                    plt.title(f'Reliability - {m}')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(self.plots_dir / f'reliability_{m}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception:
                    calib_summary.append({'model': m, 'brier': np.nan, 'ece': np.nan})
            pd.DataFrame(calib_summary).to_csv(out_dir / 'calibration_summary.csv', index=False)
        except Exception as e:
            print(f"Warning: frequentist comparisons failed: {e}")

    def _plot_ovr_acc_auc(self, df_skill: pd.DataFrame, df_auc: pd.DataFrame):
        """Create two combined plots summarizing one-vs-rest ACC and AUC by model and class.
        - Combined ACC: bar per model, with per-class markers and range whiskers; title includes per-model mean±range
        - Combined AUC: same style
        Also writes class-specific labels using disease names instead of numbers.
        Saves: ovr_acc_combined.png, ovr_auc_combined.png"""
        try:
            models = sorted(df_skill['model'].unique().tolist())
            classes = sorted(df_auc['class'].unique().tolist())
            class_names = [self.class_labels[c] if c < len(self.class_labels) else f"Class {c}" for c in classes]
            # ACC (one-vs-rest): accuracy of detecting class c vs others
            acc_by_model_class = {m: [] for m in models}
            for m in models:
                dfs = df_skill[df_skill['model']==m]
                for c in classes:
                    if dfs.empty:
                        acc_by_model_class[m].append(np.nan)
                        continue
                    y_true = (dfs['true_label'].values == c).astype(int)
                    y_pred = (dfs['predicted_label'].values == c).astype(int)
                    acc_by_model_class[m].append(accuracy_score(y_true, y_pred))
            # Plot combined ACC
            plt.figure(figsize=(10,5))
            x = np.arange(len(models))
            means = np.array([np.nanmean(acc_by_model_class[m]) for m in models])
            mins = np.array([np.nanmin(acc_by_model_class[m]) for m in models])
            maxs = np.array([np.nanmax(acc_by_model_class[m]) for m in models])
            # bar for mean
            sns.barplot(x=models, y=means, color='skyblue', edgecolor='black')
            # whiskers for range
            for i in range(len(models)):
                plt.plot([i, i], [mins[i], maxs[i]], color='black', lw=1.5)
            # per-class markers
            colors = plt.cm.tab10(np.linspace(0,1,len(classes)))
            for ci, c in enumerate(classes):
                vals = [acc_by_model_class[m][ci] for m in models]
                plt.plot(x, vals, marker='o', linestyle='-', color=colors[ci], label=class_names[ci])
            plt.ylabel('Accuracy (one-vs-rest)')
            plt.xlabel('Model')
            plt.title('One-vs-rest ACC by model (mean bar, range whiskers, class markers)')
            plt.xticks(rotation=20)
            plt.ylim(0,1)
            plt.grid(True, axis='y', alpha=0.3)
            plt.legend(title='Class', bbox_to_anchor=(1.04,1), loc='upper left')
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'ovr_acc_combined.png', dpi=300, bbox_inches='tight')
            plt.close()
            # AUC (one-vs-rest): from df_auc
            auc_by_model_class = {m: [] for m in models}
            for m in models:
                for c in classes:
                    dmc = df_auc[(df_auc['model']==m) & (df_auc['class']==c)]
                    if dmc.empty:
                        auc_by_model_class[m].append(np.nan)
                        continue
                    auc_by_model_class[m].append(roc_auc_score(dmc['y'].values, dmc['score'].values))
            plt.figure(figsize=(10,5))
            x = np.arange(len(models))
            means = np.array([np.nanmean(auc_by_model_class[m]) for m in models])
            mins = np.array([np.nanmin(auc_by_model_class[m]) for m in models])
            maxs = np.array([np.nanmax(auc_by_model_class[m]) for m in models])
            sns.barplot(x=models, y=means, color='lightgreen', edgecolor='black')
            for i in range(len(models)):
                plt.plot([i, i], [mins[i], maxs[i]], color='black', lw=1.5)
            colors = plt.cm.tab10(np.linspace(0,1,len(classes)))
            for ci, c in enumerate(classes):
                vals = [auc_by_model_class[m][ci] for m in models]
                plt.plot(x, vals, marker='s', linestyle='--', color=colors[ci], label=class_names[ci])
            plt.ylabel('AUC (one-vs-rest)')
            plt.xlabel('Model')
            plt.title('One-vs-rest AUC by model (mean bar, range whiskers, class markers)')
            plt.xticks(rotation=20)
            plt.ylim(0,1)
            plt.grid(True, axis='y', alpha=0.3)
            plt.legend(title='Class', bbox_to_anchor=(1.04,1), loc='upper left')
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'ovr_auc_combined.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: failed OVR acc/auc plots: {e}")
    
    def save_results(self, results: BayesianResults, data_dict: Dict[str, pd.DataFrame]):
        """Save all results to files."""
        print("Saving results...")
        
        # Save data
        for name, df in data_dict.items():
            df.to_csv(self.data_dir / f'{name}_data.csv', index=False)
        
        # Save accuracy results
        if results.accuracy_results:
            with open(self.results_dir / 'accuracy_results.json', 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                acc_results = {}
                for k, v in results.accuracy_results.items():
                    if k == 'idata':
                        continue
                    if isinstance(v, np.ndarray):
                        acc_results[k] = v.tolist()
                    elif isinstance(v, pd.DataFrame):
                        acc_results[k] = v.to_dict('records')
                    elif isinstance(v, (np.floating, np.integer)):
                        acc_results[k] = v.item()
                    else:
                        acc_results[k] = v
                json.dump(acc_results, f, indent=2)
            
            if 'model_comparisons' in results.accuracy_results:
                results.accuracy_results['model_comparisons'].to_csv(
                    self.results_dir / 'model_comparisons.csv', index=False)
        
        # Save other results (simplified for JSON serialization)
        if results.skill_results:
            skill_simple = {}
            for k, v in results.skill_results.items():
                if k in ('idata', 'loo'):
                    continue
                if isinstance(v, np.ndarray):
                    skill_simple[k] = v.tolist()
                elif isinstance(v, (np.floating, np.integer)):
                    skill_simple[k] = v.item()
                else:
                    skill_simple[k] = v
            with open(self.results_dir / 'skill_results.json', 'w') as f:
                json.dump(skill_simple, f, indent=2)
        
        # Save ensemble results
        if results.stacking_results:
            with open(self.results_dir / 'ensemble_results.json', 'w') as f:
                ensemble_simple = {}
                for k, v in results.stacking_results.items():
                    if isinstance(v, np.ndarray):
                        ensemble_simple[k] = v.tolist()
                    elif isinstance(v, dict):
                        # recursively handle nested dicts
                        def to_jsonable(obj):
                            if isinstance(obj, np.ndarray):
                                return obj.tolist()
                            if isinstance(obj, (np.floating, np.integer)):
                                return obj.item()
                            if isinstance(obj, dict):
                                return {kk: to_jsonable(vv) for kk, vv in obj.items()}
                            return obj
                        ensemble_simple[k] = to_jsonable(v)
                    elif isinstance(v, (np.floating, np.integer)):
                        ensemble_simple[k] = v.item()
                    else:
                        ensemble_simple[k] = v
                json.dump(ensemble_simple, f, indent=2)
            # Write a dedicated sources file for convenience
            if 'model_source_dirs' in results.stacking_results:
                with open(self.results_dir / 'ensemble_sources.json', 'w') as f:
                    json.dump(results.stacking_results['model_source_dirs'], f, indent=2)
        
        print(f"Results saved to {self.output_dir}")
    
    def run_complete_analysis(self, run_dirs: List[str], models: Optional[List[str]] = None) -> BayesianResults:
        """
        Run complete Bayesian model comparison analysis.
        
        Args:
            run_dirs: List of run directories containing model outputs
            models: Optional list of specific models to include
            
        Returns:
            BayesianResults object with all analysis results
        """
        print("Starting comprehensive Bayesian model comparison...")
        
        # Load data
        model_data = self.load_model_data(run_dirs, models)
        if not model_data:
            print("No model data found!")
            return BayesianResults({}, {}, {}, {}, {})
        
        # Prepare data
        data_dict = self.prepare_data_for_analysis(model_data)
        
        # Run analyses
        accuracy_results = self.hierarchical_accuracy_analysis(data_dict['accuracy'])
        skill_results = self.trial_level_skill_analysis(data_dict['skill'])
        calibration_results = self.bayesian_calibration_analysis(data_dict['calibration'])
        auc_results = self.bayesian_auc_analysis(data_dict['auc'])
        stacking_results = self.stacking_ensemble_analysis(model_data)
        
        # Create results object
        results = BayesianResults(
            accuracy_results=accuracy_results,
            skill_results=skill_results,
            calibration_results=calibration_results,
            auc_results=auc_results,
            stacking_results=stacking_results
        )
        
        # Create visualizations
        try:
            print("Attempting to create visualizations...")
            self.create_visualizations(results, data_dict)
            print("✅ Visualizations created successfully")
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
        
        # Save results
        try:
            print("Attempting to save results...")
            self.save_results(results, data_dict)
            print("✅ Results saved successfully")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
        
        print("Bayesian analysis complete!")
        
        # Print summary of results
        self._print_results_summary(results)
        
        return results
    
    def _print_results_summary(self, results: BayesianResults):
        """Print a summary of the analysis results."""
        print("\n" + "="*60)
        print("BAYESIAN ANALYSIS SUMMARY")
        print("="*60)
        
        if results.accuracy_results:
            print("\n📊 Hierarchical Accuracy Results:")
            models = results.accuracy_results['models']
            means = results.accuracy_results['accuracy_means']
            ci_lower = results.accuracy_results['accuracy_ci_lower']
            ci_upper = results.accuracy_results['accuracy_ci_upper']
            
            for i, model in enumerate(models):
                print(f"  {model}: {means[i]:.4f} ({ci_lower[i]:.4f}, {ci_upper[i]:.4f})")
            
            if 'model_comparisons' in results.accuracy_results:
                print("\n🔍 Model Comparison Probabilities:")
                comp_df = results.accuracy_results['model_comparisons']
                for _, row in comp_df.iterrows():
                    print(f"  P({row['model_a']} > {row['model_b']}) = {row['prob_a_better']:.4f}")
        
        if results.stacking_results:
            print(f"\n🤝 Ensemble Results:")
            
            # Individual model accuracies
            if 'individual_accuracies' in results.stacking_results:
                print("  Individual Model Accuracies:")
                for model, acc in results.stacking_results['individual_accuracies'].items():
                    print(f"    {model}: {acc:.4f}")
            
            # Ensemble comparison
            if 'ensemble_results' in results.stacking_results:
                print("\n  📊 Ensemble Comparison:")
                ensemble_results = results.stacking_results['ensemble_results']
                
                if 'equal_weight' in ensemble_results:
                    print(f"    Equal Weight: {ensemble_results['equal_weight']['accuracy']:.4f} (AUC: {ensemble_results['equal_weight']['auc_macro_ovr']:.4f})")
                
                if 'performance_weighted' in ensemble_results:
                    print(f"    Performance Weighted: {ensemble_results['performance_weighted']['accuracy']:.4f} (AUC: {ensemble_results['performance_weighted']['auc_macro_ovr']:.4f})")
                    
                    # Show the weights used
                    if 'ensemble_weights' in results.stacking_results:
                        weights = results.stacking_results['ensemble_weights']['performance_weighted']
                        models = results.stacking_results['models']
                        print("    Performance Weights:")
                        for model, weight in zip(models, weights):
                            print(f"      {model}: {weight:.3f}")
                
                if 'top2_weighted' in ensemble_results and ensemble_results['top2_weighted']:
                    print(f"    Top-2 Weighted (DenseNet + Simple3DCNN): {ensemble_results['top2_weighted']['accuracy']:.4f} (AUC: {ensemble_results['top2_weighted']['auc_macro_ovr']:.4f})")
                    
                    if 'ensemble_weights' in results.stacking_results and results.stacking_results['ensemble_weights']['top2_weighted']:
                        top2_weights = results.stacking_results['ensemble_weights']['top2_weighted']
                        print("    Top-2 Weights:")
                        for model, weight in zip(['DenseNet121_3D', 'Simple3DCNN'], top2_weights):
                            print(f"      {model}: {weight:.3f}")
            
            # Legacy support for old format
            elif 'ensemble_accuracy' in results.stacking_results:
                print(f"  Ensemble Accuracy: {results.stacking_results['ensemble_accuracy']:.4f}")
        
        # Add best ensemble summary (accuracy and AUC)
        if results.stacking_results and 'ensemble_results' in results.stacking_results:
            er = results.stacking_results['ensemble_results']
            best_name = None
            best_acc = -1.0
            best_auc = -1.0
            for name, d in er.items():
                if d is None:
                    continue
                acc = d.get('accuracy', float('nan'))
                auc = d.get('auc_macro_ovr', float('nan'))
                if acc > best_acc:
                    best_acc = acc
                    best_auc = auc
                    best_name = name
            if best_name is not None:
                print("\n🏅 Best Ensemble:")
                print(f"  {best_name.replace('_',' ').title()}: {best_acc:.4f} (AUC: {best_auc:.4f})")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Bayesian model comparison for P4P deep learning models"
    )
    parser.add_argument(
        "--run-dirs", nargs="*", required=True,
        help="Paths to run directories containing model outputs"
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Optional subset of model names to include"
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Output directory. Defaults to ~/P4P_results/bayesian_comparison/<timestamp>"
    )
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.expanduser(f"~/P4P_results/bayesian_comparison/{timestamp}")
    else:
        output_dir = args.output_dir
    
    # Run analysis
    comparator = BayesianModelComparison(output_dir, args.random_seed)
    results = comparator.run_complete_analysis(args.run_dirs, args.models)
    
    print(f"\nBayesian model comparison completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Plots saved to: {output_dir}/plots")
    print(f"Data saved to: {output_dir}/data")
    print(f"Results saved to: {output_dir}/results")


if __name__ == "__main__":
    main()
