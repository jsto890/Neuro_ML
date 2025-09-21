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
    confusion_matrix, classification_report
)

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
        pm.set_utilities_random_seed(random_seed)
        
        # Create subdirectories
        self.plots_dir = self.output_dir / "plots"
        self.results_dir = self.output_dir / "results" 
        self.data_dir = self.output_dir / "data"
        
        for d in [self.plots_dir, self.results_dir, self.data_dir]:
            d.mkdir(exist_ok=True)
    
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
                
                # Load fold data
                fold_data = self._load_model_folds(model_dir, model_name)
                model_data.extend(fold_data)
        
        print(f"Loaded data for {len(set(d.model for d in model_data))} models, {len(model_data)} folds total")
        return model_data
    
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
        
        # 1. Hierarchical accuracy data (model × site/fold level)
        accuracy_rows = []
        
        # 2. Trial-level skill data (individual predictions)
        skill_rows = []
        
        # 3. Calibration data (probabilities and outcomes)
        calib_rows = []
        
        # 4. AUC data (scores for each class)
        auc_rows = []
        
        for data in model_data:
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
            for class_idx in range(data.n_classes):
                class_mask = data.labels == class_idx
                if not np.any(class_mask):
                    continue
                    
                # Use logits for calibration (log probabilities)
                if len(data.probabilities.shape) > 1:
                    class_probs = data.probabilities[class_mask, class_idx]
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
        
        return {
            'accuracy': pd.DataFrame(accuracy_rows),
            'skill': pd.DataFrame(skill_rows),
            'calibration': pd.DataFrame(calib_rows),
            'auc': pd.DataFrame(auc_rows)
        }
    
    def hierarchical_accuracy_analysis(self, df_accuracy: pd.DataFrame) -> Dict[str, Any]:
        """
        Hierarchical accuracy estimation with beta-binomial models.
        
        Estimates performance with proper uncertainty and partial pooling across sites/folds.
        """
        print("Running hierarchical accuracy analysis...")
        
        if df_accuracy.empty:
            return {}
        
        models = df_accuracy["model"].unique()
        sites = df_accuracy["site"].unique()
        M, S = len(models), len(sites)
        
        # Create indices
        model_idx = df_accuracy["model"].astype("category").cat.codes.values
        site_idx = df_accuracy["site"].astype("category").cat.codes.values
        k = df_accuracy["k"].values
        n = df_accuracy["n"].values
        
        with pm.Model() as accuracy_model:
            # Partial pooling: logit accuracy per model with site-specific random effects
            alpha_model = pm.Normal("alpha_model", 0, 1.5, shape=M)
            u_site_raw = pm.Normal("u_site_raw", 0, 1, shape=S)
            sigma_site = pm.HalfNormal("sigma_site", 1.0)
            u_site = pm.Deterministic("u_site", u_site_raw * sigma_site)
            
            logit_theta = alpha_model[model_idx] + u_site[site_idx]
            theta = pm.Deterministic("theta", pm.math.sigmoid(logit_theta))
            
            # Observed data
            k_obs = pm.Binomial("k_obs", n=n, p=theta, observed=k)
            
            # Posterior accuracy per model (marginalised over sites)
            acc_model = pm.Deterministic("acc_model", pm.math.sigmoid(alpha_model))
            
            # Sample
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                idata = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=self.random_seed)
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
            return {}
        
        # Mixed-effects logistic: model effect + site & subject random intercepts
        bm = bmb.Model(
            "correct ~ 0 + model + (1|site) + (1|subject_id)",
            data=df_skill,
            family="bernoulli",
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = bm.fit(target_accept=0.9, random_seed=self.random_seed)
        
        # PSIS-LOO
        loo = az.loo(idata)
        
        # Extract fixed effects (model coefficients)
        model_effects = idata.posterior["model"].values.reshape(-1, -1)
        
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
        
        for class_idx in df_calib['class'].unique():
            class_data = df_calib[df_calib['class'] == class_idx]
            
            models = class_data["model"].unique()
            sites = class_data["site"].unique()
            M, S = len(models), len(sites)
            
            if M == 0:
                continue
            
            m_idx = class_data["model"].astype("category").cat.codes.values
            s_idx = class_data["site"].astype("category").cat.codes.values
            z = class_data["logit"].values
            y = class_data["y"].values
            
            with pm.Model() as calib_model:
                # Per-model intercept/slope, pooled across sites
                a = pm.Normal("a", 0, 1.5, shape=M)   # intercept per model
                b = pm.Normal("b", 1, 0.5, shape=M)   # slope per model
                u_s_raw = pm.Normal("u_s_raw", 0, 1, shape=S)
                sigma_s = pm.HalfNormal("sigma_s", 0.5)
                u_s = u_s_raw * sigma_s
                
                logit_p = a[m_idx] + b[m_idx] * z + u_s[s_idx]
                p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
                y_obs = pm.Bernoulli("y_obs", p=p, observed=y)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    idata_cal = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=self.random_seed)
            
            calibration_results[class_idx] = {
                'idata': idata_cal,
                'models': models,
                'intercept_samples': idata_cal.posterior["a"].values.reshape(-1, M),
                'slope_samples': idata_cal.posterior["b"].values.reshape(-1, M),
                'intercept_means': np.mean(idata_cal.posterior["a"].values.reshape(-1, M), axis=0),
                'slope_means': np.mean(idata_cal.posterior["b"].values.reshape(-1, M), axis=0)
            }
        
        return calibration_results
    
    def bayesian_auc_analysis(self, df_auc: pd.DataFrame) -> Dict[str, Any]:
        """
        Bayesian AUC estimation for each class and model.
        
        Uses binormal model for AUC computation.
        """
        print("Running Bayesian AUC analysis...")
        
        if df_auc.empty:
            return {}
        
        auc_results = {}
        
        for class_idx in df_auc['class'].unique():
            class_data = df_auc[df_auc['class'] == class_idx]
            
            for model in class_data['model'].unique():
                model_data = class_data[class_data['model'] == model]
                
                s_pos = model_data[model_data.y == 1]["score"].values
                s_neg = model_data[model_data.y == 0]["score"].values
                
                if len(s_pos) == 0 or len(s_neg) == 0:
                    continue
                
                with pm.Model() as auc_model:
                    mu1 = pm.Normal("mu1", 0, 2)
                    mu0 = pm.Normal("mu0", 0, 2)
                    s1 = pm.HalfNormal("s1", 1)
                    s0 = pm.HalfNormal("s0", 1)
                    
                    pm.Normal("pos", mu1, s1, observed=s_pos)
                    pm.Normal("neg", mu0, s0, observed=s_neg)
                    
                    # AUC = Phi( (mu1 - mu0) / sqrt(s1^2 + s0^2) )
                    delta = pm.Deterministic("delta", (mu1 - mu0) / pm.math.sqrt(s1**2 + s0**2))
                    auc = pm.Deterministic("auc", 0.5 * (1 + pm.math.erf(delta / pm.math.sqrt(2))))
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        idata_auc = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=self.random_seed)
                
                if class_idx not in auc_results:
                    auc_results[class_idx] = {}
                
                auc_samples = idata_auc.posterior["auc"].values.reshape(-1)
                auc_results[class_idx][model] = {
                    'idata': idata_auc,
                    'auc_samples': auc_samples,
                    'auc_mean': np.mean(auc_samples),
                    'auc_std': np.std(auc_samples),
                    'auc_ci_lower': np.percentile(auc_samples, 5.5),
                    'auc_ci_upper': np.percentile(auc_samples, 94.5)
                }
        
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
        for model in model_predictions:
            model_predictions[model]['predictions'] = np.array(model_predictions[model]['predictions'])
            model_predictions[model]['probabilities'] = np.array(model_predictions[model]['probabilities'])
            model_predictions[model]['labels'] = np.array(model_predictions[model]['labels'])
        
        # Simple ensemble using equal weights (placeholder for proper stacking)
        if len(model_predictions) > 1:
            models = list(model_predictions.keys())
            all_probs = np.array([model_predictions[model]['probabilities'] for model in models])
            ensemble_probs = np.mean(all_probs, axis=0)
            ensemble_preds = np.argmax(ensemble_probs, axis=1)
            
            # Calculate ensemble performance
            ensemble_acc = accuracy_score(
                model_predictions[models[0]]['labels'], 
                ensemble_preds
            )
            
            return {
                'models': models,
                'ensemble_predictions': ensemble_preds,
                'ensemble_probabilities': ensemble_probs,
                'ensemble_accuracy': ensemble_acc,
                'individual_accuracies': {
                    model: accuracy_score(model_predictions[model]['labels'], 
                                        model_predictions[model]['predictions'])
                    for model in models
                }
            }
        
        return {}
    
    def create_visualizations(self, results: BayesianResults, data_dict: Dict[str, pd.DataFrame]):
        """Create comprehensive visualizations for all analyses."""
        print("Creating visualizations...")
        
        # 1. Hierarchical accuracy plots
        if results.accuracy_results:
            self._plot_hierarchical_accuracy(results.accuracy_results)
        
        # 2. Calibration plots
        if results.calibration_results:
            self._plot_calibration_analysis(results.calibration_results)
        
        # 3. AUC comparison plots
        if results.auc_results:
            self._plot_auc_comparison(results.auc_results)
        
        # 4. Model comparison heatmaps
        if results.accuracy_results:
            self._plot_model_comparison_heatmap(results.accuracy_results)
    
    def _plot_hierarchical_accuracy(self, results: Dict[str, Any]):
        """Plot hierarchical accuracy results."""
        if not results:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy forest plot
        models = results['models']
        means = results['accuracy_means']
        ci_lower = results['accuracy_ci_lower']
        ci_upper = results['accuracy_ci_upper']
        
        y_pos = np.arange(len(models))
        ax1.errorbar(means, y_pos, xerr=[means - ci_lower, ci_upper - means], 
                    fmt='o', capsize=5, capthick=2)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(models)
        ax1.set_xlabel('Accuracy')
        ax1.set_title('Hierarchical Accuracy Estimates\n(95% Credible Intervals)')
        ax1.grid(True, alpha=0.3)
        
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
                acc_results = results.accuracy_results.copy()
                for key in ['accuracy_samples', 'accuracy_means', 'accuracy_std', 
                           'accuracy_ci_lower', 'accuracy_ci_upper']:
                    if key in acc_results:
                        acc_results[key] = acc_results[key].tolist()
                del acc_results['idata']  # Can't serialize PyMC objects
                json.dump(acc_results, f, indent=2)
            
            if 'model_comparisons' in results.accuracy_results:
                results.accuracy_results['model_comparisons'].to_csv(
                    self.results_dir / 'model_comparisons.csv', index=False)
        
        # Save other results (simplified for JSON serialization)
        if results.skill_results:
            skill_simple = {k: v for k, v in results.skill_results.items() 
                           if k != 'idata' and k != 'loo'}
            with open(self.results_dir / 'skill_results.json', 'w') as f:
                json.dump(skill_simple, f, indent=2)
        
        # Save ensemble results
        if results.stacking_results:
            with open(self.results_dir / 'ensemble_results.json', 'w') as f:
                ensemble_simple = {k: v for k, v in results.stacking_results.items() 
                                 if not isinstance(v, np.ndarray)}
                json.dump(ensemble_simple, f, indent=2)
        
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
        self.create_visualizations(results, data_dict)
        
        # Save results
        self.save_results(results, data_dict)
        
        print("Bayesian analysis complete!")
        return results


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
