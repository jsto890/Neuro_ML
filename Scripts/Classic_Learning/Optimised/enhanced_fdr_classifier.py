#!/usr/bin/env python3
"""
Enhanced FDR Radiomics Classification Pipeline
==============================================

This pipeline implements FDR (False Discovery Rate) feature selection and compares
three approaches:
1. FDR-based feature selection
2. Current selection (MutualInfo + RFECV)
3. No feature selection (all features)

Author: P4P Team
Date: 2024
"""

import os
import sys
import json
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import RFECV, SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, matthews_corrcoef
)
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns

# FDR correction imports
try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Install with: pip install statsmodels")

# Advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("Warning: Bayesian optimization not available. Install with: pip install scikit-optimize")

from scipy.stats import skew, kurtosis
from scipy import stats

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class EnhancedFDRRadiomicsClassifier:
    """Enhanced radiomics classifier with FDR feature selection and comprehensive comparison."""
    
    def __init__(self, input_path, output_dir, random_state=42, binary_only=True, fdr_alpha=0.05):
        """
        Initialize the Enhanced FDR Radiomics Classifier.
        
        Args:
            input_path (str): Path to radiomics CSV file
            output_dir (str): Output directory for results
            random_state (int): Random seed for reproducibility
            binary_only (bool): If True, only use labels 0 and 1
            fdr_alpha (float): FDR significance level (default: 0.05)
        """
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.binary_only = binary_only
        self.fdr_alpha = fdr_alpha
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize preprocessing components
        self.variance_selector = VarianceThreshold(threshold=0.01)
        self.scaler = RobustScaler()
        
        # Initialize models for each approach
        self.models = {
            'fdr_selection': {'svm': None, 'ensemble': None},
            'current_selection': {'svm': None, 'ensemble': None},
            'no_selection': {'svm': None, 'ensemble': None}
        }
        
        # Initialize results storage
        self.results = {}
        self.feature_importance = {}
        self.feature_engineering_results = {}
        self.comparison_results = {}
        
        # Setup logging
        self.setup_logging()
        
        # Initialize data containers
        self.data = None
        self.X = None
        self.y = None
        self.subject_ids = None
        self.feature_names = None
        self.splits = None
        
        # Check FDR availability
        if not STATSMODELS_AVAILABLE:
            self.logger.warning("statsmodels not available. FDR selection will be skipped.")
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / 'enhanced_fdr_pipeline.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data(self):
        """Stage 1: Load and validate input data."""
        self.logger.info("Stage 1: Loading data...")
        
        try:
            if not os.path.exists(self.input_path):
                raise FileNotFoundError(f"Input file not found: {self.input_path}")
            
            self.data = pd.read_csv(self.input_path)
            self.logger.info(f"Loaded {len(self.data)} samples with {len(self.data.columns)} columns")
            
            # Remove diagnostic columns
            diagnostic_cols = [col for col in self.data.columns if col.startswith('diagnostics_')]
            if diagnostic_cols:
                self.data = self.data.drop(columns=diagnostic_cols)
                self.logger.info(f"Removed {len(diagnostic_cols)} diagnostic columns")
            
            # Validate required columns
            required_cols = ['subject_id', 'label']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Filter for binary classification if requested
            if self.binary_only:
                initial_count = len(self.data)
                self.data = self.data[self.data['label'].isin([0, 1])]
                final_count = len(self.data)
                self.logger.info(f"Filtered to binary classification: {initial_count} → {final_count} samples")
                
                if final_count == 0:
                    raise ValueError("No samples remaining after binary filtering")
                
                # Verify we only have binary labels
                unique_labels = self.data['label'].unique()
                if len(unique_labels) != 2 or not all(label in [0, 1] for label in unique_labels):
                    raise ValueError(f"Expected binary labels [0, 1], got: {unique_labels}")
            
            # Extract components
            self.subject_ids = self.data['subject_id'].values
            self.y = self.data['label'].values
            self.feature_names = [col for col in self.data.columns if col not in ['subject_id', 'label']]
            self.X = self.data[self.feature_names].values
            
            self.logger.info(f"Data shape: {self.X.shape}")
            self.logger.info(f"Labels: {np.unique(self.y)} (counts: {np.bincount(self.y)})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return False
    
    def split_data(self, test_size=0.2, val_size=0.2):
        """Stage 2: Train-validation-test split."""
        self.logger.info("Stage 2: Splitting data...")
        
        try:
            # First split: train+val vs test
            X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
                self.X, self.y, self.subject_ids, 
                test_size=test_size, 
                random_state=self.random_state,
                stratify=self.y
            )
            
            # Second split: train vs val
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
                X_temp, y_temp, ids_temp,
                test_size=val_size_adjusted,
                random_state=self.random_state,
                stratify=y_temp
            )
            
            # Store splits
            self.splits = {
                'train': (X_train, y_train, ids_train),
                'val': (X_val, y_val, ids_val),
                'test': (X_test, y_test, ids_test)
            }
            
            self.logger.info(f"Data splits - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {e}")
            return False
    
    def basic_preprocessing(self, X_train, X_val, X_test):
        """Apply basic preprocessing (variance thresholding and scaling)."""
        # Remove low variance features
        X_train_var = self.variance_selector.fit_transform(X_train)
        X_val_var = self.variance_selector.transform(X_val)
        X_test_var = self.variance_selector.transform(X_test)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_var)
        X_val_scaled = self.scaler.transform(X_val_var)
        X_test_scaled = self.scaler.transform(X_test_var)
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def fdr_feature_selection(self, X_train, y_train, X_val, X_test, feature_names):
        """FDR-based feature selection using multiple testing correction."""
        self.logger.info("Applying FDR-based feature selection...")
        
        if not STATSMODELS_AVAILABLE:
            self.logger.warning("statsmodels not available, skipping FDR selection")
            return X_train, X_val, X_test, feature_names
        
        try:
            # Calculate F-statistics and p-values for all features
            f_scores, p_values = f_classif(X_train, y_train)
            
            # Apply FDR correction (Benjamini-Hochberg method)
            rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                p_values, 
                alpha=self.fdr_alpha, 
                method='fdr_bh'
            )
            
            # Select features that pass FDR correction
            selected_indices = np.where(rejected)[0]
            selected_features = [feature_names[i] for i in selected_indices]
            
            # Apply selection
            X_train_selected = X_train[:, selected_indices]
            X_val_selected = X_val[:, selected_indices]
            X_test_selected = X_test[:, selected_indices]
            
            self.logger.info(f"FDR selection: {len(feature_names)} → {len(selected_features)} features")
            self.logger.info(f"FDR alpha: {self.fdr_alpha}, significant features: {len(selected_features)}")
            
            # Store FDR results
            self.feature_engineering_results['fdr_selection'] = {
                'n_features_original': len(feature_names),
                'n_features_selected': len(selected_features),
                'fdr_alpha': self.fdr_alpha,
                'significant_features': selected_features,
                'p_values_original': p_values.tolist(),
                'p_values_corrected': p_corrected.tolist(),
                'rejected': rejected.tolist(),
                'method': 'Benjamini-Hochberg FDR correction'
            }
            
            return X_train_selected, X_val_selected, X_test_selected, selected_features
            
        except Exception as e:
            self.logger.error(f"Error in FDR feature selection: {e}")
            return X_train, X_val, X_test, feature_names
    
    def current_feature_selection(self, X_train, y_train, X_val, X_test, feature_names):
        """Current feature selection method (MutualInfo + RFECV)."""
        self.logger.info("Applying current feature selection (MutualInfo + RFECV)...")
        
        try:
            # Mutual information selection
            k_best = min(50, X_train.shape[1] // 2)
            mi_selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
            X_train_mi = mi_selector.fit_transform(X_train, y_train)
            X_val_mi = mi_selector.transform(X_val)
            X_test_mi = mi_selector.transform(X_test)
            
            # Get selected feature names
            mi_mask = mi_selector.get_support()
            selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if mi_mask[i]]
            
            # RFECV selection
            estimator = LogisticRegression(random_state=self.random_state, max_iter=1000)
            rfecv = RFECV(
                estimator=estimator,
                step=1,
                cv=5,
                scoring='roc_auc',
                min_features_to_select=10,
                n_jobs=-1
            )
            
            X_train_final = rfecv.fit_transform(X_train_mi, y_train)
            X_val_final = rfecv.transform(X_val_mi)
            X_test_final = rfecv.transform(X_test_mi)
            
            # Get final feature names
            rfecv_mask = rfecv.get_support()
            final_feature_names = [selected_feature_names[i] for i in range(len(selected_feature_names)) if rfecv_mask[i]]
            
            self.logger.info(f"Current selection: {len(feature_names)} → {len(final_feature_names)} features")
            
            # Store current selection results
            self.feature_engineering_results['current_selection'] = {
                'n_features_original': len(feature_names),
                'n_features_selected': len(final_feature_names),
                'method': 'MutualInfo + RFECV',
                'mi_k': k_best,
                'rfecv_cv': 5,
                'rfecv_scoring': 'roc_auc',
                'selected_features': final_feature_names
            }
            
            return X_train_final, X_val_final, X_test_final, final_feature_names
            
        except Exception as e:
            self.logger.error(f"Error in current feature selection: {e}")
            return X_train, X_val, X_test, feature_names
    
    def train_models_for_approach(self, approach_name, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train SVM and ensemble models for a specific feature selection approach."""
        self.logger.info(f"Training models for {approach_name}...")
        
        try:
            # Train SVM
            svm = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=self.random_state,
                max_iter=10000,
                tol=1e-3
            )
            svm.fit(X_train, y_train)
            
            # Train ensemble (simplified)
            base_models = {
                'svm_linear': SVC(kernel='linear', probability=True, random_state=self.random_state, max_iter=10000),
                'logistic': LogisticRegression(random_state=self.random_state, max_iter=1000)
            }
            
            ensemble = VotingClassifier(
                estimators=[(name, model) for name, model in base_models.items()],
                voting='soft'
            )
            ensemble.fit(X_train, y_train)
            
            # Store models
            self.models[approach_name]['svm'] = svm
            self.models[approach_name]['ensemble'] = ensemble
            
            # Evaluate models
            svm_results = self._evaluate_model(svm, f"{approach_name}_SVM", X_train, y_train, X_val, y_val, X_test, y_test)
            ensemble_results = self._evaluate_model(ensemble, f"{approach_name}_Ensemble", X_train, y_train, X_val, y_val, X_test, y_test)
            
            self.results[approach_name] = {
                'svm': svm_results,
                'ensemble': ensemble_results
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training models for {approach_name}: {e}")
            return False
    
    def _evaluate_model(self, model, model_name, X_train, y_train, X_val, y_val, X_test, y_test):
        """Evaluate a model on all splits."""
        results = {}
        
        for split_name, (X_split, y_split) in [('train', (X_train, y_train)), 
                                              ('val', (X_val, y_val)), 
                                              ('test', (X_test, y_test))]:
            try:
                y_pred = model.predict(X_split)
                y_pred_proba = model.predict_proba(X_split)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_split, y_pred)
                precision = precision_score(y_split, y_pred, average='weighted')
                recall = recall_score(y_split, y_pred, average='weighted')
                f1 = f1_score(y_split, y_pred, average='weighted')
                auc = roc_auc_score(y_split, y_pred_proba)
                mcc = matthews_corrcoef(y_split, y_pred)
                
                results[split_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'mcc': mcc
                }
                
                self.logger.info(f"{model_name} {split_name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name} on {split_name}: {e}")
                results[split_name] = None
        
        return results
    
    def compare_approaches(self):
        """Compare all three feature selection approaches."""
        self.logger.info("Comparing feature selection approaches...")
        
        comparison = {}
        
        for approach in ['fdr_selection', 'current_selection', 'no_selection']:
            if approach in self.results:
                comparison[approach] = {
                    'svm_test': self.results[approach]['svm']['test'] if self.results[approach]['svm']['test'] else {},
                    'ensemble_test': self.results[approach]['ensemble']['test'] if self.results[approach]['ensemble']['test'] else {},
                    'n_features': len(self.feature_engineering_results.get(approach, {}).get('selected_features', self.feature_names))
                }
        
        self.comparison_results = comparison
        
        # Create comparison summary
        summary = []
        for approach, results in comparison.items():
            svm_metrics = results['svm_test']
            ensemble_metrics = results['ensemble_test']
            
            summary.append({
                'approach': approach,
                'n_features': results['n_features'],
                'svm_accuracy': svm_metrics.get('accuracy', 0),
                'svm_auc': svm_metrics.get('auc', 0),
                'svm_mcc': svm_metrics.get('mcc', 0),
                'ensemble_accuracy': ensemble_metrics.get('accuracy', 0),
                'ensemble_auc': ensemble_metrics.get('auc', 0),
                'ensemble_mcc': ensemble_metrics.get('mcc', 0)
            })
        
        self.logger.info("Comparison Summary:")
        for row in summary:
            self.logger.info(f"{row['approach']}: {row['n_features']} features, "
                           f"SVM (Acc: {row['svm_accuracy']:.3f}, AUC: {row['svm_auc']:.3f}, MCC: {row['svm_mcc']:.3f}), "
                           f"Ensemble (Acc: {row['ensemble_accuracy']:.3f}, AUC: {row['ensemble_auc']:.3f}, MCC: {row['ensemble_mcc']:.3f})")
        
        return summary
    
    def save_results(self):
        """Save all results and comparison."""
        self.logger.info("Saving results...")
        
        try:
            # Save models
            for approach, models in self.models.items():
                for model_type, model in models.items():
                    if model is not None:
                        filename = f"{approach}_{model_type}_model.pkl"
                        with open(self.output_dir / filename, 'wb') as f:
                            pickle.dump(model, f)
            
            # Save scaler
            with open(self.output_dir / 'scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save feature engineering results
            with open(self.output_dir / 'feature_engineering_results.json', 'w') as f:
                json.dump(self.feature_engineering_results, f, indent=2, cls=NumpyEncoder)
            
            # Save comparison results
            with open(self.output_dir / 'comparison_results.json', 'w') as f:
                json.dump(self.comparison_results, f, indent=2, cls=NumpyEncoder)
            
            # Save detailed results
            with open(self.output_dir / 'detailed_results.json', 'w') as f:
                json.dump(self.results, f, indent=2, cls=NumpyEncoder)
            
            # Create comparison report
            self._create_comparison_report()
            
            self.logger.info("All results saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            return False
    
    def _create_comparison_report(self):
        """Create a detailed comparison report."""
        report_file = self.output_dir / 'comparison_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("ENHANCED FDR RADIOMICS CLASSIFICATION - COMPARISON REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Input File: {self.input_path}\n")
            f.write(f"FDR Alpha: {self.fdr_alpha}\n\n")
            
            f.write("FEATURE SELECTION APPROACHES COMPARED:\n")
            f.write("-" * 40 + "\n")
            f.write("1. FDR Selection: False Discovery Rate correction using Benjamini-Hochberg method\n")
            f.write("2. Current Selection: Mutual Information + RFECV\n")
            f.write("3. No Selection: All features after basic preprocessing\n\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 20 + "\n")
            
            for approach, results in self.comparison_results.items():
                f.write(f"\n{approach.upper().replace('_', ' ')}:\n")
                f.write(f"  Features: {results['n_features']}\n")
                
                if results['svm_test']:
                    svm = results['svm_test']
                    f.write(f"  SVM Test Results:\n")
                    f.write(f"    Accuracy: {svm['accuracy']:.4f}\n")
                    f.write(f"    AUC: {svm['auc']:.4f}\n")
                    f.write(f"    MCC: {svm['mcc']:.4f}\n")
                    f.write(f"    Precision: {svm['precision']:.4f}\n")
                    f.write(f"    Recall: {svm['recall']:.4f}\n")
                    f.write(f"    F1: {svm['f1']:.4f}\n")
                
                if results['ensemble_test']:
                    ensemble = results['ensemble_test']
                    f.write(f"  Ensemble Test Results:\n")
                    f.write(f"    Accuracy: {ensemble['accuracy']:.4f}\n")
                    f.write(f"    AUC: {ensemble['auc']:.4f}\n")
                    f.write(f"    MCC: {ensemble['mcc']:.4f}\n")
                    f.write(f"    Precision: {ensemble['precision']:.4f}\n")
                    f.write(f"    Recall: {ensemble['recall']:.4f}\n")
                    f.write(f"    F1: {ensemble['f1']:.4f}\n")
            
            f.write("\nRECOMMENDATIONS:\n")
            f.write("-" * 15 + "\n")
            f.write("Based on the comparison results, the best approach is typically the one with:\n")
            f.write("1. Highest MCC (Matthews Correlation Coefficient)\n")
            f.write("2. Good balance between accuracy and feature count\n")
            f.write("3. Stable performance across train/val/test splits\n")
    
    def run_complete_pipeline(self):
        """Run the complete enhanced FDR pipeline with comparison."""
        self.logger.info("Starting Enhanced FDR Radiomics Classification Pipeline")
        
        stages = [
            ("Data Loading", self.load_data),
            ("Data Splitting", self.split_data)
        ]
        
        for stage_name, stage_func in stages:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Starting {stage_name}")
            self.logger.info(f"{'='*50}")
            
            if not stage_func():
                self.logger.error(f"Pipeline failed at {stage_name}")
                return False
        
        # Get basic splits
        X_train, y_train, _ = self.splits['train']
        X_val, y_val, _ = self.splits['val']
        X_test, y_test, _ = self.splits['test']
        
        # Apply basic preprocessing
        X_train_processed, X_val_processed, X_test_processed = self.basic_preprocessing(
            X_train, X_val, X_test
        )
        
        # Approach 1: FDR Selection
        if STATSMODELS_AVAILABLE:
            self.logger.info("\n" + "="*50)
            self.logger.info("APPROACH 1: FDR FEATURE SELECTION")
            self.logger.info("="*50)
            
            X_train_fdr, X_val_fdr, X_test_fdr, features_fdr = self.fdr_feature_selection(
                X_train_processed, y_train, X_val_processed, X_test_processed, self.feature_names
            )
            
            self.train_models_for_approach('fdr_selection', X_train_fdr, y_train, X_val_fdr, y_val, X_test_fdr, y_test)
        
        # Approach 2: Current Selection
        self.logger.info("\n" + "="*50)
        self.logger.info("APPROACH 2: CURRENT FEATURE SELECTION")
        self.logger.info("="*50)
        
        X_train_current, X_val_current, X_test_current, features_current = self.current_feature_selection(
            X_train_processed, y_train, X_val_processed, X_test_processed, self.feature_names
        )
        
        self.train_models_for_approach('current_selection', X_train_current, y_train, X_val_current, y_val, X_test_current, y_test)
        
        # Approach 3: No Selection
        self.logger.info("\n" + "="*50)
        self.logger.info("APPROACH 3: NO FEATURE SELECTION")
        self.logger.info("="*50)
        
        self.feature_engineering_results['no_selection'] = {
            'n_features_original': len(self.feature_names),
            'n_features_selected': X_train_processed.shape[1],
            'method': 'No feature selection (all features)',
            'selected_features': self.feature_names
        }
        
        self.train_models_for_approach('no_selection', X_train_processed, y_train, X_val_processed, y_val, X_test_processed, y_test)
        
        # Compare approaches
        self.logger.info("\n" + "="*50)
        self.logger.info("COMPARING APPROACHES")
        self.logger.info("="*50)
        
        comparison_summary = self.compare_approaches()
        
        # Save results
        self.save_results()
        
        self.logger.info("\n" + "="*50)
        self.logger.info("ENHANCED FDR PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("="*50)
        
        return True

def main():
    """Main function to run the enhanced FDR pipeline."""
    parser = argparse.ArgumentParser(description='Enhanced FDR Radiomics Classification Pipeline')
    parser.add_argument('--input', type=str, required=True, help='Path to radiomics CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output directory for results')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--binary_only', action='store_true', help='Use only binary classification')
    parser.add_argument('--fdr_alpha', type=float, default=0.05, help='FDR significance level')
    
    args = parser.parse_args()
    
    # Create classifier and run pipeline
    classifier = EnhancedFDRRadiomicsClassifier(
        input_path=args.input,
        output_dir=args.output,
        random_state=args.random_state,
        binary_only=args.binary_only,
        fdr_alpha=args.fdr_alpha
    )
    
    success = classifier.run_complete_pipeline()
    
    if success:
        print("\n" + "="*60)
        print("Enhanced FDR pipeline completed successfully!")
        print(f"Results saved to: {args.output}")
        print("="*60)
        
        print("\nGenerated files:")
        print("  • comparison_report.txt - Detailed comparison of all approaches")
        print("  • comparison_results.json - Comparison results")
        print("  • detailed_results.json - Detailed results for each approach")
        print("  • feature_engineering_results.json - Feature engineering details")
        print("  • [approach]_[model]_model.pkl - Trained models")
        print("  • scaler.pkl - Feature scaler")
        print("  • enhanced_fdr_pipeline.log - Execution log")
        
        print("\nKey Features:")
        print("  • FDR-based feature selection with Benjamini-Hochberg correction")
        print("  • Comparison of three approaches: FDR, Current, No Selection")
        print("  • Comprehensive evaluation with MCC, AUC, and other metrics")
        print("  • Detailed comparison report and analysis")
    else:
        print("\n❌ Pipeline failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 