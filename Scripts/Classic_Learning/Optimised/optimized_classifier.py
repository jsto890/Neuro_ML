"""
Optimized Radiomics Classification Pipeline
==========================================

This optimized version focuses on:
- SVM as primary model with fine-tuned hyperparameters
- Advanced feature engineering based on cross-model importance
- Feature selection optimization
- Ensemble methods with SVM as base
- Clinical interpretability and robustness
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
    roc_curve, precision_recall_curve
)
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns

# Advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("Warning: Bayesian optimization not available. Install with: pip install scikit-optimize")

from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
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

# Stacking Ensemble class
class StackingEnsemble:
    def __init__(self, base_models, meta_learner, meta_feature_names):
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.meta_feature_names = meta_feature_names
    
    def predict_proba(self, X):
        # Get predictions from base models
        base_probs = []
        for name, model in self.base_models.items():
            prob = model.predict_proba(X)[:, 1]
            base_probs.append(prob)
        
        # Create meta-features
        meta_features = np.column_stack(base_probs)
        
        # Get meta-learner predictions
        meta_probs = self.meta_learner.predict_proba(meta_features)
        return meta_probs
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def __str__(self):
        return f"StackingEnsemble(base_models={list(self.base_models.keys())}, meta_learner={type(self.meta_learner).__name__})"

class OptimizedRadiomicsClassifier:
    """Optimized radiomics classifier focusing on SVM with advanced feature engineering."""
    
    def __init__(self, input_path, output_dir, random_state=42, binary_only=True):
        """
        Initialize the Optimized Radiomics Classifier.
        
        Args:
            input_path (str): Path to radiomics CSV file
            output_dir (str): Output directory for results
            random_state (int): Random seed for reproducibility
            binary_only (bool): If True, only use labels 0 and 1
        """
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.binary_only = binary_only
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.svm_model = None
        self.ensemble_model = None
        self.feature_importance = {}
        self.results = {}
        self.feature_engineering_results = {}
        
        # Setup logging
        self.setup_logging()
        
        # Initialize data containers
        self.data = None
        self.X = None
        self.y = None
        self.subject_ids = None
        self.feature_names = None
        self.engineered_features = None
        self.splits = None
        
        # Define top features based on cross-model analysis
        self.top_features = [
            'original_glrlm_RunLengthNonUniformity',
            'original_gldm_DependenceVariance', 
            'original_firstorder_Kurtosis',
            'original_ngtdm_Busyness',
            'original_glrlm_LongRunEmphasis',
            'original_firstorder_RobustMeanAbsoluteDeviation',
            'original_firstorder_Variance',
            'original_firstorder_Mean',
            'original_firstorder_Minimum',
            'original_gldm_DependenceNonUniformity',
            'original_glrlm_LongRunLowGrayLevelEmphasis',
            'original_gldm_LargeDependenceEmphasis',
            'original_glrlm_RunVariance',
            'original_glszm_ZoneEntropy',
            'original_firstorder_MeanAbsoluteDeviation'
        ]
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / 'optimized_pipeline.log'
        
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
        """Stage 0: Load and validate input data."""
        self.logger.info("Stage 0: Loading data...")
        
        try:
            # Check if file exists and is readable
            if not os.path.exists(self.input_path):
                raise FileNotFoundError(f"Input file not found: {self.input_path}")
            
            self.data = pd.read_csv(self.input_path)
            self.logger.info(f"Loaded {len(self.data)} samples with {len(self.data.columns)} columns")
            
            # Validate data quality
            if len(self.data) < 10:
                raise ValueError(f"Insufficient data: only {len(self.data)} samples")
            
            if len(self.data.columns) < 5:
                raise ValueError(f"Insufficient features: only {len(self.data.columns)} columns")
            
            # Check for missing values
            missing_counts = self.data.isnull().sum()
            if missing_counts.sum() > 0:
                self.logger.warning(f"Found {missing_counts.sum()} missing values")
                self.logger.warning(f"Columns with missing values: {missing_counts[missing_counts > 0].to_dict()}")
            
            # Remove diagnostic columns (PyRadiomics metadata)
            diagnostic_cols = [col for col in self.data.columns if col.startswith('diagnostics_')]
            if diagnostic_cols:
                self.data = self.data.drop(columns=diagnostic_cols)
                self.logger.info(f"Removed {len(diagnostic_cols)} diagnostic columns")
            
            # Validate required columns
            required_cols = ['subject_id', 'label']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Extract features and labels
            feature_cols = [col for col in self.data.columns if col not in ['subject_id', 'label']]
            if len(feature_cols) < 10:
                raise ValueError(f"Insufficient feature columns: only {len(feature_cols)} features")
            
            self.X = self.data[feature_cols].values
            self.y = self.data['label'].values
            self.subject_ids = self.data['subject_id'].values
            self.feature_names = feature_cols
            
            # Validate labels
            unique_labels = np.unique(self.y)
            self.logger.info(f"Unique labels: {unique_labels}")
            
            if self.binary_only:
                # Filter to binary classification
                binary_mask = np.isin(self.y, [0, 1])
                if np.sum(binary_mask) < len(self.y):
                    self.X = self.X[binary_mask]
                    self.y = self.y[binary_mask]
                    self.subject_ids = self.subject_ids[binary_mask]
                    self.logger.info(f"Filtered to binary classification: {len(self.data)} → {len(self.y)} samples")
                
                # Check class balance
                class_counts = np.bincount(self.y)
                if len(class_counts) != 2:
                    raise ValueError(f"Expected 2 classes, found {len(class_counts)}")
                
                min_class_size = min(class_counts)
                if min_class_size < 10:
                    raise ValueError(f"Class imbalance too severe: smallest class has {min_class_size} samples")
                
                imbalance_ratio = max(class_counts) / min_class_size
                if imbalance_ratio > 10:
                    self.logger.warning(f"Severe class imbalance: ratio = {imbalance_ratio:.2f}")
            
            self.logger.info(f"Data shape: {self.X.shape}")
            self.logger.info(f"Labels: {unique_labels} (counts: {[np.sum(self.y == label) for label in unique_labels]})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in data loading: {e}")
            return False
    
    def advanced_feature_engineering(self):
        """Stage 1: Simplified advanced feature engineering to prevent convergence issues."""
        self.logger.info("Stage 1: Advanced feature engineering...")
        
        try:
            # Convert X to DataFrame if it's not already
            if not isinstance(self.X, pd.DataFrame):
                self.X = pd.DataFrame(self.X, columns=self.feature_names)
            
            # 1. Variance thresholding
            variance_selector = VarianceThreshold(threshold=0.01)
            X_var_selected = variance_selector.fit_transform(self.X)
            selected_features = self.X.columns[variance_selector.get_support()].tolist()
            
            self.logger.info(f"After variance threshold: {X_var_selected.shape}")
            
            # Check if we have enough features after variance thresholding
            if len(selected_features) < 5:
                self.logger.warning(f"Too few features after variance thresholding: {len(selected_features)}")
                # Use original features if too few remain
                X_var_selected = self.X.values
                selected_features = self.feature_names
            
            # 2. Select top features based on cross-model analysis
            self.top_features = [
                'original_glrlm_RunLengthNonUniformity',
                'original_gldm_DependenceVariance', 
                'original_firstorder_Kurtosis',
                'original_ngtdm_Busyness',
                'original_glrlm_LongRunEmphasis',
                'original_firstorder_RobustMeanAbsoluteDeviation',
                'original_firstorder_Variance',
                'original_firstorder_Mean',
                'original_firstorder_Minimum',
                'original_gldm_DependenceNonUniformity',
                'original_glrlm_LongRunLowGrayLevelEmphasis',
                'original_gldm_LargeDependenceEmphasis',
                'original_glrlm_RunVariance',
                'original_glszm_ZoneEntropy',
                'original_firstorder_MeanAbsoluteDeviation'
            ]
            
            # Filter to available features
            available_top_features = [f for f in self.top_features if f in selected_features]
            self.logger.info(f"Found {len(available_top_features)} of {len(self.top_features)} top features")
            
            # 3. Simplified Feature Engineering (reduced complexity)
            engineered_features = []
            feature_names = []
            
            # Limit polynomial features to prevent memory explosion
            max_poly_features = min(8, len(available_top_features))
            top_poly_features = available_top_features[:max_poly_features]
            top_poly_indices = [selected_features.index(f) for f in top_poly_features]
            X_top_poly = X_var_selected[:, top_poly_indices]
            
            # 3a. Polynomial features (2nd degree only, reduced complexity)
            try:
                poly_2 = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
                X_poly_2 = poly_2.fit_transform(X_top_poly)
                
                # Limit polynomial features to prevent explosion
                max_features = min(20, X_poly_2.shape[1] - X_top_poly.shape[1])
                if X_poly_2.shape[1] > X_top_poly.shape[1] + max_features:
                    # Select top polynomial features by variance
                    poly_features = X_poly_2[:, X_top_poly.shape[1]:]
                    variances = np.var(poly_features, axis=0)
                    top_indices = np.argsort(variances)[-max_features:]
                    poly_features = poly_features[:, top_indices]
                    poly_2_names = [f"poly2_{i}" for i in top_indices]
                else:
                    poly_features = X_poly_2[:, X_top_poly.shape[1]:]
                    poly_2_names = [f"poly2_{i}" for i in range(poly_features.shape[1])]
                
                engineered_features.append(poly_features)
                feature_names.extend(poly_2_names)
                
            except Exception as e:
                self.logger.warning(f"Polynomial feature generation failed: {e}")
            
            # 3b. Simplified family-based interaction features (only key combinations)
            try:
                family_groups = {
                    'firstorder': [f for f in available_top_features if 'firstorder' in f],
                    'texture': [f for f in available_top_features if any(x in f for x in ['glrlm', 'gldm', 'glszm', 'ngtdm'])]
                }
                
                family_interactions = []
                family_interaction_names = []
                
                # Only create interactions between firstorder and texture families
                if family_groups['firstorder'] and family_groups['texture']:
                    indices1 = [selected_features.index(f) for f in family_groups['firstorder']]
                    indices2 = [selected_features.index(f) for f in family_groups['texture']]
                    
                    X_firstorder = X_var_selected[:, indices1]
                    X_texture = X_var_selected[:, indices2]
                    
                    # Create only a few key interactions (first feature from each family)
                    if len(X_firstorder) > 0 and len(X_texture) > 0:
                        interaction = X_firstorder[:, 0] * X_texture[:, 0]
                        family_interactions.append(interaction)
                        family_interaction_names.append("firstorder_texture_interaction")
                
                if family_interactions:
                    family_interactions = np.column_stack(family_interactions)
                    engineered_features.append(family_interactions)
                    feature_names.extend(family_interaction_names)
                    
            except Exception as e:
                self.logger.warning(f"Family interaction generation failed: {e}")
            
            # 3c. Simplified statistical summary features
            try:
                # Only for texture features
                texture_features = [f for f in available_top_features if any(x in f for x in ['glrlm', 'gldm', 'glszm', 'ngtdm'])]
                if len(texture_features) > 1:
                    texture_indices = [selected_features.index(f) for f in texture_features]
                    X_texture = X_var_selected[:, texture_indices]
                    
                    # Only mean and std (reduced from 7 features)
                    texture_mean = np.mean(X_texture, axis=1)
                    texture_std = np.std(X_texture, axis=1)
                    
                    engineered_features.append(np.column_stack([texture_mean, texture_std]))
                    feature_names.extend(['texture_mean', 'texture_std'])
                    
            except Exception as e:
                self.logger.warning(f"Statistical feature generation failed: {e}")
            
            # Combine all engineered features
            if engineered_features:
                X_engineered = np.column_stack(engineered_features)
                X_combined = np.column_stack([X_var_selected, X_engineered])
                self.feature_names = selected_features + feature_names
            else:
                X_combined = X_var_selected
                self.feature_names = selected_features
            
            self.logger.info(f"After feature engineering: {X_combined.shape}")
            self.logger.info(f"Feature types: {len(selected_features)} original, {len(feature_names)} engineered")
            
            # 4. Remove outliers to improve convergence
            try:
                z_scores = stats.zscore(X_combined, axis=0)
                outlier_mask = np.all(np.abs(z_scores) < 3, axis=1)  # Remove samples with z-score > 3
                X_cleaned = X_combined[outlier_mask]
                y_cleaned = self.y[outlier_mask]
                
                if np.sum(outlier_mask) < len(self.y):
                    self.logger.info(f"Removed {len(self.y) - np.sum(outlier_mask)} outliers")
                    self.X = X_cleaned
                    self.y = y_cleaned
                    # Update subject_ids to match the cleaned data
                    self.subject_ids = self.subject_ids[outlier_mask]
                else:
                    self.X = X_combined
                    
            except Exception as e:
                self.logger.warning(f"Outlier removal failed: {e}, using original data")
                self.X = X_combined
            
            # 5. Scaling with StandardScaler for better convergence
            try:
                self.scaler = StandardScaler()
                self.X = self.scaler.fit_transform(self.X)
                self.logger.info("Features scaled using StandardScaler")
            except Exception as e:
                self.logger.warning(f"Scaling failed: {e}, using unscaled data")
            
            # Clean up memory
            del X_var_selected, engineered_features, feature_names
            if 'X_combined' in locals():
                del X_combined
            if 'X_cleaned' in locals():
                del X_cleaned
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in advanced feature engineering: {e}")
            return False
    
    def optimized_feature_selection(self):
        """Stage 2: Simplified feature selection to prevent convergence issues."""
        self.logger.info("Stage 2: Optimized feature selection...")
        
        try:
            # Preliminary feature reduction to handle large feature set
            if self.X.shape[1] > 50:  # Reduced threshold from 100 to 50
                self.logger.info(f"Large feature set detected ({self.X.shape[1]} features), applying preliminary reduction...")
                
                # Use mutual information for initial feature selection
                from sklearn.feature_selection import SelectKBest, mutual_info_classif
                k_best = min(50, self.X.shape[1] // 2)  # Reduced from 100 to 50
                selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
                X_reduced = selector.fit_transform(self.X, self.y)
                selected_indices = selector.get_support()
                self.feature_names = [self.feature_names[i] for i in range(len(self.feature_names)) if selected_indices[i]]
                self.X = X_reduced
                
                self.logger.info(f"Preliminary reduction: {self.X.shape[1]} features selected")
            
            # Use simpler feature selection instead of RFECV
            if self.X.shape[1] > 30:
                self.logger.info("Applying additional feature selection...")
                
                # Use SelectKBest with f_classif for final selection
                from sklearn.feature_selection import SelectKBest, f_classif
                k_final = min(30, self.X.shape[1])
                selector_final = SelectKBest(score_func=f_classif, k=k_final)
                X_final = selector_final.fit_transform(self.X, self.y)
                selected_indices_final = selector_final.get_support()
                self.feature_names = [self.feature_names[i] for i in range(len(self.feature_names)) if selected_indices_final[i]]
                self.X = X_final
                
                self.logger.info(f"Final feature selection: {self.X.shape[1]} features selected")
            
            # Store feature selection results
            self.feature_engineering_results['feature_selection'] = {
                'n_features': self.X.shape[1],
                'selected_features': self.feature_names,
                'method': 'SelectKBest + f_classif'
            }
            
            self.logger.info(f"Feature selection completed: {self.X.shape[1]} features")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            return False
    
    def split_data(self, test_size=0.2, val_size=0.2):
        """Stage 3: Train-validation-test split."""
        self.logger.info("Stage 3: Splitting data...")
        
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
            
            self.splits = {
                'train': (X_train, y_train, ids_train),
                'val': (X_val, y_val, ids_val),
                'test': (X_test, y_test, ids_test)
            }
            
            self.logger.info(f"Train: {X_train.shape[0]} samples")
            self.logger.info(f"Validation: {X_val.shape[0]} samples")
            self.logger.info(f"Test: {X_test.shape[0]} samples")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in data splitting: {e}")
            return False
    
    def optimize_svm_hyperparameters(self):
        """Stage 4: Bayesian optimization of SVM hyperparameters."""
        self.logger.info("Stage 4: Optimizing SVM hyperparameters with Bayesian optimization...")
        
        try:
            X_train, y_train, _ = self.splits['train']
            
            # Create base SVM
            base_svm = SVC(probability=True, random_state=self.random_state)
            
            if BAYESIAN_AVAILABLE:
                # Bayesian optimization search space
                search_spaces = {
                    'C': Real(0.01, 100.0, prior='log-uniform'),
                    'gamma': Categorical(['scale', 'auto']),
                    'kernel': Categorical(['linear', 'rbf', 'poly']),
                    'degree': Integer(2, 3),  # for poly kernel
                    'class_weight': Categorical(['balanced']),
                    'max_iter': Integer(80000, 100000),  # Increased max iterations to 80k-100k
                    'tol': Real(1e-5, 1e-3, prior='log-uniform')  # Reduced tolerance (smaller values)
                }
                
                # Bayesian optimization
                bayes_search = BayesSearchCV(
                    estimator=base_svm,
                    search_spaces=search_spaces,
                    n_iter=50,  # Number of iterations for Bayesian optimization
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1,
                    random_state=self.random_state
                )
                
                bayes_search.fit(X_train, y_train)
                
                # Store best model and parameters
                self.svm_model = bayes_search.best_estimator_
                best_params = bayes_search.best_params_
                
                # Store optimization results
                self.feature_engineering_results['svm_optimization'] = {
                    'method': 'Bayesian Optimization',
                    'best_params': best_params,
                    'best_cv_score': bayes_search.best_score_,
                    'n_iterations': 50,
                    'cv_results': {
                        'mean_fit_time': bayes_search.cv_results_['mean_fit_time'].tolist(),
                        'mean_test_score': bayes_search.cv_results_['mean_test_score'].tolist(),
                        'rank_test_score': bayes_search.cv_results_['rank_test_score'].tolist()
                    }
                }
                
                self.logger.info(f"Bayesian optimization completed")
                self.logger.info(f"Best SVM parameters: {best_params}")
                self.logger.info(f"Best CV score: {bayes_search.best_score_:.4f}")
                
            else:
                # Fallback to simplified grid search
                self.logger.info("Bayesian optimization not available, using grid search...")
                
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'gamma': ['scale'],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'degree': [2, 3],  # for poly kernel
                    'class_weight': ['balanced'],
                    'max_iter': [80000],  # Increased max iterations to 80k
                    'tol': [1e-4]  # Reduced tolerance
                }
                
                grid_search = GridSearchCV(
                    estimator=base_svm,
                    param_grid=param_grid,
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train)
                
                # Store best model and parameters
                self.svm_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
                # Store optimization results
                self.feature_engineering_results['svm_optimization'] = {
                    'method': 'Grid Search (Fallback)',
                    'best_params': best_params,
                    'best_cv_score': grid_search.best_score_,
                    'cv_results': {
                        'mean_fit_time': grid_search.cv_results_['mean_fit_time'].tolist(),
                        'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                        'rank_test_score': grid_search.cv_results_['rank_test_score'].tolist()
                    }
                }
                
                self.logger.info(f"Grid search completed")
                self.logger.info(f"Best SVM parameters: {best_params}")
                self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in SVM optimization: {e}")
            return False
    
    def optimize_advanced_models(self):
        """Stage 4.5: Bayesian optimization of XGBoost and LightGBM hyperparameters."""
        self.logger.info("Stage 4.5: Optimizing XGBoost and LightGBM hyperparameters...")
        
        try:
            X_train, y_train, _ = self.splits['train']
            self.advanced_models = {}
            
            # Check if we have enough data for optimization
            if len(X_train) < 50:
                self.logger.warning("Insufficient training data for Bayesian optimization, using default parameters")
                return self._create_default_advanced_models()
            
            # Optimize XGBoost if available
            if XGBOOST_AVAILABLE and BAYESIAN_AVAILABLE:
                self.logger.info("Optimizing XGBoost hyperparameters...")
                
                try:
                    base_xgb = xgb.XGBClassifier(
                        random_state=self.random_state,
                        eval_metric='logloss'
                    )
                    
                    xgb_search_spaces = {
                        'n_estimators': Integer(50, 300),
                        'max_depth': Integer(3, 10),
                        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                        'subsample': Real(0.6, 1.0),
                        'colsample_bytree': Real(0.6, 1.0),
                        'reg_alpha': Real(0.001, 10.0, prior='log-uniform'),
                        'reg_lambda': Real(0.001, 10.0, prior='log-uniform'),
                        'min_child_weight': Integer(1, 10)
                    }
                    
                    # Reduce iterations for faster optimization
                    n_iter = min(20, len(X_train) // 10)  # Adaptive iterations
                    
                    xgb_bayes = BayesSearchCV(
                        estimator=base_xgb,
                        search_spaces=xgb_search_spaces,
                        n_iter=n_iter,
                        cv=min(5, len(X_train) // 10),  # Adaptive CV folds
                        scoring='accuracy',
                        n_jobs=-1,
                        verbose=1,
                        random_state=self.random_state
                    )
                    
                    xgb_bayes.fit(X_train, y_train)
                    self.advanced_models['xgboost'] = xgb_bayes.best_estimator_
                    
                    self.feature_engineering_results['xgboost_optimization'] = {
                        'method': 'Bayesian Optimization',
                        'best_params': xgb_bayes.best_params_,
                        'best_cv_score': xgb_bayes.best_score_,
                        'n_iterations': n_iter
                    }
                    
                    self.logger.info(f"XGBoost optimization completed - Best CV score: {xgb_bayes.best_score_:.4f}")
                    
                except Exception as e:
                    self.logger.warning(f"XGBoost optimization failed: {e}, using default parameters")
                    self.advanced_models['xgboost'] = xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=self.random_state,
                        eval_metric='logloss'
                    )
            
            # Optimize LightGBM if available
            if LIGHTGBM_AVAILABLE and BAYESIAN_AVAILABLE:
                self.logger.info("Optimizing LightGBM hyperparameters...")
                
                try:
                    base_lgb = lgb.LGBMClassifier(
                        random_state=self.random_state,
                        verbose=-1
                    )
                    
                    lgb_search_spaces = {
                        'n_estimators': Integer(50, 300),
                        'max_depth': Integer(3, 10),
                        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                        'subsample': Real(0.6, 1.0),
                        'colsample_bytree': Real(0.6, 1.0),
                        'reg_alpha': Real(0.001, 10.0, prior='log-uniform'),
                        'reg_lambda': Real(0.001, 10.0, prior='log-uniform'),
                        'min_child_samples': Integer(10, 100),
                        'num_leaves': Integer(20, 100)
                    }
                    
                    # Reduce iterations for faster optimization
                    n_iter = min(20, len(X_train) // 10)  # Adaptive iterations
                    
                    lgb_bayes = BayesSearchCV(
                        estimator=base_lgb,
                        search_spaces=lgb_search_spaces,
                        n_iter=n_iter,
                        cv=min(5, len(X_train) // 10),  # Adaptive CV folds
                        scoring='accuracy',
                        n_jobs=-1,
                        verbose=1,
                        random_state=self.random_state
                    )
                    
                    lgb_bayes.fit(X_train, y_train)
                    self.advanced_models['lightgbm'] = lgb_bayes.best_estimator_
                    
                    self.feature_engineering_results['lightgbm_optimization'] = {
                        'method': 'Bayesian Optimization',
                        'best_params': lgb_bayes.best_params_,
                        'best_cv_score': lgb_bayes.best_score_,
                        'n_iterations': n_iter
                    }
                    
                    self.logger.info(f"LightGBM optimization completed - Best CV score: {lgb_bayes.best_score_:.4f}")
                    
                except Exception as e:
                    self.logger.warning(f"LightGBM optimization failed: {e}, using default parameters")
                    self.advanced_models['lightgbm'] = lgb.LGBMClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=self.random_state,
                        verbose=-1
                    )
            
            # Fallback for when Bayesian optimization is not available
            if not BAYESIAN_AVAILABLE:
                self.logger.info("Bayesian optimization not available, using default parameters...")
                return self._create_default_advanced_models()
            
            self.logger.info(f"Advanced models optimization completed. Models: {list(self.advanced_models.keys())}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in advanced models optimization: {e}")
            return self._create_default_advanced_models()
    
    def _create_default_advanced_models(self):
        """Create default advanced models when optimization fails."""
        self.advanced_models = {}
        
        if XGBOOST_AVAILABLE:
            self.advanced_models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='logloss'
            )
        
        if LIGHTGBM_AVAILABLE:
            self.advanced_models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=-1
            )
        
        self.logger.info(f"Created default advanced models: {list(self.advanced_models.keys())}")
        return True
    
    def create_optimized_ensemble(self):
        """Stage 5: Creating advanced stacking ensemble with diverse base models."""
        self.logger.info("Stage 5: Creating advanced stacking ensemble...")
        
        try:
            X_train, y_train, _ = self.splits['train']
            
            # 1. Define diverse base models
            base_models = {
                'svm_linear': SVC(kernel='linear', probability=True, random_state=self.random_state, max_iter=80000, tol=1e-4),
                'svm_rbf': SVC(kernel='rbf', probability=True, random_state=self.random_state, max_iter=80000, tol=1e-4),
                'random_forest': RandomForestClassifier(
                    n_estimators=200, 
                    max_depth=10, 
                    min_samples_split=5,
                    min_samples_leaf=1,
                    random_state=self.random_state
                ),
                'logistic_regression': LogisticRegression(
                    C=1.0, 
                    class_weight='balanced', 
                    random_state=self.random_state,
                    max_iter=80000
                )
            }
            
            # Add XGBoost if available
            if XGBOOST_AVAILABLE:
                if hasattr(self, 'advanced_models') and 'xgboost' in self.advanced_models:
                    base_models['xgboost'] = self.advanced_models['xgboost']
                    self.logger.info("Using optimized XGBoost model")
                else:
                    base_models['xgboost'] = xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=self.random_state,
                        eval_metric='logloss'
                    )
                    self.logger.info("Added XGBoost with default parameters")
            
            # Add LightGBM if available
            if LIGHTGBM_AVAILABLE:
                if hasattr(self, 'advanced_models') and 'lightgbm' in self.advanced_models:
                    base_models['lightgbm'] = self.advanced_models['lightgbm']
                    self.logger.info("Using optimized LightGBM model")
                else:
                    base_models['lightgbm'] = lgb.LGBMClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=self.random_state,
                        verbose=-1
                    )
                    self.logger.info("Added LightGBM with default parameters")
            
            # 2. Train base models and get cross-validation predictions
            base_predictions = {}
            base_probabilities = {}
            
            for name, model in base_models.items():
                self.logger.info(f"Training base model: {name}")
                
                # Get cross-validation predictions
                cv_predictions = cross_val_predict(model, X_train, y_train, cv=5, method='predict')
                cv_probabilities = cross_val_predict(model, X_train, y_train, cv=5, method='predict_proba')
                
                base_predictions[name] = cv_predictions
                base_probabilities[name] = cv_probabilities[:, 1]  # Probability of positive class
                
                # Also train on full training set for final ensemble
                model.fit(X_train, y_train)
                base_models[name] = model
            
            # 3. Create meta-features matrix
            meta_features = np.column_stack(list(base_probabilities.values()))
            meta_feature_names = list(base_probabilities.keys())
            
            # 4. Train meta-learner using cross-validation
            meta_learner = LogisticRegression(
                C=1.0, 
                class_weight='balanced', 
                random_state=self.random_state,
                max_iter=80000
            )
            
            # Use cross-validation to train meta-learner
            meta_learner.fit(meta_features, y_train)
            
            # 5. Create final ensemble
            self.ensemble_model = StackingEnsemble(base_models, meta_learner, meta_feature_names)
            
            # 6. Store ensemble information
            self.ensemble_info = {
                'base_models': list(base_models.keys()),
                'meta_learner': type(meta_learner).__name__,
                'meta_feature_names': meta_feature_names,
                'base_predictions': base_predictions,
                'base_probabilities': base_probabilities,
                'diversity_models': {
                    'xgboost_available': XGBOOST_AVAILABLE,
                    'lightgbm_available': LIGHTGBM_AVAILABLE,
                    'total_models': len(base_models)
                }
            }
            
            self.logger.info("Advanced stacking ensemble created successfully")
            self.logger.info(f"Base models: {list(base_models.keys())}")
            self.logger.info(f"Meta-learner: {type(meta_learner).__name__}")
            self.logger.info(f"Total models in ensemble: {len(base_models)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating stacking ensemble: {e}")
            return False
    
    def _evaluate_model(self, model, model_name):
        """Helper method to evaluate a single model across all splits."""
        results = {}
        
        for split_name, (X_split, y_split, ids_split) in self.splits.items():
            # Predictions
            y_pred = model.predict(X_split)
            y_pred_proba = model.predict_proba(X_split)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_split, y_pred)
            precision = precision_score(y_split, y_pred, average='weighted')
            recall = recall_score(y_split, y_pred, average='weighted')
            f1 = f1_score(y_split, y_pred, average='weighted')
            auc = roc_auc_score(y_split, y_pred_proba)
            
            results[split_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'true_labels': y_split,
                'subject_ids': ids_split
            }
            
            self.logger.info(f"{model_name} {split_name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        
        return results

    def evaluate_optimized_models(self):
        """Stage 6: Evaluate optimized models including stacking ensemble."""
        self.logger.info("Stage 6: Evaluating optimized models...")
        
        try:
            self.results = {}
            
            # Evaluate Optimized SVM
            self.logger.info("Evaluating Optimized_SVM...")
            svm_results = self._evaluate_model(self.svm_model, "Optimized_SVM")
            self.results["Optimized_SVM"] = svm_results
            
            # Evaluate Stacking Ensemble
            self.logger.info("Evaluating Stacking_Ensemble...")
            ensemble_results = self._evaluate_model(self.ensemble_model, "Stacking_Ensemble")
            self.results["Stacking_Ensemble"] = ensemble_results
            
            # Evaluate individual base models from ensemble
            if hasattr(self, 'ensemble_info') and 'base_models' in self.ensemble_info:
                for base_model_name in self.ensemble_info['base_models']:
                    if hasattr(self.ensemble_model, 'base_models') and base_model_name in self.ensemble_model.base_models:
                        self.logger.info(f"Evaluating base model: {base_model_name}")
                        base_model = self.ensemble_model.base_models[base_model_name]
                        base_results = self._evaluate_model(base_model, f"Base_{base_model_name}")
                        self.results[f"Base_{base_model_name}"] = base_results
            
            # Calculate feature importance for SVM
            if hasattr(self.svm_model, 'coef_'):
                self.feature_importance = {
                    'Optimized_SVM': np.abs(self.svm_model.coef_[0])
                }
            elif hasattr(self.svm_model, 'feature_importances_'):
                self.feature_importance = {
                    'Optimized_SVM': self.svm_model.feature_importances_
                }
            
            # Store ensemble information in results
            if hasattr(self, 'ensemble_info'):
                self.feature_engineering_results['ensemble_info'] = self.ensemble_info
            
            self.logger.info("Model evaluation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error evaluating models: {e}")
            return False
    
    def generate_optimized_plots(self):
        """Generate comprehensive evaluation plots for optimized pipeline."""
        self.logger.info("Generating optimized plots...")
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Optimized Radiomics Classification Pipeline - Advanced Results', fontsize=16, fontweight='bold')
            
            # Get model names
            model_names = list(self.results.keys())
            
            # 1. ROC Curves (All Models)
            for name in model_names:
                test_results = self.results[name]['test']
                fpr, tpr, _ = roc_curve(test_results['true_labels'], test_results['probabilities'])
                auc_score = test_results['auc']
                axes[0, 0].plot(fpr, tpr, label=f'{name} (AUC={auc_score:.3f})', linewidth=2)
            
            axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[0, 0].set_xlabel('False Positive Rate')
            axes[0, 0].set_ylabel('True Positive Rate')
            axes[0, 0].set_title('ROC Curves - All Models')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Feature Importance (SVM)
            if 'Optimized_SVM' in self.feature_importance:
                importance = self.feature_importance['Optimized_SVM']
                top_indices = np.argsort(importance)[-15:]  # Top 15 features
                
                axes[0, 1].barh(range(len(top_indices)), importance[top_indices])
                axes[0, 1].set_yticks(range(len(top_indices)))
                axes[0, 1].set_yticklabels([self.feature_names[i] for i in top_indices], fontsize=8)
                axes[0, 1].set_xlabel('Feature Importance')
                axes[0, 1].set_title('Top 15 Features - Optimized SVM')
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Model Performance Comparison
            x = np.arange(len(model_names))
            width = 0.35
            
            train_acc = [self.results[name]['train']['accuracy'] for name in model_names]
            test_acc = [self.results[name]['test']['accuracy'] for name in model_names]
            
            axes[0, 2].bar(x - width/2, train_acc, width, label='Train', alpha=0.8)
            axes[0, 2].bar(x + width/2, test_acc, width, label='Test', alpha=0.8)
            axes[0, 2].set_title('Model Performance Comparison')
            axes[0, 2].set_ylabel('Accuracy')
            axes[0, 2].set_xticks(x)
            axes[0, 2].set_xticklabels(model_names, rotation=45, ha='right')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Confusion Matrix (Best Model)
            best_model = max(model_names, key=lambda x: self.results[x]['test']['auc'])
            cm = confusion_matrix(
                self.results[best_model]['test']['true_labels'],
                self.results[best_model]['test']['predictions']
            )
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
            axes[1, 0].set_title(f'Confusion Matrix - {best_model}')
            axes[1, 0].set_xlabel('Predicted')
            axes[1, 0].set_ylabel('Actual')
            
            # 5. Base Model vs Ensemble Comparison
            base_models = [name for name in model_names if name.startswith('Base_')]
            ensemble_models = [name for name in model_names if 'Ensemble' in name]
            
            if base_models and ensemble_models:
                base_acc = [self.results[name]['test']['accuracy'] for name in base_models]
                ensemble_acc = [self.results[name]['test']['accuracy'] for name in ensemble_models]
                
                all_acc = base_acc + ensemble_acc
                all_names = [name.replace('Base_', '') for name in base_models] + ensemble_models
                
                axes[1, 1].bar(range(len(all_names)), all_acc, color=['lightblue']*len(base_acc) + ['orange']*len(ensemble_acc))
                axes[1, 1].set_title('Base Models vs Ensemble')
                axes[1, 1].set_ylabel('Test Accuracy')
                axes[1, 1].set_xticks(range(len(all_names)))
                axes[1, 1].set_xticklabels(all_names, rotation=45, ha='right')
                axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Feature Engineering Summary
            if 'feature_selection' in self.feature_engineering_results:
                feature_selection_results = self.feature_engineering_results['feature_selection']
                axes[1, 2].text(0.1, 0.8, f"Original Features: {len(self.top_features)}", fontsize=12)
                axes[1, 2].text(0.1, 0.7, f"Engineered Features: {len(self.feature_names) - len(self.top_features)}", fontsize=12)
                axes[1, 2].text(0.1, 0.6, f"Selected Features: {feature_selection_results['n_features']}", fontsize=12)
                axes[1, 2].text(0.1, 0.5, f"Method: {feature_selection_results['method']}", fontsize=12)
                axes[1, 2].text(0.1, 0.4, f"Stacking Models: {len(self.ensemble_info['base_models'])}", fontsize=12)
                axes[1, 2].set_title('Advanced Feature Engineering Summary')
                axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'optimized_evaluation_plots.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Optimized plots generated and saved")
            
        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")
    
    def save_optimized_artifacts(self):
        """Stage 7: Save optimized artifacts."""
        self.logger.info("Stage 7: Saving optimized artifacts...")
        
        try:
            # Save models
            with open(self.output_dir / 'optimized_svm_model.pkl', 'wb') as f:
                pickle.dump(self.svm_model, f)
            
            with open(self.output_dir / 'optimized_ensemble_model.pkl', 'wb') as f:
                pickle.dump(self.ensemble_model, f)
            
            # Save scaler
            with open(self.output_dir / 'optimized_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save feature importance
            if 'Optimized_SVM' in self.feature_importance:
                feature_importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.feature_importance['Optimized_SVM']
                }).sort_values('importance', ascending=False)
                
                feature_importance_df.to_csv(self.output_dir / 'optimized_feature_importance.csv', index=False)
            
            # Save feature engineering results
            with open(self.output_dir / 'feature_engineering_results.json', 'w') as f:
                json.dump(self.feature_engineering_results, f, indent=2, cls=NumpyEncoder)
            
            # Save results summary
            summary = {
                'timestamp': datetime.now().isoformat(),
                'input_file': self.input_path,
                'data_shape': self.X.shape,
                'feature_names': self.feature_names,
                'top_features_used': self.top_features,
                'svm_model': str(self.svm_model),
                'ensemble_model': str(self.ensemble_model),
                'feature_engineering_results': self.feature_engineering_results,
                'results': {
                    name: {
                        split: {k: v for k, v in results.items() if k not in ['predictions', 'probabilities', 'true_labels', 'subject_ids']}
                        for split, results in model_results.items()
                    }
                    for name, model_results in self.results.items()
                }
            }
            
            with open(self.output_dir / 'optimized_results_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, cls=NumpyEncoder)
            
            # Generate plots
            self.generate_optimized_plots()
            
            self.logger.info(f"All optimized artifacts saved to {self.output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving artifacts: {e}")
            return False
    
    def run_optimized_pipeline(self):
        """Run the complete optimized pipeline."""
        self.logger.info("Starting Optimized Radiomics Classification Pipeline")
        
        stages = [
            ("Data Loading", self.load_data),
            ("Advanced Feature Engineering", self.advanced_feature_engineering),
            ("Optimized Feature Selection", self.optimized_feature_selection),
            ("Data Splitting", self.split_data),
            ("SVM Hyperparameter Optimization", self.optimize_svm_hyperparameters),
            ("Advanced Models Optimization", self.optimize_advanced_models),
            ("Ensemble Creation", self.create_optimized_ensemble),
            ("Model Evaluation", self.evaluate_optimized_models),
            ("Saving Optimized Artifacts", self.save_optimized_artifacts)
        ]
        
        for stage_name, stage_func in stages:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Starting {stage_name}")
            self.logger.info(f"{'='*50}")
            
            if not stage_func():
                self.logger.error(f"Pipeline failed at {stage_name}")
                return False
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info("Optimized pipeline completed successfully!")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info(f"{'='*50}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Optimized Radiomics Classification Pipeline')
    parser.add_argument('--input', 
                       default='~/reseng202500013-ndd-ml/data/radiomics_MRI_mri_labels.csv',
                       help='Path to radiomics CSV file')
    parser.add_argument('--output-dir', 
                       default='~/reseng202500013-ndd-ml/data/optimized_classical_results',
                       help='Output directory for results')
    parser.add_argument('--random-state', 
                       type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--binary-only', 
                       action='store_true', default=True,
                       help='Use only binary classification (labels 0 and 1)')
    
    args = parser.parse_args()
    
    # Expand user paths
    input_path = os.path.expanduser(args.input)
    output_dir = os.path.expanduser(args.output_dir)
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        sys.exit(1)
    
    print("Starting Optimized Radiomics Classification Pipeline")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Random seed: {args.random_state}")
    print(f"Classification: {'Binary (0,1)' if args.binary_only else 'Multi-class'}")
    print("=" * 60)
    
    # Initialize and run pipeline
    classifier = OptimizedRadiomicsClassifier(input_path, output_dir, args.random_state, args.binary_only)
    success = classifier.run_optimized_pipeline()
    
    if success:
        print("\n" + "=" * 60)
        print("Optimized pipeline completed successfully!")
        print(f"Results saved to: {output_dir}")
        print("\nGenerated files:")
        print(f"  • optimized_svm_model.pkl - Fine-tuned SVM model")
        print(f"  • optimized_ensemble_model.pkl - Ensemble model")
        print(f"  • optimized_scaler.pkl - Feature scaler")
        print(f"  • optimized_feature_importance.csv - Feature importance")
        print(f"  • feature_engineering_results.json - Engineering details")
        print(f"  • optimized_evaluation_plots.png - Performance plots")
        print(f"  • optimized_results_summary.json - Detailed results")
        print(f"  • optimized_pipeline.log - Execution log")
        print("\nKey Optimizations:")
        print(f"  • Advanced feature engineering based on cross-model analysis")
        print(f"  • Simplified feature selection")
        print(f"  • Fine-tuned SVM hyperparameters")
        print(f"  • Ensemble with SVM as primary model")
        print(f"  • Clinical interpretability focus")
    else:
        print("\nPipeline failed! Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 