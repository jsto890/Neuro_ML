"""
Improved Optimized Radiomics Classification Pipeline
==================================================

Key improvements:
1. Address overfitting in ensemble models
2. Fix SVM convergence issues
3. Prevent data leakage in feature engineering
4. Improve outlier detection methodology
5. Simplify ensemble to reduce overfitting
6. Add regularization and early stopping
7. FIXED: Data leakage issue - preprocessing now applied correctly after data splitting

FIXED: Data leakage issue - preprocessing now applied correctly after data splitting.
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
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")
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

# Improved Stacking Ensemble class
class ImprovedStackingEnsemble:
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
        return f"ImprovedStackingEnsemble(base_models={list(self.base_models.keys())}, meta_learner={type(self.meta_learner).__name__})"

class ImprovedOptimizedRadiomicsClassifier:
    """Improved optimized radiomics classifier with focus on preventing overfitting and data leakage."""
    
    def __init__(self, input_path, output_dir, random_state=42, binary_only=True):
        """
        Initialize the Improved Optimized Radiomics Classifier.
        
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
        
        # Initialize preprocessing components (will be fitted on training data only)
        self.variance_selector = VarianceThreshold(threshold=0.01)
        self.scaler = RobustScaler()
        self.feature_selector = None  # Will be set during preprocessing
        
        # Initialize model components
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
        log_file = self.output_dir / 'improved_optimized_pipeline.log'
        
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
            
            # Remove diagnostic columns (keep only radiomics features)
            diagnostic_cols = [col for col in self.data.columns if any(x in col.lower() for x in ['diagnostics_', 'subject_id'])]
            if diagnostic_cols:
                self.data = self.data.drop(columns=diagnostic_cols)
                self.logger.info(f"Removed {len(diagnostic_cols)} diagnostic columns")
            
            # Handle binary classification
            if self.binary_only:
                # Find label column
                label_col = None
                for col in self.data.columns:
                    if col.lower() == 'label':
                        label_col = col
                        break
                
                if label_col is None:
                    raise ValueError("No label column found")
                
                # Get unique labels
                unique_labels = self.data[label_col].unique()
                self.logger.info(f"Unique labels: {unique_labels}")
                
                # Filter to binary (0, 1)
                if len(unique_labels) > 2:
                    # Keep only classes 0 and 1
                    self.data = self.data[self.data[label_col].isin([0, 1])]
                    self.logger.info(f"Filtered to binary classification: {len(self.data)} samples")
                
                # Extract features and labels
                self.y = self.data[label_col].values
                self.X = self.data.drop(columns=[label_col]).values
                self.feature_names = self.data.drop(columns=[label_col]).columns.tolist()
                self.subject_ids = np.arange(len(self.data))
                
                # Log final data shape
                self.logger.info(f"Data shape: {self.X.shape}")
                self.logger.info(f"Labels: {np.unique(self.y)} (counts: {[np.sum(self.y == label) for label in np.unique(self.y)]})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return False
    
    def split_data(self, test_size=0.2, val_size=0.2):
        """Stage 2: Split data with stratification (BEFORE any preprocessing)."""
        self.logger.info("Stage 2: Splitting data (before preprocessing)...")
        
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
            
            # Store raw splits (before preprocessing)
            self.splits = {
                'train': (X_train, y_train, ids_train),
                'val': (X_val, y_val, ids_val),
                'test': (X_test, y_test, ids_test)
            }
            
            self.logger.info(f"Raw data splits:")
            self.logger.info(f"  Train: {len(X_train)} samples")
            self.logger.info(f"  Validation: {len(X_val)} samples")
            self.logger.info(f"  Test: {len(X_test)} samples")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in data splitting: {e}")
            return False
    
    def improved_feature_engineering(self):
        """Stage 3: Improved feature engineering with data leakage prevention."""
        self.logger.info("Stage 3: Improved feature engineering (training data only)...")
        
        try:
            X_train, y_train, ids_train = self.splits['train']
            X_val, y_val, ids_val = self.splits['val']
            X_test, y_test, ids_test = self.splits['test']
            
            # Handle missing values with imputation (fit on train, apply to all)
            if hasattr(X_train, 'dtype') and np.issubdtype(X_train.dtype, np.number):
                missing_count = np.isnan(X_train).sum()
            else:
                X_train_numeric = pd.DataFrame(X_train, columns=self.feature_names).apply(pd.to_numeric, errors='coerce')
                missing_count = X_train_numeric.isnull().sum().sum()
            
            if missing_count > 0:
                self.logger.warning(f"Found {missing_count} missing values in training data")
                
                # Use imputation fitted on training data
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                
                if hasattr(X_train, 'dtype') and np.issubdtype(X_train.dtype, np.number):
                    X_train = imputer.fit_transform(X_train)
                    X_val = imputer.transform(X_val)
                    X_test = imputer.transform(X_test)
                else:
                    X_train_numeric = pd.DataFrame(X_train, columns=self.feature_names).apply(pd.to_numeric, errors='coerce')
                    X_val_numeric = pd.DataFrame(X_val, columns=self.feature_names).apply(pd.to_numeric, errors='coerce')
                    X_test_numeric = pd.DataFrame(X_test, columns=self.feature_names).apply(pd.to_numeric, errors='coerce')
                    
                    X_train = imputer.fit_transform(X_train_numeric)
                    X_val = imputer.transform(X_val_numeric)
                    X_test = imputer.transform(X_test_numeric)
                
                self.logger.info(f"Imputed missing values using median strategy (fitted on training data)")
            
            # 1. Apply variance thresholding first (fitted on train, applied to all)
            X_train_var = self.variance_selector.fit_transform(X_train)
            X_val_var = self.variance_selector.transform(X_val)
            X_test_var = self.variance_selector.transform(X_test)
            
            # Get feature names after variance filtering
            var_mask = self.variance_selector.get_support()
            var_feature_names = [self.feature_names[i] for i in range(len(self.feature_names)) if var_mask[i]]
            
            self.logger.info(f"After variance threshold: {X_train_var.shape}")
            
            # 2. Find top features from variance-filtered data
            found_top_features = [f for f in self.top_features if f in var_feature_names]
            self.logger.info(f"Found {len(found_top_features)} of {len(self.top_features)} top features")
            
            # 3. Conservative polynomial features (only degree 2, interaction_only=True)
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            X_train_poly = poly.fit_transform(X_train_var)
            X_val_poly = poly.transform(X_val_var)
            X_test_poly = poly.transform(X_test_var)
            
            # Get polynomial feature names
            poly_feature_names = []
            for i, feature_name in enumerate(var_feature_names):
                poly_feature_names.append(feature_name)
            
            # Add interaction terms
            for i in range(len(var_feature_names)):
                for j in range(i+1, len(var_feature_names)):
                    poly_feature_names.append(f"{var_feature_names[i]}_{var_feature_names[j]}")
            
            # 4. Statistical summary features (only for top features)
            if found_top_features:
                top_feature_indices = [var_feature_names.index(f) for f in found_top_features]
                top_features_train = X_train_var[:, top_feature_indices]
                top_features_val = X_val_var[:, top_feature_indices]
                top_features_test = X_test_var[:, top_feature_indices]
                
                # Calculate statistical measures
                texture_mean_train = np.mean(top_features_train, axis=1, keepdims=True)
                texture_std_train = np.std(top_features_train, axis=1, keepdims=True)
                texture_mean_val = np.mean(top_features_val, axis=1, keepdims=True)
                texture_std_val = np.std(top_features_val, axis=1, keepdims=True)
                texture_mean_test = np.mean(top_features_test, axis=1, keepdims=True)
                texture_std_test = np.std(top_features_test, axis=1, keepdims=True)
                
                # Combine all features
                X_train_engineered = np.hstack([X_train_poly, texture_mean_train, texture_std_train])
                X_val_engineered = np.hstack([X_val_poly, texture_mean_val, texture_std_val])
                X_test_engineered = np.hstack([X_test_poly, texture_mean_test, texture_std_test])
                engineered_feature_names = poly_feature_names + ['texture_mean', 'texture_std']
            else:
                X_train_engineered = X_train_poly
                X_val_engineered = X_val_poly
                X_test_engineered = X_test_poly
                engineered_feature_names = poly_feature_names
            
            self.logger.info(f"After feature engineering: {X_train_engineered.shape}")
            self.logger.info(f"Feature types: {len(var_feature_names)} original, {len(engineered_feature_names) - len(var_feature_names)} engineered")
            
            # 5. Improved outlier detection (IQR-based, more conservative) - only on training data
            Q1 = np.percentile(X_train_engineered, 25, axis=0)
            Q3 = np.percentile(X_train_engineered, 75, axis=0)
            IQR = Q3 - Q1
            
            # More conservative outlier detection (3*IQR instead of 1.5*IQR)
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outlier_mask = np.any((X_train_engineered < lower_bound) | (X_train_engineered > upper_bound), axis=1)
            outlier_indices = np.where(outlier_mask)[0]
            
            # Remove outliers from training data only
            X_train_clean = X_train_engineered[~outlier_mask]
            y_train_clean = y_train[~outlier_mask]
            ids_train_clean = ids_train[~outlier_mask]
            
            self.logger.info(f"Removed {len(outlier_indices)} outliers from training data (conservative IQR method)")
            
            # 6. Improved scaling (RobustScaler for better outlier handling) - fitted on clean training data
            X_train_scaled = self.scaler.fit_transform(X_train_clean)
            X_val_scaled = self.scaler.transform(X_val_engineered)
            X_test_scaled = self.scaler.transform(X_test_engineered)
            
            # Update splits with preprocessed data
            self.splits = {
                'train': (X_train_scaled, y_train_clean, ids_train_clean),
                'val': (X_val_scaled, y_val, self.splits['val'][2]),
                'test': (X_test_scaled, y_test, self.splits['test'][2])
            }
            
            # Update feature names
            self.feature_names = engineered_feature_names
            
            # Store feature engineering results
            self.feature_engineering_results = {
                'variance_threshold': {
                    'n_features_before': len(self.feature_names),
                    'n_features_after': len(var_feature_names),
                    'threshold': 0.01
                },
                'polynomial_features': {
                    'degree': 2,
                    'interaction_only': True,
                    'n_features_before': len(var_feature_names),
                    'n_features_after': len(poly_feature_names)
                },
                'outlier_detection': {
                    'method': 'IQR_3x',
                    'n_outliers_removed': len(outlier_indices),
                    'outlier_indices': outlier_indices.tolist(),
                    'applied_to': 'training_data_only'
                },
                'scaling': {
                    'method': 'RobustScaler',
                    'n_features': len(engineered_feature_names),
                    'fitted_on': 'clean_training_data'
                }
            }
            
            self.logger.info(f"Features scaled using RobustScaler (fitted on clean training data)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in improved feature engineering: {e}")
            return False
    
    def improved_feature_selection(self):
        """Stage 4: Improved feature selection with cross-validation (training data only)."""
        self.logger.info("Stage 4: Improved feature selection (training data only)...")
        
        try:
            X_train, y_train, _ = self.splits['train']
            X_val, y_val, _ = self.splits['val']
            X_test, y_test, _ = self.splits['test']
            
            # Use mutual information for feature selection (more robust than f-statistic)
            k_best = min(50, X_train.shape[1] // 2)  # Select top 50% of features
            
            # Apply mutual information feature selection (fitted on train, applied to all)
            mi_selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
            X_train_mi = mi_selector.fit_transform(X_train, y_train)
            X_val_mi = mi_selector.transform(X_val)
            X_test_mi = mi_selector.transform(X_test)
            
            # Get selected feature names
            mi_mask = mi_selector.get_support()
            selected_feature_names = [self.feature_names[i] for i in range(len(self.feature_names)) if mi_mask[i]]
            
            # Apply RFECV for final selection (fitted on train, applied to all)
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
            
            # Update splits with final selected features
            self.splits = {
                'train': (X_train_final, y_train, self.splits['train'][2]),
                'val': (X_val_final, y_val, self.splits['val'][2]),
                'test': (X_test_final, y_test, self.splits['test'][2])
            }
            
            # Update feature names
            self.feature_names = final_feature_names
            
            # Store feature selection results
            self.feature_engineering_results['feature_selection'] = {
                'n_features': len(final_feature_names),
                'selected_features': final_feature_names,
                'method': 'MutualInfo + RFECV',
                'mi_k': k_best,
                'rfecv_cv': 5,
                'rfecv_scoring': 'roc_auc',
                'fitted_on': 'training_data_only'
            }
            
            self.logger.info(f"Feature selection completed: {len(final_feature_names)} features (fitted on training data)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            return False
    
    def optimize_svm_hyperparameters(self):
        """Stage 5: Optimize SVM with improved convergence handling."""
        self.logger.info("Stage 5: Optimizing SVM hyperparameters...")
        
        try:
            X_train, y_train, _ = self.splits['train']
            
            # Create base SVM with improved settings
            base_svm = SVC(probability=True, random_state=self.random_state)
            
            if BAYESIAN_AVAILABLE:
                # Improved search space for better convergence
                search_spaces = {
                    'C': Real(0.1, 50.0, prior='log-uniform'),  # Reduced upper bound
                    'gamma': Categorical(['scale', 'auto']),
                    'kernel': Categorical(['linear', 'rbf']),  # Removed poly kernel
                    'class_weight': Categorical(['balanced']),
                    'max_iter': Integer(5000, 15000),  # Reduced max_iter
                    'tol': Real(1e-4, 1e-2, prior='log-uniform')  # Increased tolerance
                }
                
                # Bayesian optimization with fewer iterations
                bayes_search = BayesSearchCV(
                    estimator=base_svm,
                    search_spaces=search_spaces,
                    n_iter=30,  # Reduced iterations
                    cv=5,
                    scoring='roc_auc',  # Changed to roc_auc
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
                    'method': 'Bayesian Optimization (Improved)',
                    'best_params': best_params,
                    'best_cv_score': bayes_search.best_score_,
                    'n_iterations': 30,
                    'scoring': 'roc_auc'
                }
                
                self.logger.info(f"Bayesian optimization completed")
                self.logger.info(f"Best SVM parameters: {best_params}")
                self.logger.info(f"Best CV score: {bayes_search.best_score_:.4f}")
                
            else:
                # Fallback to grid search
                param_grid = {
                    'C': [0.1, 1.0, 10.0, 50.0],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['linear', 'rbf'],
                    'class_weight': ['balanced'],
                    'max_iter': [10000],
                    'tol': [1e-3]
                }
                
                grid_search = GridSearchCV(
                    estimator=base_svm,
                    param_grid=param_grid,
                    cv=5,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train)
                
                self.svm_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
                self.feature_engineering_results['svm_optimization'] = {
                    'method': 'Grid Search (Fallback)',
                    'best_params': best_params,
                    'best_cv_score': grid_search.best_score_,
                    'scoring': 'roc_auc'
                }
                
                self.logger.info(f"Grid search completed")
                self.logger.info(f"Best SVM parameters: {best_params}")
                self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in SVM optimization: {e}")
            return False
    
    def create_improved_ensemble(self):
        """Stage 6: Create simplified ensemble to prevent overfitting."""
        self.logger.info("Stage 6: Creating improved ensemble...")
        
        try:
            X_train, y_train, _ = self.splits['train']
            
            # Simplified base models with regularization
            base_models = {
                'svm_linear': SVC(
                    kernel='linear', 
                    probability=True, 
                    random_state=self.random_state, 
                    max_iter=10000, 
                    tol=1e-3,
                    C=1.0  # Conservative C value
                ),
                'svm_rbf': SVC(
                    kernel='rbf', 
                    probability=True, 
                    random_state=self.random_state, 
                    max_iter=10000, 
                    tol=1e-3,
                    C=1.0,  # Conservative C value
                    gamma='scale'
                ),
                'logistic_regression': LogisticRegression(C=0.1,  # Strong regularization
                    class_weight='balanced', 
                    random_state=self.random_state,
                    max_iter=1000
                )
            }
            
            # Add XGBoost with regularization if available
            if XGBOOST_AVAILABLE:
                base_models['xgboost'] = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,  # Reduced depth
                    learning_rate=0.1,
                    subsample=0.8,  # Add regularization
                    colsample_bytree=0.8,  # Add regularization
                    reg_alpha=0.1,  # L1 regularization
                    reg_lambda=1.0,  # L2 regularization
                    random_state=self.random_state,
                    eval_metric='logloss'
                )
            
            # Train base models and get cross-validation predictions
            base_predictions = {}
            base_probabilities = {}
            
            for name, model in base_models.items():
                self.logger.info(f"Training base model: {name}")
                
                # Get cross-validation predictions
                cv_predictions = cross_val_predict(model, X_train, y_train, cv=5, method='predict')
                cv_probabilities = cross_val_predict(model, X_train, y_train, cv=5, method='predict_proba')
                
                base_predictions[name] = cv_predictions
                base_probabilities[name] = cv_probabilities[:, 1]
                
                # Train on full training set
                model.fit(X_train, y_train)
                base_models[name] = model
            
            # Create meta-features matrix
            meta_features = np.column_stack(list(base_probabilities.values()))
            meta_feature_names = list(base_probabilities.keys())
            
            # Train meta-learner with regularization
            meta_learner = LogisticRegression(
                C=0.1,  # Strong regularization
                class_weight='balanced', 
                random_state=self.random_state,
                max_iter=1000
            )
            
            meta_learner.fit(meta_features, y_train)
            
            # Create final ensemble
            self.ensemble_model = ImprovedStackingEnsemble(base_models, meta_learner, meta_feature_names)
            
            # Store ensemble information
            self.ensemble_info = {
                'base_models': list(base_models.keys()),
                'meta_learner': type(meta_learner).__name__,
                'meta_feature_names': meta_feature_names,
                'total_models': len(base_models),
                'regularization': 'enabled'
            }
            
            self.logger.info("Improved ensemble created successfully")
            self.logger.info(f"Base models: {list(base_models.keys())}")
            self.logger.info(f"Meta-learner: {type(meta_learner).__name__}")
            self.logger.info(f"Total models in ensemble: {len(base_models)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating improved ensemble: {e}")
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

    def evaluate_improved_models(self):
        """Stage 7: Evaluate improved models."""
        self.logger.info("Stage 7: Evaluating improved models...")
        
        try:
            self.results = {}
            
            # Evaluate Optimized SVM
            self.logger.info("Evaluating Optimized_SVM...")
            svm_results = self._evaluate_model(self.svm_model, "Optimized_SVM")
            self.results["Optimized_SVM"] = svm_results
            
            # Evaluate Improved Ensemble
            self.logger.info("Evaluating Improved_Ensemble...")
            ensemble_results = self._evaluate_model(self.ensemble_model, "Improved_Ensemble")
            self.results["Improved_Ensemble"] = ensemble_results
            
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
    
    def save_improved_artifacts(self):
        """Stage 8: Save improved artifacts."""
        self.logger.info("Stage 8: Saving improved artifacts...")
        
        try:
            # Save models
            with open(self.output_dir / 'improved_svm_model.pkl', 'wb') as f:
                pickle.dump(self.svm_model, f)
            
            with open(self.output_dir / 'improved_ensemble_model.pkl', 'wb') as f:
                pickle.dump(self.ensemble_model, f)
            
            # Save scaler
            with open(self.output_dir / 'improved_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save feature importance
            if self.feature_importance:
                feature_importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.feature_importance.get('Optimized_SVM', [0] * len(self.feature_names))
                })
                feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
                feature_importance_df.to_csv(self.output_dir / 'improved_feature_importance.csv', index=False)
            
            # Save feature engineering results
            with open(self.output_dir / 'improved_feature_engineering_results.json', 'w') as f:
                json.dump(self.feature_engineering_results, f, indent=2, cls=NumpyEncoder)
            
            # Save results summary
            results_summary = {
                'timestamp': datetime.now().isoformat(),
                'input_file': self.input_path,
                'data_shape': list(self.X.shape),
                'final_feature_count': len(self.feature_names),
                'feature_names': self.feature_names,
                'svm_model': str(self.svm_model),
                'ensemble_model': str(self.ensemble_model),
                'feature_engineering_results': self.feature_engineering_results,
                'data_leakage_fixed': True,
                'preprocessing_info': {
                    'variance_threshold': 0.01,
                    'scaling_method': 'RobustScaler',
                    'feature_selection': 'MutualInfo + RFECV',
                    'outlier_detection': 'IQR_3x (training only)',
                    'data_leakage_fixed': True
                },
                'results': self.results
            }
            
            with open(self.output_dir / 'improved_results_summary.json', 'w') as f:
                json.dump(results_summary, f, indent=2, cls=NumpyEncoder)
            
            self.logger.info("All improved artifacts saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving artifacts: {e}")
            return False
    
    def run_improved_pipeline(self):
        """Run the complete improved optimized pipeline."""
        self.logger.info("Starting Improved Optimized Radiomics Classification Pipeline (Data Leakage Fixed)")
        
        stages = [
            ("Data Loading", self.load_data),
            ("Data Splitting", self.split_data),
            ("Improved Feature Engineering", self.improved_feature_engineering),
            ("Improved Feature Selection", self.improved_feature_selection),
            ("SVM Hyperparameter Optimization", self.optimize_svm_hyperparameters),
            ("Improved Ensemble Creation", self.create_improved_ensemble),
            ("Model Evaluation", self.evaluate_improved_models),
            ("Saving Improved Artifacts", self.save_improved_artifacts)
        ]
        
        for stage_name, stage_func in stages:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Starting {stage_name}")
            self.logger.info(f"{'='*50}")
            
            if not stage_func():
                self.logger.error(f"Pipeline failed at {stage_name}")
                return False
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info("Improved pipeline completed successfully!")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info(f"{'='*50}")
        
        return True

def main():
    """Main function to run the improved optimized pipeline."""
    parser = argparse.ArgumentParser(description='Improved Optimized Radiomics Classification Pipeline')
    parser.add_argument('--input', type=str, required=True, help='Path to radiomics CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output directory for results')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--binary_only', action='store_true', help='Use only binary classification')
    
    args = parser.parse_args()
    
    # Create classifier and run pipeline
    classifier = ImprovedOptimizedRadiomicsClassifier(
        input_path=args.input,
        output_dir=args.output,
        random_state=args.random_state,
        binary_only=args.binary_only
    )
    
    success = classifier.run_improved_pipeline()
    
    if success:
        print("\n" + "="*60)
        print("Improved pipeline completed successfully!")
        print(f"Results saved to: {args.output}")
        print("="*60)
        
        print("\nGenerated files:")
        print("  • improved_svm_model.pkl - Optimized SVM model")
        print("  • improved_ensemble_model.pkl - Improved ensemble model")
        print("  • improved_scaler.pkl - Feature scaler")
        print("  • improved_feature_importance.csv - Feature importance")
        print("  • improved_feature_engineering_results.json - Engineering details")
        print("  • improved_results_summary.json - Detailed results")
        print("  • improved_optimized_pipeline.log - Execution log")
        
        print("\nKey Improvements:")
        print("  • Reduced overfitting through regularization")
        print("  • Improved SVM convergence handling")
        print("  • Simplified ensemble (removed overfitting models)")
        print("  • Better outlier detection (IQR-based)")
        print("  • Mutual information feature selection")
        print("  • ✅ Data leakage prevention (FIXED)")
        print("  • ✅ Preprocessing fitted on training data only")
        
        print("\nClinical Recommendations:")
        print("  • Use improved_svm_model.pkl for primary predictions")
        print("  • Review improved_feature_importance.csv for key biomarkers")
        print("  • Check improved_feature_engineering_results.json for insights")
        print("  • Ensemble model provides robust backup predictions")
    else:
        print("Pipeline failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 