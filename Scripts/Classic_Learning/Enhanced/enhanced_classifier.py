"""
Enhanced Radiomics Classification Pipeline
==========================================

This enhanced version includes:
- Multiple algorithms (Random Forest, SVM, Logistic Regression, Gradient Boosting)
- Advanced feature engineering and selection
- Better regularization to prevent overfitting
- Comprehensive model comparison
- Ensemble methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
import logging
import argparse
import sys
from datetime import datetime
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_score,
    train_test_split, RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, 
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, matthews_corrcoef
)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')
import os

# Optional advanced libraries
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

class EnhancedRadiomicsClassifier:
    """Enhanced radiomics classifier with multiple algorithms and advanced feature engineering."""
    
    def __init__(self, input_path, output_dir, random_state=42, binary_only=True, binary_labels=(0, 1), ml_threads=None):
        """
        Initialize the Enhanced Radiomics Classifier.
        
        Args:
            input_path (str): Path to radiomics CSV file
            output_dir (str): Output directory for results
            random_state (int): Random seed for reproducibility
            binary_only (bool): If True, only use specified binary labels
            binary_labels (tuple): Labels to use for binary classification (default: (0, 1))
            ml_threads (int|None): Max threads for LightGBM/XGBoost. If None, defaults to min(4, CPU cores).
        """
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.binary_only = binary_only
        self.binary_labels = binary_labels
        self.ml_threads = ml_threads
        
        # Disease label mapping
        self.label_to_disease = {0: 'CN', 1: 'AD', 2: 'PD'}
        self.disease_to_label = {v: k for k, v in self.label_to_disease.items()}
        
        # Label remapping for training (to handle non-consecutive labels)
        self.original_to_training_labels = {}
        self.training_to_original_labels = {}
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_selector = None
        self.models = {}
        self.best_models = {}
        self.feature_importance = {}
        self.results = {}
        
        # Setup logging
        self.setup_logging()
        
        # Initialize data containers
        self.data = None
        self.X = None
        self.y = None
        self.subject_ids = None
        self.feature_names = None
        self.selected_features = None
        self.splits = None
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / 'enhanced_pipeline.log'
        
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
            self.data = pd.read_csv(self.input_path)
            self.logger.info(f"Loaded {len(self.data)} samples with {len(self.data.columns)} columns")
            
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
            
            # Filter for binary classification if requested
            if self.binary_only:
                initial_count = len(self.data)
                self.data = self.data[self.data['label'].isin(self.binary_labels)]
                final_count = len(self.data)
                self.logger.info(f"Filtered to binary classification (labels {self.binary_labels}): {initial_count} → {final_count} samples")
                
                if final_count == 0:
                    raise ValueError(f"No samples remaining after binary filtering for labels {self.binary_labels}")
                
                # Create label remapping for training (to handle non-consecutive labels)
                unique_labels = sorted(self.binary_labels)
                self.original_to_training_labels = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
                self.training_to_original_labels = {v: k for k, v in self.original_to_training_labels.items()}
                
                self.logger.info(f"Label remapping: {self.original_to_training_labels}")
            
            # Extract components
            self.subject_ids = self.data['subject_id'].values
            self.y = self.data['label'].values.astype(int)  # Ensure labels are integers
            self.feature_names = [col for col in self.data.columns if col not in ['subject_id', 'label']]
            # Preserve full list for per-fold resets in outer CV
            self._original_feature_names = list(self.feature_names)
            self.X = self.data[self.feature_names].values
            
            self.logger.info(f"Data shape: {self.X.shape}")
            unique_labels = np.unique(self.y)
            disease_counts = {self.label_to_disease.get(int(label), str(label)): np.sum(self.y == label) for label in unique_labels}
            self.logger.info(f"Labels: {unique_labels} -> {disease_counts}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return False
    
    def advanced_preprocessing(self):
        """Stage 1: Advanced preprocessing with feature engineering."""
        self.logger.info("Stage 1: Advanced preprocessing...")
        
        try:
            # Handle missing values with imputation
            if hasattr(self.X, 'dtype') and np.issubdtype(self.X.dtype, np.number):
                missing_count = np.isnan(self.X).sum()
            else:
                X_numeric = pd.DataFrame(self.X, columns=self.feature_names).apply(pd.to_numeric, errors='coerce')
                missing_count = X_numeric.isnull().sum().sum()
            
            if missing_count > 0:
                self.logger.info(f"Found {missing_count} missing values - using imputation")
                imputer = SimpleImputer(strategy='median')
                
                if hasattr(self.X, 'dtype') and np.issubdtype(self.X.dtype, np.number):
                    self.X = imputer.fit_transform(self.X)
                else:
                    X_numeric = pd.DataFrame(self.X, columns=self.feature_names).apply(pd.to_numeric, errors='coerce')
                    X_imputed = imputer.fit_transform(X_numeric)
                    self.X = X_imputed
            
            # Remove constant and quasi-constant features
            variance_selector = VarianceThreshold(threshold=0.01)
            self.X = variance_selector.fit_transform(self.X)
            kept_features = variance_selector.get_support()
            self.feature_names = [f for f, keep in zip(self.feature_names, kept_features) if keep]
            self.logger.info(f"After variance threshold: {self.X.shape}")
            
            # Advanced feature selection
            self.logger.info("Performing advanced feature selection...")
            
            # Method 1: Mutual Information (captures non-linear relationships)
            mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(50, self.X.shape[1]))
            X_mi = mi_selector.fit_transform(self.X, self.y)
            mi_scores = mi_selector.scores_
            mi_features = [f for f, selected in zip(self.feature_names, mi_selector.get_support()) if selected]
            
            # Method 2: F-statistic (captures linear relationships)
            f_selector = SelectKBest(score_func=f_classif, k=min(50, self.X.shape[1]))
            X_f = f_selector.fit_transform(self.X, self.y)
            f_scores = f_selector.scores_
            f_features = [f for f, selected in zip(self.feature_names, f_selector.get_support()) if selected]
            
            # Combine both methods (union of selected features)
            combined_features = list(set(mi_features + f_features))
            feature_indices = [self.feature_names.index(f) for f in combined_features]
            
            self.X = self.X[:, feature_indices]
            self.feature_names = combined_features
            self.logger.info(f"After feature selection: {self.X.shape}")
            
            # Scale features
            self.X = self.scaler.fit_transform(self.X)
            self.logger.info("Features scaled using RobustScaler")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {e}")
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
    
    def define_models(self):
        """Define multiple models with regularization-focused parameters."""
        self.logger.info("Defining models with regularization...")
        
        # Random Forest (regularized)
        rf_params = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [2, 4],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced']
        }
        
        # Logistic Regression (L2)
        lr_params = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'class_weight': ['balanced']
        }
        
        # SVM (kernel)
        svm_params = {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto'],
            'class_weight': ['balanced']
        }
        # Calibrated LinearSVC (probabilistic)
        linsvc_calibrated = CalibratedClassifierCV(estimator=LinearSVC(max_iter=10000), cv=5, method='sigmoid')
        linsvc_params = {
            'estimator__C': [0.1, 1.0, 10.0],
            'estimator__loss': ['squared_hinge']
        }
        
        # Gradient Boosting
        gb_params = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 0.9],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [2, 4]
        }
        
        # Extra Trees
        et_params = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }

        # KNN (features are scaled earlier)
        knn_params = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }



        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            # Cap internal threads to avoid oversubscription
            max_threads = self.ml_threads if self.ml_threads is not None else min(4, (os.cpu_count() or 4))
            xgb_model = XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                n_jobs=max_threads,
                tree_method='hist',
                verbosity=0
            )
            xgb_params = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'reg_alpha': [0.0, 0.1],
                'reg_lambda': [1.0, 2.0]
            }
        else:
            xgb_model, xgb_params = None, None

        # LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            # Cap internal threads to avoid oversubscription
            max_threads = self.ml_threads if self.ml_threads is not None else min(4, (os.cpu_count() or 4))
            lgbm_model = LGBMClassifier(
                random_state=self.random_state,
                n_jobs=max_threads,
                verbosity=-1,
                force_col_wise=True
            )
            lgbm_params = {
                'n_estimators': [100, 200],
                'num_leaves': [31, 63],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'reg_alpha': [0.0, 0.1],
                'reg_lambda': [1.0, 2.0]
            }
        else:
            lgbm_model, lgbm_params = None, None
        
        self.models = {
            'RandomForest': (RandomForestClassifier(random_state=self.random_state), rf_params),
            'ExtraTrees': (ExtraTreesClassifier(random_state=self.random_state), et_params),
            'LogisticRegression': (LogisticRegression(random_state=self.random_state), lr_params),
            'SVM': (SVC(random_state=self.random_state, probability=True), svm_params),
            'LinearSVC_Calibrated': (linsvc_calibrated, linsvc_params),
            'GradientBoosting': (GradientBoostingClassifier(random_state=self.random_state), gb_params),
            'KNN': (KNeighborsClassifier(), knn_params)
        }
        if XGBOOST_AVAILABLE and xgb_model is not None:
            self.models['XGBoost'] = (xgb_model, xgb_params)
        else:
            self.logger.info("XGBoost not available. Skipping.")
        if LIGHTGBM_AVAILABLE and lgbm_model is not None:
            self.models['LightGBM'] = (lgbm_model, lgbm_params)
        else:
            self.logger.info("LightGBM not available. Skipping.")
        
        self.logger.info(f"Defined {len(self.models)} models")
        return True
    
    def _remap_labels_for_training(self, y):
        """Remap labels to consecutive integers for training."""
        if hasattr(self, 'original_to_training_labels') and self.original_to_training_labels:
            return np.array([self.original_to_training_labels[label] for label in y])
        return y
    
    def _restore_original_labels(self, y):
        """Restore original labels from training labels."""
        if hasattr(self, 'training_to_original_labels') and self.training_to_original_labels:
            try:
                # Convert input to list of integers first
                y_list = [int(label) for label in y]
                # Check if we need to restore labels (if labels are in training space [0,1])
                if set(y_list).issubset({0, 1}):
                    restored = [self.training_to_original_labels[label] for label in y_list]
                    result = np.array(restored, dtype=int)
                    self.logger.debug(f"Label restoration: {y} -> {result}")
                    return result
                else:
                    # Labels are already in original space, just ensure they're integers
                    result = np.array([int(x) for x in y], dtype=int)
                    self.logger.debug(f"Labels already in original space: {y} -> {result}")
                    return result
            except Exception as e:
                self.logger.warning(f"Error in label restoration: {e}, returning original labels")
                # Ensure we return regular Python integers
                return np.array([int(x) for x in y], dtype=int)
        # If no remapping, ensure we return regular Python integers
        result = np.array([int(x) for x in y], dtype=int)
        self.logger.debug(f"No remapping, converted to int: {y} -> {result}")
        return result
    
    def train_models(self):
        """Stage 3: Train multiple models with cross-validation."""
        self.logger.info("Stage 3: Training multiple models...")
        
        try:
            X_train, y_train, _ = self.splits['train']
            # Remap labels for training
            y_train_remapped = self._remap_labels_for_training(y_train)
            
            # Log class distribution for debugging
            unique_labels, counts = np.unique(y_train_remapped, return_counts=True)
            self.logger.info(f"Training set class distribution: {dict(zip(unique_labels, counts))}")
            if len(unique_labels) < 3 and not self.binary_only:
                self.logger.warning(f"Warning: Only {len(unique_labels)} classes in training set for multiclass classification")
            
            for name, (model, param_grid) in self.models.items():
                self.logger.info(f"Training {name}...")
                
                # Use RandomizedSearchCV for faster search
                # Avoid nested parallelism with multi-threaded learners (LightGBM/XGBoost)
                # by limiting the search parallelism to a single job for those models.
                search_n_jobs = -1
                if ('LightGBM' in name) or ('XGBoost' in name):
                    search_n_jobs = 1
                # Choose appropriate scoring metric based on classification type
                if self.binary_only:
                    scoring = 'roc_auc'
                else:
                    # For multiclass, use accuracy which is more reliable
                    scoring = 'accuracy'
                
                search = RandomizedSearchCV(
                    model, param_grid, n_iter=20, cv=5,
                    scoring=scoring, n_jobs=search_n_jobs,
                    random_state=self.random_state, verbose=0
                )
                
                # Suppress verbose library-level stdout for certain learners
                if (LIGHTGBM_AVAILABLE and 'LightGBM' in name) or (XGBOOST_AVAILABLE and 'XGBoost' in name):
                    import contextlib, os, sys
                    with open(os.devnull, 'w') as devnull:
                        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                            search.fit(X_train, y_train_remapped)
                else:
                    search.fit(X_train, y_train_remapped)
                
                # Check if we got a valid score
                if np.isnan(search.best_score_):
                    self.logger.warning(f"{name} got NaN score. This might indicate class imbalance in CV folds.")
                    # Try to get a fallback score
                    try:
                        # Use the model to predict on training data and calculate accuracy
                        y_pred = search.best_estimator_.predict(X_train)
                        fallback_score = accuracy_score(y_train_remapped, y_pred)
                        self.logger.info(f"{name} - Fallback accuracy score: {fallback_score:.4f}")
                    except Exception as e:
                        self.logger.warning(f"Could not calculate fallback score for {name}: {e}")
                        fallback_score = 0.0
                else:
                    fallback_score = search.best_score_
                
                self.best_models[name] = search.best_estimator_
                self.logger.info(f"{name} - Best CV score: {fallback_score:.4f}")
                self.logger.info(f"{name} - Best params: {search.best_params_}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in model training: {e}")
            return False
    
    def evaluate_models(self):
        """Stage 4: Evaluate all models."""
        self.logger.info("Stage 4: Evaluating models...")
        
        try:
            for name, model in self.best_models.items():
                self.logger.info(f"Evaluating {name}...")
                
                results = {}
                for split_name, (X_split, y_split, ids_split) in self.splits.items():
                    try:
                        # Predictions
                        y_pred = model.predict(X_split)
                        
                        # Handle probabilities for both binary and multiclass
                        if self.binary_only:
                            # Binary: extract probability of positive class
                            y_pred_proba = model.predict_proba(X_split)[:, 1]
                        else:
                            # Multiclass: use probability of predicted class
                            proba_matrix = model.predict_proba(X_split)
                            y_pred_proba = np.max(proba_matrix, axis=1)
                        
                        # Restore original labels for evaluation if remapping was used
                        y_split_original = self._restore_original_labels(y_split)
                        y_pred_original = self._restore_original_labels(y_pred)
                        
                        # Log label information for debugging
                        self.logger.debug(f"{name} {split_name} - Original labels: {np.unique(y_split_original)}, Predicted labels: {np.unique(y_pred_original)}")
                        
                        # Calculate metrics using original labels
                        accuracy = accuracy_score(y_split_original, y_pred_original)
                        precision = precision_score(y_split_original, y_pred_original, average='weighted')
                        recall = recall_score(y_split_original, y_pred_original, average='weighted')
                        f1 = f1_score(y_split_original, y_pred_original, average='weighted')
                        
                        # For ROC AUC, handle both binary and multiclass scenarios
                        try:
                            if len(np.unique(y_split_original)) == 2:
                                # Binary classification - handle non-consecutive labels
                                if set(np.unique(y_split_original)) != {0, 1}:
                                    y_binary = (y_split_original == y_split_original.max()).astype(int)
                                    auc = roc_auc_score(y_binary, y_pred_proba)
                                    self.logger.debug(f"Binary: converted labels {np.unique(y_split_original)} to {np.unique(y_binary)} for AUC")
                                else:
                                    auc = roc_auc_score(y_split_original, y_pred_proba)
                            else:
                                # Multiclass classification - use one-vs-rest
                                auc = roc_auc_score(y_split_original, y_pred_proba, multi_class='ovr', average='weighted')
                                self.logger.debug(f"Multiclass: using OVR AUC with {len(np.unique(y_split_original))} classes")
                        except Exception as e:
                            self.logger.warning(f"Could not compute AUC: {e}, setting to 0.0")
                            auc = 0.0
                        
                        results[split_name] = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'auc': auc,
                            'predictions': y_pred_original,
                            'probabilities': y_pred_proba,
                            'true_labels': y_split_original,
                            'subject_ids': ids_split
                        }
                        
                        self.logger.info(f"{name} {split_name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
                        
                    except Exception as e:
                        self.logger.error(f"Error evaluating {name} on {split_name}: {e}")
                        self.logger.error(f"y_split type: {type(y_split)}, y_pred type: {type(y_pred)}")
                        self.logger.error(f"y_split_original type: {type(y_split_original) if 'y_split_original' in locals() else 'Not created'}")
                        self.logger.error(f"y_pred_original type: {type(y_pred_original) if 'y_pred_original' in locals() else 'Not created'}")
                        raise
                
                self.results[name] = results
                
                # Extract feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    self.feature_importance[name] = np.abs(model.coef_[0])
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {e}")
            return False
    
    def create_ensemble(self):
        """Stage 5: Create ensemble model."""
        self.logger.info("Stage 5: Creating ensemble model...")
        
        try:
            X_train, y_train, _ = self.splits['train']
            
            # Simple voting ensemble (average probabilities)
            ensemble_results = {}
            
            for split_name, (X_split, y_split, ids_split) in self.splits.items():
                try:
                    # Get probabilities from all models
                    all_probs = []
                    for name, model in self.best_models.items():
                        if self.binary_only:
                            # Binary: extract probability of positive class
                            probs = model.predict_proba(X_split)[:, 1]
                        else:
                            # Multiclass: use probability of predicted class
                            proba_matrix = model.predict_proba(X_split)
                            probs = np.max(proba_matrix, axis=1)
                        all_probs.append(probs)
                    
                    # Average probabilities
                    ensemble_probs = np.mean(all_probs, axis=0)
                    ensemble_preds = (ensemble_probs > 0.5).astype(int)
                    
                    # Restore original labels for evaluation if remapping was used
                    y_split_original = self._restore_original_labels(y_split)
                    ensemble_preds_original = self._restore_original_labels(ensemble_preds)
                    
                    # Log label information for debugging
                    self.logger.debug(f"Ensemble {split_name} - Original labels: {np.unique(y_split_original)}, Predicted labels: {np.unique(ensemble_preds_original)}")
                    
                    # Calculate metrics using original labels
                    accuracy = accuracy_score(y_split_original, ensemble_preds_original)
                    precision = precision_score(y_split_original, ensemble_preds_original, average='weighted')
                    recall = recall_score(y_split_original, ensemble_preds_original, average='weighted')
                    f1 = f1_score(y_split_original, ensemble_preds_original, average='weighted')
                    
                    # For ROC AUC, handle both binary and multiclass scenarios
                    try:
                        if len(np.unique(y_split_original)) == 2:
                            # Binary classification - handle non-consecutive labels
                            if set(np.unique(y_split_original)) != {0, 1}:
                                y_binary = (y_split_original == y_split_original.max()).astype(int)
                                auc = roc_auc_score(y_binary, ensemble_probs)
                                self.logger.debug(f"Binary: converted labels {np.unique(y_split_original)} to {y_binary} for AUC")
                            else:
                                auc = roc_auc_score(y_split_original, ensemble_probs)
                        else:
                            # Multiclass classification - use one-vs-rest
                            auc = roc_auc_score(y_split_original, ensemble_probs, multi_class='ovr', average='weighted')
                            self.logger.debug(f"Multiclass: using OVR AUC with {len(np.unique(y_split_original))} classes")
                    except Exception as e:
                        self.logger.warning(f"Could not compute AUC: {e}, setting to 0.0")
                        auc = 0.0
                
                    ensemble_results[split_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'auc': auc,
                        'predictions': ensemble_preds_original,
                        'probabilities': ensemble_probs,
                        'true_labels': y_split_original,
                        'subject_ids': ids_split
                    }
                    
                    self.logger.info(f"Ensemble {split_name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating Ensemble on {split_name}: {e}")
                    self.logger.error(f"y_split type: {type(y_split)}, ensemble_preds type: {type(ensemble_preds)}")
                    self.logger.error(f"y_split_original type: {type(y_split_original) if 'y_split_original' in locals() else 'Not created'}")
                    self.logger.error(f"ensemble_preds_original type: {type(ensemble_preds_original) if 'ensemble_preds_original' in locals() else 'Not created'}")
                    raise
            
            self.results['Ensemble'] = ensemble_results
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating ensemble: {e}")
            return False
    
    def generate_plots(self):
        """Generate comprehensive visualization plots."""
        self.logger.info("Generating plots...")
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Enhanced Radiomics Classification Results', fontsize=16)
            
            # 1. Model Comparison (Accuracy)
            model_names = list(self.results.keys())
            test_accuracies = [self.results[name]['test']['accuracy'] for name in model_names]
            
            axes[0, 0].bar(model_names, test_accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            axes[0, 0].set_title('Model Comparison - Test Accuracy')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].set_xticklabels(model_names, ha='right')
            
            # 2. ROC Curves
            for name in model_names:
                y_true = self.results[name]['test']['true_labels']
                y_probs = self.results[name]['test']['probabilities']
                
                if self.binary_only:
                    # Binary: use probabilities directly
                    y_true_binary = y_true
                    y_probs_binary = y_probs
                else:
                    # Multiclass: convert to one-vs-rest for ROC curves
                    # Use the highest probability class as positive
                    y_true_binary = (y_true == y_true.max()).astype(int)
                    y_probs_binary = y_probs
                    self.logger.debug(f"Multiclass ROC: converting labels {np.unique(y_true)} to {np.unique(y_true_binary)}")
                
                fpr, tpr, _ = roc_curve(y_true_binary, y_probs_binary)
                auc_score = self.results[name]['test']['auc']
                axes[0, 1].plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
            
            axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[0, 1].set_title('ROC Curves - Test Set')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 3. Feature Importance (Random Forest)
            if 'RandomForest' in self.feature_importance:
                rf_importance = self.feature_importance['RandomForest']
                top_indices = np.argsort(rf_importance)[-10:]
                top_features = [self.feature_names[i] for i in top_indices]
                top_importance = rf_importance[top_indices]
                
                axes[0, 2].barh(range(len(top_features)), top_importance)
                axes[0, 2].set_yticks(range(len(top_features)))
                axes[0, 2].set_yticklabels([f.split('_')[-1] for f in top_features])
                axes[0, 2].set_title('Top 10 Features - Random Forest')
                axes[0, 2].set_xlabel('Importance')
            
            # 4. Confusion Matrix (Best Model)
            best_model = max(model_names, key=lambda x: self.results[x]['test']['auc'])
            cm = confusion_matrix(
                self.results[best_model]['test']['true_labels'],
                self.results[best_model]['test']['predictions']
            )
            
            # Get unique labels in the confusion matrix
            unique_labels = np.unique(np.concatenate([np.unique(self.results[best_model]['test']['true_labels']), 
                                                    np.unique(self.results[best_model]['test']['predictions'])]))
            disease_labels = [self.label_to_disease.get(int(label), str(label)) for label in unique_labels]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                       xticklabels=disease_labels, yticklabels=disease_labels)
            axes[1, 0].set_title(f'Confusion Matrix - {best_model}')
            axes[1, 0].set_xlabel('Predicted')
            axes[1, 0].set_ylabel('Actual')
            
            # 5. Performance Metrics Comparison
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
            metric_data = []
            
            for name in model_names:
                row = [self.results[name]['test'][metric] for metric in metrics]
                metric_data.append(row)
            
            metric_df = pd.DataFrame(metric_data, index=model_names, columns=metrics)
            sns.heatmap(metric_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1, 1])
            axes[1, 1].set_title('Performance Metrics - Test Set')
            
            # 6. Train vs Test Performance
            train_acc = [self.results[name]['train']['accuracy'] for name in model_names]
            test_acc = [self.results[name]['test']['accuracy'] for name in model_names]
            
            x = np.arange(len(model_names))
            width = 0.35
            
            axes[1, 2].bar(x - width/2, train_acc, width, label='Train', alpha=0.8)
            axes[1, 2].bar(x + width/2, test_acc, width, label='Test', alpha=0.8)
            axes[1, 2].set_title('Train vs Test Accuracy')
            axes[1, 2].set_ylabel('Accuracy')
            axes[1, 2].set_xticks(x)
            axes[1, 2].set_xticklabels(model_names, rotation=45, ha='right')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'enhanced_evaluation_plots.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Plots generated and saved")
            
        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")
    
    def save_artifacts(self):
        """Stage 6: Save all artifacts."""
        self.logger.info("Stage 6: Saving artifacts...")
        
        try:
            # Save models
            for name, model in self.best_models.items():
                with open(self.output_dir / f'{name.lower()}_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
            
            # Save scaler
            with open(self.output_dir / 'scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save feature importance
            feature_importance_df = pd.DataFrame()
            for name, importance in self.feature_importance.items():
                if len(importance) == len(self.feature_names):
                    df_temp = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': importance,
                        'model': name
                    })
                    feature_importance_df = pd.concat([feature_importance_df, df_temp], ignore_index=True)
            
            feature_importance_df.to_csv(self.output_dir / 'feature_importance_comparison.csv', index=False)
            
            # Save results summary
            summary = {
                'timestamp': datetime.now().isoformat(),
                'input_file': self.input_path,
                'data_shape': self.X.shape,
                'feature_names': self.feature_names,
                'best_models': {name: str(model) for name, model in self.best_models.items()},
                'results': {
                    name: {
                        split: {k: v for k, v in results.items() if k not in ['predictions', 'probabilities', 'true_labels', 'subject_ids']}
                        for split, results in model_results.items()
                    }
                    for name, model_results in self.results.items()
                }
            }
            
            with open(self.output_dir / 'enhanced_results_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Generate plots
            self.generate_plots()
            
            self.logger.info(f"All artifacts saved to {self.output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving artifacts: {e}")
            return False
    
    def run_pipeline(self):
        """Run the complete enhanced pipeline."""
        self.logger.info("Starting Enhanced Radiomics Classification Pipeline")
        
        stages = [
            ("Data Loading", self.load_data),
            ("Advanced Preprocessing", self.advanced_preprocessing),
            ("Data Splitting", self.split_data),
            ("Model Definition", self.define_models),
            ("Model Training", self.train_models),
            ("Model Evaluation", self.evaluate_models),
            ("Ensemble Creation", self.create_ensemble),
            ("Saving Artifacts", self.save_artifacts)
        ]
        
        for stage_name, stage_func in stages:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Starting {stage_name}")
            self.logger.info(f"{'='*50}")
            
            if not stage_func():
                self.logger.error(f"Pipeline failed at {stage_name}")
                return False
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info("Enhanced pipeline completed successfully!")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info(f"{'='*50}")
        
        return True

    def _preprocess_splits_train_only(self):
        """Preprocess Train/Val/Test with transformers fit on Train only to avoid leakage.

        Steps:
        - SimpleImputer (median) fit on Train, transform Val/Test
        - VarianceThreshold fit on Train, transform Val/Test
        - SelectKBest (mutual_info, f_classif) fit on Train, union of selected features applied to all splits
        - RobustScaler fit on Train, transform Val/Test
        """
        self.logger.info("Preprocessing splits (fit on Train only)...")
        try:
            X_train, y_train, ids_train = self.splits['train']
            has_val = 'val' in self.splits
            X_val, y_val, ids_val = self.splits['val'] if has_val else (None, None, None)
            X_test, y_test, ids_test = self.splits['test']

            # 1) Imputation on Train, apply to Val/Test
            imputer = SimpleImputer(strategy='median')
            X_train_imp = imputer.fit_transform(X_train)
            X_val_imp = imputer.transform(X_val) if has_val else None
            X_test_imp = imputer.transform(X_test)

            # 2) Variance threshold on Train, apply to Val/Test
            variance_selector = VarianceThreshold(threshold=0.01)
            X_train_var = variance_selector.fit_transform(X_train_imp)
            X_val_var = variance_selector.transform(X_val_imp) if has_val else None
            X_test_var = variance_selector.transform(X_test_imp)

            kept_mask = variance_selector.get_support()
            feature_names_after_var = [f for f, keep in zip(self.feature_names, kept_mask) if keep]

            # 3) Advanced selection on Train only (MI + F-stat), union of features
            k_mi = min(50, X_train_var.shape[1]) if X_train_var.shape[1] > 0 else 0
            k_f = min(50, X_train_var.shape[1]) if X_train_var.shape[1] > 0 else 0

            if k_mi > 0:
                mi_selector = SelectKBest(score_func=mutual_info_classif, k=k_mi)
                mi_selector.fit(X_train_var, y_train)
                mi_mask = mi_selector.get_support()
                mi_features = [f for f, m in zip(feature_names_after_var, mi_mask) if m]
            else:
                mi_features = []

            if k_f > 0:
                f_selector = SelectKBest(score_func=f_classif, k=k_f)
                f_selector.fit(X_train_var, y_train)
                f_mask = f_selector.get_support()
                f_features = [f for f, m in zip(feature_names_after_var, f_mask) if m]
            else:
                f_features = []

            combined_features = list(set(mi_features + f_features)) if (mi_features or f_features) else feature_names_after_var
            # Map combined feature names to indices in X_train_var order
            name_to_index = {name: idx for idx, name in enumerate(feature_names_after_var)}
            selected_indices = [name_to_index[name] for name in combined_features if name in name_to_index]

            X_train_sel = X_train_var[:, selected_indices] if selected_indices else X_train_var
            X_val_sel = (X_val_var[:, selected_indices] if selected_indices else X_val_var) if has_val else None
            X_test_sel = X_test_var[:, selected_indices] if selected_indices else X_test_var

            self.selected_features = combined_features if selected_indices else feature_names_after_var

            # 4) Scale with RobustScaler fit on Train
            self.scaler = RobustScaler()
            X_train_scaled = self.scaler.fit_transform(X_train_sel)
            X_val_scaled = self.scaler.transform(X_val_sel) if has_val else None
            X_test_scaled = self.scaler.transform(X_test_sel)

            # Store back
            self.feature_names = self.selected_features
            new_splits = {
                'train': (X_train_scaled, y_train, ids_train),
                'test': (X_test_scaled, y_test, ids_test)
            }
            if has_val:
                new_splits['val'] = (X_val_scaled, y_val, ids_val)
            self.splits = new_splits

            self.logger.info(f"Preprocessing complete. Final features: {len(self.feature_names)}")
            return True
        except Exception as e:
            self.logger.error(f"Error preprocessing splits: {e}")
            return False

    def run_outer_cv(self, k_folds: int = 5, val_ratio: float = 0.2):
        """Run outer Stratified K-Fold evaluation with per-fold Train/Val split.

        Preprocessing and model selection are fit exclusively on Train to avoid leakage.
        """
        import numpy as np
        
        self.logger.info("Starting Enhanced Outer Stratified K-Fold evaluation")

        # Stage 0: Load data once
        if not self.load_data():
            self.logger.error("Failed to load data for outer CV")
            return False

        # Prepare outer folds (~20% test when k_folds=5)
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.random_state)

        outer_results = []
        # Collect per-model test results across folds
        per_model_results = {}

        for fold_idx, (train_pool_idx, test_idx) in enumerate(skf.split(self.X, self.y), start=1):
            self.logger.info(f"\n{'='*60}\nStarting OUTER FOLD {fold_idx}/{k_folds}\n{'='*60}")

            # Reset per-fold feature name state to the full original list to avoid progressive shrink
            if hasattr(self, '_original_feature_names'):
                self.feature_names = list(self._original_feature_names)
            # Reset selection cache
            self.selected_features = None

            X_train_pool = self.X[train_pool_idx]
            y_train_pool = self.y[train_pool_idx]
            ids_train_pool = self.subject_ids[train_pool_idx]

            X_test_raw = self.X[test_idx]
            y_test_raw = self.y[test_idx]
            ids_test_raw = self.subject_ids[test_idx]

            # Inner Train/Val split of train pool
            if val_ratio and val_ratio > 0:
                X_train_raw, X_val_raw, y_train_raw, y_val_raw, ids_train_raw, ids_val_raw = train_test_split(
                    X_train_pool,
                    y_train_pool,
                    ids_train_pool,
                    test_size=val_ratio,
                    random_state=self.random_state,
                    stratify=y_train_pool
                )
                # Set raw splits
                self.splits = {
                    'train': (X_train_raw, y_train_raw, ids_train_raw),
                    'val': (X_val_raw, y_val_raw, ids_val_raw),
                    'test': (X_test_raw, y_test_raw, ids_test_raw)
                }
            else:
                self.splits = {
                    'train': (X_train_pool, y_train_pool, ids_train_pool),
                    'test': (X_test_raw, y_test_raw, ids_test_raw)
                }

            # Fold-specific output directory
            original_output_dir = self.output_dir
            fold_output_dir = original_output_dir / f"outercv_fold_{fold_idx}"
            self.output_dir = fold_output_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.setup_logging()

            # Preprocess with fit-on-train only
            if not self._preprocess_splits_train_only():
                self.logger.error(f"Fold {fold_idx}: preprocessing failed")
                # restore and continue
                self.output_dir = original_output_dir
                self.setup_logging()
                continue
            
            # Create label remapping for this fold if binary classification
            if self.binary_only and hasattr(self, 'binary_labels'):
                unique_labels = sorted(self.binary_labels)
                self.original_to_training_labels = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
                self.training_to_original_labels = {v: k for k, v in self.original_to_training_labels.items()}
                self.logger.info(f"Fold {fold_idx} label remapping: {self.original_to_training_labels}")
                self.logger.debug(f"Fold {fold_idx} training_to_original_labels: {self.training_to_original_labels}")
                self.logger.debug(f"Fold {fold_idx} y type: {type(self.y)}, y unique: {np.unique(self.y)}")

            # Proceed with modeling stages (no leakage)
            stages = [
                ("Model Definition", self.define_models),
                ("Model Training", self.train_models),
                ("Model Evaluation", self.evaluate_models),
                ("Ensemble Creation", self.create_ensemble),
                ("Saving Artifacts", self.save_artifacts)
            ]

            fold_success = True
            for stage_name, stage_func in stages:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"Starting {stage_name}")
                self.logger.info(f"{'='*50}")
                if not stage_func():
                    self.logger.error(f"Fold {fold_idx} failed at {stage_name}")
                    fold_success = False
                    break

            if fold_success and 'Ensemble' in self.results:
                test_metrics = self.results['Ensemble']['test']
                outer_results.append({
                    'fold': fold_idx,
                    'accuracy': float(test_metrics.get('accuracy', 0.0)),
                    'precision': float(test_metrics.get('precision', 0.0)),
                    'recall': float(test_metrics.get('recall', 0.0)),
                    'f1': float(test_metrics.get('f1', 0.0)),
                    'auc': float(test_metrics.get('auc', 0.0)),
                    'mcc': float(matthews_corrcoef(test_metrics.get('true_labels'), test_metrics.get('predictions'))) if (test_metrics.get('true_labels') is not None and test_metrics.get('predictions') is not None) else None
                })

                # Aggregate per-model results for summary PNGs
                for model_name, results in self.results.items():
                    # Each entry has splits; we want test split
                    if 'test' not in results:
                        continue
                    test_res = results['test']
                    y_true = test_res.get('true_labels')
                    y_pred = test_res.get('predictions')
                    # Compute MCC if not present
                    mcc = None
                    try:
                        if y_true is not None and y_pred is not None:
                            mcc = float(matthews_corrcoef(y_true, y_pred))
                    except Exception:
                        mcc = None
                    # Compute confusion matrix using original labels
                    try:
                        # Ensure labels are regular integers for confusion_matrix
                        if y_true is not None and y_pred is not None:
                            y_true_int = y_true.astype(int)
                            y_pred_int = y_pred.astype(int)
                            # Ensure we have valid labels for confusion matrix
                            if len(np.unique(y_true_int)) > 0 and len(np.unique(y_pred_int)) > 0:
                                cm = confusion_matrix(y_true_int, y_pred_int)
                            else:
                                cm = None
                        else:
                            cm = None
                    except Exception as e:
                        self.logger.warning(f"Could not compute confusion matrix: {e}")
                        cm = None

                    entry = {
                        'accuracy': float(test_res.get('accuracy', 0.0)),
                        'precision': float(test_res.get('precision', 0.0)),
                        'recall': float(test_res.get('recall', 0.0)),
                        'f1': float(test_res.get('f1', 0.0)),
                        'auc': float(test_res.get('auc', 0.0)),
                        'mcc': mcc,
                        'cm': cm
                    }
                    # Store ROC data only for Ensemble to plot per-fold ROC curves
                    if model_name == 'Ensemble':
                        entry['y_true'] = y_true
                        entry['y_prob'] = test_res.get('probabilities')
                    per_model_results.setdefault(model_name, []).append(entry)

            # restore output dir
            self.output_dir = original_output_dir
            self.setup_logging()

        # Save outer CV summary
        try:
            import json
            from statistics import mean
            summary = {
                'k_folds': k_folds,
                'val_ratio': val_ratio,
                'random_state': self.random_state,
                'folds': outer_results
            }
            if outer_results:
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc']:
                    vals = [f[metric] for f in outer_results if f.get(metric) is not None]
                    if len(vals) > 0:
                        summary[f'{metric}_mean'] = float(mean(vals))
            (self.output_dir / 'outer_cv_summary.json').write_text(json.dumps(summary, indent=2))
            self.logger.info(f"Outer CV summary saved to {self.output_dir / 'outer_cv_summary.json'}")
            # Create PNG summary plot with means, SD and 95% CI
            try:
                # Compute averaged confusion matrix and collect ROC data for the Ensemble across folds (if available)
                ensemble_entries = per_model_results.get('Ensemble', []) if 'Ensemble' in per_model_results else []
                ensemble_cms = []
                ensemble_rocs = []
                try:
                    for entry in ensemble_entries:
                        cm = entry.get('cm')
                        if cm is not None:
                            ensemble_cms.append(cm)
                        y_true = entry.get('y_true')
                        y_prob = entry.get('y_prob')
                        if y_true is not None and y_prob is not None:
                            ensemble_rocs.append((y_true, y_prob))
                except Exception:
                    pass

                avg_cm = None
                if len(ensemble_cms) > 0:
                    try:
                        import numpy as np
                        avg_cm = np.mean(np.stack(ensemble_cms, axis=0), axis=0)
                    except Exception:
                        avg_cm = ensemble_cms[0]

                self._save_outer_cv_summary_plot(
                    outer_results,
                    output_path=self.output_dir / 'outer_cv_summary.png',
                    ensemble_avg_cm=avg_cm,
                    ensemble_rocs=ensemble_rocs
                )
                self.logger.info(f"Outer CV summary plot saved to {self.output_dir / 'outer_cv_summary.png'}")
            except Exception as plot_err:
                self.logger.error(f"Failed to write outer CV summary plot: {plot_err}")

            # Per-model summary PNGs removed as requested
        except Exception as e:
            self.logger.error(f"Failed to write outer CV summary: {e}")

        self.logger.info("Enhanced Outer Stratified K-Fold evaluation complete")
        return True

    def _save_outer_cv_summary_plot(self, outer_results, output_path, ensemble_avg_cm=None, ensemble_rocs=None):
        """Create a PNG summarizing outer CV results with:
        - Mean and 95% CI per metric, with per-fold points and numeric annotations
        - Averaged confusion matrix across folds for the Ensemble (if provided)
        - Color heatmap table of per-fold metric values
        - ROC curves for the 5 ensemble models
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc']
        values_by_metric = {}
        for m in metrics:
            vals = [float(d[m]) for d in outer_results if d.get(m) is not None]
            if len(vals) == 0:
                vals = [0.0]
            values_by_metric[m] = np.array(vals, dtype=float)

        means = [values_by_metric[m].mean() for m in metrics]
        stds = [values_by_metric[m].std(ddof=1) if len(values_by_metric[m]) > 1 else 0.0 for m in metrics]
        ns = [len(values_by_metric[m]) for m in metrics]
        cis = []
        for mean_val, std_val, n in zip(means, stds, ns):
            if n > 1:
                se = std_val / np.sqrt(n)
                ci = 1.96 * se
            else:
                ci = 0.0
            cis.append(ci)

        # Build per-fold table data
        rows = []
        for i, fold_dict in enumerate(outer_results, start=1):
            rows.append([
                f"Fold {i}",
                f"{fold_dict.get('accuracy', 0.0):.3f}",
                f"{fold_dict.get('precision', 0.0):.3f}",
                f"{fold_dict.get('recall', 0.0):.3f}",
                f"{fold_dict.get('f1', 0.0):.3f}",
                f"{fold_dict.get('auc', 0.0):.3f}",
                f"{fold_dict.get('mcc', 0.0):.3f}"
            ])
        table_df = pd.DataFrame(rows, columns=['Fold', 'ACC', 'PREC', 'REC', 'F1', 'AUC', 'MCC'])

        # Layout: 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('K Fold Ensemble Model Comparison', fontsize=18)

        # Top-left: bar chart with 95% CI and points
        x = np.arange(len(metrics))
        ax0 = axes[0, 0]
        bars = ax0.bar(x, means, yerr=cis, capsize=6, color='#4C78A8', alpha=0.85, label='Mean (95% CI)')

        # Overlay fold points
        for i, m in enumerate(metrics):
            y_points = values_by_metric[m]
            jitter = (np.random.rand(len(y_points)) - 0.5) * 0.15
            ax0.scatter(np.full_like(y_points, x[i]) + jitter, y_points, color='#F58518', alpha=0.8, s=30, label='Fold scores' if i == 0 else None)

        # Annotate mean values above bars and SD below
        for i, (b, mean_val, sd_val) in enumerate(zip(bars, means, stds)):
            ax0.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{mean_val:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax0.text(b.get_x() + b.get_width() / 2, max(0.01, mean_val) - 0.06, f"SD={sd_val:.3f}", ha='center', va='top', fontsize=9)

        ax0.set_xticks(x)
        ax0.set_xticklabels([m.upper() for m in metrics], ha='right')
        ax0.set_ylim(0.0, 1.05)
        ax0.set_ylabel('Score')
        ax0.set_title('Test Metrics (Means, 95% CI, and Fold Scores)')
        ax0.legend()
        ax0.grid(True, axis='y', alpha=0.3)

        # Top-right: Averaged confusion matrix (if provided)
        ax1 = axes[0, 1]
        if ensemble_avg_cm is not None:
            # Use disease labels for confusion matrix based on actual labels
            if hasattr(self, 'binary_labels') and self.binary_only:
                disease_labels = [self.label_to_disease.get(label, str(label)) for label in sorted(self.binary_labels)]
            else:
                disease_labels = [self.label_to_disease.get(label, str(label)) for label in sorted(set(self.y))]
            
            sns.heatmap(ensemble_avg_cm, annot=True, fmt='.1f', cmap='Blues', cbar=True, ax=ax1,
                       xticklabels=disease_labels, yticklabels=disease_labels)
            ax1.set_title('Averaged Confusion Matrix (Ensemble, Test)')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')
        else:
            ax1.axis('off')
            ax1.text(0.5, 0.5, 'Averaged confusion matrix unavailable', ha='center', va='center', fontsize=11, alpha=0.7)

        # Bottom-left: Per-fold values table as color heatmap
        ax2 = axes[1, 0]
        # Convert table data to numeric for heatmap
        heatmap_data = table_df.iloc[:, 1:].astype(float).values
        heatmap_labels = table_df.iloc[:, 0].values
        
        sns.heatmap(heatmap_data, 
                    xticklabels=table_df.columns[1:], 
                    yticklabels=heatmap_labels,
                    annot=True, fmt='.3f', cmap='YlOrRd', 
                    cbar=True, ax=ax2)
        ax2.set_title('Per-fold Test Metrics (Color Heatmap)', pad=10)

        # Bottom-right: ROC curves for the 5 ensemble models
        ax3 = axes[1, 1]
        if ensemble_rocs and len(ensemble_rocs) > 0:
            for i, (y_true, y_prob) in enumerate(ensemble_rocs):
                # Convert labels to {0,1} for roc_curve if they're not consecutive
                if set(np.unique(y_true)) != {0, 1}:
                    y_true_binary = (y_true == y_true.max()).astype(int)
                else:
                    y_true_binary = y_true
                
                fpr, tpr, _ = roc_curve(y_true_binary, y_prob)
                auc_score = roc_auc_score(y_true_binary, y_prob)
                ax3.plot(fpr, tpr, label=f'Fold {i+1} (AUC = {auc_score:.3f})', alpha=0.8, linewidth=2)
            
            ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('ROC Curves - Ensemble Models (5 Folds)')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.axis('off')
            ax3.text(0.5, 0.5, 'ROC curves unavailable', ha='center', va='center', fontsize=11, alpha=0.7)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _save_per_model_summary_plot(self, model_name, folds, output_path):
        """Summarize a single model across outer folds with metrics and confusion matrix.

        folds: list of dicts with keys: accuracy, precision, recall, f1, auc, mcc, cm
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc']
        values_by_metric = {}
        for m in metrics:
            vals = [float(d[m]) for d in folds if d.get(m) is not None]
            if len(vals) == 0:
                vals = [0.0]
            values_by_metric[m] = np.array(vals, dtype=float)

        means = [values_by_metric[m].mean() for m in metrics]
        stds = [values_by_metric[m].std(ddof=1) if len(values_by_metric[m]) > 1 else 0.0 for m in metrics]
        ns = [len(values_by_metric[m]) for m in metrics]
        cis = []
        for mean_val, std_val, n in zip(means, stds, ns):
            if n > 1:
                se = std_val / np.sqrt(n)
                ci = 1.96 * se
            else:
                ci = 0.0
            cis.append(ci)

        # Aggregate confusion matrix by summing across folds
        cms = [d['cm'] for d in folds if d.get('cm') is not None]
        if cms:
            try:
                agg_cm = np.sum(np.stack(cms, axis=0), axis=0)
            except Exception:
                agg_cm = cms[0]
        else:
            agg_cm = np.array([[0, 0], [0, 0]])

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle(f'Model Summary: {model_name}', fontsize=14)

        # Bar chart with 95% CI
        x = np.arange(len(metrics))
        ax0 = axes[0, 0]
        ax0.bar(x, means, yerr=cis, capsize=6, color='#4C78A8', alpha=0.9)
        ax0.set_xticks(x)
        ax0.set_xticklabels([m.upper() for m in metrics], rotation=0)
        ax0.set_ylim(0.0, 1.05)
        ax0.set_ylabel('Score')
        ax0.set_title('Mean (95% CI) across folds')
        ax0.grid(True, axis='y', alpha=0.3)

        # Overlay individual fold points
        for i, m in enumerate(metrics):
            pts = values_by_metric[m]
            jitter = (np.random.rand(len(pts)) - 0.5) * 0.15
            ax0.scatter(np.full_like(pts, x[i]) + jitter, pts, color='#F58518', alpha=0.7, s=30)

        # SD annotations
        for i, s in enumerate(stds):
            ax0.text(x[i], max(0.01, means[i]) - 0.05, f"SD={s:.3f}", ha='center', va='top', fontsize=9)

        # Confusion matrix heatmap (aggregated)
        ax1 = axes[0, 1]
        # Use disease labels for confusion matrix based on actual labels
        if hasattr(self, 'binary_labels') and self.binary_only:
            disease_labels = [self.label_to_disease.get(label, str(label)) for label in sorted(self.binary_labels)]
        else:
            disease_labels = [self.label_to_disease.get(label, str(label)) for label in sorted(set(self.y))]
        
        sns.heatmap(agg_cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1,
                   xticklabels=disease_labels, yticklabels=disease_labels)
        ax1.set_title('Aggregated Confusion Matrix (Test)')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')

        # Text panel with exact means/SD/CIs
        ax2 = axes[1, 0]
        ax2.axis('off')
        lines = [
            f"{m.upper()}: mean={means[i]:.3f}, sd={stds[i]:.3f}, 95% CI=±{cis[i]:.3f}" for i, m in enumerate(metrics)
        ]
        ax2.text(0.0, 1.0, "\n".join(lines), fontsize=11, va='top')

        # Placeholder for ROC would require storing probabilities and labels per fold;
        # keep final axis empty or future extension
        ax3 = axes[1, 1]
        ax3.axis('off')
        ax3.text(0.5, 0.5, 'See per-fold plots for ROC curves', ha='center', va='center', fontsize=10, alpha=0.6)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Enhanced Radiomics Classification Pipeline')
    parser.add_argument('--input', 
                       default='~/reseng202500013-ndd-ml/data/radiomics_MRI_mri_labels.csv',
                       help='Path to radiomics CSV file')
    parser.add_argument('--output-dir', 
                       default='~/reseng202500013-ndd-ml/data/classic_results/enhanced_run',
                       help='Output directory for results')
    parser.add_argument('--random-state', 
                       type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--binary-only', 
                       action='store_true', default=True,
                       help='Use only binary classification (labels 0 and 1)')
    parser.add_argument('--multi-class', 
                       action='store_true', default=False,
                       help='Use multi-class classification (all labels)')
    
    args = parser.parse_args()
    
    # Handle binary vs multi-class
    if args.multi_class:
        binary_only = False
    else:
        binary_only = True
    
    # Expand user paths
    input_path = os.path.expanduser(args.input)
    output_dir = os.path.expanduser(args.output_dir)
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        sys.exit(1)
    
    print("Starting Enhanced Radiomics Classification Pipeline")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Random seed: {args.random_state}")
    print(f"Classification: {'Binary (0,1)' if binary_only else 'Multi-class'}")
    print("=" * 60)
    
    # Initialize and run pipeline
    classifier = EnhancedRadiomicsClassifier(input_path, output_dir, args.random_state, binary_only)
    success = classifier.run_pipeline()
    
    if success:
        print("\n" + "=" * 60)
        print("Enhanced pipeline completed successfully!")
        print(f"Results saved to: {output_dir}")
        print("\nGenerated files:")
        print(f"  • Multiple model files (.pkl)")
        print(f"  • Feature importance comparison")
        print(f"  • Enhanced evaluation plots")
        print(f"  • Comprehensive results summary")
        print(f"  • Ensemble model results")
    else:
        print("\nPipeline failed! Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 