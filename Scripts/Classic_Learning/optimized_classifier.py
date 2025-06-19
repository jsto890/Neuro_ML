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
    train_test_split, RandomizedSearchCV, validation_curve
)
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, 
    RFE, SelectFromModel, VarianceThreshold, RFECV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, make_scorer
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
import os
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

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
                self.data = self.data[self.data['label'].isin([0, 1])]
                final_count = len(self.data)
                self.logger.info(f"Filtered to binary classification: {initial_count} → {final_count} samples")
                
                if final_count == 0:
                    raise ValueError("No samples remaining after binary filtering")
            
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
    
    def advanced_feature_engineering(self):
        """Stage 1: Advanced feature engineering with polynomial features, family interactions, and statistical summaries."""
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
            
            # 3. Enhanced Feature Engineering
            engineered_features = []
            feature_names = []
            
            # Get top 10 features for polynomial features
            top_10_features = available_top_features[:10]
            top_10_indices = [selected_features.index(f) for f in top_10_features]
            X_top_10 = X_var_selected[:, top_10_indices]
            
            # 3a. Polynomial features (2nd and 3rd degree)
            poly_2 = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            poly_3 = PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)
            
            X_poly_2 = poly_2.fit_transform(X_top_10)
            X_poly_3 = poly_3.fit_transform(X_top_10)
            
            # Get feature names for polynomial features
            poly_2_names = [f"poly2_{i}" for i in range(X_poly_2.shape[1] - X_top_10.shape[1])]
            poly_3_names = [f"poly3_{i}" for i in range(X_poly_3.shape[1] - X_top_10.shape[1])]
            
            engineered_features.extend([X_poly_2[:, X_top_10.shape[1]:], X_poly_3[:, X_top_10.shape[1]:]])
            feature_names.extend(poly_2_names + poly_3_names)
            
            # 3b. Family-based interaction features
            family_groups = {
                'firstorder': [f for f in available_top_features if 'firstorder' in f],
                'glrlm': [f for f in available_top_features if 'glrlm' in f],
                'gldm': [f for f in available_top_features if 'gldm' in f],
                'glszm': [f for f in available_top_features if 'glszm' in f],
                'ngtdm': [f for f in available_top_features if 'ngtdm' in f]
            }
            
            family_interactions = []
            family_interaction_names = []
            
            for family1 in family_groups:
                for family2 in family_groups:
                    if family1 < family2:  # Avoid duplicates
                        features1 = family_groups[family1]
                        features2 = family_groups[family2]
                        
                        if features1 and features2:
                            indices1 = [selected_features.index(f) for f in features1]
                            indices2 = [selected_features.index(f) for f in features2]
                            
                            X_family1 = X_var_selected[:, indices1]
                            X_family2 = X_var_selected[:, indices2]
                            
                            # Create interaction features (element-wise multiplication)
                            for i, feat1 in enumerate(features1):
                                for j, feat2 in enumerate(features2):
                                    interaction = X_family1[:, i] * X_family2[:, j]
                                    family_interactions.append(interaction)
                                    family_interaction_names.append(f"family_interaction_{family1}_{family2}_{i}_{j}")
            
            if family_interactions:
                family_interactions = np.column_stack(family_interactions)
                engineered_features.append(family_interactions)
                feature_names.extend(family_interaction_names)
            
            # 3c. Statistical summary features
            # Percentiles, skewness, kurtosis across feature groups
            for family_name, family_features in family_groups.items():
                if len(family_features) > 1:
                    indices = [selected_features.index(f) for f in family_features]
                    X_family = X_var_selected[:, indices]
                    
                    # Percentiles
                    p25 = np.percentile(X_family, 25, axis=1)
                    p75 = np.percentile(X_family, 75, axis=1)
                    p90 = np.percentile(X_family, 90, axis=1)
                    
                    # Skewness and kurtosis
                    skewness = skew(X_family, axis=1)
                    kurt = kurtosis(X_family, axis=1)
                    
                    # Range and IQR
                    feature_range = np.ptp(X_family, axis=1)
                    iqr = p75 - p25
                    
                    engineered_features.append(np.column_stack([p25, p75, p90, skewness, kurt, feature_range, iqr]))
                    feature_names.extend([
                        f"{family_name}_p25", f"{family_name}_p75", f"{family_name}_p90",
                        f"{family_name}_skewness", f"{family_name}_kurtosis",
                        f"{family_name}_range", f"{family_name}_iqr"
                    ])
            
            # 3d. Original interaction features (simplified)
            X_top_10_df = pd.DataFrame(X_top_10, columns=top_10_features)
            interactions = []
            interaction_names = []
            
            for i in range(len(top_10_features)):
                for j in range(i+1, len(top_10_features)):
                    interaction = X_top_10_df.iloc[:, i] * X_top_10_df.iloc[:, j]
                    interactions.append(interaction.values)
                    interaction_names.append(f"interaction_{i}_{j}")
            
            if interactions:
                interactions = np.column_stack(interactions)
                engineered_features.append(interactions)
                feature_names.extend(interaction_names)
            
            # 3e. Summary features
            texture_features = [f for f in available_top_features if any(x in f for x in ['glrlm', 'gldm', 'glszm', 'ngtdm'])]
            if texture_features:
                texture_indices = [selected_features.index(f) for f in texture_features]
                X_texture = X_var_selected[:, texture_indices]
                texture_mean = np.mean(X_texture, axis=1)
                texture_std = np.std(X_texture, axis=1)
                engineered_features.append(np.column_stack([texture_mean, texture_std]))
                feature_names.extend(['texture_mean', 'texture_std'])
            
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
            
            # 4. Scaling
            self.scaler = RobustScaler()
            self.X = self.scaler.fit_transform(X_combined)
            self.logger.info("Features scaled using RobustScaler")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in advanced feature engineering: {e}")
            return False
    
    def optimized_feature_selection(self):
        """Stage 2: Optimized feature selection with RFECV and preliminary reduction."""
        self.logger.info("Stage 2: Optimized feature selection...")
        
        try:
            # Preliminary feature reduction to handle large feature set
            if self.X.shape[1] > 100:
                self.logger.info(f"Large feature set detected ({self.X.shape[1]} features), applying preliminary reduction...")
                
                # Use mutual information for initial feature selection
                from sklearn.feature_selection import SelectKBest, mutual_info_classif
                k_best = min(100, self.X.shape[1] // 2)  # Select top 50% or 100 features, whichever is smaller
                selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
                X_reduced = selector.fit_transform(self.X, self.y)
                selected_indices = selector.get_support()
                self.feature_names = [self.feature_names[i] for i in range(len(self.feature_names)) if selected_indices[i]]
                self.X = X_reduced
                
                self.logger.info(f"Preliminary reduction: {self.X.shape[1]} features selected")
            
            # RFECV with optimized SVM parameters
            base_svm = SVC(
                kernel='linear',
                C=1.0,
                class_weight='balanced',
                probability=True,
                random_state=self.random_state,
                max_iter=2000  # Increased max_iter to prevent convergence warnings
            )
            
            rfecv = RFECV(
                estimator=base_svm,
                step=1,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                min_features_to_select=10
            )
            
            self.X = rfecv.fit_transform(self.X, self.y)
            selected_features = [self.feature_names[i] for i in range(len(self.feature_names)) if rfecv.support_[i]]
            self.feature_names = selected_features
            
            # Store RFECV results
            self.feature_engineering_results['rfecv'] = {
                'n_features': rfecv.n_features_,
                'cv_score': rfecv.cv_results_['mean_test_score'].max(),
                'selected_features': selected_features,
                'feature_ranking': rfecv.ranking_.tolist()
            }
            
            self.logger.info(f"RFECV selected {rfecv.n_features_} features")
            self.logger.info(f"Optimal number of features: {rfecv.n_features_}")
            self.logger.info(f"Cross-validation score: {rfecv.cv_results_['mean_test_score'].max():.4f}")
            
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
        """Stage 4: Optimize SVM hyperparameters with extended parameter grid."""
        self.logger.info("Stage 4: Optimizing SVM hyperparameters...")
        
        try:
            X_train, y_train, _ = self.splits['train']
            
            # Extended parameter grid with higher max_iter
            param_grid = {
                'C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['linear', 'rbf', 'poly'],
                'degree': [2, 3],  # for poly kernel
                'class_weight': ['balanced', None],
                'max_iter': [2000]  # Higher max_iter to prevent convergence warnings
            }
            
            # Create base SVM
            base_svm = SVC(probability=True, random_state=self.random_state)
            
            # Grid search with cross-validation
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
                'best_params': best_params,
                'best_cv_score': grid_search.best_score_,
                'cv_results': {
                    'mean_fit_time': grid_search.cv_results_['mean_fit_time'].tolist(),
                    'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                    'rank_test_score': grid_search.cv_results_['rank_test_score'].tolist()
                }
            }
            
            self.logger.info(f"Best SVM parameters: {best_params}")
            self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in SVM optimization: {e}")
            return False
    
    def create_optimized_ensemble(self):
        """Stage 5: Creating advanced stacking ensemble with cross-validation."""
        self.logger.info("Stage 5: Creating advanced stacking ensemble...")
        
        try:
            X_train, y_train, _ = self.splits['train']
            
            # 1. Define base models
            base_models = {
                'svm_linear': SVC(kernel='linear', probability=True, random_state=self.random_state, max_iter=2000),
                'svm_rbf': SVC(kernel='rbf', probability=True, random_state=self.random_state, max_iter=2000),
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
                    max_iter=2000
                )
            }
            
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
                max_iter=1000
            )
            
            # Use cross-validation to train meta-learner
            meta_learner.fit(meta_features, y_train)
            
            # 5. Create stacking ensemble class
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
            
            # 6. Create final ensemble
            self.ensemble_model = StackingEnsemble(base_models, meta_learner, meta_feature_names)
            
            # 7. Store ensemble information
            self.ensemble_info = {
                'base_models': list(base_models.keys()),
                'meta_learner': type(meta_learner).__name__,
                'meta_feature_names': meta_feature_names,
                'base_predictions': base_predictions,
                'base_probabilities': base_probabilities
            }
            
            self.logger.info("Advanced stacking ensemble created successfully")
            self.logger.info(f"Base models: {list(base_models.keys())}")
            self.logger.info(f"Meta-learner: {type(meta_learner).__name__}")
            
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
            if 'rfecv' in self.feature_engineering_results:
                rfecv_results = self.feature_engineering_results['rfecv']
                axes[1, 2].text(0.1, 0.8, f"Original Features: {len(self.top_features)}", fontsize=12)
                axes[1, 2].text(0.1, 0.7, f"Engineered Features: {len(self.feature_names) - len(self.top_features)}", fontsize=12)
                axes[1, 2].text(0.1, 0.6, f"Selected Features: {rfecv_results['n_features']}", fontsize=12)
                axes[1, 2].text(0.1, 0.5, f"CV Score: {rfecv_results['cv_score']:.4f}", fontsize=12)
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
        print(f"  • RFECV feature selection optimized for SVM")
        print(f"  • Fine-tuned SVM hyperparameters")
        print(f"  • Ensemble with SVM as primary model")
        print(f"  • Clinical interpretability focus")
    else:
        print("\nPipeline failed! Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 