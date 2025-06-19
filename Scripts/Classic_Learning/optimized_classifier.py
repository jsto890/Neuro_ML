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
        """Stage 1: Advanced feature engineering based on top features."""
        self.logger.info("Stage 1: Advanced feature engineering...")
        
        try:
            # Handle missing values
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
            
            # Remove constant features
            variance_selector = VarianceThreshold(threshold=0.01)
            self.X = variance_selector.fit_transform(self.X)
            kept_features = variance_selector.get_support()
            self.feature_names = [f for f, keep in zip(self.feature_names, kept_features) if keep]
            self.logger.info(f"After variance threshold: {self.X.shape}")
            
            # Focus on top features identified from cross-model analysis
            available_top_features = [f for f in self.top_features if f in self.feature_names]
            self.logger.info(f"Found {len(available_top_features)} of {len(self.top_features)} top features")
            
            # Get indices of top features
            top_feature_indices = [self.feature_names.index(f) for f in available_top_features]
            
            # Create feature engineering matrix
            X_top = self.X[:, top_feature_indices]
            feature_names_top = available_top_features
            
            # Advanced feature engineering
            engineered_features = []
            engineered_names = []
            
            # 1. Original top features
            engineered_features.append(X_top)
            engineered_names.extend(feature_names_top)
            
            # 2. Polynomial features (interactions between top features)
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            X_poly = poly.fit_transform(X_top)
            poly_names = [f"interaction_{i}" for i in range(X_poly.shape[1] - X_top.shape[1])]
            engineered_features.append(X_poly[:, X_top.shape[1]:])  # Only interaction terms
            engineered_names.extend(poly_names)
            
            # 3. Statistical aggregations
            # Mean of texture features
            texture_features = [f for f in feature_names_top if any(x in f for x in ['glrlm', 'gldm', 'glszm', 'ngtdm'])]
            if texture_features:
                texture_indices = [feature_names_top.index(f) for f in texture_features]
                texture_mean = np.mean(X_top[:, texture_indices], axis=1, keepdims=True)
                engineered_features.append(texture_mean)
                engineered_names.append('texture_mean')
            
            # Variance of first-order features
            firstorder_features = [f for f in feature_names_top if 'firstorder' in f]
            if firstorder_features:
                firstorder_indices = [feature_names_top.index(f) for f in firstorder_features]
                firstorder_var = np.var(X_top[:, firstorder_indices], axis=1, keepdims=True)
                engineered_features.append(firstorder_var)
                engineered_names.append('firstorder_variance')
            
            # 4. Ratio features
            if 'original_firstorder_Mean' in feature_names_top and 'original_firstorder_Variance' in feature_names_top:
                mean_idx = feature_names_top.index('original_firstorder_Mean')
                var_idx = feature_names_top.index('original_firstorder_Variance')
                mean_var_ratio = (X_top[:, mean_idx:mean_idx+1] / (X_top[:, var_idx:var_idx+1] + 1e-8))
                engineered_features.append(mean_var_ratio)
                engineered_names.append('mean_variance_ratio')
            
            # 5. Z-score features (normalized versions)
            for i, feature in enumerate(feature_names_top):
                feature_data = X_top[:, i:i+1]
                z_score = (feature_data - np.mean(feature_data)) / (np.std(feature_data) + 1e-8)
                engineered_features.append(z_score)
                engineered_names.append(f'{feature}_zscore')
            
            # Combine all engineered features
            self.X = np.hstack(engineered_features)
            self.feature_names = engineered_names
            
            self.logger.info(f"After feature engineering: {self.X.shape}")
            self.logger.info(f"Feature types: {len(feature_names_top)} original, {len(engineered_names) - len(feature_names_top)} engineered")
            
            # Scale features
            self.X = self.scaler.fit_transform(self.X)
            self.logger.info("Features scaled using RobustScaler")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {e}")
            return False
    
    def optimized_feature_selection(self):
        """Stage 2: Optimized feature selection for SVM."""
        self.logger.info("Stage 2: Optimized feature selection...")
        
        try:
            # Use RFECV (Recursive Feature Elimination with Cross-Validation) for SVM
            from sklearn.svm import LinearSVC
            
            # Create a linear SVM for feature selection
            svm_selector = LinearSVC(random_state=self.random_state, max_iter=10000)
            
            # RFECV to find optimal number of features
            rfecv = RFECV(
                estimator=svm_selector,
                step=1,
                cv=StratifiedKFold(5, shuffle=True, random_state=self.random_state),
                scoring='roc_auc',
                n_jobs=-1,
                min_features_to_select=10
            )
            
            rfecv.fit(self.X, self.y)
            
            # Get selected features
            selected_features = rfecv.get_support()
            self.X = self.X[:, selected_features]
            self.feature_names = [f for f, selected in zip(self.feature_names, selected_features) if selected]
            
            self.logger.info(f"RFECV selected {len(self.feature_names)} features")
            self.logger.info(f"Optimal number of features: {rfecv.n_features_}")
            self.logger.info(f"Cross-validation score: {rfecv.cv_results_['mean_test_score'].max():.4f}")
            
            # Store feature selection results
            self.feature_engineering_results['rfecv'] = {
                'n_features': rfecv.n_features_,
                'cv_score': rfecv.cv_results_['mean_test_score'].max(),
                'selected_features': self.feature_names
            }
            
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
        """Stage 4: Fine-tune SVM hyperparameters."""
        self.logger.info("Stage 4: Optimizing SVM hyperparameters...")
        
        try:
            X_train, y_train, _ = self.splits['train']
            
            # Extended parameter grid for SVM optimization
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                'class_weight': ['balanced', None],
                'probability': [True]
            }
            
            # Use GridSearchCV for thorough search
            svm = SVC(random_state=self.random_state)
            
            grid_search = GridSearchCV(
                svm, param_grid, cv=5, scoring='roc_auc',
                n_jobs=-1, verbose=1, refit=True
            )
            
            grid_search.fit(X_train, y_train)
            
            self.svm_model = grid_search.best_estimator_
            
            self.logger.info(f"Best SVM parameters: {grid_search.best_params_}")
            self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            # Store optimization results
            self.feature_engineering_results['svm_optimization'] = {
                'best_params': grid_search.best_params_,
                'best_cv_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in SVM optimization: {e}")
            return False
    
    def create_optimized_ensemble(self):
        """Stage 5: Create optimized ensemble with SVM as base."""
        self.logger.info("Stage 5: Creating optimized ensemble...")
        
        try:
            X_train, y_train, _ = self.splits['train']
            
            # Create multiple SVM models with different parameters
            svm_models = [
                ('svm_linear', SVC(kernel='linear', C=1.0, probability=True, random_state=self.random_state)),
                ('svm_rbf', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=self.random_state)),
                ('svm_optimized', self.svm_model)
            ]
            
            # Create voting ensemble
            self.ensemble_model = VotingClassifier(
                estimators=svm_models,
                voting='soft',
                weights=[0.3, 0.3, 0.4]  # Give more weight to optimized SVM
            )
            
            # Train ensemble
            self.ensemble_model.fit(X_train, y_train)
            
            self.logger.info("Ensemble model trained successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating ensemble: {e}")
            return False
    
    def evaluate_optimized_models(self):
        """Stage 6: Evaluate optimized models."""
        self.logger.info("Stage 6: Evaluating optimized models...")
        
        try:
            models = {
                'Optimized_SVM': self.svm_model,
                'Ensemble': self.ensemble_model
            }
            
            for name, model in models.items():
                self.logger.info(f"Evaluating {name}...")
                
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
                    
                    self.logger.info(f"{name} {split_name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
                
                self.results[name] = results
                
                # Extract feature importance for SVM
                if hasattr(model, 'coef_'):
                    self.feature_importance[name] = np.abs(model.coef_[0])
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {e}")
            return False
    
    def generate_optimized_plots(self):
        """Generate optimized visualization plots."""
        self.logger.info("Generating optimized plots...")
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Optimized Radiomics Classification Results', fontsize=16)
            
            # 1. Model Comparison
            model_names = list(self.results.keys())
            test_accuracies = [self.results[name]['test']['accuracy'] for name in model_names]
            test_aucs = [self.results[name]['test']['auc'] for name in model_names]
            
            x = np.arange(len(model_names))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, test_accuracies, width, label='Accuracy', alpha=0.8)
            axes[0, 0].bar(x + width/2, test_aucs, width, label='AUC', alpha=0.8)
            axes[0, 0].set_title('Model Performance Comparison')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(model_names)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. ROC Curves
            for name in model_names:
                fpr, tpr, _ = roc_curve(
                    self.results[name]['test']['true_labels'],
                    self.results[name]['test']['probabilities']
                )
                auc_score = self.results[name]['test']['auc']
                axes[0, 1].plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
            
            axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[0, 1].set_title('ROC Curves - Test Set')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 3. Feature Importance (SVM coefficients)
            if 'Optimized_SVM' in self.feature_importance:
                svm_importance = self.feature_importance['Optimized_SVM']
                top_indices = np.argsort(svm_importance)[-15:]
                top_features = [self.feature_names[i] for i in top_indices]
                top_importance = svm_importance[top_indices]
                
                axes[0, 2].barh(range(len(top_features)), top_importance)
                axes[0, 2].set_yticks(range(len(top_features)))
                axes[0, 2].set_yticklabels([f.split('_')[-1] for f in top_features])
                axes[0, 2].set_title('Top 15 Features - Optimized SVM')
                axes[0, 2].set_xlabel('Coefficient Magnitude')
            
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
            
            # 5. Train vs Test Performance
            train_acc = [self.results[name]['train']['accuracy'] for name in model_names]
            test_acc = [self.results[name]['test']['accuracy'] for name in model_names]
            
            axes[1, 1].bar(x - width/2, train_acc, width, label='Train', alpha=0.8)
            axes[1, 1].bar(x + width/2, test_acc, width, label='Test', alpha=0.8)
            axes[1, 1].set_title('Train vs Test Accuracy')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(model_names)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Feature Engineering Summary
            if 'rfecv' in self.feature_engineering_results:
                rfecv_results = self.feature_engineering_results['rfecv']
                axes[1, 2].text(0.1, 0.8, f"Original Features: {len(self.top_features)}", fontsize=12)
                axes[1, 2].text(0.1, 0.7, f"Engineered Features: {len(self.feature_names)}", fontsize=12)
                axes[1, 2].text(0.1, 0.6, f"Selected Features: {rfecv_results['n_features']}", fontsize=12)
                axes[1, 2].text(0.1, 0.5, f"CV Score: {rfecv_results['cv_score']:.4f}", fontsize=12)
                axes[1, 2].set_title('Feature Engineering Summary')
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