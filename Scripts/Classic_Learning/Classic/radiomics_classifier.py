#!/usr/bin/env python3
"""
Radiomics-based Classical Learning Pipeline
===========================================

A comprehensive pipeline for training and evaluating Random Forest models
on radiomics features extracted from neuroimaging data.

FIXED: Data leakage issue - preprocessing now applied correctly after data splitting.

Stages:
1. Data Loading
2. Data Splitting (FIRST - before any preprocessing)
3. Preprocessing (fitted on training data only)
4. Model Training (Random Forest)
5. Evaluation
6. Interpretation
7. Output Artifacts

Usage:
    python radiomics_classifier.py --input ~/reseng202500013-ndd-ml/data/radiomics_MRI_mri_labels.csv --output-dir results/
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import logging

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RadiomicsClassifier:
    def __init__(self, input_path, output_dir, random_state=42, binary_only=True):
        """
        Initialize the RadiomicsClassifier.
        
        Args:
            input_path (str): Path to radiomics CSV file
            output_dir (str): Output directory for results
            random_state (int): Random seed for reproducibility
            binary_only (bool): If True, only use labels 0 and 1 (binary classification)
        """
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.binary_only = binary_only
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize preprocessing components (will be fitted on training data only)
        self.variance_selector = VarianceThreshold(threshold=0.01)
        self.scaler = StandardScaler()
        self.feature_selector = None  # Will be set during preprocessing
        
        # Initialize model components
        self.model = None
        self.best_params = None
        self.cv_results = None
        self.results = None
        self.feature_importance_df = None
        
        # Setup logging
        self.setup_logging()
        
        # Initialize data containers
        self.data = None
        self.X = None
        self.y = None
        self.subject_ids = None
        self.feature_names = None
        self.splits = None
        
    def setup_logging(self):
        """Set up logging to both console and file."""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'pipeline.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data(self):
        """Stage 1: Load and validate input data."""
        self.logger.info("Stage 1: Loading data...")
        
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
    
    def split_data(self, test_size=0.2, val_size=0.2):
        """Stage 2: Train-validation-test split (BEFORE any preprocessing)."""
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
            self.logger.info(f"  Train: {X_train.shape[0]} samples")
            self.logger.info(f"  Validation: {X_val.shape[0]} samples")
            self.logger.info(f"  Test: {X_test.shape[0]} samples")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in data splitting: {e}")
            return False
    
    def preprocess_data(self):
        """Stage 3: Data preprocessing (fitted on training data only)."""
        self.logger.info("Stage 3: Preprocessing data (training data only)...")
        
        try:
            X_train, y_train, _ = self.splits['train']
            X_val, y_val, _ = self.splits['val']
            X_test, y_test, _ = self.splits['test']
            
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
            
            # Remove constant features (fitted on train, applied to all)
            X_train_var = self.variance_selector.fit_transform(X_train)
            X_val_var = self.variance_selector.transform(X_val)
            X_test_var = self.variance_selector.transform(X_test)
            
            # Update feature names after variance thresholding
            kept_features = self.variance_selector.get_support()
            self.feature_names = [f for f, keep in zip(self.feature_names, kept_features) if keep]
            
            self.logger.info(f"After variance thresholding: {X_train_var.shape}")
            
            # Standardize features (fitted on train, applied to all)
            X_train_scaled = self.scaler.fit_transform(X_train_var)
            X_val_scaled = self.scaler.transform(X_val_var)
            X_test_scaled = self.scaler.transform(X_test_var)
            
            self.logger.info("Features standardized (fitted on training data)")
            
            # Feature selection (fitted on train, applied to all)
            if X_train_scaled.shape[1] > 100:
                self.feature_selector = SelectKBest(score_func=f_classif, k=100)
                X_train_final = self.feature_selector.fit_transform(X_train_scaled, y_train)
                X_val_final = self.feature_selector.transform(X_val_scaled)
                X_test_final = self.feature_selector.transform(X_test_scaled)
                
                # Update feature names after feature selection
                kept_features = self.feature_selector.get_support()
                self.feature_names = [f for f, keep in zip(self.feature_names, kept_features) if keep]
                
                self.logger.info(f"Selected top 100 features (fitted on training data): {X_train_final.shape}")
            else:
                X_train_final = X_train_scaled
                X_val_final = X_val_scaled
                X_test_final = X_test_scaled
            
            # Update splits with preprocessed data
            self.splits = {
                'train': (X_train_final, y_train, self.splits['train'][2]),
                'val': (X_val_final, y_val, self.splits['val'][2]),
                'test': (X_test_final, y_test, self.splits['test'][2])
            }
            
            self.logger.info(f"Preprocessing completed. Final feature count: {len(self.feature_names)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {e}")
            return False
    
    def train_model(self):
        """Stage 4: Train Random Forest model with hyperparameter tuning."""
        self.logger.info("Stage 4: Training Random Forest model...")
        
        try:
            X_train, y_train, _ = self.splits['train']
            
            # Define parameter grid for GridSearchCV
            param_grid = {
                'n_estimators': [100, 200, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Initialize base model
            base_rf = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            # Grid search with cross-validation
            self.logger.info("Performing grid search...")
            grid_search = GridSearchCV(
                base_rf, param_grid, cv=5, scoring='roc_auc',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.cv_results = grid_search.cv_results_
            
            self.logger.info(f"Best parameters: {self.best_params}")
            self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in model training: {e}")
            return False
    
    def evaluate_model(self):
        """Stage 5: Evaluate model performance."""
        self.logger.info("Stage 5: Evaluating model...")
        
        try:
            results = {}
            
            for split_name, (X_split, y_split, ids_split) in self.splits.items():
                self.logger.info(f"Evaluating on {split_name} set...")
                
                # Predictions
                y_pred = self.model.predict(X_split)
                y_pred_proba = self.model.predict_proba(X_split)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_split, y_pred)
                precision = precision_score(y_split, y_pred, average='weighted')
                recall = recall_score(y_split, y_pred, average='weighted')
                f1 = f1_score(y_split, y_pred, average='weighted')
                auc = roc_auc_score(y_split, y_pred_proba)
                
                # Store results
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
                
                self.logger.info(f"{split_name.capitalize()} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            
            self.results = results
            return True
            
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {e}")
            return False
    
    def interpret_model(self):
        """Stage 6: Model interpretation and feature importance."""
        self.logger.info("Stage 6: Model interpretation...")
        
        try:
            # Extract feature importances
            importances = self.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Save feature importance
            feature_importance_df.to_csv(self.output_dir / 'feature_importance.csv', index=False)
            
            # Top features
            top_features = feature_importance_df.head(20)
            self.logger.info(f"Top 5 features: {top_features['feature'].head().tolist()}")
            
            self.feature_importance_df = feature_importance_df
            return True
            
        except Exception as e:
            self.logger.error(f"Error in model interpretation: {e}")
            return False
    
    def save_artifacts(self):
        """Stage 7: Save all artifacts and generate plots."""
        self.logger.info("Stage 7: Saving artifacts...")
        
        try:
            # Save model
            with open(self.output_dir / 'random_forest_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save preprocessing components
            with open(self.output_dir / 'variance_selector.pkl', 'wb') as f:
                pickle.dump(self.variance_selector, f)
            
            with open(self.output_dir / 'scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            if self.feature_selector:
                with open(self.output_dir / 'feature_selector.pkl', 'wb') as f:
                    pickle.dump(self.feature_selector, f)
            
            # Save results summary
            summary = {
                'timestamp': datetime.now().isoformat(),
                'input_file': self.input_path,
                'data_shape': self.X.shape,
                'final_feature_count': len(self.feature_names),
                'best_parameters': self.best_params,
                'feature_names': self.feature_names,
                'preprocessing_info': {
                    'variance_threshold': 0.01,
                    'scaling_method': 'StandardScaler',
                    'feature_selection': 'SelectKBest (f_classif, k=100)' if self.feature_selector else 'None',
                    'data_leakage_fixed': True
                },
                'results': {
                    split: {k: v for k, v in results.items() if k not in ['predictions', 'probabilities', 'true_labels', 'subject_ids']}
                    for split, results in self.results.items()
                }
            }
            
            with open(self.output_dir / 'results_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Generate plots
            self.generate_plots()
            
            self.logger.info(f"All artifacts saved to {self.output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving artifacts: {e}")
            return False
    
    def generate_plots(self):
        """Generate evaluation plots."""
        try:
            # Set up figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Radiomics Random Forest Model Evaluation (Data Leakage Fixed)', fontsize=16)
            
            # 1. ROC Curves
            ax1 = axes[0, 0]
            for split_name, results in self.results.items():
                fpr, tpr, _ = roc_curve(results['true_labels'], results['probabilities'])
                auc = results['auc']
                ax1.plot(fpr, tpr, label=f'{split_name.capitalize()} (AUC = {auc:.3f})')
            
            ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('ROC Curves')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Confusion Matrix (Test set)
            ax2 = axes[0, 1]
            cm = confusion_matrix(self.results['test']['true_labels'], self.results['test']['predictions'])
            im = ax2.imshow(cm, cmap='Blues', interpolation='nearest')
            ax2.set_title('Confusion Matrix (Test Set)')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
            
            # Add text annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax2.text(j, i, str(cm[i, j]), ha='center', va='center')
            
            # Add colorbar
            plt.colorbar(im, ax=ax2)
            
            # 3. Feature Importance
            ax3 = axes[1, 0]
            top_10 = self.feature_importance_df.head(10)
            ax3.barh(range(len(top_10)), top_10['importance'])
            ax3.set_yticks(range(len(top_10)))
            ax3.set_yticklabels(top_10['feature'], fontsize=8)
            ax3.set_xlabel('Feature Importance')
            ax3.set_title('Top 10 Feature Importances')
            ax3.invert_yaxis()
            
            # 4. Performance Metrics
            ax4 = axes[1, 1]
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
            splits = list(self.results.keys())
            
            x = np.arange(len(metrics))
            width = 0.25
            
            for i, split in enumerate(splits):
                values = [self.results[split][metric] for metric in metrics]
                ax4.bar(x + i*width, values, width, label=split.capitalize())
            
            ax4.set_xlabel('Metrics')
            ax4.set_ylabel('Score')
            ax4.set_title('Performance Metrics by Split')
            ax4.set_xticks(x + width)
            ax4.set_xticklabels(metrics)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'evaluation_plots.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Plots generated and saved")
            
        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")
    
    def run_pipeline(self):
        """Run the complete pipeline."""
        self.logger.info("Starting Radiomics Classification Pipeline (Data Leakage Fixed)")
        
        stages = [
            ("Data Loading", self.load_data),
            ("Data Splitting", self.split_data),
            ("Preprocessing", self.preprocess_data),
            ("Model Training", self.train_model),
            ("Model Evaluation", self.evaluate_model),
            ("Model Interpretation", self.interpret_model),
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
        self.logger.info("Pipeline completed successfully!")
        self.logger.info("✅ Data leakage issue has been fixed!")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info(f"{'='*50}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Extract radiomics features from neuroimaging data')
    parser.add_argument('--input', required=True, help='Path to radiomics CSV file')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize and run pipeline
    classifier = RadiomicsClassifier(args.input, args.output_dir)
    success = classifier.run_pipeline()
    
    if success:
        print(f"Feature extraction completed successfully!")
        print(f"Results saved to: {args.output_dir}")
    else:
        print(f"Feature extraction failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 