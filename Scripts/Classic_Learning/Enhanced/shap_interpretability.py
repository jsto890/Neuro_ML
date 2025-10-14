"""
SHAP Interpretability Module for Classical Machine Learning Models
===================================================================

This module provides comprehensive SHAP (SHapley Additive exPlanations) analysis
for classical machine learning models in the neurodegenerative disease detection pipeline.

SHAP works by computing Shapley values from game theory to explain model predictions,
showing which features contribute most to each prediction.

Supported Models:
- Tree-based: RandomForest, XGBoost, LightGBM, GradientBoosting, ExtraTrees
- Linear: LogisticRegression, LinearSVM
- Other: SVM (with KernelExplainer), KNN (with KernelExplainer)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
import json

# SHAP library
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP library not available. Install with: pip install shap")

# ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# Optional: XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class SHAPInterpreter:
    """
    SHAP-based interpretability for classical ML models.
    
    This class provides methods to:
    - Generate SHAP values for model predictions
    - Create summary plots showing feature importance
    - Generate dependence plots showing feature interactions
    - Create force plots for individual predictions
    - Export SHAP values for further analysis
    """
    
    def __init__(self, model: Any, X_train: np.ndarray, feature_names: List[str], 
                 output_dir: Union[str, Path], model_name: str = "model",
                 class_names: Optional[List[str]] = None):
        """
        Initialize SHAP interpreter.
        
        Args:
            model: Trained sklearn-compatible model
            X_train: Training data (used as background for SHAP)
            feature_names: List of feature names
            output_dir: Directory to save SHAP outputs
            model_name: Name of the model (for file naming)
            class_names: Names of classes (e.g., ['CN', 'AD', 'PD'])
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library is required. Install with: pip install shap")
        
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.class_names = class_names or ['Class 0', 'Class 1']
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize explainer (will be set based on model type)
        self.explainer = None
        self.explainer_type = None
        self.shap_values = None
        
        # Initialize explainer
        self._initialize_explainer()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(f"SHAP_{self.model_name}")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.output_dir / f"shap_{self.model_name}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type."""
        model_class = type(self.model).__name__
        self.logger.info(f"Initializing SHAP explainer for {model_class}")
        
        try:
            # Tree-based models: use TreeExplainer (fast and exact)
            if isinstance(self.model, (RandomForestClassifier, ExtraTreesClassifier)):
                self.explainer = shap.TreeExplainer(self.model)
                self.explainer_type = "tree"
                self.logger.info(f"Using TreeExplainer for {model_class}")
            
            # GradientBoosting: TreeExplainer only supports binary, use KernelExplainer for multi-class
            elif isinstance(self.model, GradientBoostingClassifier):
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                    self.explainer_type = "tree"
                    self.logger.info(f"Using TreeExplainer for {model_class}")
                except Exception as e:
                    # Fallback to KernelExplainer for multi-class
                    self.logger.warning(f"TreeExplainer failed for GradientBoosting (likely multi-class): {e}")
                    self.logger.info(f"Falling back to KernelExplainer (slower but supports multi-class)")
                    background_size = min(100, len(self.X_train))
                    background = shap.kmeans(self.X_train, background_size)
                    self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
                    self.explainer_type = "kernel"
                    self.logger.warning(f"KernelExplainer can be slow for large datasets")
            
            # XGBoost
            elif XGBOOST_AVAILABLE and isinstance(self.model, XGBClassifier):
                self.explainer = shap.TreeExplainer(self.model)
                self.explainer_type = "tree"
                self.logger.info(f"Using TreeExplainer for XGBoost")
            
            # LightGBM
            elif LIGHTGBM_AVAILABLE and isinstance(self.model, LGBMClassifier):
                self.explainer = shap.TreeExplainer(self.model)
                self.explainer_type = "tree"
                self.logger.info(f"Using TreeExplainer for LightGBM")
            
            # Linear models: use LinearExplainer (fast and exact)
            elif isinstance(self.model, (LogisticRegression, LinearSVC)):
                # For linear models, we need to pass both model and data
                self.explainer = shap.LinearExplainer(self.model, self.X_train)
                self.explainer_type = "linear"
                self.logger.info(f"Using LinearExplainer for {model_class}")
            
            # Other models: use KernelExplainer (slower but model-agnostic)
            else:
                # Use a subset of training data as background for efficiency
                # KernelExplainer can be slow, so we use kmeans to summarize data
                background_size = min(100, len(self.X_train))
                background = shap.kmeans(self.X_train, background_size)
                
                # For classifiers, we typically want to explain probability predictions
                if hasattr(self.model, 'predict_proba'):
                    self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
                else:
                    self.explainer = shap.KernelExplainer(self.model.predict, background)
                
                self.explainer_type = "kernel"
                self.logger.info(f"Using KernelExplainer for {model_class}")
                self.logger.warning(f"KernelExplainer can be slow. Consider using tree-based or linear models for faster explanations.")
        
        except Exception as e:
            self.logger.error(f"Error initializing SHAP explainer: {e}")
            raise
    
    def compute_shap_values(self, X_test: np.ndarray, max_samples: Optional[int] = None) -> np.ndarray:
        """
        Compute SHAP values for test data.
        
        Args:
            X_test: Test data to explain
            max_samples: Maximum number of samples to compute (for efficiency)
        
        Returns:
            SHAP values array
        """
        if max_samples is not None and len(X_test) > max_samples:
            self.logger.info(f"Computing SHAP for {max_samples} samples (out of {len(X_test)})")
            X_test = X_test[:max_samples]
        else:
            self.logger.info(f"Computing SHAP values for {len(X_test)} samples")
        
        try:
            if self.explainer_type == "tree":
                # For tree models, shap_values returns values for each class
                self.shap_values = self.explainer.shap_values(X_test)
                
                # For binary classification, some models return a single array
                if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
                    # Use the positive class (index 1) for binary classification
                    self.logger.info("Binary classification detected (tree model)")
            
            elif self.explainer_type == "linear":
                self.shap_values = self.explainer.shap_values(X_test)
            
            elif self.explainer_type == "kernel":
                # KernelExplainer returns values for each output
                self.shap_values = self.explainer.shap_values(X_test)
            
            self.logger.info(f"SHAP values computed successfully")
            return self.shap_values
        
        except Exception as e:
            self.logger.error(f"Error computing SHAP values: {e}")
            raise
    
    def plot_summary(self, X_test: np.ndarray, max_display: int = 20, 
                     plot_type: str = "dot", class_idx: Optional[int] = None,
                     save_name: Optional[str] = None):
        """
        Create SHAP summary plot.
        
        Args:
            X_test: Test data
            max_display: Maximum number of features to display
            plot_type: Type of plot ("dot", "bar", "violin")
            class_idx: For multiclass, which class to explain (None for all)
            save_name: Custom save name (default: summary_plot.png)
        """
        if self.shap_values is None:
            self.compute_shap_values(X_test)
        
        try:
            plt.figure(figsize=(12, 8))
            
            # Handle different SHAP value formats
            if isinstance(self.shap_values, list):
                # Multi-output model
                if class_idx is not None:
                    shap_vals = self.shap_values[class_idx]
                    title = f"SHAP Summary - {self.model_name} - {self.class_names[class_idx]}"
                else:
                    # For binary, use positive class
                    shap_vals = self.shap_values[1] if len(self.shap_values) == 2 else self.shap_values[0]
                    title = f"SHAP Summary - {self.model_name}"
            else:
                shap_vals = self.shap_values
                title = f"SHAP Summary - {self.model_name}"
            
            shap.summary_plot(
                shap_vals,
                X_test,
                feature_names=self.feature_names,
                max_display=max_display,
                plot_type=plot_type,
                show=False
            )
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            save_path = self.output_dir / (save_name or f"shap_summary_{self.model_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Summary plot saved to {save_path}")
            plt.close()
        
        except Exception as e:
            self.logger.error(f"Error creating summary plot: {e}")
            plt.close()
    
    def plot_bar(self, X_test: np.ndarray, max_display: int = 20,
                 class_idx: Optional[int] = None, save_name: Optional[str] = None):
        """
        Create SHAP bar plot showing mean absolute SHAP values.
        
        Args:
            X_test: Test data
            max_display: Maximum number of features to display
            class_idx: For multiclass, which class to explain
            save_name: Custom save name
        """
        if self.shap_values is None:
            self.compute_shap_values(X_test)
        
        try:
            plt.figure(figsize=(10, 8))
            
            # Handle different SHAP value formats
            if isinstance(self.shap_values, list):
                if class_idx is not None:
                    shap_vals = self.shap_values[class_idx]
                    title = f"SHAP Feature Importance - {self.class_names[class_idx]}"
                else:
                    shap_vals = self.shap_values[1] if len(self.shap_values) == 2 else self.shap_values[0]
                    title = f"SHAP Feature Importance - {self.model_name}"
            else:
                shap_vals = self.shap_values
                title = f"SHAP Feature Importance - {self.model_name}"
            
            shap.summary_plot(
                shap_vals,
                X_test,
                feature_names=self.feature_names,
                max_display=max_display,
                plot_type="bar",
                show=False
            )
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            save_path = self.output_dir / (save_name or f"shap_bar_{self.model_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Bar plot saved to {save_path}")
            plt.close()
        
        except Exception as e:
            self.logger.error(f"Error creating bar plot: {e}")
            plt.close()
    
    def plot_dependence(self, X_test: np.ndarray, feature_idx: int, 
                       interaction_idx: Optional[int] = None,
                       class_idx: Optional[int] = None,
                       save_name: Optional[str] = None):
        """
        Create SHAP dependence plot for a specific feature.
        
        Args:
            X_test: Test data
            feature_idx: Index of feature to plot
            interaction_idx: Index of feature to color by (None for auto)
            class_idx: For multiclass, which class to explain
            save_name: Custom save name
        """
        if self.shap_values is None:
            self.compute_shap_values(X_test)
        
        try:
            plt.figure(figsize=(10, 6))
            
            # Handle different SHAP value formats
            if isinstance(self.shap_values, list):
                if class_idx is not None:
                    shap_vals = self.shap_values[class_idx]
                else:
                    shap_vals = self.shap_values[1] if len(self.shap_values) == 2 else self.shap_values[0]
            else:
                shap_vals = self.shap_values
            
            shap.dependence_plot(
                feature_idx,
                shap_vals,
                X_test,
                feature_names=self.feature_names,
                interaction_index=interaction_idx,
                show=False
            )
            
            feature_name = self.feature_names[feature_idx]
            plt.title(f"SHAP Dependence - {feature_name}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            save_path = self.output_dir / (save_name or f"shap_dependence_{feature_name}_{self.model_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Dependence plot saved to {save_path}")
            plt.close()
        
        except Exception as e:
            self.logger.error(f"Error creating dependence plot: {e}")
            plt.close()
    
    def plot_force(self, X_test: np.ndarray, sample_idx: int = 0,
                   class_idx: Optional[int] = None, save_name: Optional[str] = None):
        """
        Create SHAP force plot for a single prediction.
        
        Args:
            X_test: Test data
            sample_idx: Index of sample to explain
            class_idx: For multiclass, which class to explain
            save_name: Custom save name
        """
        if self.shap_values is None:
            self.compute_shap_values(X_test)
        
        try:
            # Handle different SHAP value formats
            if isinstance(self.shap_values, list):
                if class_idx is not None:
                    shap_vals = self.shap_values[class_idx]
                    base_value = self.explainer.expected_value[class_idx] if isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value
                else:
                    shap_vals = self.shap_values[1] if len(self.shap_values) == 2 else self.shap_values[0]
                    base_value = self.explainer.expected_value[1] if len(self.shap_values) == 2 else self.explainer.expected_value[0]
            else:
                shap_vals = self.shap_values
                base_value = self.explainer.expected_value
            
            # Create force plot
            force_plot = shap.force_plot(
                base_value,
                shap_vals[sample_idx],
                X_test[sample_idx],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            
            save_path = self.output_dir / (save_name or f"shap_force_sample{sample_idx}_{self.model_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Force plot saved to {save_path}")
            plt.close()
        
        except Exception as e:
            self.logger.error(f"Error creating force plot: {e}")
            plt.close()
    
    def plot_waterfall(self, X_test: np.ndarray, sample_idx: int = 0,
                      class_idx: Optional[int] = None, max_display: int = 20,
                      save_name: Optional[str] = None):
        """
        Create SHAP waterfall plot for a single prediction.
        
        Args:
            X_test: Test data
            sample_idx: Index of sample to explain
            class_idx: For multiclass, which class to explain
            max_display: Maximum number of features to display
            save_name: Custom save name
        """
        if self.shap_values is None:
            self.compute_shap_values(X_test)
        
        try:
            plt.figure(figsize=(10, 8))
            
            # Handle different SHAP value formats
            if isinstance(self.shap_values, list):
                if class_idx is not None:
                    shap_vals = self.shap_values[class_idx]
                    base_value = self.explainer.expected_value[class_idx] if isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value
                else:
                    shap_vals = self.shap_values[1] if len(self.shap_values) == 2 else self.shap_values[0]
                    base_value = self.explainer.expected_value[1] if len(self.shap_values) == 2 else self.explainer.expected_value[0]
            else:
                shap_vals = self.shap_values
                base_value = self.explainer.expected_value
            
            # Create Explanation object for waterfall plot
            explanation = shap.Explanation(
                values=shap_vals[sample_idx],
                base_values=base_value,
                data=X_test[sample_idx],
                feature_names=self.feature_names
            )
            
            shap.waterfall_plot(explanation, max_display=max_display, show=False)
            
            plt.title(f"SHAP Waterfall - Sample {sample_idx}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            save_path = self.output_dir / (save_name or f"shap_waterfall_sample{sample_idx}_{self.model_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Waterfall plot saved to {save_path}")
            plt.close()
        
        except Exception as e:
            self.logger.error(f"Error creating waterfall plot: {e}")
            plt.close()
    
    def export_shap_values(self, X_test: np.ndarray, y_test: Optional[np.ndarray] = None,
                          save_name: Optional[str] = None) -> pd.DataFrame:
        """
        Export SHAP values to CSV for further analysis.
        
        Args:
            X_test: Test data
            y_test: Test labels (optional)
            save_name: Custom save name
        
        Returns:
            DataFrame with SHAP values
        """
        if self.shap_values is None:
            self.compute_shap_values(X_test)
        
        try:
            # Handle different SHAP value formats
            if isinstance(self.shap_values, list):
                # For binary, use positive class
                shap_vals = self.shap_values[1] if len(self.shap_values) == 2 else self.shap_values[0]
            else:
                shap_vals = self.shap_values
            
            # Handle 3D arrays (samples, features, classes) - take last class
            if shap_vals.ndim == 3:
                self.logger.info(f"Multi-class SHAP values detected (shape: {shap_vals.shape}), using class 1")
                shap_vals = shap_vals[:, :, 1]  # Use class 1 (positive class)
            
            # Create DataFrame
            shap_df = pd.DataFrame(shap_vals, columns=self.feature_names)
            
            # Add labels if provided
            if y_test is not None:
                shap_df.insert(0, 'true_label', y_test[:len(shap_vals)])
            
            # Save to CSV
            save_path = self.output_dir / (save_name or f"shap_values_{self.model_name}.csv")
            shap_df.to_csv(save_path, index=False)
            self.logger.info(f"SHAP values exported to {save_path}")
            
            return shap_df
        
        except Exception as e:
            self.logger.error(f"Error exporting SHAP values: {e}")
            return pd.DataFrame()
    
    def generate_comprehensive_report(self, X_test: np.ndarray, y_test: Optional[np.ndarray] = None,
                                     max_display: int = 20, top_features: int = 5):
        """
        Generate a comprehensive SHAP analysis report with all visualizations.
        
        Args:
            X_test: Test data
            y_test: Test labels (optional)
            max_display: Maximum number of features in plots
            top_features: Number of top features for dependence plots
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Generating comprehensive SHAP report for {self.model_name}")
        self.logger.info("=" * 80)
        
        # Compute SHAP values
        self.compute_shap_values(X_test)
        
        # 1. Summary plot (dot)
        self.logger.info("Creating summary plot...")
        self.plot_summary(X_test, max_display=max_display, plot_type="dot")
        
        # 2. Bar plot
        self.logger.info("Creating bar plot...")
        self.plot_bar(X_test, max_display=max_display)
        
        # 3. Export SHAP values
        self.logger.info("Exporting SHAP values...")
        shap_df = self.export_shap_values(X_test, y_test)
        
        # 4. Dependence plots for top features
        self.logger.info(f"Creating dependence plots for top {top_features} features...")
        
        # Get top features by mean absolute SHAP value
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1] if len(self.shap_values) == 2 else self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        # Handle 3D arrays (samples, features, classes)
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]  # Use class 1 (positive class)
        
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_features:][::-1]
        
        for idx in top_indices:
            idx_int = int(idx)  # Ensure it's a Python int, not numpy int
            feature_name = self.feature_names[idx_int]
            self.logger.info(f"  Creating dependence plot for {feature_name}...")
            self.plot_dependence(X_test, idx_int)
        
        # 5. Waterfall plot for first sample
        self.logger.info("Creating waterfall plot for sample 0...")
        self.plot_waterfall(X_test, sample_idx=0, max_display=max_display)
        
        # 6. Create summary statistics
        self.logger.info("Creating summary statistics...")
        summary_stats = {
            'model_name': self.model_name,
            'n_samples': len(X_test),
            'n_features': len(self.feature_names),
            'explainer_type': self.explainer_type,
            'top_features': []
        }
        
        for idx in top_indices:
            summary_stats['top_features'].append({
                'feature': self.feature_names[idx],
                'mean_abs_shap': float(mean_abs_shap[idx])
            })
        
        # Save summary stats
        summary_path = self.output_dir / f"shap_summary_stats_{self.model_name}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        self.logger.info("=" * 80)
        self.logger.info(f"SHAP report completed! All outputs saved to {self.output_dir}")
        self.logger.info("=" * 80)


def load_model_and_generate_shap(model_path: Union[str, Path], 
                                 X_train: np.ndarray, 
                                 X_test: np.ndarray,
                                 y_test: Optional[np.ndarray],
                                 feature_names: List[str],
                                 output_dir: Union[str, Path],
                                 model_name: str = "model",
                                 class_names: Optional[List[str]] = None):
    """
    Convenience function to load a model and generate SHAP explanations.
    
    Args:
        model_path: Path to pickled model
        X_train: Training data (for SHAP background)
        X_test: Test data to explain
        y_test: Test labels (optional)
        feature_names: List of feature names
        output_dir: Output directory
        model_name: Name of model
        class_names: Class names (optional)
    """
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Create interpreter
    interpreter = SHAPInterpreter(
        model=model,
        X_train=X_train,
        feature_names=feature_names,
        output_dir=output_dir,
        model_name=model_name,
        class_names=class_names
    )
    
    # Generate comprehensive report
    interpreter.generate_comprehensive_report(X_test, y_test)
    
    return interpreter


if __name__ == "__main__":
    print("SHAP Interpretability Module")
    print("=" * 60)
    print("This module provides SHAP-based interpretability for classical ML models.")
    print("\nUsage:")
    print("  from shap_interpretability import SHAPInterpreter, load_model_and_generate_shap")
    print("\nFor examples, see: run_shap_analysis.py")

