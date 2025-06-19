"""
Data Preprocessing Utilities for Radiomics Classification
=======================================================

This module contains utilities for preprocessing radiomics data including:
- Missing value handling
- Feature selection
- Scaling and normalization
- Data validation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Comprehensive data preprocessor for radiomics features."""
    
    def __init__(self, config=None):
        """
        Initialize the preprocessor.
        
        Args:
            config (dict): Configuration dictionary with preprocessing parameters
        """
        self.config = config or {}
        self.scaler = None
        self.feature_selector = None
        self.feature_names = None
        self.preprocessing_steps = []
        
    def validate_data(self, data):
        """
        Validate input data structure and content.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required columns
            required_cols = ['subject_id', 'label']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Check for features
            feature_cols = [col for col in data.columns if col not in required_cols]
            if len(feature_cols) == 0:
                logger.error("No feature columns found")
                return False
            
            # Check data types
            if not data['label'].dtype in ['int64', 'int32']:
                logger.warning("Label column is not integer type, converting...")
                data['label'] = data['label'].astype(int)
            
            # Check for infinite values
            feature_data = data[feature_cols]
            if np.isinf(feature_data.values).any():
                logger.warning("Found infinite values in features")
            
            logger.info(f"Data validation passed: {data.shape[0]} samples, {len(feature_cols)} features")
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False
    
    def handle_missing_values(self, data, strategy='drop'):
        """
        Handle missing values in the dataset.
        
        Args:
            data (pd.DataFrame): Input data
            strategy (str): 'drop' or 'impute'
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        feature_cols = [col for col in data.columns if col not in ['subject_id', 'label']]
        feature_data = data[feature_cols]
        
        missing_count = feature_data.isnull().sum().sum()
        if missing_count == 0:
            logger.info("No missing values found")
            return data
        
        logger.info(f"Found {missing_count} missing values")
        
        if strategy == 'drop':
            # Drop rows with any missing values
            initial_count = len(data)
            data_clean = data.dropna()
            final_count = len(data_clean)
            dropped_count = initial_count - final_count
            
            logger.info(f"Dropped {dropped_count} rows with missing values")
            logger.info(f"Remaining samples: {final_count}")
            
            return data_clean
        
        elif strategy == 'impute':
            # Simple imputation with median
            for col in feature_cols:
                if data[col].isnull().any():
                    median_val = data[col].median()
                    data[col].fillna(median_val, inplace=True)
                    logger.info(f"Imputed {col} with median: {median_val:.4f}")
            
            return data
        
        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}")
    
    def remove_constant_features(self, data, threshold=0.01):
        """
        Remove features with low variance (constant or near-constant).
        
        Args:
            data (pd.DataFrame): Input data
            threshold (float): Variance threshold
            
        Returns:
            pd.DataFrame: Data with constant features removed
        """
        feature_cols = [col for col in data.columns if col not in ['subject_id', 'label']]
        feature_data = data[feature_cols]
        
        initial_features = len(feature_cols)
        
        # Calculate variance
        variances = feature_data.var()
        low_var_features = variances[variances < threshold].index.tolist()
        
        if low_var_features:
            logger.info(f"Removing {len(low_var_features)} low-variance features (threshold: {threshold})")
            data_clean = data.drop(columns=low_var_features)
            final_features = len([col for col in data_clean.columns if col not in ['subject_id', 'label']])
            logger.info(f"Remaining features: {final_features}")
        else:
            logger.info("No low-variance features found")
            data_clean = data
        
        return data_clean
    
    def select_features(self, data, method='k_best', n_features=100, target_col='label'):
        """
        Perform feature selection.
        
        Args:
            data (pd.DataFrame): Input data
            method (str): 'k_best', 'mutual_info', or 'pca'
            n_features (int): Number of features to select
            target_col (str): Target column name
            
        Returns:
            pd.DataFrame: Data with selected features
        """
        feature_cols = [col for col in data.columns if col not in ['subject_id', 'label']]
        feature_data = data[feature_cols]
        target_data = data[target_col]
        
        initial_features = len(feature_cols)
        
        if method == 'k_best':
            # SelectKBest with f_classif
            selector = SelectKBest(score_func=f_classif, k=min(n_features, initial_features))
            selected_features = selector.fit_transform(feature_data, target_data)
            selected_indices = selector.get_support()
            selected_cols = [col for col, selected in zip(feature_cols, selected_indices) if selected]
            
        elif method == 'mutual_info':
            # Mutual information
            selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, initial_features))
            selected_features = selector.fit_transform(feature_data, target_data)
            selected_indices = selector.get_support()
            selected_cols = [col for col, selected in zip(feature_cols, selected_indices) if selected]
            
        elif method == 'pca':
            # PCA dimensionality reduction
            pca = PCA(n_components=min(n_features, initial_features))
            selected_features = pca.fit_transform(feature_data)
            selected_cols = [f"PC_{i+1}" for i in range(selected_features.shape[1])]
            
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Create new dataframe with selected features
        result_data = data[['subject_id', 'label']].copy()
        for i, col in enumerate(selected_cols):
            if method == 'pca':
                result_data[col] = selected_features[:, i]
            else:
                result_data[col] = data[col]
        
        logger.info(f"Feature selection: {initial_features} → {len(selected_cols)} features")
        return result_data
    
    def scale_features(self, data, method='standard'):
        """
        Scale features using specified method.
        
        Args:
            data (pd.DataFrame): Input data
            method (str): 'standard', 'robust', or 'minmax'
            
        Returns:
            tuple: (scaled_data, scaler)
        """
        feature_cols = [col for col in data.columns if col not in ['subject_id', 'label']]
        feature_data = data[feature_cols]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform
        scaled_features = scaler.fit_transform(feature_data)
        
        # Create new dataframe
        scaled_data = data[['subject_id', 'label']].copy()
        for i, col in enumerate(feature_cols):
            scaled_data[col] = scaled_features[:, i]
        
        logger.info(f"Features scaled using {method} scaling")
        return scaled_data, scaler
    
    def preprocess_pipeline(self, data, config=None):
        """
        Run complete preprocessing pipeline.
        
        Args:
            data (pd.DataFrame): Input data
            config (dict): Configuration dictionary
            
        Returns:
            tuple: (processed_data, preprocessing_info)
        """
        config = config or self.config
        
        logger.info("Starting preprocessing pipeline...")
        
        # Validate data
        if not self.validate_data(data):
            raise ValueError("Data validation failed")
        
        # Handle missing values
        if config.get('preprocessing', {}).get('remove_missing', True):
            data = self.handle_missing_values(data, strategy='drop')
        
        # Remove constant features
        variance_threshold = config.get('preprocessing', {}).get('variance_threshold', 0.01)
        data = self.remove_constant_features(data, threshold=variance_threshold)
        
        # Feature selection
        feature_selection_config = config.get('preprocessing', {}).get('feature_selection', {})
        if feature_selection_config.get('enabled', True):
            method = feature_selection_config.get('method', 'k_best')
            n_features = feature_selection_config.get('n_features', 100)
            data = self.select_features(data, method=method, n_features=n_features)
        
        # Scaling
        scaling_config = config.get('preprocessing', {}).get('scaling', {})
        method = scaling_config.get('method', 'standard')
        data, scaler = self.scale_features(data, method=method)
        
        # Store preprocessing info
        preprocessing_info = {
            'scaler': scaler,
            'feature_names': [col for col in data.columns if col not in ['subject_id', 'label']],
            'n_samples': len(data),
            'n_features': len([col for col in data.columns if col not in ['subject_id', 'label']])
        }
        
        logger.info(f"Preprocessing completed: {data.shape[0]} samples, {preprocessing_info['n_features']} features")
        
        return data, preprocessing_info

def load_and_preprocess_data(input_path, config_path=None):
    """
    Convenience function to load and preprocess data.
    
    Args:
        input_path (str): Path to input CSV file
        config_path (str): Path to configuration file
        
    Returns:
        tuple: (processed_data, preprocessing_info)
    """
    # Load data
    logger.info(f"Loading data from {input_path}")
    data = pd.read_csv(input_path)
    
    # Load config if provided
    config = None
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Preprocess
    preprocessor = DataPreprocessor(config)
    processed_data, preprocessing_info = preprocessor.preprocess_pipeline(data, config)
    
    return processed_data, preprocessing_info 