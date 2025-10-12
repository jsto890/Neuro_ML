"""
SHAP Analysis Runner Script
============================

This script loads trained classical ML models and generates comprehensive SHAP
interpretability reports.

Usage:
    python run_shap_analysis.py --model_dir /path/to/models --data /path/to/data.csv
    
Or run interactively for all models in a directory:
    python run_shap_analysis.py --model_dir /path/to/models --data /path/to/data.csv --all
"""

import argparse
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional
import warnings

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from shap_interpretability import SHAPInterpreter, load_model_and_generate_shap, SHAP_AVAILABLE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_path: str, test_size: float = 0.2, random_state: int = 42) -> Dict:
    """
    Load radiomics data and split into train/test sets.
    
    Args:
        data_path: Path to CSV file with radiomics features
        test_size: Proportion of data to use for testing
        random_state: Random seed
    
    Returns:
        Dictionary with X_train, X_test, y_train, y_test, feature_names
    """
    logger.info(f"Loading data from {data_path}")
    
    # Load CSV
    df = pd.read_csv(data_path)
    
    # Identify label column
    label_col = None
    for col in ['label', 'Label', 'diagnosis', 'Diagnosis', 'class', 'Class']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError("Could not find label column. Expected one of: label, Label, diagnosis, Diagnosis, class, Class")
    
    # Separate features and labels
    y = df[label_col].values
    
    # Remove non-feature columns (metadata, IDs, etc.)
    exclude_cols = [
        label_col, 'subject_id', 'Subject_ID', 'SubjectID', 'ID', 'id',
        'Subject', 'subject', 'Patient', 'patient', 'PatientID', 'patient_id'
    ]
    
    # Get potential feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Filter to only numeric columns (exclude version strings, etc.)
    numeric_cols = []
    for col in feature_cols:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            # Try to detect if it's accidentally stored as object but is numeric
            try:
                pd.to_numeric(df[col], errors='raise')
                numeric_cols.append(col)
            except (ValueError, TypeError):
                logger.warning(f"Skipping non-numeric column: {col} (example value: {df[col].iloc[0]})")
    
    feature_cols = numeric_cols
    
    if not feature_cols:
        raise ValueError("No numeric feature columns found in CSV!")
    
    X = df[feature_cols].values.astype(np.float64)
    feature_names = feature_cols
    
    logger.info(f"Loaded {len(X)} samples with {len(feature_names)} features")
    logger.info(f"Label distribution: {np.bincount(y.astype(int))}")
    
    # Split into train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names
    }


def load_saved_data(npz_path: str) -> Dict:
    """
    Load preprocessed data from NPZ file (if saved during training).
    
    Args:
        npz_path: Path to NPZ file
    
    Returns:
        Dictionary with X_train, X_test, y_train, y_test, feature_names
    """
    logger.info(f"Loading preprocessed data from {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    return {
        'X_train': data['X_train'],
        'X_test': data['X_test'],
        'y_train': data['y_train'],
        'y_test': data['y_test'],
        'feature_names': data['feature_names'].tolist() if 'feature_names' in data else None
    }


def find_model_files(model_dir: Path) -> List[Path]:
    """
    Find all pickled model files in directory.
    
    Args:
        model_dir: Directory to search
    
    Returns:
        List of model file paths
    """
    model_files = []
    
    # Exclude non-model files
    exclude_patterns = ['scaler', 'selector', 'imputer', 'encoder', 'preprocessor']
    
    # Look for .pkl files
    for pattern in ['*.pkl', '*.pickle']:
        for file_path in model_dir.glob(pattern):
            # Skip non-model files
            filename_lower = file_path.stem.lower()
            if any(exclude in filename_lower for exclude in exclude_patterns):
                logger.info(f"Skipping non-model file: {file_path.name}")
                continue
            model_files.append(file_path)
    
    logger.info(f"Found {len(model_files)} model files in {model_dir}")
    return sorted(model_files)


def load_selected_features(model_dir: Path) -> Optional[List[str]]:
    """
    Load the list of selected features used during training.
    
    Args:
        model_dir: Directory containing the model and feature importance files
    
    Returns:
        List of selected feature names, or None if not found
    """
    # Look for feature importance CSV
    feature_csv = model_dir / "feature_importance_comparison.csv"
    
    if feature_csv.exists():
        try:
            df = pd.read_csv(feature_csv)
            if 'feature' in df.columns:
                features = df['feature'].tolist()
                logger.info(f"Loaded {len(features)} selected features from {feature_csv.name}")
                return features
        except Exception as e:
            logger.warning(f"Could not load features from {feature_csv}: {e}")
    
    return None


def filter_data_to_selected_features(data: Dict, selected_features: List[str]) -> Dict:
    """
    Filter data to only include selected features.
    
    Args:
        data: Dictionary with X_train, X_test, feature_names
        selected_features: List of features to keep
    
    Returns:
        Filtered data dictionary
    """
    all_feature_names = data['feature_names']
    
    # Find indices of selected features
    feature_indices = []
    found_features = []
    
    for feat in selected_features:
        if feat in all_feature_names:
            idx = all_feature_names.index(feat)
            feature_indices.append(idx)
            found_features.append(feat)
        else:
            logger.warning(f"Selected feature '{feat}' not found in data")
    
    if not feature_indices:
        raise ValueError("No selected features found in data!")
    
    logger.info(f"Filtering data from {len(all_feature_names)} to {len(feature_indices)} features")
    
    # Filter data
    filtered_data = {
        'X_train': data['X_train'][:, feature_indices],
        'X_test': data['X_test'][:, feature_indices],
        'y_train': data['y_train'],
        'y_test': data['y_test'],
        'feature_names': found_features
    }
    
    return filtered_data


def run_shap_for_model(model_path: Path, data: Dict, output_dir: Path, 
                       class_names: Optional[List[str]] = None,
                       model_dir: Optional[Path] = None):
    """
    Run SHAP analysis for a single model.
    
    Args:
        model_path: Path to model file
        data: Dictionary with train/test data
        output_dir: Output directory
        class_names: Class names (optional)
        model_dir: Directory containing model (for finding selected features)
    """
    model_name = model_path.stem
    logger.info("=" * 80)
    logger.info(f"Running SHAP analysis for: {model_name}")
    logger.info("=" * 80)
    
    # Create model-specific output directory
    model_output_dir = output_dir / f"shap_{model_name}"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Determine model's expected feature count
        expected_features = None
        if hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
        elif hasattr(model, 'coef_'):
            expected_features = model.coef_.shape[1] if model.coef_.ndim > 1 else model.coef_.shape[0]
        
        # Get data
        current_data = data.copy()
        feature_names = data.get('feature_names')
        
        if feature_names is None:
            n_features = data['X_train'].shape[1]
            feature_names = [f"feature_{i}" for i in range(n_features)]
            current_data['feature_names'] = feature_names
            logger.warning("No feature names found, using default names")
        
        # Check for feature mismatch
        data_features = current_data['X_train'].shape[1]
        
        if expected_features is not None and data_features != expected_features:
            logger.warning(f"Feature mismatch: model expects {expected_features}, data has {data_features}")
            logger.info("Attempting to load selected features from training...")
            
            # Try to load selected features
            if model_dir is None:
                model_dir = model_path.parent
            
            selected_features = load_selected_features(model_dir)
            
            if selected_features and len(selected_features) == expected_features:
                logger.info(f"✓ Found matching selected features ({len(selected_features)})")
                current_data = filter_data_to_selected_features(current_data, selected_features)
            else:
                logger.error(f"✗ Could not resolve feature mismatch. Skipping {model_name}.")
                return False
        
        # Create interpreter
        interpreter = SHAPInterpreter(
            model=model,
            X_train=current_data['X_train'],
            feature_names=current_data['feature_names'],
            output_dir=model_output_dir,
            model_name=model_name,
            class_names=class_names
        )
        
        # Generate comprehensive report
        interpreter.generate_comprehensive_report(
            X_test=current_data['X_test'],
            y_test=current_data['y_test'],
            max_display=20,
            top_features=5
        )
        
        logger.info(f"✓ SHAP analysis completed for {model_name}")
        return True
    
    except Exception as e:
        logger.error(f"✗ Error analyzing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate SHAP interpretability reports for classical ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single model
  python run_shap_analysis.py --model path/to/model.pkl --data data.csv --output shap_results
  
  # Analyze all models in a directory
  python run_shap_analysis.py --model_dir path/to/models/ --data data.csv --output shap_results --all
  
  # Use preprocessed data (NPZ file)
  python run_shap_analysis.py --model path/to/model.pkl --npz_data preprocessed_data.npz --output shap_results
  
  # Specify class names
  python run_shap_analysis.py --model path/to/model.pkl --data data.csv --output shap_results --class_names CN AD PD
        """
    )
    
    # Model arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model', type=str, help='Path to single model file (.pkl)')
    model_group.add_argument('--model_dir', type=str, help='Directory containing model files')
    
    # Data arguments
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--data', type=str, help='Path to CSV file with radiomics features')
    data_group.add_argument('--npz_data', type=str, help='Path to preprocessed NPZ file')
    
    # Other arguments
    parser.add_argument('--output', type=str, default='shap_results',
                       help='Output directory for SHAP results (default: shap_results)')
    parser.add_argument('--all', action='store_true',
                       help='Analyze all models in model_dir')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--class_names', nargs='+', type=str,
                       help='Class names (e.g., --class_names CN AD PD)')
    
    args = parser.parse_args()
    
    # Check SHAP availability
    if not SHAP_AVAILABLE:
        logger.error("SHAP library not found. Install with: pip install shap")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    if args.data:
        data = load_data(args.data, test_size=args.test_size, random_state=args.random_state)
    else:
        data = load_saved_data(args.npz_data)
    
    # Get model files
    if args.model:
        model_files = [Path(args.model)]
    else:
        model_dir = Path(args.model_dir)
        model_files = find_model_files(model_dir)
        
        if not args.all and len(model_files) > 1:
            logger.warning(f"Found {len(model_files)} models. Use --all to analyze all, or specify --model for a single model.")
            logger.info("Available models:")
            for i, mf in enumerate(model_files, 1):
                logger.info(f"  {i}. {mf.name}")
            sys.exit(0)
    
    if not model_files:
        logger.error("No model files found!")
        sys.exit(1)
    
    # Run SHAP analysis
    logger.info(f"Starting SHAP analysis for {len(model_files)} model(s)...")
    
    # Determine model directory (for finding selected features)
    if args.model:
        model_dir = Path(args.model).parent
    else:
        model_dir = Path(args.model_dir)
    
    success_count = 0
    for model_file in model_files:
        success = run_shap_for_model(
            model_path=model_file,
            data=data,
            output_dir=output_dir,
            class_names=args.class_names,
            model_dir=model_dir
        )
        if success:
            success_count += 1
    
    # Summary
    logger.info("=" * 80)
    logger.info(f"SHAP analysis completed: {success_count}/{len(model_files)} models successful")
    logger.info(f"Results saved to: {output_dir.absolute()}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

