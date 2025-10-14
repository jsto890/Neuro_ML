"""
Example: SHAP Interpretability for Classical Models
====================================================

This script demonstrates how to use SHAP interpretability with trained
classical machine learning models.

Quick start example that you can adapt for your use case.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle

# Import SHAP module
from shap_interpretability import SHAPInterpreter, load_model_and_generate_shap, SHAP_AVAILABLE

# Check if SHAP is available
if not SHAP_AVAILABLE:
    print("❌ SHAP not installed. Install with: pip install shap")
    exit(1)


def example_basic_usage():
    """Example 1: Basic SHAP analysis for a single model."""
    
    print("=" * 80)
    print("Example 1: Basic SHAP Analysis")
    print("=" * 80)
    
    # Load your trained model
    model_path = "path/to/your/model.pkl"  # Change this!
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load your data (this should match your training data preprocessing)
    # For this example, we'll create dummy data
    n_samples = 200
    n_features = 50
    X_train = np.random.randn(n_samples, n_features)
    X_test = np.random.randn(50, n_features)
    y_test = np.random.randint(0, 2, 50)
    
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    # Create SHAP interpreter
    interpreter = SHAPInterpreter(
        model=model,
        X_train=X_train,  # Background data for SHAP
        feature_names=feature_names,
        output_dir="shap_results_example1",
        model_name="RandomForest",
        class_names=["CN", "AD"]
    )
    
    # Generate all SHAP plots and analysis
    interpreter.generate_comprehensive_report(X_test, y_test)
    
    print("\n✓ Results saved to shap_results_example1/")


def example_custom_plots():
    """Example 2: Create specific SHAP plots."""
    
    print("\n" + "=" * 80)
    print("Example 2: Custom SHAP Plots")
    print("=" * 80)
    
    # Load model and data (placeholder - replace with your actual data)
    model_path = "path/to/your/model.pkl"
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Dummy data
    n_samples = 200
    n_features = 50
    X_train = np.random.randn(n_samples, n_features)
    X_test = np.random.randn(50, n_features)
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    # Create interpreter
    interpreter = SHAPInterpreter(
        model=model,
        X_train=X_train,
        feature_names=feature_names,
        output_dir="shap_results_example2",
        model_name="SVM"
    )
    
    # Compute SHAP values once
    shap_values = interpreter.compute_shap_values(X_test)
    
    # Create specific plots
    print("\n📊 Creating summary plot...")
    interpreter.plot_summary(X_test, max_display=15, plot_type='dot')
    
    print("📊 Creating bar plot...")
    interpreter.plot_bar(X_test, max_display=20)
    
    print("📊 Creating dependence plot for feature 5...")
    interpreter.plot_dependence(X_test, feature_idx=5)
    
    print("📊 Creating waterfall plot for first sample...")
    interpreter.plot_waterfall(X_test, sample_idx=0, max_display=15)
    
    # Export SHAP values to CSV
    print("💾 Exporting SHAP values...")
    shap_df = interpreter.export_shap_values(X_test)
    
    print(f"\n✓ Custom analysis saved to shap_results_example2/")
    print(f"   SHAP values exported to CSV with shape: {shap_df.shape}")


def example_compare_models():
    """Example 3: Compare SHAP values across multiple models."""
    
    print("\n" + "=" * 80)
    print("Example 3: Compare SHAP Across Models")
    print("=" * 80)
    
    # Load multiple models
    model_paths = [
        "path/to/rf_model.pkl",
        "path/to/svm_model.pkl",
        "path/to/lr_model.pkl"
    ]
    model_names = ["RandomForest", "SVM", "LogisticRegression"]
    
    # Dummy data (replace with your actual data)
    n_samples = 200
    n_features = 50
    X_train = np.random.randn(n_samples, n_features)
    X_test = np.random.randn(50, n_features)
    y_test = np.random.randint(0, 2, 50)
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    shap_dfs = []
    
    for model_path, model_name in zip(model_paths, model_names):
        print(f"\n🔬 Analyzing {model_name}...")
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Create interpreter
        interpreter = SHAPInterpreter(
            model=model,
            X_train=X_train,
            feature_names=feature_names,
            output_dir=f"shap_comparison_{model_name}",
            model_name=model_name
        )
        
        # Get SHAP values
        shap_values = interpreter.compute_shap_values(X_test)
        
        # Create plots
        interpreter.plot_summary(X_test, max_display=15)
        interpreter.plot_bar(X_test, max_display=15)
        
        # Export and store
        shap_df = interpreter.export_shap_values(X_test, y_test)
        shap_dfs.append((model_name, shap_df))
    
    # Compare top features across models
    print("\n" + "=" * 80)
    print("Top 10 Features by Model:")
    print("=" * 80)
    
    for model_name, shap_df in shap_dfs:
        # Calculate mean absolute SHAP for each feature
        mean_abs_shap = shap_df.iloc[:, 1:].abs().mean().sort_values(ascending=False)
        top_10 = mean_abs_shap.head(10)
        
        print(f"\n{model_name}:")
        for i, (feat, val) in enumerate(top_10.items(), 1):
            print(f"  {i:2d}. {feat:30s} {val:.4f}")
    
    print("\n✓ Model comparison complete!")


def example_quick_analysis():
    """Example 4: Quick one-liner analysis."""
    
    print("\n" + "=" * 80)
    print("Example 4: Quick Analysis (One Function Call)")
    print("=" * 80)
    
    # Dummy data
    n_samples = 200
    n_features = 50
    X_train = np.random.randn(n_samples, n_features)
    X_test = np.random.randn(50, n_features)
    y_test = np.random.randint(0, 2, 50)
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    # One-liner: load model and generate full report
    interpreter = load_model_and_generate_shap(
        model_path="path/to/your/model.pkl",  # Change this!
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        output_dir="shap_quick_analysis",
        model_name="MyModel",
        class_names=["Healthy", "Disease"]
    )
    
    print("\n✓ Quick analysis complete!")


def example_real_workflow():
    """Example 5: Realistic workflow with CSV data."""
    
    print("\n" + "=" * 80)
    print("Example 5: Realistic Workflow with CSV Data")
    print("=" * 80)
    
    # Paths - CHANGE THESE to your actual paths!
    csv_path = "path/to/radiomics_features.csv"
    model_path = "path/to/trained_model.pkl"
    output_dir = "shap_analysis_real"
    
    print(f"\n📂 Loading data from: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Identify label column
    label_col = 'label'  # or 'diagnosis', 'class', etc.
    
    # Separate features and labels
    y = df[label_col].values
    
    # Remove non-feature columns
    exclude_cols = [label_col, 'subject_id', 'Subject_ID']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].values
    feature_names = feature_cols
    
    print(f"   Loaded {len(X)} samples with {len(feature_names)} features")
    
    # Split into train/test (or load from saved splits)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Load trained model
    print(f"\n🤖 Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Create SHAP interpreter
    print(f"\n🔬 Running SHAP analysis...")
    interpreter = SHAPInterpreter(
        model=model,
        X_train=X_train,
        feature_names=feature_names,
        output_dir=output_dir,
        model_name=Path(model_path).stem,
        class_names=['CN', 'AD']  # Adjust based on your labels
    )
    
    # Generate comprehensive report
    interpreter.generate_comprehensive_report(
        X_test=X_test,
        y_test=y_test,
        max_display=20,
        top_features=5
    )
    
    print(f"\n✓ Analysis complete! Results in: {output_dir}/")


def main():
    """Run examples."""
    
    print("\n" + "="*80)
    print("SHAP Interpretability Examples for Classical ML Models")
    print("="*80)
    
    print("\n⚠️  Note: These examples use dummy data for demonstration.")
    print("    Update the paths to use your actual trained models and data.")
    
    # Uncomment the example you want to run:
    
    # example_basic_usage()           # Basic usage
    # example_custom_plots()          # Custom plots
    # example_compare_models()        # Compare multiple models
    # example_quick_analysis()        # One-liner
    # example_real_workflow()         # Realistic workflow
    
    print("\n" + "="*80)
    print("To run these examples:")
    print("  1. Update the file paths in the example functions")
    print("  2. Uncomment the example you want to run in main()")
    print("  3. Run: python example_shap_usage.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

