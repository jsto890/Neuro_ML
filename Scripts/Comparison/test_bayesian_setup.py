#!/usr/bin/env python3
"""
Test script to verify Bayesian analysis setup and dependencies.

Run this to check if all required packages are installed and working.
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported."""
    
    required_packages = [
        'numpy',
        'pandas', 
        'matplotlib',
        'seaborn',
        'sklearn',
        'pymc',
        'arviz',
        'bambi'
    ]
    
    print("Testing package imports...")
    print("=" * 50)
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    print("\n" + "=" * 50)
    
    if failed_imports:
        print(f"❌ Failed to import: {', '.join(failed_imports)}")
        print("\nTo install missing packages, run:")
        print("pip install -r requirements_bayesian.txt")
        return False
    else:
        print("✅ All packages imported successfully!")
        return True


def test_bayesian_functionality():
    """Test basic Bayesian functionality."""
    
    try:
        import numpy as np
        import pymc as pm
        import arviz as az
        
        print("\nTesting basic Bayesian functionality...")
        print("=" * 50)
        
        # Test PyMC model creation
        with pm.Model() as test_model:
            x = pm.Normal('x', 0, 1)
            y = pm.Normal('y', x, 1, observed=np.random.randn(10))
            
            # Test sampling (just a few samples for speed)
            with pm.Model() as quick_test:
                test_var = pm.Normal('test_var', 0, 1)
                idata = pm.sample(100, tune=100, progressbar=False, random_seed=42)
        
        print("✅ PyMC model creation and sampling")
        
        # Test ArviZ
        summary = az.summary(idata)
        print("✅ ArviZ summary generation")
        
        # Test Bambi
        import bambi as bmb
        
        # Create simple test data
        test_data = {
            'y': np.random.binomial(1, 0.5, 100),
            'x': np.random.randn(100),
            'group': np.random.choice(['A', 'B'], 100)
        }
        
        # Test Bambi model (convert dict to DataFrame)
        import pandas as pd
        test_df = pd.DataFrame(test_data)
        model = bmb.Model('y ~ x + (1|group)', test_df, family='bernoulli')
        print("✅ Bambi model creation")
        
        print("\n✅ All Bayesian functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Bayesian functionality test failed: {e}")
        return False


def test_data_structures():
    """Test that our data structures work correctly."""
    
    try:
        import numpy as np
        import pandas as pd
        
        print("\nTesting data structures...")
        print("=" * 50)
        
        # Test ModelFoldData-like structure
        n_samples = 50
        n_classes = 3
        
        predictions = np.random.randint(0, n_classes, n_samples)
        probabilities = np.random.dirichlet([1, 1, 1], n_samples)
        labels = np.random.randint(0, n_classes, n_samples)
        subject_ids = [f"subject_{i}" for i in range(n_samples)]
        
        # Test DataFrame creation
        df = pd.DataFrame({
            'model': ['TestModel'] * n_samples,
            'site': ['fold_1'] * n_samples,
            'correct': (predictions == labels).astype(int),
            'true_label': labels,
            'predicted_label': predictions
        })
        
        print("✅ Data structure creation")
        print(f"   - Predictions shape: {predictions.shape}")
        print(f"   - Probabilities shape: {probabilities.shape}")
        print(f"   - DataFrame shape: {df.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False


def main():
    """Run all tests."""
    
    print("Bayesian Model Comparison Setup Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Setup incomplete. Please install missing packages.")
        return False
    
    # Test functionality
    functionality_ok = test_bayesian_functionality()
    
    # Test data structures
    data_ok = test_data_structures()
    
    print("\n" + "=" * 60)
    
    if imports_ok and functionality_ok and data_ok:
        print("🎉 All tests passed! Bayesian analysis setup is ready.")
        print("\nYou can now run:")
        print("  python bayesian_model_comparison.py --help")
        print("  python example_bayesian_usage.py")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
