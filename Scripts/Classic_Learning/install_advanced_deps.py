#!/usr/bin/env python3
"""
Advanced Dependencies Installation Script
========================================

This script installs the advanced dependencies needed for the optimized pipeline:
- scikit-optimize (for Bayesian optimization)
- xgboost (for gradient boosting)
- lightgbm (for fast gradient boosting)
"""

import subprocess
import sys
import os

def install_package(package_name, pip_name=None):
    """Install a package and return success status."""
    if pip_name is None:
        pip_name = package_name
    
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        print(f"✓ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package_name}: {e}")
        return False

def check_package(package_name):
    """Check if a package is already installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    """Install advanced dependencies."""
    print("Advanced Dependencies Installation")
    print("=" * 40)
    
    # Core dependencies (required)
    core_packages = [
        ("scikit-learn", "scikit-learn"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scipy", "scipy")
    ]
    
    # Advanced dependencies (optional but recommended)
    advanced_packages = [
        ("skopt", "scikit-optimize"),  # Bayesian optimization
        ("xgboost", "xgboost"),        # Gradient boosting
        ("lightgbm", "lightgbm")       # Fast gradient boosting
    ]
    
    # Check and install core packages
    print("\nChecking core dependencies...")
    core_missing = []
    for package, pip_name in core_packages:
        if check_package(package):
            print(f"✓ {package} already installed")
        else:
            core_missing.append((package, pip_name))
    
    if core_missing:
        print(f"\nInstalling {len(core_missing)} missing core packages...")
        for package, pip_name in core_missing:
            install_package(package, pip_name)
    else:
        print("All core dependencies are already installed!")
    
    # Check and install advanced packages
    print("\nChecking advanced dependencies...")
    advanced_missing = []
    for package, pip_name in advanced_packages:
        if check_package(package):
            print(f"✓ {package} already installed")
        else:
            advanced_missing.append((package, pip_name))
    
    if advanced_missing:
        print(f"\nInstalling {len(advanced_missing)} advanced packages...")
        print("These packages enable Bayesian optimization and advanced models:")
        print("  • scikit-optimize: Bayesian hyperparameter optimization")
        print("  • xgboost: Gradient boosting with regularization")
        print("  • lightgbm: Fast gradient boosting")
        print()
        
        for package, pip_name in advanced_missing:
            install_package(package, pip_name)
    else:
        print("All advanced dependencies are already installed!")
    
    # Final verification
    print("\n" + "=" * 40)
    print("Final verification...")
    
    all_packages = core_packages + advanced_packages
    all_installed = True
    
    for package, _ in all_packages:
        if check_package(package):
            print(f"✓ {package} - OK")
        else:
            print(f"✗ {package} - MISSING")
            all_installed = False
    
    print("\n" + "=" * 40)
    if all_installed:
        print("🎉 All dependencies installed successfully!")
        print("\nYou can now run the optimized pipeline:")
        print("  python run_optimized.py")
        print("\nThe pipeline will use:")
        print("  • Bayesian optimization for hyperparameter tuning")
        print("  • XGBoost and LightGBM for advanced models")
        print("  • Stacking ensemble with diverse base models")
    else:
        print("⚠️  Some dependencies failed to install.")
        print("The pipeline will still work but with limited functionality.")
        print("You can manually install missing packages using:")
        print("  pip install <package_name>")

if __name__ == "__main__":
    main() 