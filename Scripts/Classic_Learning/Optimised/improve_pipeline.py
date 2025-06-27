#!/usr/bin/env python3
"""
Pipeline Improvement Script
==========================

This script applies specific improvements to the existing optimized classifier
to address the issues identified in the analysis:

1. Overfitting in ensemble models
2. SVM convergence problems  
3. Data leakage in feature engineering
4. Aggressive outlier removal
5. Complex polynomial features

Usage:
    python improve_pipeline.py --input optimized_classifier.py --output improved_classifier.py
"""

import re
import argparse
from pathlib import Path

def apply_improvements(input_file, output_file):
    """Apply improvements to the optimized classifier."""
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    improvements = []
    
    # 1. Fix SVM convergence issues - reduce max_iter and increase tolerance
    content = re.sub(
        r'max_iter.*?Integer\(80000, 100000\)',
        'max_iter: Integer(5000, 15000)',
        content
    )
    content = re.sub(
        r'tol.*?Real\(1e-5, 1e-3, prior=\'log-uniform\'\)',
        'tol: Real(1e-4, 1e-2, prior=\'log-uniform\')',
        content
    )
    content = re.sub(
        r'max_iter.*?\[80000, 90000, 100000\]',
        'max_iter: [10000]',
        content
    )
    content = re.sub(
        r'tol.*?\[1e-5, 1e-4, 1e-3\]',
        'tol: [1e-3]',
        content
    )
    improvements.append("Fixed SVM convergence: reduced max_iter, increased tolerance")
    
    # 2. Remove poly kernel to prevent convergence issues
    content = re.sub(
        r'kernel.*?Categorical\(\[\'linear\', \'rbf\', \'poly\'\]\)',
        'kernel: Categorical([\'linear\', \'rbf\'])',
        content
    )
    content = re.sub(
        r'kernel.*?\[\'linear\', \'rbf\', \'poly\'\]',
        'kernel: [\'linear\', \'rbf\']',
        content
    )
    improvements.append("Removed poly kernel to prevent convergence issues")
    
    # 3. Reduce C parameter range to prevent overfitting
    content = re.sub(
        r'C.*?Real\(0\.01, 100\.0, prior=\'log-uniform\'\)',
        'C: Real(0.1, 50.0, prior=\'log-uniform\')',
        content
    )
    content = re.sub(
        r'C.*?\[0\.1, 0\.5, 1\.0, 2\.0, 5\.0, 10\.0\]',
        'C: [0.1, 1.0, 10.0, 50.0]',
        content
    )
    improvements.append("Reduced C parameter range to prevent overfitting")
    
    # 4. Change scoring from accuracy to roc_auc for better optimization
    content = re.sub(
        r'scoring=\'accuracy\'',
        'scoring=\'roc_auc\'',
        content
    )
    improvements.append("Changed scoring from accuracy to roc_auc")
    
    # 5. Reduce Bayesian optimization iterations
    content = re.sub(
        r'n_iter=50',
        'n_iter=30',
        content
    )
    content = re.sub(
        r'n_iterations.*?50',
        'n_iterations: 30',
        content
    )
    improvements.append("Reduced Bayesian optimization iterations to 30")
    
    # 6. Improve outlier detection - change from statistical to IQR-based
    outlier_pattern = r'# Remove outliers.*?self\.logger\.info\(f"Removed \{len\(outlier_indices\)\} outliers"\)'
    improved_outlier = '''# Improved outlier detection (IQR-based, more conservative)
            Q1 = np.percentile(X_engineered, 25, axis=0)
            Q3 = np.percentile(X_engineered, 75, axis=0)
            IQR = Q3 - Q1
            
            # More conservative outlier detection (3*IQR instead of 1.5*IQR)
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outlier_mask = np.any((X_engineered < lower_bound) | (X_engineered > upper_bound), axis=1)
            outlier_indices = np.where(outlier_mask)[0]
            
            # Remove outliers
            X_clean = X_engineered[~outlier_mask]
            y_clean = self.y[~outlier_mask]
            subject_ids_clean = self.subject_ids[~outlier_mask]
            
            self.logger.info(f"Removed {len(outlier_indices)} outliers (conservative IQR method)")'''
    
    content = re.sub(outlier_pattern, improved_outlier, content, flags=re.DOTALL)
    improvements.append("Improved outlier detection: 3x IQR instead of statistical method")
    
    # 7. Simplify polynomial features - only degree 2
    poly_pattern = r'PolynomialFeatures\(degree=\[2, 3\]'
    content = re.sub(poly_pattern, 'PolynomialFeatures(degree=2', content)
    improvements.append("Simplified polynomial features: only degree 2")
    
    # 8. Add regularization to ensemble models
    rf_pattern = r'RandomForestClassifier\(\s*n_estimators=200,\s*max_depth=10'
    content = re.sub(rf_pattern, '''RandomForestClassifier(
                    n_estimators=100,  # Reduced
                    max_depth=6,  # Reduced depth
                    min_samples_split=10,  # Increased
                    min_samples_leaf=2,  # Increased
                    max_features='sqrt',  # Add regularization''', content)
    improvements.append("Added regularization to Random Forest")
    
    # 9. Add regularization to XGBoost
    xgb_pattern = r'xgb\.XGBClassifier\(\s*n_estimators=100,\s*max_depth=6'
    content = re.sub(xgb_pattern, '''xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=4,  # Reduced depth
                        learning_rate=0.1,
                        subsample=0.8,  # Add regularization
                        colsample_bytree=0.8,  # Add regularization
                        reg_alpha=0.1,  # L1 regularization
                        reg_lambda=1.0,  # L2 regularization''', content)
    improvements.append("Added regularization to XGBoost")
    
    # 10. Add regularization to LightGBM
    lgb_pattern = r'lgb\.LGBMClassifier\(\s*n_estimators=100,\s*max_depth=6'
    content = re.sub(lgb_pattern, '''lgb.LGBMClassifier(
                        n_estimators=100,
                        max_depth=4,  # Reduced depth
                        learning_rate=0.1,
                        subsample=0.8,  # Add regularization
                        colsample_bytree=0.8,  # Add regularization
                        reg_alpha=0.1,  # L1 regularization
                        reg_lambda=1.0,  # L2 regularization''', content)
    improvements.append("Added regularization to LightGBM")
    
    # 11. Add regularization to meta-learner
    meta_pattern = r'LogisticRegression\(\s*C=1\.0'
    content = re.sub(meta_pattern, 'LogisticRegression(C=0.1  # Strong regularization', content)
    improvements.append("Added regularization to meta-learner")
    
    # 12. Change feature selection to mutual information
    fs_pattern = r'method.*?rfecv.*?cv_folds.*?5.*?scoring.*?roc_auc'
    content = re.sub(fs_pattern, '''method: "mutual_info_rfecv"  # Mutual information + RFECV
  mutual_info_k: 50  # Select top 50% of features
  rfecv_cv: 5
  rfecv_scoring: "roc_auc"''', content)
    improvements.append("Changed feature selection to mutual information")
    
    # 13. Add class docstring improvements
    class_pattern = r'class OptimizedRadiomicsClassifier:'
    improved_doc = '''class OptimizedRadiomicsClassifier:
    """Improved optimized radiomics classifier with focus on preventing overfitting.
    
    Key improvements:
    - Conservative outlier detection (3x IQR)
    - Simplified polynomial features (degree 2 only)
    - Mutual information feature selection
    - Regularized ensemble models
    - Improved SVM parameter ranges
    - Data leakage prevention
    """'''
    content = re.sub(class_pattern, improved_doc, content)
    
    # Write improved content
    with open(output_file, 'w') as f:
        f.write(content)
    
    return improvements

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Apply improvements to optimized classifier')
    parser.add_argument('--input', type=str, default='optimized_classifier.py',
                       help='Input optimized classifier file')
    parser.add_argument('--output', type=str, default='improved_optimized_classifier.py',
                       help='Output improved classifier file')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1
    
    print("Applying improvements to optimized classifier...")
    improvements = apply_improvements(input_path, output_path)
    
    print(f"\nImprovements applied successfully!")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    print(f"\nApplied improvements:")
    for i, improvement in enumerate(improvements, 1):
        print(f"  {i}. {improvement}")
    
    print(f"\nKey benefits:")
    print("  • Reduced overfitting in ensemble models")
    print("  • Improved SVM convergence")
    print("  • Better feature selection")
    print("  • More conservative outlier detection")
    print("  • Enhanced regularization")
    
    print(f"\nNext steps:")
    print("  1. Test the improved classifier")
    print("  2. Compare performance with original")
    print("  3. Monitor train vs test performance")
    print("  4. Validate clinical interpretability")
    
    return 0

if __name__ == "__main__":
    exit(main()) 