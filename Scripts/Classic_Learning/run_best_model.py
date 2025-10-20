#!/usr/bin/env python3
"""
Complete Radiomics Classification Workflow - Current Best Model
==============================================================

This script provides a complete workflow from radiomics features to the current best model
(Improved Optimised Pipeline) for neurodegenerative disease detection.

Author: P4P Team
Date: 2024
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add the Optimised directory to the path
sys.path.append(str(Path(__file__).parent / "Optimised"))

try:
    from improved_optimised_classifier import ImprovedOptimizedRadiomicsClassifier
except ImportError as e:
    print(f"Error importing ImprovedOptimizedRadiomicsClassifier: {e}")
    print("Please ensure you're running this from the Classic_Learning directory")
    sys.exit(1)

def setup_logging(output_dir):
    """Setup comprehensive logging for the workflow."""
    log_file = output_dir / f"best_model_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def validate_input_data(input_path):
    """Validate that the input radiomics file exists and has required format."""
    logger = logging.getLogger(__name__)
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Check file extension
    if not input_path.endswith('.csv'):
        logger.warning(f"Input file {input_path} doesn't have .csv extension")
    
    logger.info(f"Input file validated: {input_path}")
    return True

def create_output_directory(output_dir):
    """Create output directory with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(output_dir) / f"best_model_results_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Created output directory: {output_path}")
    return output_path

def load_config(config_path=None):
    """Load configuration, using default if none provided."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded configuration from: {config_path}")
    else:
        # Default configuration for the improved optimised pipeline
        config = {
            'data': {
                'binary_only': True,
                'random_state': 42
            },
            'feature_engineering': {
                'variance_threshold': 0.01,
                'polynomial_degrees': [2],
                'polynomial_interaction_only': True,
                'outlier_method': 'iqr_3x',
                'outlier_threshold': 3.0,
                'enable_statistical_features': True,
                'normalization': {
                    'enabled': True,
                    'scaler': 'robust'
                }
            },
            'feature_selection': {
                'method': 'mutual_info_rfecv',
                'mutual_info_k': 50,
                'rfecv_cv': 5,
                'rfecv_scoring': 'roc_auc',
                'min_features': 10
            },
            'svm_optimization': {
                'cv_folds': 5,
                'scoring': 'roc_auc',
                'search_method': 'bayesian',
                'n_iter': 30
            },
            'ensemble': {
                'enabled': True,
                'method': 'stacking'
            },
            'splitting': {
                'test_size': 0.2,
                'val_size': 0.2,
                'random_state': 42,
                'stratify': True
            },
            'evaluation': {
                'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
                'plots': {
                    'model_comparison': True,
                    'roc_curves': True,
                    'feature_importance': True,
                    'confusion_matrix': True,
                    'train_vs_test': True,
                    'overfitting_analysis': True
                }
            },
            'output': {
                'save_models': True,
                'save_scaler': True,
                'save_feature_importance': True,
                'save_feature_engineering': True,
                'plot_dpi': 300,
                'plot_format': 'png',
                'save_ensemble': True
            },
            'clinical': {
                'interpretability': True,
                'feature_importance_threshold': 0.01,
                'confidence_threshold': 0.8,
                'save_predictions': True,
                'save_probabilities': True
            }
        }
        logger = logging.getLogger(__name__)
        logger.info("Using default configuration for improved optimised pipeline")
    
    return config

def run_complete_workflow(input_path, output_dir, config_path=None, random_state=42):
    """
    Complete workflow from radiomics features to trained model.
    
    Args:
        input_path (str): Path to radiomics CSV file
        output_dir (str): Base output directory
        config_path (str, optional): Path to configuration file
        random_state (int): Random seed for reproducibility
    
    Returns:
        dict: Results summary
    """
    logger = logging.getLogger(__name__)
    
    # Step 1: Setup and validation
    logger.info("=" * 60)
    logger.info("STARTING COMPLETE RADIOMICS CLASSIFICATION WORKFLOW")
    logger.info("USING CURRENT BEST MODEL: IMPROVED OPTIMIZED PIPELINE")
    logger.info("=" * 60)
    
    # Validate input
    validate_input_data(input_path)
    
    # Create output directory
    output_path = create_output_directory(output_dir)
    
    # Load configuration
    config = load_config(config_path)
    
    # Step 2: Initialize the improved optimised classifier
    logger.info("Step 2: Initializing Improved Optimized Radiomics Classifier...")
    classifier = ImprovedOptimizedRadiomicsClassifier(
        input_path=input_path,
        output_dir=str(output_path),
        random_state=random_state,
        binary_only=config['data']['binary_only']
    )
    
    # Step 3: Run the complete pipeline
    logger.info("Step 3: Running complete pipeline...")
    success = classifier.run_improved_pipeline()
    
    if not success:
        logger.error("Pipeline failed to complete successfully")
        return {'success': False, 'error': 'Pipeline execution failed'}
    
    # Step 4: Generate summary report
    logger.info("Step 4: Generating summary report...")
    summary = generate_summary_report(classifier, output_path)
    
    # Step 5: Save workflow results
    logger.info("Step 5: Saving workflow results...")
    save_workflow_results(summary, output_path)
    
    logger.info("=" * 60)
    logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    
    return summary

def generate_summary_report(classifier, output_path):
    """Generate a comprehensive summary report."""
    logger = logging.getLogger(__name__)
    
    try:
        # Get model performance
        if hasattr(classifier, 'svm_model') and classifier.svm_model is not None:
            svm_results = classifier._evaluate_model(classifier.svm_model, 'SVM')
        else:
            svm_results = None
        
        if hasattr(classifier, 'ensemble_model') and classifier.ensemble_model is not None:
            ensemble_results = classifier._evaluate_model(classifier.ensemble_model, 'Ensemble')
        else:
            ensemble_results = None
        
        # Get feature engineering results
        fe_results = getattr(classifier, 'feature_engineering_results', {})
        
        # Create summary
        summary = {
            'workflow_info': {
                'timestamp': datetime.now().isoformat(),
                'input_file': classifier.input_path,
                'output_directory': str(output_path),
                'pipeline_version': 'Improved Optimized Pipeline (Current Best)',
                'random_state': classifier.random_state
            },
            'data_info': {
                'n_samples': len(classifier.data) if hasattr(classifier, 'data') and classifier.data is not None else 'Unknown',
                'n_features_original': len(classifier.feature_names) if hasattr(classifier, 'feature_names') else 'Unknown',
                'n_features_final': len(classifier.feature_names) if hasattr(classifier, 'feature_names') else 'Unknown',
                'class_distribution': 'Available in results files'
            },
            'model_performance': {
                'svm': svm_results,
                'ensemble': ensemble_results
            },
            'feature_engineering': fe_results,
            'output_files': {
                'models': list(output_path.glob('*.pkl')),
                'plots': list(output_path.glob('*.png')),
                'results': list(output_path.glob('*.json')),
                'logs': list(output_path.glob('*.log'))
            },
            'recommendations': {
                'primary_model': 'SVM model (optimised_svm_model.pkl)',
                'backup_model': 'Ensemble model (optimised_ensemble_model.pkl)',
                'clinical_use': 'Use SVM model for clinical predictions',
                'research_use': 'Use ensemble model for research and validation'
            }
        }
        
        logger.info("Summary report generated successfully")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary report: {e}")
        return {'error': str(e)}

def save_workflow_results(summary, output_path):
    """Save workflow results and summary."""
    logger = logging.getLogger(__name__)
    
    try:
        # Save summary as JSON
        summary_file = output_path / 'workflow_summary.json'
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save summary as text report
        report_file = output_path / 'workflow_report.txt'
        with open(report_file, 'w') as f:
            f.write("COMPLETE RADIOMICS CLASSIFICATION WORKFLOW REPORT\n")
            f.write("CURRENT BEST MODEL: IMPROVED OPTIMIZED PIPELINE\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Timestamp: {summary['workflow_info']['timestamp']}\n")
            f.write(f"Pipeline Version: {summary['workflow_info']['pipeline_version']}\n")
            f.write(f"Input File: {summary['workflow_info']['input_file']}\n")
            f.write(f"Output Directory: {summary['workflow_info']['output_directory']}\n\n")
            
            f.write("MODEL PERFORMANCE\n")
            f.write("-" * 20 + "\n")
            if summary['model_performance']['svm']:
                f.write(f"SVM Model:\n")
                for metric, value in summary['model_performance']['svm'].items():
                    f.write(f"  {metric}: {value}\n")
                f.write("\n")
            
            if summary['model_performance']['ensemble']:
                f.write(f"Ensemble Model:\n")
                for metric, value in summary['model_performance']['ensemble'].items():
                    f.write(f"  {metric}: {value}\n")
                f.write("\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            for key, value in summary['recommendations'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("OUTPUT FILES\n")
            f.write("-" * 12 + "\n")
            for file_type, files in summary['output_files'].items():
                f.write(f"{file_type.upper()}:\n")
                for file_path in files:
                    f.write(f"  {file_path.name}\n")
                f.write("\n")
        
        logger.info(f"Workflow results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving workflow results: {e}")

def main():
    """Main function to run the complete workflow."""
    parser = argparse.ArgumentParser(
        description="Complete Radiomics Classification Workflow - Current Best Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python run_best_model.py --input radiomics_features.csv --output results/
  
  # Run with custom configuration
  python run_best_model.py --input radiomics_features.csv --output results/ --config my_config.yaml
  
  # Run with custom random seed
  python run_best_model.py --input radiomics_features.csv --output results/ --random-state 123
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to radiomics CSV file (must contain subject_id, label columns)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Base output directory for results'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Path to configuration YAML file (optional, uses defaults if not provided)'
    )
    
    parser.add_argument(
        '--random-state', '-r',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    output_path = create_output_directory(args.output)
    logger = setup_logging(output_path)
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Run the complete workflow
        results = run_complete_workflow(
            input_path=args.input,
            output_dir=args.output,
            config_path=args.config,
            random_state=args.random_state
        )
        
        if results.get('success', True):
            print(f"\n✅ Workflow completed successfully!")
            print(f"📁 Results saved to: {output_path}")
            print(f"🎯 Primary model: SVM (optimised_svm_model.pkl)")
            print(f"🔄 Backup model: Ensemble (optimised_ensemble_model.pkl)")
            print(f"📊 Check workflow_report.txt for detailed results")
        else:
            print(f"\n❌ Workflow failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Workflow failed with error: {e}")
        print(f"\n❌ Workflow failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 