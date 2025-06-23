#!/usr/bin/env python3
"""
Standalone script to generate evaluation plots from deep learning training results.
This can be used to troubleshoot plotting issues or generate plots after training.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import argparse

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

def create_training_plots(folds_data, output_dir="./deep_learning_plots"):
    """Create comprehensive training plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating plots in: {output_path}")
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Deep Learning Training Results - AD vs CN Classification', fontsize=16, fontweight='bold')
    
    # 1. Training Loss
    ax1 = axes[0, 0]
    for fold_data in folds_data:
        epochs = [d['epoch'] for d in fold_data['data']]
        losses = [d['train_loss'] for d in fold_data['data']]
        ax1.plot(epochs, losses, alpha=0.7, label=f"Fold {fold_data['fold']}")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss by Fold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Training Accuracy
    ax2 = axes[0, 1]
    for fold_data in folds_data:
        epochs = [d['epoch'] for d in fold_data['data']]
        accs = [d['train_acc'] for d in fold_data['data']]
        ax2.plot(epochs, accs, alpha=0.7, label=f"Fold {fold_data['fold']}")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Accuracy')
    ax2.set_title('Training Accuracy by Fold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Validation AUC
    ax3 = axes[0, 2]
    for fold_data in folds_data:
        epochs = [d['epoch'] for d in fold_data['data']]
        aucs = [d['val_auc'] for d in fold_data['data']]
        ax3.plot(epochs, aucs, alpha=0.7, label=f"Fold {fold_data['fold']}")
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Validation AUC')
    ax3.set_title('Validation AUC by Fold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Best AUC per fold
    ax4 = axes[1, 0]
    best_aucs = []
    fold_numbers = []
    for fold_data in folds_data:
        best_auc = max([d['val_auc'] for d in fold_data['data']])
        best_aucs.append(best_auc)
        fold_numbers.append(fold_data['fold'])
    
    bars = ax4.bar(fold_numbers, best_aucs, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Fold')
    ax4.set_ylabel('Best Validation AUC')
    ax4.set_title('Best AUC per Fold')
    ax4.set_ylim(0.5, 1.0)
    
    # Add value labels on bars
    for bar, auc in zip(bars, best_aucs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc:.3f}', ha='center', va='bottom')
    
    ax4.grid(True, alpha=0.3)
    
    # 5. Final metrics comparison
    ax5 = axes[1, 1]
    final_metrics = []
    metric_names = []
    
    for fold_data in folds_data:
        final_epoch = fold_data['data'][-1]
        final_metrics.extend([
            final_epoch['val_auc'],
            final_epoch['val_acc'],
            final_epoch['train_acc']
        ])
        metric_names.extend(['Val AUC', 'Val Acc', 'Train Acc'])
    
    # Reshape for plotting
    metrics_array = np.array(final_metrics).reshape(len(folds_data), 3)
    
    x = np.arange(len(folds_data))
    width = 0.25
    
    ax5.bar(x - width, metrics_array[:, 0], width, label='Val AUC', alpha=0.8)
    ax5.bar(x, metrics_array[:, 1], width, label='Val Acc', alpha=0.8)
    ax5.bar(x + width, metrics_array[:, 2], width, label='Train Acc', alpha=0.8)
    
    ax5.set_xlabel('Fold')
    ax5.set_ylabel('Score')
    ax5.set_title('Final Metrics by Fold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(fold_numbers)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Training vs Validation Performance
    ax6 = axes[1, 2]
    final_train_accs = [fold_data['data'][-1]['train_acc'] for fold_data in folds_data]
    final_val_accs = [fold_data['data'][-1]['val_acc'] for fold_data in folds_data]
    
    ax6.scatter(final_train_accs, final_val_accs, s=100, alpha=0.7, c='green')
    ax6.plot([0.5, 1.0], [0.5, 1.0], 'k--', alpha=0.5)
    
    for i, fold_num in enumerate(fold_numbers):
        ax6.annotate(f'F{fold_num}', (final_train_accs[i], final_val_accs[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax6.set_xlabel('Final Training Accuracy')
    ax6.set_ylabel('Final Validation Accuracy')
    ax6.set_title('Training vs Validation Performance')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / 'deep_learning_training_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Main plot saved to: {plot_path}")
    
    # Create summary statistics
    summary = {
        'total_folds': len(folds_data),
        'average_best_auc': np.mean(best_aucs),
        'std_best_auc': np.std(best_aucs),
        'min_best_auc': np.min(best_aucs),
        'max_best_auc': np.max(best_aucs),
        'fold_results': []
    }
    
    for fold_data in folds_data:
        fold_result = {
            'fold': fold_data['fold'],
            'epochs_trained': len(fold_data['data']),
            'best_val_auc': max([d['val_auc'] for d in fold_data['data']]),
            'final_val_auc': fold_data['data'][-1]['val_auc'],
            'final_val_acc': fold_data['data'][-1]['val_acc'],
            'final_train_acc': fold_data['data'][-1]['train_acc']
        }
        summary['fold_results'].append(fold_result)
    
    # Save summary
    summary_path = output_path / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("DEEP LEARNING TRAINING SUMMARY")
    print("="*60)
    print(f"Total folds: {summary['total_folds']}")
    print(f"Average best AUC: {summary['average_best_auc']:.4f} ± {summary['std_best_auc']:.4f}")
    print(f"AUC range: {summary['min_best_auc']:.4f} - {summary['max_best_auc']:.4f}")
    print("\nFOLD DETAILS:")
    for fold_result in summary['fold_results']:
        print(f"Fold {fold_result['fold']}: {fold_result['epochs_trained']} epochs, "
              f"Best AUC: {fold_result['best_val_auc']:.4f}, "
              f"Final Val Acc: {fold_result['final_val_acc']:.4f}")
    print("="*60)
    
    return summary

def create_sample_data():
    """Create sample training data for testing."""
    print("Creating sample training data for demonstration...")
    
    folds_data = []
    for fold in range(1, 6):
        fold_data = {
            'fold': fold,
            'data': []
        }
        
        # Generate realistic training progression
        epochs = 50 + np.random.randint(0, 30)  # Random number of epochs
        for epoch in range(1, epochs + 1):
            # Simulate training progression
            train_loss = 0.05 * np.exp(-epoch/20) + 0.001 + np.random.normal(0, 0.002)
            train_acc = 0.5 + 0.4 * (1 - np.exp(-epoch/15)) + np.random.normal(0, 0.02)
            val_auc = 0.6 + 0.25 * (1 - np.exp(-epoch/25)) + np.random.normal(0, 0.01)
            val_acc = 0.55 + 0.2 * (1 - np.exp(-epoch/20)) + np.random.normal(0, 0.02)
            
            # Ensure values are in reasonable ranges
            train_loss = max(0.001, min(0.1, train_loss))
            train_acc = max(0.4, min(1.0, train_acc))
            val_auc = max(0.5, min(0.95, val_auc))
            val_acc = max(0.4, min(0.9, val_acc))
            
            epoch_data = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_auc': val_auc,
                'val_acc': val_acc,
                'lr': 0.0002
            }
            fold_data['data'].append(epoch_data)
        
        folds_data.append(fold_data)
    
    return folds_data

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation plots from training results")
    parser.add_argument("--output_dir", type=str, default="./deep_learning_plots",
                        help="Directory to save plots")
    parser.add_argument("--sample", action="store_true",
                        help="Generate sample plots for testing")
    parser.add_argument("--folds_data", type=str, default=None,
                        help="Path to folds_data.json from training")
    
    args = parser.parse_args()
    
    if args.folds_data:
        print(f"Loading folds_data from: {args.folds_data}")
        with open(args.folds_data, 'r') as f:
            folds_data = json.load(f)
    elif args.sample:
        print("Generating sample plots...")
        folds_data = create_sample_data()
    else:
        print("Error: No training data provided.")
        print("Use --sample to generate sample plots for testing.")
        print("Or use --folds_data to load your actual training data.")
        return
    
    try:
        summary = create_training_plots(folds_data, args.output_dir)
        print(f"\n✅ Successfully generated plots in: {args.output_dir}")
    except Exception as e:
        print(f"\n❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 