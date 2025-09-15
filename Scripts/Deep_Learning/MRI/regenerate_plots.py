import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reuse plotting helpers from training script
try:
    from train_smri import create_training_plots, create_test_summary_plots, get_label_description
except Exception:
    # Fallback: minimal stubs to avoid import failure; will exit if not available
    create_training_plots = None
    create_test_summary_plots = None
    get_label_description = None


def load_folds_data(model_dir: str, model_name: str) -> Optional[List[Dict[str, Any]]]:
    """Load folds_data JSON saved during training, or return None if missing."""
    candidates = [
        os.path.join(model_dir, f"{model_name}_folds_data.json"),
        os.path.join(model_dir, "folds_data.json"),
    ]
    for folds_path in candidates:
        if os.path.exists(folds_path):
            with open(folds_path, "r") as f:
                return json.load(f)
    return None


def _pad_metrics_with_all_labels(metrics: Dict[str, Any], all_labels: Optional[List[int]]) -> Dict[str, Any]:
    """Ensure confusion matrix and classification report include all labels.
    If all_labels is None, returns metrics unchanged.
    """
    if not all_labels:
        return metrics
    try:
        # Confusion matrix padding
        cm = metrics.get("confusion_matrix")
        report = metrics.get("classification_report", {})
        if cm is not None and isinstance(cm, list):
            import numpy as np
            present_labels = []
            # Extract numeric class keys from classification_report if available
            for k in report.keys():
                if isinstance(k, str) and k.isdigit():
                    present_labels.append(int(k))
            # Fallback to range of current cm size if report absent
            if not present_labels:
                present_labels = list(range(len(cm)))
            label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
            new_cm = np.zeros((len(all_labels), len(all_labels)), dtype=int)
            cm_arr = np.array(cm)
            for i_src, lbl_true in enumerate(present_labels):
                for j_src, lbl_pred in enumerate(present_labels):
                    if lbl_true in label_to_idx and lbl_pred in label_to_idx:
                        new_cm[label_to_idx[lbl_true], label_to_idx[lbl_pred]] = int(cm_arr[i_src, j_src])
            metrics["confusion_matrix"] = new_cm.tolist()

        # Ensure each class exists in classification_report
        if isinstance(report, dict):
            for lbl in all_labels:
                key = str(lbl)
                if key not in report:
                    report[key] = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}
            metrics["classification_report"] = report
    except Exception:
        # Do not fail regen if padding fails
        return metrics
    return metrics


def load_fold_test_metrics(model_dir: str, all_labels: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Reconstruct fold_test_metrics from per-fold test_metrics JSON files."""
    fold_test_metrics: List[Dict[str, Any]] = []

    # Prefer explicit metrics files: test_metrics_fold_{i}.json
    metrics_files = [f for f in os.listdir(model_dir) if re.match(r"^test_metrics_fold_\d+\.json$", f)]
    if metrics_files:
        for fname in sorted(metrics_files, key=lambda x: int(re.findall(r"\d+", x)[0])):
            fold_idx = int(re.findall(r"\d+", fname)[0])
            metrics_path = os.path.join(model_dir, fname)
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            metrics = _pad_metrics_with_all_labels(metrics, all_labels)
            fold_test_metrics.append({
                "fold": fold_idx,
                "metrics": metrics,
                "threshold_used": None,
                "temperature_scaling": None,
            })
    else:
        # Fallback: infer folds from plot directories
        for entry in sorted(os.listdir(model_dir)):
            m = re.match(r"^test_evaluation_plots_fold_(\d+)$", entry)
            if not m:
                continue
            fold_idx = int(m.group(1))
            metrics_path = os.path.join(model_dir, f"test_metrics_fold_{fold_idx}.json")
            if not os.path.exists(metrics_path):
                continue
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            metrics = _pad_metrics_with_all_labels(metrics, all_labels)
            fold_test_metrics.append({
                "fold": fold_idx,
                "metrics": metrics,
                "threshold_used": None,
                "temperature_scaling": None,
            })

    if not fold_test_metrics:
        raise FileNotFoundError(f"No per-fold test metrics found in: {model_dir}")

    fold_test_metrics.sort(key=lambda x: x["fold"])  # Ensure sorted
    return fold_test_metrics


def main():
    parser = argparse.ArgumentParser(description="Regenerate evaluation plots from saved JSONs")
    parser.add_argument("--model_dir", required=True, help="Path to model directory for a single model inside a run (e.g., .../run_xxx/Simple3DCNN)")
    parser.add_argument("--model_name", default=None, help="Model name (inferred from folder name if not provided)")
    parser.add_argument("--labels", nargs="+", type=int, default=None, help="Label set used (optional, for plot titles)")
    args = parser.parse_args()

    model_dir = os.path.abspath(os.path.expanduser(args.model_dir))
    if not os.path.isdir(model_dir):
        raise NotADirectoryError(f"Not a directory: {model_dir}")

    model_name = args.model_name or os.path.basename(model_dir.rstrip(os.sep))

    # Import checks
    if create_training_plots is None or create_test_summary_plots is None:
        raise RuntimeError("Could not import plotting helpers from train_smri.py. Please run from the same environment/repo.")

    # Prepare output dir
    evaluation_dir = os.path.join(model_dir, "evaluation_plots")
    os.makedirs(evaluation_dir, exist_ok=True)

    # Load and regenerate training plots (if folds data is available)
    folds_data = load_folds_data(model_dir, model_name)
    if folds_data is not None:
        create_training_plots(folds_data, evaluation_dir, model_name)
    else:
        print("[INFO] Training folds data not found; skipping training plots.")

    # Rebuild fold_test_metrics and regenerate test summary plots
    fold_test_metrics = load_fold_test_metrics(model_dir, args.labels)
    classification_description = get_label_description(args.labels) if (args.labels and get_label_description) else None
    # Generate the default summary first
    create_test_summary_plots(fold_test_metrics, evaluation_dir, model_name, classification_description=classification_description)
    
    # Now create an alternate summary PNG that replaces the confusion matrix panel
    # with the cumulative (sum) confusion matrix using fixed labels
    try:
        # Import plotting helper from evaluation
        from evaluate_model import create_evaluation_plots as _create_eval_plot
    except Exception:
        _create_eval_plot = None

    # Build cumulative metrics from fold_test_metrics
    try:
        cms = [np.array(item['metrics']['confusion_matrix']) for item in fold_test_metrics if 'metrics' in item and 'confusion_matrix' in item['metrics']]
        if cms:
            agg_cm = np.sum(cms, axis=0)
            # Build a pseudo metrics dict using averaged scalar metrics but cumulative CM
            # Use the first fold's scalar metrics as a baseline
            base = fold_test_metrics[0]['metrics']
            pseudo_metrics = {
                'accuracy': float(np.mean([m['metrics']['accuracy'] for m in fold_test_metrics])),
                'precision': float(np.mean([m['metrics']['precision'] for m in fold_test_metrics])),
                'recall': float(np.mean([m['metrics']['recall'] for m in fold_test_metrics])),
                'f1_score': float(np.mean([m['metrics']['f1_score'] for m in fold_test_metrics])),
                'auc': float(np.mean([m['metrics']['auc'] for m in fold_test_metrics])),
                'mcc': float(np.mean([m['metrics']['mcc'] for m in fold_test_metrics])),
                'confusion_matrix': agg_cm,
                'classification_report': base.get('classification_report', {})
            }
            # Create a replacement panel PNG to be used alongside the default summary
            cm_summary_path = os.path.join(evaluation_dir, 'confusion_matrix_summary_fixed.png')
            _save_fixed_cm(agg_cm.tolist(), cm_summary_path, disease_names)
    except Exception:
        pass

    # Additionally, force per-fold confusion matrix labels to include all classes
    def _disease_names_from_labels(lbls: Optional[List[int]]) -> List[str]:
        if not lbls:
            return []
        name_map = {0: 'CN', 1: 'AD', 2: 'PD'}
        return [name_map.get(i, f'Class {i}') for i in lbls]

    disease_names = _disease_names_from_labels(args.labels)

    def _save_fixed_cm(cm: List[List[int]], out_path: str, names: List[str]):
        try:
            cm_arr = np.array(cm)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=names if names else 'auto',
                        yticklabels=names if names else 'auto')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix (Fixed Labels)')
            fig.tight_layout()
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
        except Exception:
            pass

    # Write per-fold fixed confusion matrices
    for item in fold_test_metrics:
        fold_idx = item.get('fold')
        metrics = item.get('metrics', {})
        cm = metrics.get('confusion_matrix')
        if cm is None:
            continue
        fold_dir = os.path.join(model_dir, f"test_evaluation_plots_fold_{fold_idx}")
        out_path = os.path.join(fold_dir, 'confusion_matrix_fixed.png')
        _save_fixed_cm(cm, out_path, disease_names)

    # Also write an aggregated (sum) confusion matrix across folds
    try:
        cms = [np.array(item['metrics']['confusion_matrix']) for item in fold_test_metrics if 'metrics' in item and 'confusion_matrix' in item['metrics']]
        if cms:
            agg_cm = np.sum(cms, axis=0)
            out_path_summary = os.path.join(evaluation_dir, 'confusion_matrix_summary_fixed.png')
            _save_fixed_cm(agg_cm.tolist(), out_path_summary, disease_names)
    except Exception:
        pass

    print(f"Regenerated plots saved to: {evaluation_dir}")


if __name__ == "__main__":
    main()


