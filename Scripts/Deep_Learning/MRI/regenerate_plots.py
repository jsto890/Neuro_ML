import os
import json
import argparse
from typing import List, Dict, Any

# Reuse plotting helpers from training script
try:
    from train_smri import create_training_plots, create_test_summary_plots, get_label_description
except Exception:
    # Fallback: minimal stubs to avoid import failure; will exit if not available
    create_training_plots = None
    create_test_summary_plots = None
    get_label_description = None


def load_folds_data(model_dir: str, model_name: str) -> List[Dict[str, Any]]:
    """Load folds_data JSON saved during training."""
    folds_path = os.path.join(model_dir, f"{model_name}_folds_data.json")
    if not os.path.exists(folds_path):
        raise FileNotFoundError(f"folds_data file not found: {folds_path}")
    with open(folds_path, "r") as f:
        folds_data = json.load(f)
    return folds_data


def load_fold_test_metrics(model_dir: str) -> List[Dict[str, Any]]:
    """Reconstruct fold_test_metrics from per-fold test_metrics JSON files."""
    fold_test_metrics: List[Dict[str, Any]] = []

    # Find files like test_metrics_fold_{i}.json
    for entry in sorted(os.listdir(model_dir)):
        if not entry.startswith("test_evaluation_plots_fold_"):
            continue
        # Extract fold index
        try:
            fold_str = entry.split("test_evaluation_plots_fold_")[-1]
            fold_idx = int(fold_str)
        except Exception:
            continue

        metrics_path = os.path.join(model_dir, f"test_metrics_fold_{fold_idx}.json")
        if not os.path.exists(metrics_path):
            # Skip if metrics missing
            continue
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        fold_test_metrics.append({
            "fold": fold_idx,
            "metrics": metrics,
            "threshold_used": None,
            "temperature_scaling": None,
        })

    if not fold_test_metrics:
        raise FileNotFoundError(f"No per-fold test metrics found in: {model_dir}")

    # Ensure sorted by fold
    fold_test_metrics.sort(key=lambda x: x["fold"])
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

    # Load and regenerate training plots
    folds_data = load_folds_data(model_dir, model_name)
    create_training_plots(folds_data, evaluation_dir, model_name)

    # Rebuild fold_test_metrics and regenerate test summary plots
    fold_test_metrics = load_fold_test_metrics(model_dir)
    classification_description = get_label_description(args.labels) if (args.labels and get_label_description) else None
    create_test_summary_plots(fold_test_metrics, evaluation_dir, model_name, classification_description=classification_description)

    print(f"Regenerated plots saved to: {evaluation_dir}")


if __name__ == "__main__":
    main()


