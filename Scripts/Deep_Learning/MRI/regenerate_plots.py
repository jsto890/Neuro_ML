import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional

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


def load_fold_test_metrics(model_dir: str) -> List[Dict[str, Any]]:
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
    fold_test_metrics = load_fold_test_metrics(model_dir)
    classification_description = get_label_description(args.labels) if (args.labels and get_label_description) else None
    create_test_summary_plots(fold_test_metrics, evaluation_dir, model_name, classification_description=classification_description)

    print(f"Regenerated plots saved to: {evaluation_dir}")


if __name__ == "__main__":
    main()


