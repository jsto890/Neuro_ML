#!/usr/bin/env python3
"""
Clinical Prediction Script (Multiclass CN/AD/PD) for Classical Models
=====================================================================

Enhancements:
- Multiclass support (CN, AD, PD) with configurable label mapping
- Accept a single-subject radiomics CSV row (or CSV with subject_id filter)
- JSON report output with probabilities and confidence summary
- Optional SHAP feature contributions for classical models

Note: This script is for classical (tabular radiomics) models. For deep learning
image inference and 3D interpretability volumes (Grad-CAM, occlusion, saliency, SHAP),
use the deep clinical script in `Scripts/Clinical_Deploy/Deep/`.
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def expand_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


class ClinicalPredictor:
    """Clinical predictor using a pre-trained classical model and scaler."""

    def __init__(self,
                 model_path: str = "~/reseng202500013-ndd-ml/data/optimized_classical_results/optimized_svm_model.pkl",
                 scaler_path: str = "~/reseng202500013-ndd-ml/data/optimized_classical_results/optimized_scaler.pkl",
                 label_map: dict = None):
        self.model = None
        self.scaler = None
        self.feature_names = None
        # Default mapping aligned with prior code: 0->AD, 1->CN, 2->PD
        self.label_map = label_map or {0: "AD", 1: "CN", 2: "PD"}
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}
        self.model_path = expand_path(model_path)
        self.scaler_path = expand_path(scaler_path)
        self.load_model()

    def load_model(self):
        """Load the serialized model and scaler."""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            print(" Model loaded successfully")
            print(f"  Model: {type(self.model).__name__}")
            print(f"  Scaler: {type(self.scaler).__name__}")
        except Exception as e:
            print(f" Error loading model artifacts: {e}")
            sys.exit(1)

    def _confidence_level(self, confidence: float) -> str:
        if confidence >= 0.9:
            return "Very High"
        if confidence >= 0.8:
            return "High"
        if confidence >= 0.7:
            return "Moderate"
        return "Low"

    def predict_single(self, feature_vector: np.ndarray, subject_id: str = "Unknown") -> dict:
        """Predict multiclass disease label and probabilities for one subject."""
        try:
            # Ensure shape [1, n_features]
            if feature_vector.ndim == 1:
                feature_vector = feature_vector.reshape(1, -1)

            X_scaled = self.scaler.transform(feature_vector)

            # Predictions and probabilities
            y_pred = self.model.predict(X_scaled)[0]
            proba = None
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_scaled)[0]

            # Build probabilities dict with label names when available
            class_probabilities = {}
            confidence = None
            pred_label_name = None

            if proba is not None and hasattr(self.model, 'classes_'):
                classes = list(self.model.classes_)
                confidence = float(np.max(proba))
                # Map to label names if possible
                for idx, cls in enumerate(classes):
                    name = self.label_map.get(int(cls), str(cls))
                    class_probabilities[name] = float(proba[idx])
                pred_label_name = self.label_map.get(int(y_pred), str(y_pred))
            else:
                # Fallback without probabilities
                pred_label_name = self.label_map.get(int(y_pred), str(y_pred))
                confidence = 0.0

            # Clinical text
            recommendation = "Consider further clinical evaluation" if pred_label_name in {"AD", "PD"} else "Routine monitoring"

            return {
                "subject_id": subject_id,
                "predicted_label_index": int(y_pred),
                "predicted_label_name": pred_label_name,
                "confidence": float(confidence),
                "confidence_level": self._confidence_level(confidence),
                "probabilities": class_probabilities,
                "recommendation": recommendation
            }
        except Exception as e:
            print(f" Error making prediction: {e}")
            return None

    def predict_from_csv(self, csv_path: str, subject_id: str = None) -> dict:
        """Load a radiomics CSV and predict for a specified row/subject."""
        df = pd.read_csv(expand_path(csv_path))
        # If subject_id specified, filter; else use the first row
        row = None
        subj = subject_id
        if subject_id is not None and 'subject_id' in df.columns:
            match = df[df['subject_id'].astype(str) == str(subject_id)]
            if len(match) == 0:
                raise ValueError(f"subject_id {subject_id} not found in CSV")
            row = match.iloc[0]
        else:
            row = df.iloc[0]
            if 'subject_id' in df.columns:
                subj = str(row['subject_id'])
            else:
                subj = subj or "Unknown"

        # Remove non-feature columns if present
        drop_cols = [c for c in ['subject_id', 'label', 'image_path'] if c in df.columns]
        feature_names = [c for c in df.columns if c not in drop_cols]
        self.feature_names = feature_names
        features = row[feature_names].astype(float).values
        return self.predict_single(features, subject_id=subj)


def compute_shap_contributions(model, scaler, feature_vector_1d: np.ndarray, feature_names: list):
    """Compute SHAP contributions for a single instance (best-effort)."""
    try:
        import shap  # type: ignore
    except Exception:
        return None

    try:
        # Scale the single instance
        X = feature_vector_1d.reshape(1, -1)
        X_scaled = scaler.transform(X)

        # Use generic Explainer (will pick a model-specific one if available)
        explainer = shap.Explainer(model)
        shap_values = explainer(X_scaled)

        # shap_values values shape: [n_samples, n_features, n_classes] for multiclass models
        values = getattr(shap_values, 'values', None)
        if values is None:
            return None

        # Aggregate absolute contributions across classes for ranking
        if values.ndim == 3:
            contrib = np.mean(np.abs(values[0]), axis=1)
        else:
            contrib = np.abs(values[0])

        top_indices = np.argsort(contrib)[::-1][:10]
        top = [{"feature": feature_names[i], "contribution": float(contrib[i])} for i in top_indices]
        return {
            "top_features": top,
            "all_features": {feature_names[i]: float(contrib[i]) for i in range(len(feature_names))}
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Clinical prediction (classical radiomics, CN/AD/PD)")
    parser.add_argument("--radiomics-file", type=str, required=False, help="Path to single-subject radiomics CSV or CSV containing the subject row")
    parser.add_argument("--subject-id", type=str, required=False, help="Subject identifier to select from CSV (if CSV has multiple rows)")
    parser.add_argument("--model-path", type=str, default="~/reseng202500013-ndd-ml/data/optimized_classical_results/optimized_svm_model.pkl", help="Path to serialized classical model (.pkl)")
    parser.add_argument("--scaler-path", type=str, default="~/reseng202500013-ndd-ml/data/optimized_classical_results/optimized_scaler.pkl", help="Path to serialized scaler (.pkl)")
    parser.add_argument("--output-dir", type=str, default="~/reseng202500013-ndd-ml/clinical_outputs/classical", help="Directory to write outputs (JSON report)")
    parser.add_argument("--compute-shap", action="store_true", help="Compute SHAP feature contributions (tabular)")
    parser.add_argument("--label-map-json", type=str, required=False, help="Optional JSON mapping of numeric labels to names, e.g., {\"0\":\"AD\",\"1\":\"CN\",\"2\":\"PD\"}")

    args = parser.parse_args()

    # If no radiomics input provided, fall back to legacy interactive mode
    if not args.radiomics_file:
        print("Interactive prediction mode (classical model)")
        print("Enter patient features (comma-separated) or 'quit' to exit")
        predictor = ClinicalPredictor(args.model_path, args.scaler_path)
        while True:
            try:
                user_input = input("Enter features (comma-separated) or 'quit': ").strip()
                if user_input.lower() == 'quit':
                    break
                features = np.array([float(x.strip()) for x in user_input.split(',')], dtype=float)
                result = predictor.predict_single(features)
                if result:
                    print(json.dumps(result, indent=2))
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f" Error: {e}")
        return

    # Prepare label mapping
    label_map = None
    if args.label_map_json:
        try:
            with open(expand_path(args.label_map_json), 'r') as f:
                raw_map = json.load(f)
            # Ensure keys become ints
            label_map = {int(k): str(v) for k, v in raw_map.items()}
        except Exception as e:
            print(f"Warning: could not load label map JSON: {e}. Using default mapping.")

    predictor = ClinicalPredictor(args.model_path, args.scaler_path, label_map)

    # Load CSV and predict
    result = predictor.predict_from_csv(args.radiomics_file, args.subject_id)
    if result is None:
        print(" No result produced")
        sys.exit(1)

    # Optionally compute SHAP on the same row
    shap_payload = None
    if args.compute_shap:
        try:
            df = pd.read_csv(expand_path(args.radiomics_file))
            # Select row used
            if args.subject_id is not None and 'subject_id' in df.columns:
                row = df[df['subject_id'].astype(str) == str(args.subject_id)].iloc[0]
            else:
                row = df.iloc[0]
            drop_cols = [c for c in ['subject_id', 'label', 'image_path'] if c in df.columns]
            feature_names = [c for c in df.columns if c not in drop_cols]
            features = row[feature_names].astype(float).values
            shap_payload = compute_shap_contributions(predictor.model, predictor.scaler, features, feature_names)
        except Exception:
            shap_payload = None

    # Prepare output directory
    output_dir = Path(expand_path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON report
    report = {
        "modality": "radiomics",
        "model_type": "classical",
        "model_name": type(predictor.model).__name__ if predictor.model is not None else "Unknown",
        "scaler_name": type(predictor.scaler).__name__ if predictor.scaler is not None else "Unknown",
        "result": result,
        "shap": shap_payload
    }

    # Use subject_id in filename
    sid = result.get("subject_id", "subject")
    json_path = output_dir / f"{sid}_clinical_prediction.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f" JSON report saved to: {json_path}")

    # Probability visualisation (bar chart) next to JSON
    try:
        result_probs = result.get("probabilities", {}) if isinstance(result, dict) else {}
        if result_probs:
            labels = list(result_probs.keys())
            values = [float(result_probs[k]) for k in labels]
            s = sum(values)
            disp_vals = [v / s if s > 0 else 0.0 for v in values]

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(labels, disp_vals, color=['#2ca02c' if l == 'CN' else '#1f77b4' if l == 'AD' else '#d62728' for l in labels])
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel('Probability')
            pred_name = result.get("predicted_label_name", "") if isinstance(result, dict) else ""
            conf = float(result.get("confidence", 0.0)) if isinstance(result, dict) else 0.0
            sid_title = sid if isinstance(sid, str) else "subject"
            ax.set_title(f"{sid_title} • Prediction: {pred_name} ({conf:.0%})")
            for rect, v in zip(bars, disp_vals):
                ax.annotate(f"{v*100:.0f}%",
                            xy=(rect.get_x() + rect.get_width() / 2, v),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
            fig.tight_layout()
            prob_png = output_dir / f"{sid}_probabilities.png"
            plt.savefig(str(prob_png), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f" Probabilities plot saved to: {prob_png}")
    except Exception as _e:
        print(f"Warning: failed to save probabilities plot: {_e}")


if __name__ == "__main__":
    main()