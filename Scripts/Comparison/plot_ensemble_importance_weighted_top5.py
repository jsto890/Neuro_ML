#!/usr/bin/env python3
"""
Create publication-ready Top-5 ensemble feature-importance plots from precomputed CSVs.

Outputs one PNG per modality with an overlaid (paired-bar) comparison on a single axis:
- Stability-weighted ensemble SHAP importance (`ensemble_weighted`) (normalised 0-1)
- Voting-based ensemble frequency (`vote_frequency`) (normalised 0-1)

This script is aligned with the style used in
`Scripts/Clinical_Deploy/Classic/run_shap_comprehensive.py`, but limits to top 5.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt


def _wrap_label(s: str, width: int) -> str:
    if width <= 0:
        return s
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False))


def _read_topn(
    csv_path: Path,
    *,
    value_col: str,
    top_n: int = 5,
    wrap_width: int = 42,
) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} appears empty or missing header row")

        required = {"feature", value_col}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}; found: {sorted(reader.fieldnames)}")

        for r in reader:
            feat = (r.get("feature") or "").strip()
            val_raw = (r.get(value_col) or "").strip()
            if not feat or not val_raw:
                continue
            try:
                val = float(val_raw)
            except ValueError:
                continue
            rows.append((feat, val))

    rows = sorted(rows, key=lambda x: x[1], reverse=True)[:top_n]
    features_wrapped = [_wrap_label(s, wrap_width) for s, _ in rows]
    return [(fw, v) for fw, (_, v) in zip(features_wrapped, rows)]


def _read_metric_by_feature(csv_path: Path, *, value_col: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return out
        if "feature" not in reader.fieldnames or value_col not in reader.fieldnames:
            return out
        for r in reader:
            feat = (r.get("feature") or "").strip()
            val_raw = (r.get(value_col) or "").strip()
            if not feat or not val_raw:
                continue
            try:
                out[feat] = float(val_raw)
            except ValueError:
                continue
    return out


def _is_excluded_feature(
    feature_name: str,
    *,
    exclude_shape: bool,
    exclude_firstorder_minmax: bool,
) -> bool:
    f = feature_name.lower()

    if exclude_shape and "shape" in f:
        return True

    if exclude_firstorder_minmax and (
        "firstorder_minimum" in f
        or f.endswith("_minimum")
        or "firstorder_maximum" in f
        or f.endswith("_maximum")
    ):
        return True

    return False


def _top_features(
    csv_path: Path,
    *,
    metric_col: str,
    top_n: int,
    exclude_shape: bool,
    exclude_firstorder_minmax: bool,
) -> list[str]:
    metric = _read_metric_by_feature(csv_path, value_col=metric_col)
    items: list[tuple[str, float]] = []
    for feat, val in metric.items():
        if _is_excluded_feature(
            feat,
            exclude_shape=exclude_shape,
            exclude_firstorder_minmax=exclude_firstorder_minmax,
        ):
            continue
        items.append((feat, val))
    items.sort(key=lambda x: x[1], reverse=True)
    return [feat for feat, _ in items[:top_n]]


def _setup_pub_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.titleweight": "bold",
        }
    )

def _barh_panel(ax: plt.Axes, labels: list[str], values: list[float], *, color: str, xlabel: str, title: str) -> None:
    y_pos = list(range(len(values)))
    ax.barh(y_pos, values, alpha=0.85, color=color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=10)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0.0)

    xmax = float(max(values)) if values else 1.0
    ax.set_xlim(0, xmax * 1.12 if xmax > 0 else 1.0)
    for i, v in enumerate(values):
        ax.text(
            v + xmax * 0.02,
            i,
            f"{v:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#222222",
        )


def plot_top5_weighted_vs_voting_overlay(
    csv_path: Path,
    modality: str,
    out_path: Path,
    *,
    top_n: int = 5,
    wrap_width: int = 42,
    weighted_color: str = "purple",
    voting_color: str = "darkorange",
    exclude_shape: bool = True,
    exclude_firstorder_minmax: bool = True,
) -> dict[str, list[tuple[str, float]]]:
    _setup_pub_style()

    # Option A: use the union of the top-N stability-weighted features and the top-N voting features
    # (after exclusions), then plot both metrics for direct comparison.
    voting_map = _read_metric_by_feature(csv_path, value_col="vote_frequency")
    weighted_map = _read_metric_by_feature(csv_path, value_col="ensemble_weighted")

    top_weighted = _top_features(
        csv_path,
        metric_col="ensemble_weighted",
        top_n=top_n,
        exclude_shape=exclude_shape,
        exclude_firstorder_minmax=exclude_firstorder_minmax,
    )
    top_voting = _top_features(
        csv_path,
        metric_col="vote_frequency",
        top_n=top_n,
        exclude_shape=exclude_shape,
        exclude_firstorder_minmax=exclude_firstorder_minmax,
    )

    # Preserve order but include voting-only features as well
    union_features: list[str] = []
    for f in top_weighted + top_voting:
        if f not in union_features:
            union_features.append(f)

    # Compute raw values and a combined normalised score (max of normalised metrics) for ordering
    w_vals_raw_all = [float(weighted_map.get(f, 0.0)) for f in union_features]
    v_vals_raw_all = [float(voting_map.get(f, 0.0)) for f in union_features]

    w_max = max(w_vals_raw_all) if w_vals_raw_all else 1.0
    v_max = max(v_vals_raw_all) if v_vals_raw_all else 1.0
    w_norm_all = [(v / w_max) if w_max > 0 else 0.0 for v in w_vals_raw_all]
    v_norm_all = [(v / v_max) if v_max > 0 else 0.0 for v in v_vals_raw_all]
    combined = [max(wn, vn) for wn, vn in zip(w_norm_all, v_norm_all)]

    # Order by combined score (desc), then stability-weighted (desc), then voting (desc)
    order = sorted(
        range(len(union_features)),
        key=lambda i: (combined[i], w_norm_all[i], v_norm_all[i]),
        reverse=True,
    )
    top_features = [union_features[i] for i in order]
    labels = [_wrap_label(f, wrap_width) for f in top_features]

    w_vals_raw = [float(weighted_map.get(f, 0.0)) for f in top_features]
    v_vals_raw = [float(voting_map.get(f, 0.0)) for f in top_features]

    # Normalise each metric to 0..1 for single-axis readability (within the union set)
    w_max2 = max(w_vals_raw) if w_vals_raw else 1.0
    v_max2 = max(v_vals_raw) if v_vals_raw else 1.0
    w_vals = [(v / w_max2) if w_max2 > 0 else 0.0 for v in w_vals_raw]
    v_vals = [(v / v_max2) if v_max2 > 0 else 0.0 for v in v_vals_raw]

    fig, ax = plt.subplots(1, 1, figsize=(12.5, 6.8))
    y = list(range(len(labels)))

    # paired horizontal bars
    offset = 0.18
    ax.barh(
        [yy - offset for yy in y],
        w_vals,
        height=0.32,
        color=weighted_color,
        alpha=0.88,
        label="Stability-weighted (normalised)",
    )
    ax.barh(
        [yy + offset for yy in y],
        v_vals,
        height=0.32,
        color=voting_color,
        alpha=0.88,
        label="Voting frequency (normalised)",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Normalised score (0–1 within modality)")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0.0)

    ax.legend(loc="lower right", frameon=True, fontsize=10)

    fig.suptitle(
        f"Union of Top {top_n} (stability-weighted) + Top {top_n} (voting)\nStability-weighted vs Voting Ensemble — {modality}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "ensemble_weighted_raw": list(zip(top_features, w_vals_raw)),
        "vote_frequency_raw": list(zip(top_features, v_vals_raw)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Top-5 ensemble_importance_weighted.png plots from precomputed CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mri", type=str, required=True, help="Path to ensemble_feature_importance_mri.csv")
    parser.add_argument("--pet", type=str, required=True, help="Path to ensemble_feature_importance_pet.csv")
    parser.add_argument("--spect", type=str, required=True, help="Path to ensemble_feature_importance_spect.csv")
    parser.add_argument(
        "--outdir",
        type=str,
        default="/Users/josephstorey/P4P_outputs/shap_top5_weighted",
        help="Output directory for the three PNGs",
    )
    parser.add_argument("--top-n", type=int, default=5, help="Number of top features to show")
    parser.add_argument(
        "--include-shape",
        action="store_true",
        help="Include shape features (default: excluded)",
    )
    parser.add_argument(
        "--include-firstorder-minmax",
        action="store_true",
        help="Include firstorder min/max features (default: excluded)",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()

    specs = [
        (
            "MRI",
            Path(args.mri).expanduser().resolve(),
            outdir / "MRI_weighted_vs_voting_overlay_union_top5_no_shape_no_minmax.png",
            "purple",
        ),
        (
            "PET",
            Path(args.pet).expanduser().resolve(),
            outdir / "PET_weighted_vs_voting_overlay_union_top5_no_shape_no_minmax.png",
            "darkgreen",
        ),
        (
            "SPECT",
            Path(args.spect).expanduser().resolve(),
            outdir / "SPECT_weighted_vs_voting_overlay_union_top5_no_shape_no_minmax.png",
            "steelblue",
        ),
    ]

    for modality, csv_path, out_path, color in specs:
        top = plot_top5_weighted_vs_voting_overlay(
            csv_path,
            modality,
            out_path,
            top_n=args.top_n,
            weighted_color=color,
            exclude_shape=not args.include_shape,
            exclude_firstorder_minmax=not args.include_firstorder_minmax,
        )
        print(f"\n[{modality}] wrote: {out_path}")
        print("  Features shown (chosen by stability-weighted top-N), with raw values:")
        for i, ((feat, wv), (_, vv)) in enumerate(
            zip(top["ensemble_weighted_raw"], top["vote_frequency_raw"]),
            start=1,
        ):
            print(f"   {i:>2d}. {feat} | ensemble_weighted={wv:.6f} | vote_frequency={vv:.6f}")


if __name__ == "__main__":
    main()

