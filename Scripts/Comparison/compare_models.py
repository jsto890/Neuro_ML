import argparse
import json
import os
import sys
import glob
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

from scipy.stats import wilcoxon, friedmanchisquare, chi2, rankdata

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


plt.style.use("default")
sns.set_palette("husl")


@dataclass
class ModelFoldMetrics:
    model: str
    fold: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    mcc: float


@dataclass
class ModelPredictions:
    model: str
    subject_ids: List[str]
    y_true: np.ndarray
    y_prob_pos: np.ndarray
    y_pred: np.ndarray
    is_binary: bool


def expand_path(p: str) -> str:
    if p is None:
        return p
    return os.path.abspath(os.path.expanduser(p))


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def discover_model_dirs(run_dir: str) -> Dict[str, str]:
    """Find model subdirectories that contain a run summary JSON.

    Returns mapping: model_name -> model_dir
    """
    model_dirs: Dict[str, str] = {}
    if not os.path.isdir(run_dir):
        return model_dirs
    for entry in sorted(os.listdir(run_dir)):
        mdir = os.path.join(run_dir, entry)
        if not os.path.isdir(mdir):
            continue
        summary = os.path.join(mdir, f"{entry}_run_summary.json")
        # Some models might save with different casing; be lenient
        if not os.path.exists(summary):
            # Try to find any *_run_summary.json in the folder
            matches = glob.glob(os.path.join(mdir, "*_run_summary.json"))
            # If no summary, still allow if we can find test_evaluation_plots_fold_* metrics
            if matches:
                summary = matches[0]
            else:
                metrics_glob = glob.glob(os.path.join(mdir, "test_evaluation_plots_fold_*", "evaluation_metrics.json"))
                if not metrics_glob:
                    continue
        model_name = os.path.basename(mdir)
        model_dirs[model_name] = mdir
    return model_dirs


def load_per_fold_test_metrics(model_dir: str) -> List[ModelFoldMetrics]:
    """Load per-fold test metrics using paths from the model's run summary JSON.

    Expects each fold result to provide 'test_metrics_path'.
    """
    candidates = glob.glob(os.path.join(model_dir, "*_run_summary.json"))
    results: List[ModelFoldMetrics] = []
    model_name = os.path.basename(model_dir)

    # Preferred: read from run summary to get explicit test_metrics_path
    if candidates:
        try:
            with open(candidates[0], "r") as f:
                summary = json.load(f)
            for fr in summary.get("fold_results", []):
                metrics_path = fr.get("test_metrics_path")
                if not metrics_path or not os.path.exists(metrics_path):
                    continue
                with open(metrics_path, "r") as mf:
                    m = json.load(mf)
                results.append(
                    ModelFoldMetrics(
                        model=model_name,
                        fold=int(fr.get("fold", len(results) + 1)),
                        accuracy=float(m.get("accuracy", np.nan)),
                        precision=float(m.get("precision", np.nan)),
                        recall=float(m.get("recall", np.nan)),
                        f1=float(m.get("f1_score", np.nan)),
                        auc=float(m.get("auc", np.nan)),
                        mcc=float(m.get("mcc", np.nan)),
                    )
                )
        except Exception:
            results = []

    # Fallback: glob test_evaluation_plots_fold_*/evaluation_metrics.json
    if not results:
        import re
        pattern = os.path.join(model_dir, "test_evaluation_plots_fold_*", "evaluation_metrics.json")
        for mp in sorted(glob.glob(pattern)):
            try:
                with open(mp, "r") as mf:
                    m = json.load(mf)
                # extract fold number from parent dir name
                parent = os.path.basename(os.path.dirname(mp))
                m_fold = None
                m_fold_match = re.search(r"fold_(\d+)", parent)
                if m_fold_match:
                    m_fold = int(m_fold_match.group(1))
                else:
                    # fallback incremental
                    m_fold = len(results) + 1
                results.append(
                    ModelFoldMetrics(
                        model=model_name,
                        fold=m_fold,
                        accuracy=float(m.get("accuracy", np.nan)),
                        precision=float(m.get("precision", np.nan)),
                        recall=float(m.get("recall", np.nan)),
                        f1=float(m.get("f1_score", np.nan)),
                        auc=float(m.get("auc", np.nan)),
                        mcc=float(m.get("mcc", np.nan)),
                    )
                )
            except Exception:
                continue
    return results


def aggregate_fold_metrics(all_metrics: List[ModelFoldMetrics]) -> pd.DataFrame:
    if not all_metrics:
        return pd.DataFrame(columns=["model", "fold", "auc", "mcc", "f1", "acc", "precision", "recall"])
    rows = []
    for r in all_metrics:
        rows.append({
            "model": r.model,
            "fold": r.fold,
            "auc": r.auc,
            "mcc": r.mcc,
            "f1": r.f1,
            "acc": r.accuracy,
            "precision": r.precision,
            "recall": r.recall,
        })
    df = pd.DataFrame(rows)
    return df


def compute_ci_from_folds(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "std": np.nan}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    z = 1.96
    return {
        "mean": mean,
        "std": std,
        "ci_lower": mean - z * se,
        "ci_upper": mean + z * se,
    }


def compute_ece(y_true: np.ndarray, y_prob_pos: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(y_prob_pos, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = inds == b
        if not np.any(mask):
            continue
        conf = np.mean(y_prob_pos[mask])
        acc = np.mean((y_prob_pos[mask] >= 0.5) == (y_true[mask] == 1))
        ece += (np.sum(mask) / len(y_prob_pos)) * abs(acc - conf)
    return float(ece)


def load_predictions_from_dir(pred_dir: str, model_name: Optional[str] = None) -> Optional[ModelPredictions]:
    """Load predictions from a directory that contains predictions.csv.

    Expected CSV columns (best-effort inference):
    - subject_id (or id)
    - true_label (or label, y_true)
    - predicted_label (or prediction, y_pred)
    - positive-class probability (e.g., pd_probability, positive_probability, prob1, probability_1)
    """
    csv_path = os.path.join(pred_dir, "predictions.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        # infer required columns
        sid_col = None
        for c in ["subject_id", "id", "subject", "sid"]:
            if c in df.columns:
                sid_col = c
                break
        ytrue_col = None
        for c in ["true_label", "label", "y_true", "y"]:
            if c in df.columns:
                ytrue_col = c
                break
        ypred_col = None
        for c in ["predicted_label", "prediction", "y_pred", "pred"]:
            if c in df.columns:
                ypred_col = c
                break
        prob_col = None
        for c in [
            "pd_probability",
            "positive_probability",
            "prob_pos",
            "pos_prob",
            "probability_1",
            "prob1",
            "p1",
        ]:
            if c in df.columns:
                prob_col = c
                break
        # Special case: cn/pd pair
        if prob_col is None and {"cn_probability", "pd_probability"}.issubset(df.columns):
            prob_col = "pd_probability"

        if sid_col and ytrue_col and ypred_col and prob_col:
            y_prob_pos = df[prob_col].astype(float).values
            y_true = df[ytrue_col].astype(int).values
            y_pred = df[ypred_col].astype(int).values
            model = model_name or os.path.basename(os.path.normpath(pred_dir))
            is_binary = len(np.unique(y_true)) == 2
            return ModelPredictions(
                model=model,
                subject_ids=df[sid_col].astype(str).tolist(),
                y_true=y_true,
                y_prob_pos=y_prob_pos,
                y_pred=y_pred,
                is_binary=is_binary,
            )
    except Exception:
        return None
    return None


# DeLong implementation (paired AUC test). Simplified for binary classification.
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    result = np.empty(n, dtype=float)
    result[order] = midranks
    return result


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> Tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
    tz = np.empty([k, m + n])
    for r in range(k):
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v10)
    sy = np.cov(v01)
    s = sx / m + sy / n
    return aucs, s


def delong_roc_variance(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    order = np.argsort(-y_scores)
    y_true_sorted = y_true[order]
    y_scores_sorted = y_scores[order]
    label_1_count = np.sum(y_true_sorted)
    if label_1_count == 0 or label_1_count == len(y_true_sorted):
        return float("nan"), float("nan")
    preds = np.vstack((y_scores_sorted,))
    aucs, covariance = _fast_delong(preds, int(label_1_count))
    auc = float(aucs[0])
    var = float(covariance[0, 0])
    return auc, var


def delong_test(y_true: np.ndarray, y_scores_a: np.ndarray, y_scores_b: np.ndarray) -> Tuple[float, float, float]:
    auc_a, var_a = delong_roc_variance(y_true, y_scores_a)
    auc_b, var_b = delong_roc_variance(y_true, y_scores_b)
    if any(map(lambda v: (v is None) or np.isnan(v), [auc_a, var_a, auc_b, var_b])):
        return float("nan"), float("nan"), float("nan")
    cov_ab = 0.0  # Approximate independence if covariance unavailable
    diff = auc_a - auc_b
    var = var_a + var_b - 2 * cov_ab
    if var <= 0:
        return diff, float("inf"), 0.0
    z = diff / math.sqrt(var)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return diff, z, p


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Tuple[int, int, float]:
    n01 = int(np.sum((y_pred_a == y_true) & (y_pred_b != y_true)))
    n10 = int(np.sum((y_pred_a != y_true) & (y_pred_b == y_true)))
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10 + 1e-12)
    p = 1 - chi2.cdf(stat, df=1)
    return n01, n10, float(p)


def bowker_test(conf_matrix: np.ndarray) -> float:
    k = conf_matrix.shape[0]
    stat = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            nij = conf_matrix[i, j]
            nji = conf_matrix[j, i]
            if nij + nji > 0:
                stat += (nij - nji) ** 2 / (nij + nji)
    p = 1 - chi2.cdf(stat, df=k * (k - 1) / 2)
    return float(p)


def adjust_pvalues(pvals: List[float], method: str = "holm") -> List[float]:
    if HAS_STATSMODELS:
        res = multipletests(pvals, method=("holm" if method == "holm" else "fdr_bh"))
        return list(res[1])
    # Fallback simple Holm-Bonferroni
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    for i, idx in enumerate(order):
        adjusted[idx] = min(1.0, pvals[idx] * (m - i))
    return list(adjusted)


def compute_rank_table(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    models = sorted(df["model"].unique())
    data = {}
    for metric in metrics:
        agg = df.groupby("model")[metric].mean()
        ranks = rankdata(-agg.values, method="average")  # higher is better
        data[metric] = ranks
    rank_df = pd.DataFrame(data, index=models)
    return rank_df


def plot_bars_with_ci(df: pd.DataFrame, metric: str, out_path: str, title: str) -> None:
    stats_rows = []
    for model, g in df.groupby("model"):
        ci = compute_ci_from_folds(g[metric].values)
        stats_rows.append({"model": model, **ci})
    stats = pd.DataFrame(stats_rows).sort_values("mean", ascending=False)
    x = np.arange(len(stats))
    means = stats["mean"].values
    yerr = np.vstack((means - stats["ci_lower"].values, stats["ci_upper"].values - means))
    plt.figure(figsize=(10, 6))
    plt.bar(x, means, yerr=yerr, capsize=4, color="skyblue", edgecolor="black")
    plt.xticks(x, stats["model"].tolist(), rotation=20)
    plt.ylabel(metric.upper())
    plt.ylim(0, 1)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_violin(df: pd.DataFrame, metric: str, out_path: str, title: str) -> None:
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="model", y=metric, inner="box")
    plt.xticks(rotation=20)
    plt.ylim(0, 1)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_paired_delta(df: pd.DataFrame, metric: str, baseline: str, out_path: str, title: str) -> None:
    pivot = df.pivot_table(index="fold", columns="model", values=metric)
    if baseline not in pivot.columns:
        return
    base_vals = pivot[baseline]
    deltas = pivot.subtract(base_vals, axis=0).drop(columns=[baseline], errors="ignore")
    deltas = deltas.melt(ignore_index=False, var_name="model", value_name="delta").reset_index()
    plt.figure(figsize=(10, 6))
    sns.stripplot(data=deltas, x="model", y="delta", jitter=0.1)
    sns.pointplot(data=deltas, x="model", y="delta", estimator=np.mean, errorbar=("ci", 95), join=False, color="red")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xticks(rotation=20)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_rank_heatmap(rank_df: pd.DataFrame, out_path: str, title: str) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(rank_df, annot=True, cmap="YlGnBu", cbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_roc_pr_overlays(pred_list: List[ModelPredictions], out_dir: str) -> None:
    roc_path = os.path.join(out_dir, "roc_overlay.png")
    pr_path = os.path.join(out_dir, "pr_overlay.png")
    # ROC
    plt.figure(figsize=(8, 6))
    any_curve = False
    for s in pred_list:
        if not s.is_binary:
            continue
        fpr, tpr, _ = roc_curve(s.y_true, s.y_prob_pos)
        auc = roc_auc_score(s.y_true, s.y_prob_pos)
        plt.plot(fpr, tpr, lw=2, label=f"{s.model} (AUC={auc:.3f})")
        any_curve = True
    if any_curve:
        plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves (predictions provided)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(roc_path, dpi=200, bbox_inches="tight")
        plt.close()
    # PR
    plt.figure(figsize=(8, 6))
    any_curve = False
    for s in pred_list:
        if not s.is_binary:
            continue
        precision, recall, _ = precision_recall_curve(s.y_true, s.y_prob_pos)
        ap = average_precision_score(s.y_true, s.y_prob_pos)
        plt.plot(recall, precision, lw=2, label=f"{s.model} (AP={ap:.3f})")
        any_curve = True
    if any_curve:
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves (predictions provided)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(pr_path, dpi=200, bbox_inches="tight")
        plt.close()


def plot_calibration(pred_list: List[ModelPredictions], out_dir: str) -> None:
    calib_path = os.path.join(out_dir, "calibration_reliability.png")
    plt.figure(figsize=(8, 6))
    any_curve = False
    for s in pred_list:
        if not s.is_binary:
            continue
        frac_pos, mean_pred = calibration_curve(s.y_true, s.y_prob_pos, n_bins=10, strategy="uniform")
        plt.plot(mean_pred, frac_pos, marker="o", lw=2, label=f"{s.model}")
        any_curve = True
    if any_curve:
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Reliability Diagram (predictions provided)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(calib_path, dpi=200, bbox_inches="tight")
        plt.close()


def paired_tests_and_tables(df: pd.DataFrame, metrics: List[str], baseline: Optional[str], fdr_method: str, out_dir: str) -> None:
    models = sorted(df["model"].unique())
    pairs = []
    pvals = {m: [] for m in metrics}
    records = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            joined = df[df["model"].isin([m1, m2])].pivot_table(index="fold", columns="model", values=metrics)
            joined = joined.dropna()
            if joined.empty:
                continue
            rec = {"model_a": m1, "model_b": m2}
            for metric in metrics:
                if (m1 not in joined[metric].columns) or (m2 not in joined[metric].columns):
                    rec[f"p_{metric}"] = np.nan
                    continue
                a = joined[metric][m1].values
                b = joined[metric][m2].values
                try:
                    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                except Exception:
                    p = np.nan
                rec[f"p_{metric}"] = p
                pvals[metric].append(p if not np.isnan(p) else 1.0)
            records.append(rec)
            pairs.append((m1, m2))
    if not records:
        return
    p_df = pd.DataFrame(records)
    for metric in metrics:
        if len(pvals[metric]) == 0:
            continue
        adj = adjust_pvalues(pvals[metric], method=fdr_method)
        col = f"p_{metric}"
        # Assign adjusted in order they were collected
        idxs = [k for k in range(len(p_df)) if not np.isnan(p_df.loc[k, col])]
        for idx, ap in zip(idxs, adj):
            p_df.loc[idx, f"p_{metric}_adj_{fdr_method}"] = ap
    ensure_dir(out_dir)
    p_df.to_csv(os.path.join(out_dir, "paired_wilcoxon_pvalues.csv"), index=False)

    # Friedman across models (for each metric)
    fr_rows = []
    for metric in metrics:
        pivot = df.pivot_table(index="fold", columns="model", values=metric)
        pivot = pivot.dropna(axis=0)
        if pivot.shape[1] < 3:
            continue
        try:
            stat, p = friedmanchisquare(*[pivot[c].values for c in pivot.columns])
        except Exception:
            stat, p = np.nan, np.nan
        fr_rows.append({"metric": metric, "friedman_stat": stat, "friedman_p": p})
    if fr_rows:
        pd.DataFrame(fr_rows).to_csv(os.path.join(out_dir, "friedman_tests.csv"), index=False)

    # Paired delta plots vs baseline
    if baseline and baseline in models:
        for metric in metrics:
            plot_paired_delta(df, metric, baseline, os.path.join(out_dir, f"paired_delta_{metric}.png"), f"Per-fold {metric.upper()} delta vs {baseline}")


def predictions_pairwise_stats(pred_list: List[ModelPredictions], out_dir: str, fdr_method: str) -> None:
    if len(pred_list) < 2:
        return
    rows_delong = []
    rows_mcnemar = []
    for i in range(len(pred_list)):
        for j in range(i + 1, len(pred_list)):
            a = pred_list[i]
            b = pred_list[j]
            # Align subjects by ID if possible
            df = pd.DataFrame({
                "sid": a.subject_ids,
                "y_a": a.y_true,
                "p_a": a.y_prob_pos,
                "pred_a": a.y_pred,
            })
            df_b = pd.DataFrame({
                "sid": b.subject_ids,
                "y_b": b.y_true,
                "p_b": b.y_prob_pos,
                "pred_b": b.y_pred,
            })
            merged = df.merge(df_b, on="sid", how="inner")
            if merged.empty:
                continue
            y = merged["y_a"].values
            if not np.array_equal(merged["y_a"].values, merged["y_b"].values):
                # If labels mismatch after merge, skip
                continue
            # DeLong AUC diff
            try:
                diff, z, p = delong_test(y, merged["p_a"].values, merged["p_b"].values)
            except Exception:
                diff, z, p = np.nan, np.nan, np.nan
            rows_delong.append({"model_a": a.model, "model_b": b.model, "auc_diff": diff, "z": z, "p": p})
            # McNemar (binary)
            try:
                n01, n10, p_m = mcnemar_test(y, merged["pred_a"].values, merged["pred_b"].values)
            except Exception:
                n01, n10, p_m = 0, 0, np.nan
            rows_mcnemar.append({"model_a": a.model, "model_b": b.model, "n01": n01, "n10": n10, "p": p_m})

    if rows_delong:
        df_delong = pd.DataFrame(rows_delong)
        df_delong["p_adj_{}".format(fdr_method)] = adjust_pvalues(df_delong["p"].fillna(1.0).tolist(), method=fdr_method)
        df_delong.to_csv(os.path.join(out_dir, "predictions_delong_auc_tests.csv"), index=False)
    if rows_mcnemar:
        df_m = pd.DataFrame(rows_mcnemar)
        df_m["p_adj_{}".format(fdr_method)] = adjust_pvalues(df_m["p"].fillna(1.0).tolist(), method=fdr_method)
        df_m.to_csv(os.path.join(out_dir, "predictions_mcnemar_tests.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Compare MRI/PET/SPECT (or any) models from run directories and optional predictions.")
    parser.add_argument("--run-dirs", nargs="*", default=[], help="Paths to run directories containing per-model subfolders and *_run_summary.json files (MRI/PET/SPECT).")
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of model names to include.")
    parser.add_argument("--pred-dirs", nargs="*", default=[], help="Optional directories with predictions.csv to enable ROC/PR/calibration and paired tests (any modality).")
    parser.add_argument("--spect-dirs", nargs="*", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory. Defaults to ~/P4P_results/model_comparison/<timestamp>.")
    parser.add_argument("--baseline", default="Simple3DCNN", help="Baseline model name for paired delta plots.")
    parser.add_argument("--fdr-method", choices=["holm", "bh"], default="holm", help="Multiple comparison correction method.")
    args = parser.parse_args()

    run_dirs = [expand_path(p) for p in args.run_dirs]
    run_dirs = [p for p in run_dirs if p and os.path.isdir(p)]

    out_dir = expand_path(args.output_dir) if args.output_dir else os.path.join(os.path.expanduser("~"), "P4P_results", "model_comparison", timestamp_str())
    plots_dir = os.path.join(out_dir, "plots")
    stats_dir = os.path.join(out_dir, "stats")
    tables_dir = os.path.join(out_dir, "tables")
    raw_dir = os.path.join(out_dir, "raw")
    for d in [out_dir, plots_dir, stats_dir, tables_dir, raw_dir]:
        ensure_dir(d)

    # Discover and load per-fold test metrics for MRI/PET models
    all_fold_metrics: List[ModelFoldMetrics] = []
    discovered_models = {}
    for rdir in run_dirs:
        model_dirs = discover_model_dirs(rdir)
        for mname, mdir in model_dirs.items():
            if args.models and mname not in args.models:
                continue
            if (mname, mdir) in discovered_models.items():
                continue
            fold_metrics = load_per_fold_test_metrics(mdir)
            if fold_metrics:
                all_fold_metrics.extend(fold_metrics)
                discovered_models[mname] = mdir

    df_folds = aggregate_fold_metrics(all_fold_metrics)
    if not df_folds.empty:
        df_folds.sort_values(["model", "fold"]).to_csv(os.path.join(raw_dir, "per_fold_test_metrics.csv"), index=False)

    # Per-model CIs across folds (MRI/PET per-fold stats)
    if not df_folds.empty:
        summary_rows = []
        for model, g in df_folds.groupby("model"):
            for metric in ["auc", "mcc", "f1", "acc", "precision", "recall"]:
                ci = compute_ci_from_folds(g[metric].values)
                summary_rows.append({"model": model, "metric": metric, **ci})
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(os.path.join(tables_dir, "fold_metric_summary_ci.csv"), index=False)

        # Plots for bars and violins
        for metric in ["auc", "mcc", "f1", "acc"]:
            plot_bars_with_ci(df_folds, metric, os.path.join(plots_dir, f"bars_{metric}.png"), f"Test {metric.upper()} (mean ± 95% CI)")
            plot_violin(df_folds, metric, os.path.join(plots_dir, f"violin_{metric}.png"), f"Per-fold {metric.upper()} distribution")

        # Pairwise tests and delta plots
        paired_tests_and_tables(df_folds, ["auc", "mcc"], args.baseline, args.fdr_method, stats_dir)

        # Rank heatmap across key metrics
        rank_df = compute_rank_table(df_folds, ["auc", "mcc", "f1", "acc"]) if not df_folds.empty else pd.DataFrame()
        if not rank_df.empty:
            rank_df.to_csv(os.path.join(tables_dir, "rank_table.csv"))
            plot_rank_heatmap(rank_df, os.path.join(plots_dir, "rank_heatmap.png"), "Average Ranks across Metrics")

    # Predictions (any modality): ROC/PR overlays, calibration, DeLong/McNemar
    pred_dirs: List[str] = []
    if args.pred_dirs:
        pred_dirs.extend([expand_path(p) for p in args.pred_dirs])
    if args.spect_dirs:
        pred_dirs.extend([expand_path(p) for p in args.spect_dirs])
    pred_dirs = [p for p in pred_dirs if p and os.path.isdir(p)]

    pred_list: List[ModelPredictions] = []
    for pdir in pred_dirs:
        s = load_predictions_from_dir(pdir)
        if s is not None:
            pred_list.append(s)
    if pred_list:
        # Save per-model SPECT summary
        rows = []
        for s in pred_list:
            auc = roc_auc_score(s.y_true, s.y_prob_pos) if s.is_binary else np.nan
            ap = average_precision_score(s.y_true, s.y_prob_pos) if s.is_binary else np.nan
            acc = accuracy_score(s.y_true, s.y_pred)
            f1 = f1_score(s.y_true, s.y_pred, average="binary" if s.is_binary else "macro")
            mcc = 0.0
            try:
                from sklearn.metrics import matthews_corrcoef
                mcc = float(matthews_corrcoef(s.y_true, s.y_pred))
            except Exception:
                mcc = float("nan")
            brier = brier_score_loss(s.y_true, s.y_prob_pos) if s.is_binary else np.nan
            ece = compute_ece(s.y_true, s.y_prob_pos) if s.is_binary else np.nan
            rows.append({
                "model": s.model,
                "auc": auc,
                "ap": ap,
                "acc": acc,
                "f1": f1,
                "mcc": mcc,
                "brier": brier,
                "ece": ece,
            })
        pd.DataFrame(rows).to_csv(os.path.join(tables_dir, "predictions_summary.csv"), index=False)

        # Plots
        plot_roc_pr_overlays(pred_list, plots_dir)
        plot_calibration(pred_list, plots_dir)

        # Pairwise statistical tests for SPECT
        predictions_pairwise_stats(pred_list, stats_dir, args.fdr_method)

    print(f"\nComparison completed. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()


