"""
utils/metrics.py — Shared evaluation helpers used by all layers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)
import json, time
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR


def evaluate_classifier(y_true, y_pred, y_prob=None, label="model", save=True):
    """Compute and save full classifier metrics. Returns dict of results."""
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    results = {
        "label":     label,
        "accuracy":  round(acc, 4),
        "f1_score":  round(f1, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "timestamp": datetime.utcnow().isoformat(),
    }

    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            results["roc_auc"] = round(auc, 4)
        except Exception:
            pass

    print(f"\n{'='*50}")
    print(f"Results — {label}")
    print(f"{'='*50}")
    for k, v in results.items():
        if k not in ("label", "timestamp"):
            print(f"  {k:15s}: {v}")
    print(classification_report(y_true, y_pred, zero_division=0))

    if save:
        out = RESULTS_DIR / f"{label}_metrics.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

    return results


def plot_confusion_matrix(y_true, y_pred, class_names, label="model", save=True):
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {label}", fontsize=14)
    plt.tight_layout()
    if save:
        out = RESULTS_DIR / f"{label}_confusion_matrix.png"
        fig.savefig(out, dpi=150)
        print(f"  Saved confusion matrix → {out}")
    plt.close()
    return fig


def plot_comparison_bar(results_list, metric="f1_score", title="Model Comparison", save=True):
    """
    Plot a bar chart comparing multiple models.
    results_list: list of dicts with 'label' and metric keys.
    """
    labels  = [r["label"] for r in results_list]
    values  = [r.get(metric, 0) for r in results_list]
    colours = ["#2563EB", "#16A34A", "#D97706", "#DC2626", "#7C3AED"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colours[:len(labels)], edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save:
        out = RESULTS_DIR / f"{title.replace(' ', '_').lower()}.png"
        fig.savefig(out, dpi=150)
        print(f"  Saved comparison chart → {out}")
    plt.close()
    return fig


def save_results_csv(results_list, filename):
    """Save a list of result dicts to CSV."""
    df = pd.DataFrame(results_list)
    out = RESULTS_DIR / filename
    df.to_csv(out, index=False)
    print(f"  Saved results table → {out}")
    return df


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, name=""):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"  {self.name} took {self.elapsed:.2f}s")
    @property
    def minutes(self):
        return self.elapsed / 60
