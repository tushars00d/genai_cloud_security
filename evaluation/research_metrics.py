"""Research-grade metrics and plots for multiclass IDS experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


def save_per_class_report(y_true, y_pred, class_names, out_csv: Path) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=[str(c) for c in class_names],
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for cls in class_names:
        vals = report.get(str(cls), {})
        rows.append({
            "class": str(cls),
            "precision": vals.get("precision", 0.0),
            "recall": vals.get("recall", 0.0),
            "f1_score": vals.get("f1-score", 0.0),
            "support": vals.get("support", 0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def save_confusion_matrix(y_true, y_pred, class_names, out_png: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Improved IDS Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def save_multiclass_roc(y_true, y_prob, class_names, out_png: Path, out_json: Path):
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    roc_auc = {}
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, cls in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[str(cls)] = float(auc(fpr, tpr))
        if i < 8:
            ax.plot(fpr, tpr, lw=1.2, label=f"{cls} AUC={roc_auc[str(cls)]:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    out_json.write_text(json.dumps(roc_auc, indent=2))
    return roc_auc
