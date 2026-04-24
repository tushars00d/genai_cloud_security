"""Imbalance handling utilities that avoid heavyweight dependencies."""

from __future__ import annotations

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def random_oversample(X, y, max_per_class: int, random_state: int = 42):
    """Oversample minority classes up to a capped class count."""
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    target = min(max(int(counts.max()), int(np.median(counts) * 2)), max_per_class)
    xs, ys = [X], [y]
    for cls, count in zip(classes, counts):
        if count >= target:
            continue
        idx = np.flatnonzero(y == cls)
        sampled = rng.choice(idx, size=target - count, replace=True)
        xs.append(X[sampled])
        ys.append(y[sampled])
    return np.vstack(xs).astype(np.float32), np.concatenate(ys).astype(y.dtype)


def balanced_class_weights(y, num_classes: int):
    classes = np.arange(num_classes)
    present = np.unique(y)
    weights = np.ones(num_classes, dtype=np.float32)
    computed = compute_class_weight(class_weight="balanced", classes=present, y=y)
    for cls, weight in zip(present, computed):
        weights[int(cls)] = weight
    return weights
