"""Evaluation utilities."""

from __future__ import annotations

import random

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def safe_auprc(y_true, y_score):
    return np.nan if len(np.unique(y_true)) < 2 else average_precision_score(y_true, y_score)


def safe_auroc(y_true, y_score):
    return np.nan if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_score)
