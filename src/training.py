"""Training helpers for masked multi-task binary classification."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader
except Exception as exc:  # pragma: no cover
    torch = None
    F = None
    DataLoader = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def masked_bce_with_logits_loss(logits, targets, pos_weight=None):
    if targets.shape != logits.shape:
        targets = targets.view_as(logits)
    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return None
    targets_filled = torch.nan_to_num(targets, nan=0.0)
    if pos_weight is None:
        full_loss = F.binary_cross_entropy_with_logits(logits, targets_filled, reduction="none")
    else:
        full_loss = F.binary_cross_entropy_with_logits(
            logits, targets_filled, reduction="none", pos_weight=pos_weight
        )
    return full_loss[mask].mean()


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.best_state = None

    def step(self, val_loss: float, model) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, loader, optimizer, device, pos_weight=None):
    model.train()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = masked_bce_with_logits_loss(logits, batch.y, pos_weight=pos_weight)
        if loss is None:
            continue
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return np.nan if n_batches == 0 else total_loss / n_batches


@torch.no_grad()
def evaluate_loss(model, loader, device, pos_weight=None):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = masked_bce_with_logits_loss(logits, batch.y, pos_weight=pos_weight)
        if loss is None:
            continue
        total_loss += float(loss.item())
        n_batches += 1
    return np.nan if n_batches == 0 else total_loss / n_batches


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    all_targets, all_scores = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        targets = batch.y
        if targets.shape != logits.shape:
            targets = targets.view_as(logits)
        all_targets.append(targets.cpu())
        all_scores.append(torch.sigmoid(logits).cpu())
    return torch.cat(all_targets, dim=0).numpy(), torch.cat(all_scores, dim=0).numpy()
