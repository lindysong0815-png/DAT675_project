"""GCN model definitions."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool
except Exception as exc:  # pragma: no cover
    torch = None
    nn = object  # type: ignore
    F = None
    GCNConv = None
    global_mean_pool = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        if _IMPORT_ERROR is not None:  # pragma: no cover
            raise RuntimeError("PyTorch Geometric is not available") from _IMPORT_ERROR
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.lin(x)
