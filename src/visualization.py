"""Visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_publication_style(ax, grid_axis: str = "y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, linestyle="--", alpha=0.25)
    return ax


def save_current_figure(path: str, dpi: int = 300):
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
