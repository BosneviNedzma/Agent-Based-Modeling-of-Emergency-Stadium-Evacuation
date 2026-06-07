"""Visualization helpers for evacuation simulation results.

Provides density heatmaps, evacuation time histograms, and
bottleneck zone plots using Matplotlib and Seaborn.
"""
from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_heatmap(
    history_positions: List[np.ndarray],
    area: Tuple[float, float, float, float],
    bins: int = 80,
    cmap: str = "magma",
    title: str = "Density heatmap (aggregated)",
) -> plt.Figure:
    """Plot an aggregated spatial density heatmap from position history.

    Stacks all position snapshots and computes a 2D histogram to show
    where agents spent the most time — highlighting bottleneck zones.

    Args:
        history_positions: List of (N, 2) position arrays per timestep.
        area: (x_min, y_min, x_max, y_max) of the arena [m].
        bins: Number of histogram bins along each axis.
        cmap: Matplotlib colormap name.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    x1, y1, x2, y2 = area
    all_pos = np.vstack(history_positions)
    valid = ~np.isnan(all_pos[:, 0])
    all_pos = all_pos[valid]

    H, xedges, yedges = np.histogram2d(
        all_pos[:, 0], all_pos[:, 1],
        bins=bins,
        range=[[x1, x2], [y1, y2]],
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    img = ax.imshow(
        H.T,
        origin="lower",
        extent=[x1, x2, y1, y2],
        aspect="auto",
        cmap=cmap,
        interpolation="gaussian",
    )
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Akumulirana gustina (broj agenata × frejmovi)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_bottleneck_heatmap(
    history_positions: List[np.ndarray],
    area: Tuple[float, float, float, float],
    n_agents: int,
    title: str = "Bottleneck heatmap",
    bins: int = 80,
) -> plt.Figure:
    
    x1, y1, x2, y2 = area
    mid_frames = []
    for frame in history_positions:
        valid = ~np.isnan(frame[:, 0])
        frac_active = np.sum(valid) / n_agents
        if 0.20 <= frac_active <= 0.80:
            mid_frames.append(frame[valid])

    if not mid_frames:
        mid_frames = [f[~np.isnan(f[:, 0])] for f in history_positions]

    all_pos = np.vstack(mid_frames)
    H, _, _ = np.histogram2d(
        all_pos[:, 0], all_pos[:, 1],
        bins=bins,
        range=[[x1, x2], [y1, y2]],
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    img = ax.imshow(
        H.T,
        origin="lower",
        extent=[x1, x2, y1, y2],
        aspect="auto",
        cmap="hot_r",
        interpolation="gaussian",
    )
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Gustina tokom kritične faze evakuacije")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_evac_times(
    evac_times: dict,
    title: str = "Distribucija vremena evakuacije agenata",
) -> plt.Figure:
    """Plot a histogram of individual agent evacuation times.

    Args:
        evac_times: Dictionary {agent_id: evacuation_time | None}.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    times = [t for t in evac_times.values() if t is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(times, bins=40, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Vrijeme evakuacije (s)")
    ax.set_ylabel("Broj agenata")
    ax.set_title(title)
    fig.tight_layout()
    return fig