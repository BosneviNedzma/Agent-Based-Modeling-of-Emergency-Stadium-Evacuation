"""Visualization helpers: heatmap of density and evacuation plots."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


def plot_heatmap(history_positions: List[np.ndarray], area: tuple, bins: int = 80, cmap: str = "magma"):
    # stack positions over last frames for density
    all_pos = np.vstack(history_positions)
    x1, y1, x2, y2 = area
    H, xedges, yedges = np.histogram2d(all_pos[:,0], all_pos[:,1], bins=bins, range=[[x1,x2],[y1,y2]])
    H = H.T  # transpose for correct orientation
    plt.figure(figsize=(6,5))
    sns.heatmap(H, cmap=cmap)
    plt.title('Density heatmap (aggregated)')
    plt.tight_layout()


def plot_evac_times(evac_times: dict):
    times = [t for t in evac_times.values() if t is not None]
    plt.figure()
    plt.hist(times, bins=30)
    plt.xlabel('Evacuation time (s)')
    plt.ylabel('Count')
    plt.title('Distribution of agent evacuation times')
    plt.tight_layout()

