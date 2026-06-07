"""Analysis utilities for evacuation simulation results.

Provides basic statistical metrics and density-based bottleneck detection.
"""
from __future__ import annotations
import json
from typing import Dict, Any, List, Tuple
import numpy as np


def compute_evac_metrics(res: Dict[str, Any], n_agents: int) -> Dict[str, Any]:
    """Compute basic evacuation statistics from simulation result.

    Returns a dictionary with mean/median/std, percentiles, throughput, and
    times to evacuate fraction of population.
    """
    evac_times = [t for t in res.get("evac_times", {}).values() if t is not None]
    evac_times = np.array(evac_times)
    metrics = {}
    metrics["n_agents"] = n_agents
    metrics["n_evacuated"] = int(len(evac_times))
    if len(evac_times) == 0:
        return metrics
    metrics["total_time"] = float(res.get("total_time", float(np.max(evac_times))))
    metrics["mean_time"] = float(np.mean(evac_times))
    metrics["median_time"] = float(np.median(evac_times))
    metrics["std_time"] = float(np.std(evac_times))
    metrics["p90_time"] = float(np.percentile(evac_times, 90))
    metrics["max_time"] = float(np.max(evac_times))
    metrics["throughput_avg"] = metrics["n_evacuated"] / metrics["total_time"]
    # times to evacuate fraction
    metrics["time_to_25pct"] = float(np.percentile(evac_times, 25))
    metrics["time_to_50pct"] = float(np.percentile(evac_times, 50))
    metrics["time_to_75pct"] = float(np.percentile(evac_times, 75))
    metrics["time_to_90pct"] = float(np.percentile(evac_times, 90))
    return metrics


def compute_max_density(history_positions: List[np.ndarray], area: Tuple[float, float, float, float], bins: int = 80) -> Dict[str, Any]:
    """Compute maximum per-cell density observed across timeline.

    Returns max cell count and its frame index and cell coordinates.
    """
    x1, y1, x2, y2 = area
    max_count = 0
    max_info = {"frame_index": None, "cell": None, "count": 0}
    for i, pos in enumerate(history_positions):
        if pos is None or pos.size == 0:
            continue
        H, xedges, yedges = np.histogram2d(pos[:, 0], pos[:, 1], bins=bins, range=[[x1, x2], [y1, y2]])
        current_max = int(H.max())
        if current_max > max_count:
            max_count = current_max
            idx = np.unravel_index(int(H.argmax()), H.shape)
            max_info = {"frame_index": i, "cell": idx, "count": current_max}
    return max_info


def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    def _np_encoder(obj):
        # convert numpy types to native python types for json
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_np_encoder)
