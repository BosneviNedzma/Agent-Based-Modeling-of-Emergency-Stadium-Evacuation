from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np


def compute_evac_metrics(
    res: Dict[str, Any],
    n_agents: int,
) -> Dict[str, Any]:
    
    evac_times = np.array([
        t for t in res.get("evac_times", {}).values() if t is not None
    ])

    metrics: Dict[str, Any] = {"n_agents": n_agents}
    metrics["n_evacuated"] = int(len(evac_times))

    if len(evac_times) == 0:
        return metrics

    total_time = float(res.get("total_time", np.max(evac_times)))
    metrics["total_time"] = total_time
    metrics["mean_time"] = float(np.mean(evac_times))
    metrics["median_time"] = float(np.median(evac_times))
    metrics["std_time"] = float(np.std(evac_times, ddof=1))
    metrics["p90_time"] = float(np.percentile(evac_times, 90))
    metrics["max_time"] = float(np.max(evac_times))
    metrics["throughput_avg"] = float(
        metrics["n_evacuated"] / total_time
    ) if total_time > 0 else 0.0
    metrics["time_to_25pct"] = float(np.percentile(evac_times, 25))
    metrics["time_to_50pct"] = float(np.percentile(evac_times, 50))
    metrics["time_to_75pct"] = float(np.percentile(evac_times, 75))
    metrics["time_to_90pct"] = float(np.percentile(evac_times, 90))
    return metrics


def compute_max_density(
    history_positions: List[np.ndarray],
    area: Tuple[float, float, float, float],
    bins: int = 80,
) -> Dict[str, Any]:
    
    x1, y1, x2, y2 = area
    cell_area = ((x2 - x1) / bins) * ((y2 - y1) / bins)  # m^2

    max_count = 0
    max_info: Dict[str, Any] = {
        "frame_index": None,
        "cell": None,
        "count": 0,
        "density_m2": 0.0,
    }

    for i, pos in enumerate(history_positions):
        if pos is None or pos.size == 0:
            continue
        valid = ~np.isnan(pos[:, 0])
        if not np.any(valid):
            continue

        H, _, _ = np.histogram2d(
            pos[valid, 0],
            pos[valid, 1],
            bins=bins,
            range=[[x1, x2], [y1, y2]],
        )
        current_max = int(H.max())
        if current_max > max_count:
            max_count = current_max
            idx = np.unravel_index(int(H.argmax()), H.shape)
            max_info = {
                "frame_index": i,
                "cell": list(idx),
                "count": current_max,
                "density_m2": round(current_max / cell_area, 2),
            }

    return max_info


def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    
    def _encoder(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not JSON serialisable")

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=_encoder)
    except OSError as exc:
        raise OSError(
            f"Nije moguće snimiti metriku u '{path}': {exc}"
        ) from exc