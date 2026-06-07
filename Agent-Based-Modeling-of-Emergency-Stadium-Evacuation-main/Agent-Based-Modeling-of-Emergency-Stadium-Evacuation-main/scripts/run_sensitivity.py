from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from src.sim.analysis import save_metrics
from src.sim.scenarios import (
    ARENA_H,
    ARENA_W,
    WARMUP_FRACTION,
    build_scenario_a,
    build_scenario_b,
    run_single_replicate,
    RANDOM_STATE,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_REPS: int = 5
OUTPUT_DIR: str = "outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mean_evac_time(exits: list, n_reps: int) -> tuple[float, float]:
    """Run n_reps replications and return mean and std of total evacuation time.

    Args:
        exits: Exit configuration.
        n_reps: Number of independent replications.

    Returns:
        Tuple of (mean_time, std_time) in seconds.
    """
    times = [
        run_single_replicate(exits, seed=RANDOM_STATE + i)["total_time"]
        for i in range(n_reps)
    ]
    return float(np.mean(times)), float(np.std(times, ddof=1))


def plot_bottleneck_heatmap(
    positions: list,
    area: tuple,
    title: str,
    filepath: str,
) -> None:
    """Plot a density heatmap highlighting bottleneck zones near exits.

    Uses only frames where occupancy is between 20% and 80% of total
    agents — the period when crowding at exits is most intense.

    Args:
        positions: List of (N, 2) position arrays (post warm-up).
        area: (x_min, y_min, x_max, y_max) of the arena [m].
        title: Plot title.
        filepath: Output file path.
    """
    x_min, y_min, x_max, y_max = area
    bins_x, bins_y = 80, 60

    n_agents = positions[0].shape[0] if positions else 1
    mid_frames = [
        frame
        for frame in positions
        if 0.20 <= np.sum(~np.isnan(frame[:, 0])) / n_agents <= 0.80
    ]
    if not mid_frames:
        mid_frames = positions  # fallback if phase not captured

    all_pos = np.vstack([
        frame[~np.isnan(frame[:, 0])] for frame in mid_frames
    ])

    heatmap, _, _ = np.histogram2d(
        all_pos[:, 0],
        all_pos[:, 1],
        bins=[bins_x, bins_y],
        range=[[x_min, x_max], [y_min, y_max]],
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    img = ax.imshow(
        heatmap.T,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        aspect="auto",
        cmap="hot_r",
        interpolation="gaussian",
    )
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Akumulirana gustina (broj agenata × frejmovi)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Heatmap spremljen: {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run sensitivity sweep and generate all output files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    widths = np.arange(3.0, 9.5, 1.0)  # 3 m to 9 m, step 1 m
    results: dict = {"2_exits": [], "4_exits": []}

    print("=" * 60)
    print("Senzitivna analiza: širina izlaza")
    print("=" * 60)

    for w in widths:
        print(f"\nŠirina izlaza = {w:.1f} m")

        exits_2 = build_scenario_a(exit_width=w)
        m2, s2 = mean_evac_time(exits_2, N_REPS)
        results["2_exits"].append({"width": float(w), "mean": m2, "std": s2})
        print(f"  2 izlaza: {m2:.1f}s ± {s2:.1f}s")

        exits_4 = build_scenario_b(exit_width=w)
        m4, s4 = mean_evac_time(exits_4, N_REPS)
        results["4_exits"].append({"width": float(w), "mean": m4, "std": s4})
        print(f"  4 izlaza: {m4:.1f}s ± {s4:.1f}s")

    save_metrics(results, os.path.join(OUTPUT_DIR, "sensitivity_results.json"))

    # ------------------------------------------------------------------
    # Plot: sensitivity curve
    # ------------------------------------------------------------------
    w_arr = np.array([r["width"] for r in results["2_exits"]])
    m2_arr = np.array([r["mean"] for r in results["2_exits"]])
    s2_arr = np.array([r["std"] for r in results["2_exits"]])
    m4_arr = np.array([r["mean"] for r in results["4_exits"]])
    s4_arr = np.array([r["std"] for r in results["4_exits"]])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(w_arr, m2_arr, "o-", label="2 široka izlaza", color="steelblue")
    ax.fill_between(w_arr, m2_arr - s2_arr, m2_arr + s2_arr, alpha=0.2, color="steelblue")
    ax.plot(w_arr, m4_arr, "s-", label="4 uska izlaza", color="darkorange")
    ax.fill_between(w_arr, m4_arr - s4_arr, m4_arr + s4_arr, alpha=0.2, color="darkorange")
    ax.set_xlabel("Širina svakog izlaza (m)")
    ax.set_ylabel("Srednje vrijeme evakuacije (s)")
    ax.set_title("Senzitivna analiza — uticaj širine izlaza na evakuacijsko vrijeme")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sensitivity_width.png"), dpi=150)
    plt.close()
    print("\nSenzitivna kriva: outputs/sensitivity_width.png")

    # ------------------------------------------------------------------
    # Bottleneck heatmaps
    # ------------------------------------------------------------------
    print("\nGeneriranje bottleneck heatmapa...")
    area = (0.0, 0.0, ARENA_W, ARENA_H)

    res_a = run_single_replicate(build_scenario_a(exit_width=6.0), seed=RANDOM_STATE)
    plot_bottleneck_heatmap(
        res_a["positions"],
        area=area,
        title="Heatmap uskih grla — Scenarij A (2 široka izlaza, 6 m)",
        filepath=os.path.join(OUTPUT_DIR, "bottleneck_heatmap_a.png"),
    )

    res_b = run_single_replicate(build_scenario_b(exit_width=3.0), seed=RANDOM_STATE)
    plot_bottleneck_heatmap(
        res_b["positions"],
        area=area,
        title="Heatmap uskih grla — Scenarij B (4 uska izlaza, 3 m)",
        filepath=os.path.join(OUTPUT_DIR, "bottleneck_heatmap_b.png"),
    )

    print("\nZavršeno. Svi outputi u 'outputs/':")
    for fname in [
        "sensitivity_results.json",
        "sensitivity_width.png",
        "bottleneck_heatmap_a.png",
        "bottleneck_heatmap_b.png",
    ]:
        print(f"  {fname}")


if __name__ == "__main__":
    main()