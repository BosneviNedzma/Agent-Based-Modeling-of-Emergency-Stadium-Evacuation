"""Run two evacuation scenarios and save figures and metrics.

Scenario A: 2 wide exits (6 m each) on left and right walls.
Scenario B: 4 narrow exits (3 m each) on left and right walls.

Usage::

    python scripts/run_sim.py
"""
from __future__ import annotations

import os

from src.sim.analysis import compute_evac_metrics, compute_max_density, save_metrics
from src.sim.scenarios import (
    ARENA_H,
    ARENA_W,
    NUM_AGENTS,
    build_scenario_a,
    build_scenario_b,
    run_single_replicate,
)
from src.visualization.visualize import (
    plot_bottleneck_heatmap,
    plot_evac_times,
    plot_heatmap,
)

OUTPUT_DIR: str = "outputs"


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    area = (0.0, 0.0, ARENA_W, ARENA_H)

    # ------------------------------------------------------------------
    # Scenario A: 2 wide exits (6 m each)
    # ------------------------------------------------------------------
    print("Pokretanje Scenarija A (2 široka izlaza)...")
    res_a = run_single_replicate(build_scenario_a(exit_width=6.0), seed=42)

    plot_heatmap(
        res_a["positions"],
        area=area,
        title="Scenarij A — prostorna gustina agenata (2 široka izlaza)",
    ).savefig(os.path.join(OUTPUT_DIR, "scenario_a_heatmap.png"), dpi=150)

    plot_bottleneck_heatmap(
        res_a["positions"],
        area=area,
        n_agents=NUM_AGENTS,
        title="Scenarij A — Bottleneck heatmap (kritična faza 20%–80%)",
    ).savefig(os.path.join(OUTPUT_DIR, "bottleneck_a.png"), dpi=150)

    plot_evac_times(res_a["evac_times"]).savefig(
        os.path.join(OUTPUT_DIR, "scenario_a_evac_hist.png"), dpi=150
    )

    metrics_a = compute_evac_metrics(res_a, n_agents=NUM_AGENTS)
    metrics_a["density_peak"] = compute_max_density(res_a["positions"], area=area)
    metrics_a["warmup_time"] = res_a.get("warmup_time", 0.0)
    save_metrics(metrics_a, os.path.join(OUTPUT_DIR, "metrics_a.json"))

    # ------------------------------------------------------------------
    # Scenario B: 4 narrow exits (3 m each)
    # ------------------------------------------------------------------
    print("Pokretanje Scenarija B (4 uska izlaza)...")
    res_b = run_single_replicate(build_scenario_b(exit_width=3.0), seed=42)

    plot_heatmap(
        res_b["positions"],
        area=area,
        title="Scenarij B — prostorna gustina agenata (4 uska izlaza)",
    ).savefig(os.path.join(OUTPUT_DIR, "scenario_b_heatmap.png"), dpi=150)

    plot_bottleneck_heatmap(
        res_b["positions"],
        area=area,
        n_agents=NUM_AGENTS,
        title="Scenarij B — Bottleneck heatmap (kritična faza 20%–80%)",
    ).savefig(os.path.join(OUTPUT_DIR, "bottleneck_b.png"), dpi=150)

    plot_evac_times(res_b["evac_times"]).savefig(
        os.path.join(OUTPUT_DIR, "scenario_b_evac_hist.png"), dpi=150
    )

    metrics_b = compute_evac_metrics(res_b, n_agents=NUM_AGENTS)
    metrics_b["density_peak"] = compute_max_density(res_b["positions"], area=area)
    metrics_b["warmup_time"] = res_b.get("warmup_time", 0.0)
    save_metrics(metrics_b, os.path.join(OUTPUT_DIR, "metrics_b.json"))

    # ------------------------------------------------------------------
    # Combined summary
    # ------------------------------------------------------------------
    save_metrics(
        {"scenario_a": metrics_a, "scenario_b": metrics_b},
        os.path.join(OUTPUT_DIR, "summary.json"),
    )

    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("Evacuation simulation summary\n")
        f.write(f"Warm-up period: {res_a['warmup_time']:.1f}s odbačeno\n")
        f.write(
            f"Scenarij A total_time: {metrics_a.get('total_time', 0):.2f}s\n"
        )
        f.write(
            f"Scenarij B total_time: {metrics_b.get('total_time', 0):.2f}s\n"
        )

    print(f"Scenarij A ukupno: {res_a['total_time']:.2f}s")
    print(f"Scenarij B ukupno: {res_b['total_time']:.2f}s")
    print(f"Warm-up: {res_a['warmup_time']:.1f}s odbačeno iz analize")
    print(f"\nOutputi u '{OUTPUT_DIR}/'")