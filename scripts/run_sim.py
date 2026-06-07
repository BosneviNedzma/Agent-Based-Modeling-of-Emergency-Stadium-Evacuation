"""Example script to run two scenarios and save basic figures.
Usage: python scripts/run_sim.py
"""
from __future__ import annotations
import numpy as np
from src.model.environment import Environment, Exit
from src.sim.runner import create_agents_grid, Simulation
from src.visualization.visualize import plot_heatmap, plot_evac_times
from src.sim.analysis import compute_evac_metrics, compute_max_density, save_metrics
import json
import os
import matplotlib.pyplot as plt


def run_scenario(num_agents=400, exits=[]):
    env = Environment(width=40.0, height=25.0, exits=exits)
    agents = create_agents_grid(num_agents, area=(5.0, 5.0, 35.0, 20.0))
    sim = Simulation(env, agents, dt=0.05, max_steps=8000)
    res = sim.run()
    # attach area info for analysis
    res["area"] = (0.0, 0.0, 40.0, 25.0)
    res["dt"] = sim.dt
    return env, res


if __name__ == '__main__':
    # Scenario A: 2 wide exits
    exits_a = [Exit(0.0, 10.0, 0.0, 15.0), Exit(40.0, 10.0, 40.0, 15.0)]
    env_a, res_a = run_scenario(600, exits=exits_a)
    plot_heatmap(res_a['positions'][-200:], area=(0,0,40,25))
    plot_evac_times(res_a['evac_times'])
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(os.path.join('outputs','scenario_a.png'), dpi=150)
    # analysis
    metrics_a = compute_evac_metrics(res_a, n_agents=600)
    metrics_a['density_peak'] = compute_max_density(res_a['positions'], area=(0,0,40,25), bins=80)
    save_metrics(metrics_a, os.path.join('outputs','metrics_a.json'))

    # Scenario B: 4 narrow exits
    exits_b = [Exit(0.0, 6.0, 0.0, 8.0), Exit(0.0, 17.0, 0.0, 19.0), Exit(40.0, 6.0, 40.0, 8.0), Exit(40.0, 17.0, 40.0, 19.0)]
    env_b, res_b = run_scenario(600, exits=exits_b)
    plot_heatmap(res_b['positions'][-200:], area=(0,0,40,25))
    plot_evac_times(res_b['evac_times'])
    plt.savefig(os.path.join('outputs','scenario_b.png'), dpi=150)
    metrics_b = compute_evac_metrics(res_b, n_agents=600)
    metrics_b['density_peak'] = compute_max_density(res_b['positions'], area=(0,0,40,25), bins=80)
    save_metrics(metrics_b, os.path.join('outputs','metrics_b.json'))

    print('Scenario A total_time:', res_a['total_time'])
    print('Scenario B total_time:', res_b['total_time'])

    # write short summary to README
    summary = {
        'scenario_a': metrics_a,
        'scenario_b': metrics_b,
    }
    # save machine-readable summary
    save_metrics(summary, os.path.join('outputs','summary.json'))
    # write short human-readable summary
    with open('outputs/summary.txt','w',encoding='utf-8') as f:
        f.write('Evacuation simulation summary\n')
        f.write(f"Scenario A total_time: {metrics_a.get('total_time', float('nan')):.2f}s\n")
        f.write(f"Scenario B total_time: {metrics_b.get('total_time', float('nan')):.2f}s\n")
    # append short human-readable summary to project README
    with open('../README.md','a',encoding='utf-8') as f:
        f.write('\n\n## Quick Results\n')
        f.write(f"Scenario A total_time: {res_a['total_time']:.2f}s\n")
        f.write(f"Scenario B total_time: {res_b['total_time']:.2f}s\n")
        f.write('\nSee outputs/metrics_a.json and outputs/metrics_b.json for full metrics.\n')
