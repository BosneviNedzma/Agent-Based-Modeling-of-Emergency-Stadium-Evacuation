"""Run multiple replicates for each scenario, compute statistics and plots.

Usage: python scripts/run_replicas.py
"""
from __future__ import annotations
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from src.model.environment import Environment, Exit
from src.sim.runner import create_agents_grid, Simulation
from src.sim.analysis import save_metrics
from scipy import stats


def run_single(num_agents: int, exits: List[Exit]) -> Dict:
    env = Environment(width=40.0, height=25.0, exits=exits)
    agents = create_agents_grid(num_agents, area=(5.0, 5.0, 35.0, 20.0))
    sim = Simulation(env, agents, dt=0.05, max_steps=8000)
    res = sim.run()
    return res


def aggregate_survival(all_evac_times: List[Dict[int, float]], dt: float = 0.5) -> Dict[str, List]:
    # determine max time across reps
    max_t = 0.0
    for d in all_evac_times:
        times = [t for t in d.values() if t is not None]
        if len(times) > 0:
            max_t = max(max_t, max(times))
    t_bins = np.arange(0.0, max_t + dt, dt)
    frac_matrix = []
    for d in all_evac_times:
        evac_arr = np.array([t if t is not None else np.inf for t in d.values()])
        frac = [float(np.sum(evac_arr <= t) / len(evac_arr)) for t in t_bins]
        frac_matrix.append(frac)
    frac_matrix = np.array(frac_matrix)
    mean_frac = np.mean(frac_matrix, axis=0).tolist()
    std_frac = np.std(frac_matrix, axis=0).tolist()
    return {"t_bins": t_bins.tolist(), "mean_frac": mean_frac, "std_frac": std_frac}


def ci_for_mean(data: np.ndarray, alpha: float = 0.05):
    n = len(data)
    mean = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n))
    h = se * stats.t.ppf(1 - alpha / 2, n - 1)
    return mean, mean - h, mean + h


def main(reps: int = 30, num_agents: int = 600):
    os.makedirs('outputs', exist_ok=True)
    # define scenarios
    exits_a = [Exit(0.0, 10.0, 0.0, 15.0), Exit(40.0, 10.0, 40.0, 15.0)]
    exits_b = [Exit(0.0, 6.0, 0.0, 8.0), Exit(0.0, 17.0, 0.0, 19.0), Exit(40.0, 6.0, 40.0, 8.0), Exit(40.0, 17.0, 40.0, 19.0)]

    results = {"A": [], "B": []}
    evac_times_all = {"A": [], "B": []}

    for i in range(reps):
        print(f"Running replica {i+1}/{reps} for scenario A")
        res_a = run_single(num_agents, exits_a)
        results["A"].append(res_a["total_time"])
        evac_times_all["A"].append(res_a["evac_times"])

        print(f"Running replica {i+1}/{reps} for scenario B")
        res_b = run_single(num_agents, exits_b)
        results["B"].append(res_b["total_time"])
        evac_times_all["B"].append(res_b["evac_times"])

    # stats
    arr_a = np.array(results["A"])
    arr_b = np.array(results["B"])
    mean_a, lo_a, hi_a = ci_for_mean(arr_a)
    mean_b, lo_b, hi_b = ci_for_mean(arr_b)

    # t-test and mann-whitney
    t_stat, p_t = stats.ttest_ind(arr_a, arr_b, equal_var=False)
    try:
        u_stat, p_u = stats.mannwhitneyu(arr_a, arr_b, alternative='two-sided')
    except Exception:
        u_stat, p_u = None, None

    summary = {
        "reps": reps,
        "num_agents": num_agents,
        "scenario_a": {"mean": float(mean_a), "ci95": [float(lo_a), float(hi_a)], "all_times": arr_a.tolist()},
        "scenario_b": {"mean": float(mean_b), "ci95": [float(lo_b), float(hi_b)], "all_times": arr_b.tolist()},
        "t_test": {"t_stat": float(t_stat), "p_value": float(p_t)},
        "mannwhitney": {"u_stat": (float(u_stat) if u_stat is not None else None), "p_value": (float(p_u) if p_u is not None else None)},
    }

    # save summary
    save_metrics(summary, os.path.join('outputs', 'replicates_summary.json'))

    # boxplot
    plt.figure()
    plt.boxplot([arr_a, arr_b], labels=['A (2 wide)', 'B (4 narrow)'])
    plt.ylabel('Total evacuation time (s)')
    plt.title('Replicates: total evacuation time')
    plt.tight_layout()
    plt.savefig('outputs/replicates_boxplot.png', dpi=150)
    plt.close()

    # survival curves aggregated
    surv_a = aggregate_survival(evac_times_all['A'], dt=0.5)
    surv_b = aggregate_survival(evac_times_all['B'], dt=0.5)
    # plot
    t = np.array(surv_a['t_bins'])
    mean_a_frac = np.array(surv_a['mean_frac'])
    std_a_frac = np.array(surv_a['std_frac'])
    t_b = np.array(surv_b['t_bins'])
    mean_b_frac = np.array(surv_b['mean_frac'])
    std_b_frac = np.array(surv_b['std_frac'])

    plt.figure()
    plt.plot(t, mean_a_frac, label='A mean')
    plt.fill_between(t, mean_a_frac - std_a_frac, mean_a_frac + std_a_frac, alpha=0.2)
    plt.plot(t_b, mean_b_frac, label='B mean')
    plt.fill_between(t_b, mean_b_frac - std_b_frac, mean_b_frac + std_b_frac, alpha=0.2)
    plt.xlabel('Time (s)')
    plt.ylabel('Fraction evacuated (mean across reps)')
    plt.legend()
    plt.title('Aggregated survival curves')
    plt.tight_layout()
    plt.savefig('outputs/replicates_survival.png', dpi=150)
    plt.close()

    print('Summary saved to outputs/replicates_summary.json')
    print('Boxplot saved to outputs/replicates_boxplot.png')
    print('Survival plot saved to outputs/replicates_survival.png')


if __name__ == '__main__':
    main(reps=30, num_agents=600)
