"""Run multiple replicates for each scenario, compute statistics and plots.

Compares Scenario A (2 wide exits) vs Scenario B (4 narrow exits) over
multiple Monte Carlo replications and produces boxplots, survival curves,
confidence interval plots, and a replication convergence chart.

Usage::

    python scripts/run_replicas.py
"""
from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.sim.analysis import save_metrics
from src.sim.scenarios import (
    WARMUP_FRACTION,
    build_scenario_a,
    build_scenario_b,
    run_single_replicate,
)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def ci_for_mean(
    data: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Compute a two-sided confidence interval for the mean via t-distribution.

    Args:
        data: 1D array of observations.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        Tuple (mean, lower_bound, upper_bound).

    Raises:
        ValueError: If data has fewer than 2 observations.
    """
    n = len(data)
    if n < 2:
        raise ValueError("CI zahtijeva barem 2 opservacije.")
    mean = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n))
    h = se * stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
    return mean, mean - h, mean + h


def aggregate_survival(
    all_evac_times: List[Dict[int, float | None]],
    dt: float = 0.5,
) -> Dict[str, list]:
    """Aggregate survival curves across replicates using NumPy broadcasting.

    Replaces the original Python-level double loop with a vectorised
    (R, N, T) boolean broadcast, which is significantly faster for
    large R, N, and T.

    Args:
        all_evac_times: List of {agent_id: evac_time | None} dicts,
                        one per replicate.
        dt: Time bin width for the survival curve [s].

    Returns:
        Dict with 't_bins', 'mean_frac', and 'std_frac' lists.
    """
    # Build replicate matrix R x N, replacing None/missing with inf
    matrices = []
    for d in all_evac_times:
        arr = np.array(
            [t if t is not None else np.inf for t in d.values()],
            dtype=float,
        )
        matrices.append(arr)

    evac_matrix = np.array(matrices)  # (R, N)
    finite = evac_matrix[evac_matrix != np.inf]
    max_t = float(finite.max()) if finite.size > 0 else 0.0
    t_bins = np.arange(0.0, max_t + dt, dt)  # (T,)

    n_agents = evac_matrix.shape[1]

    # Broadcasting: (R, N, 1) <= (1, 1, T)  ->  (R, N, T) bool
    # Sum over N axis, divide by N  ->  (R, T) fraction evacuated
    frac_matrix = (
        evac_matrix[:, :, np.newaxis] <= t_bins[np.newaxis, np.newaxis, :]
    ).sum(axis=1) / n_agents  # (R, T)

    return {
        "t_bins": t_bins.tolist(),
        "mean_frac": np.mean(frac_matrix, axis=0).tolist(),
        "std_frac": np.std(frac_matrix, axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(reps: int = 30, num_agents: int = 2000) -> None:
    """Run full replication study and save all outputs.

    Args:
        reps: Number of Monte Carlo replications per scenario.
        num_agents: Number of agents per simulation run.
    """
    os.makedirs("outputs", exist_ok=True)

    exits_a = build_scenario_a(exit_width=6.0)
    exits_b = build_scenario_b(exit_width=3.0)

    evac_totals: Dict[str, list] = {"A": [], "B": []}
    evac_times_all: Dict[str, list] = {"A": [], "B": []}

    for i in range(reps):
        print(f"Replikacija {i + 1}/{reps} — Scenarij A")
        res_a = run_single_replicate(exits_a, seed=42 + i, num_agents=num_agents)
        evac_totals["A"].append(res_a["total_time"])
        evac_times_all["A"].append(res_a["evac_times"])

        print(f"Replikacija {i + 1}/{reps} — Scenarij B")
        res_b = run_single_replicate(exits_b, seed=42 + i, num_agents=num_agents)
        evac_totals["B"].append(res_b["total_time"])
        evac_times_all["B"].append(res_b["evac_times"])

    arr_a = np.array(evac_totals["A"])
    arr_b = np.array(evac_totals["B"])

    mean_a, lo_a, hi_a = ci_for_mean(arr_a)
    mean_b, lo_b, hi_b = ci_for_mean(arr_b)

    t_stat, p_t = stats.ttest_ind(arr_a, arr_b, equal_var=False)
    try:
        u_stat, p_u = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
    except Exception:
        u_stat, p_u = None, None

    pooled_std = np.sqrt(
        (np.std(arr_a, ddof=1) ** 2 + np.std(arr_b, ddof=1) ** 2) / 2.0
    )
    cohens_d = float((mean_a - mean_b) / pooled_std) if pooled_std > 0 else 0.0

    summary = {
        "reps": reps,
        "num_agents": num_agents,
        "warmup_fraction": WARMUP_FRACTION,
        "scenario_a": {
            "mean": float(mean_a),
            "ci95": [float(lo_a), float(hi_a)],
            "all_times": arr_a.tolist(),
        },
        "scenario_b": {
            "mean": float(mean_b),
            "ci95": [float(lo_b), float(hi_b)],
            "all_times": arr_b.tolist(),
        },
        "t_test": {"t_stat": float(t_stat), "p_value": float(p_t)},
        "mannwhitney": {
            "u_stat": float(u_stat) if u_stat is not None else None,
            "p_value": float(p_u) if p_u is not None else None,
        },
        "cohens_d": cohens_d,
    }
    save_metrics(summary, os.path.join("outputs", "replicates_summary.json"))

    # ------------------------------------------------------------------
    # Plot 1: Boxplot
    # ------------------------------------------------------------------
    plt.figure()
    plt.boxplot([arr_a, arr_b], labels=["A (2 široka)", "B (4 uska)"])
    plt.ylabel("Ukupno vrijeme evakuacije (s)")
    plt.title("Replikacije — ukupno vrijeme evakuacije")
    plt.tight_layout()
    plt.savefig("outputs/replicates_boxplot.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 2: Survival curves
    # ------------------------------------------------------------------
    surv_a = aggregate_survival(evac_times_all["A"])
    surv_b = aggregate_survival(evac_times_all["B"])

    t_a = np.array(surv_a["t_bins"])
    m_a = np.array(surv_a["mean_frac"])
    s_a = np.array(surv_a["std_frac"])
    t_b = np.array(surv_b["t_bins"])
    m_b = np.array(surv_b["mean_frac"])
    s_b = np.array(surv_b["std_frac"])

    plt.figure()
    plt.plot(t_a, m_a, label="A (2 široka izlaza)")
    plt.fill_between(t_a, m_a - s_a, m_a + s_a, alpha=0.2)
    plt.plot(t_b, m_b, label="B (4 uska izlaza)")
    plt.fill_between(t_b, m_b - s_b, m_b + s_b, alpha=0.2)
    plt.xlabel("Vrijeme (s)")
    plt.ylabel("Frakcija evakuiranih (srednja vrijednost)")
    plt.legend()
    plt.title(f"Agregirane survival krive ({reps} replikacija)")
    plt.tight_layout()
    plt.savefig("outputs/replicates_survival.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 3: CI plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.errorbar(
        ["A (2 široka)", "B (4 uska)"],
        [mean_a, mean_b],
        yerr=[
            [mean_a - lo_a, mean_b - lo_b],
            [hi_a - mean_a, hi_b - mean_b],
        ],
        fmt="o",
        capsize=10,
        capthick=2,
        markersize=8,
        color="steelblue",
    )
    ax.set_ylabel("Srednje vrijeme evakuacije (s)")
    ax.set_title("Srednja vrijednost ± 95% interval pouzdanosti")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("outputs/replicates_ci_plot.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 4: Konvergencija standardne greške
    # ------------------------------------------------------------------
    se_a = [
        float(np.std(arr_a[:n], ddof=1) / np.sqrt(n)) for n in range(2, reps + 1)
    ]
    se_b = [
        float(np.std(arr_b[:n], ddof=1) / np.sqrt(n)) for n in range(2, reps + 1)
    ]
    ns = list(range(2, reps + 1))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, se_a, label="A (2 široka izlaza)", color="steelblue")
    ax.plot(ns, se_b, label="B (4 uska izlaza)", color="darkorange")
    ax.set_xlabel("Broj replikacija")
    ax.set_ylabel("Standardna greška srednje vrijednosti (s)")
    ax.set_title("Konvergencija: standardna greška po broju replikacija")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("outputs/replicates_convergence.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print(f"\nWarm-up period: {WARMUP_FRACTION * 100:.0f}% frejmova odbačeno")
    print(f"Scenarij A: {mean_a:.1f}s [95% CI: {lo_a:.1f}s – {hi_a:.1f}s]")
    print(f"Scenarij B: {mean_b:.1f}s [95% CI: {lo_b:.1f}s – {hi_b:.1f}s]")
    print(f"Welch t-test: t={t_stat:.3f}, p={p_t:.4f}")
    if p_u is not None:
        print(f"Mann-Whitney U: U={u_stat:.1f}, p={p_u:.4f}")
    print(f"Cohen's d: {cohens_d:.3f}")
    print("\nSpremljeno u outputs/:")
    for fname in [
        "replicates_summary.json",
        "replicates_boxplot.png",
        "replicates_survival.png",
        "replicates_ci_plot.png",
        "replicates_convergence.png",
    ]:
        print(f"  {fname}")


if __name__ == "__main__":
    main(reps=30, num_agents=2000)