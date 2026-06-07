"""Metamodel-based optimizacija konfiguracije izlaza stadiona.

Koristi već natrenirani Random Forest kao surrogate (brzu aproksimaciju)
simulacijske funkcije i traži globalni minimum evakuacijskog vremena
pomoću scipy.optimize.differential_evolution.

Tok:
    1. Generiraj design matricu (ili učitaj postojeću)
    2. Treniraj RF surrogate model
    3. Pokreni differential_evolution nad surrogate-om
    4. Verificiraj optimum stvarnom simulacijom

Usage::

    python scripts/run_optimization.py
"""
from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, OptimizeResult
from sklearn.ensemble import RandomForestRegressor

from src.sim.analysis import save_metrics
from src.sim.scenarios import (
    build_exits_symmetric,
    run_single_replicate,
    RANDOM_STATE,
)


# ---------------------------------------------------------------------------
# Konstante
# ---------------------------------------------------------------------------
OUTPUT_DIR: str = "outputs"
N_REPS_VERIFY: int = 5       # replikacija za verifikaciju optima
N_REPS_DESIGN: int = 3       # replikacija po design točki


# ---------------------------------------------------------------------------
# Design matrica (ista kao u run_metamodel.py)
# ---------------------------------------------------------------------------

def _simulate_point(n_exits: int, exit_width: float, n_reps: int) -> float:
    """Pokreni n_reps replikacija za jednu design točku, vrati srednju vrijednost.

    Args:
        n_exits: Broj izlaza (parni, 2–8).
        exit_width: Širina izlaza [m].
        n_reps: Broj replikacija.

    Returns:
        Srednje ukupno evakuacijsko vrijeme [s].
    """
    exits = build_exits_symmetric(n_exits, exit_width)
    times = [
        run_single_replicate(exits, seed=RANDOM_STATE + i)["total_time"]
        for i in range(n_reps)
    ]
    return float(np.mean(times))


def build_design_matrix() -> tuple[np.ndarray, np.ndarray]:
    """Generiraj full-factorial design matricu (n_exits × exit_width).

    Features: [n_exits, exit_width, total_width].

    Returns:
        X: Feature matrica oblika (n_points, 3).
        y: Vektor srednjih evakuacijskih vremena [s].
    """
    exit_counts = list(range(2, 10, 2))           # 2, 4, 6, 8
    widths = np.linspace(3.0, 9.0, 5)             # 3, 4.5, 6, 7.5, 9

    rows, vals = [], []
    total = len(exit_counts) * len(widths)
    done = 0

    for n_ex in exit_counts:
        for w in widths:
            done += 1
            print(f"  Design točka {done}/{total}: n_exits={n_ex}, w={w:.1f} m")
            t = _simulate_point(n_ex, w, N_REPS_DESIGN)
            rows.append([n_ex, w, n_ex * w])
            vals.append(t)

    return np.array(rows), np.array(vals)


# ---------------------------------------------------------------------------
# Surrogate model
# ---------------------------------------------------------------------------

def train_surrogate(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    """Treniraj Random Forest surrogate model na design matrici.

    Args:
        X: Feature matrica (n_points, 3).
        y: Ciljne vrijednosti — evakuacijska vremena [s].

    Returns:
        Natrenirani RandomForestRegressor.
    """
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
    )
    rf.fit(X, y)
    return rf


# ---------------------------------------------------------------------------
# Surrogate objective funkcija
# ---------------------------------------------------------------------------

def make_objective(rf: RandomForestRegressor):
    """Vrati objective funkciju za scipy.optimize nad RF surrogate-om.

    Kontinuirani n_exits se zaokružuje na najbliži parni broj kako bi
    optimizator (koji radi u kontinuiranom prostoru) uvažavao diskretnu
    prirodu broja izlaza. total_width se računa automatski kao treći feature.

    Args:
        rf: Natrenirani RandomForestRegressor surrogate.

    Returns:
        Callable(x) -> float — predviđeno evakuacijsko vrijeme [s].
    """
    def objective(x: np.ndarray) -> float:
        n_exits_cont, exit_width = x
        # Zaokruži na najbliži parni broj, klampaj u [2, 8]
        n_exits = int(np.clip(round(n_exits_cont / 2) * 2, 2, 8))
        total_width = n_exits * exit_width
        features = np.array([[n_exits, exit_width, total_width]])
        return float(rf.predict(features)[0])

    return objective


# ---------------------------------------------------------------------------
# Verifikacija optima stvarnom simulacijom
# ---------------------------------------------------------------------------

def verify_optimum(
    n_exits: int,
    exit_width: float,
    n_reps: int = N_REPS_VERIFY,
) -> dict[str, Any]:
    """Verificiraj optimalne parametre stvarnom simulacijom.

    Args:
        n_exits: Optimalan broj izlaza.
        exit_width: Optimalna širina izlaza [m].
        n_reps: Broj verifikacijskih replikacija.

    Returns:
        Rječnik s mean, std i svim vremenima [s].
    """
    exits = build_exits_symmetric(n_exits, exit_width)
    times = [
        run_single_replicate(exits, seed=RANDOM_STATE + 100 + i)["total_time"]
        for i in range(n_reps)
    ]
    return {
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times, ddof=1)),
        "all_times": times,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Pokreni metamodel-based optimizaciju i spremi sve rezultate."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Design matrica + surrogate
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Generiranje design matrice...")
    print("=" * 60)
    X, y = build_design_matrix()

    print("\nTrening surrogate modela (Random Forest)...")
    rf = train_surrogate(X, y)
    print(f"  Surrogate treniran na {X.shape[0]} design točkama.")

    # ------------------------------------------------------------------
    # 2. Globalna optimizacija — differential_evolution
    #    Bounds: n_exits ∈ [2, 8] (kontinuirano, zaokružuje se interno)
    #            exit_width ∈ [3.0, 9.0] m
    # ------------------------------------------------------------------
    print("\nPokretanje scipy.optimize.differential_evolution...")
    bounds = [(2.0, 8.0), (3.0, 9.0)]
    objective = make_objective(rf)

    result: OptimizeResult = differential_evolution(
        objective,
        bounds=bounds,
        seed=RANDOM_STATE,
        maxiter=500,
        tol=1e-4,
        polish=True,          # L-BFGS-B poliranje na kraju
        workers=1,
        disp=False,
    )

    # Dekodiraj optimalne parametre
    n_exits_opt = int(np.clip(round(result.x[0] / 2) * 2, 2, 8))
    width_opt = float(result.x[1])
    surrogate_pred = float(result.fun)

    print(f"\n  Optimalni broj izlaza : {n_exits_opt}")
    print(f"  Optimalna širina      : {width_opt:.2f} m")
    print(f"  Surrogate predikcija  : {surrogate_pred:.1f} s")

    # ------------------------------------------------------------------
    # 3. Verifikacija stvarnom simulacijom
    # ------------------------------------------------------------------
    print(f"\nVerifikacija simulacijom ({N_REPS_VERIFY} replikacija)...")
    verif = verify_optimum(n_exits_opt, width_opt, N_REPS_VERIFY)
    print(f"  Simulirano (mean ± std): {verif['mean_time']:.1f} s ± {verif['std_time']:.1f} s")
    print(f"  Surrogate greška      : {abs(surrogate_pred - verif['mean_time']):.1f} s")

    # ------------------------------------------------------------------
    # 4. Spremi rezultate
    # ------------------------------------------------------------------
    optimization_results = {
        "method": "differential_evolution + RandomForest surrogate",
        "bounds": {"n_exits": [2, 8], "exit_width_m": [3.0, 9.0]},
        "optimal": {
            "n_exits": n_exits_opt,
            "exit_width_m": round(width_opt, 4),
            "total_width_m": round(n_exits_opt * width_opt, 4),
        },
        "surrogate_prediction_s": round(surrogate_pred, 2),
        "verification": {
            "mean_time_s": round(verif["mean_time"], 2),
            "std_time_s": round(verif["std_time"], 2),
            "n_reps": N_REPS_VERIFY,
            "surrogate_error_s": round(
                abs(surrogate_pred - verif["mean_time"]), 2
            ),
        },
        "optimizer_info": {
            "success": bool(result.success),
            "n_iterations": int(result.nit),
            "n_evaluations": int(result.nfev),
            "message": result.message,
        },
    }
    save_metrics(
        optimization_results,
        os.path.join(OUTPUT_DIR, "optimization_results.json"),
    )

    # ------------------------------------------------------------------
    # 5. Vizualizacija — surrogate površina + označeni optimum
    # ------------------------------------------------------------------
    n_vals = np.array([2, 4, 6, 8])
    w_vals = np.linspace(3.0, 9.0, 60)
    W, N = np.meshgrid(w_vals, n_vals)
    Z = np.array([
        rf.predict([[int(n), w, int(n) * w]])[0]
        for n, w in zip(N.ravel(), W.ravel())
    ]).reshape(N.shape)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, n_ex in enumerate(n_vals):
        ax.plot(
            w_vals, Z[i],
            label=f"{n_ex} izlaza",
            linewidth=1.8,
        )

    ax.axvline(width_opt, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.scatter(
        [width_opt], [surrogate_pred],
        color="red", zorder=5, s=80,
        label=f"Optimum: {n_exits_opt} izl., {width_opt:.1f} m → {surrogate_pred:.0f}s",
    )
    ax.set_xlabel("Širina izlaza (m)")
    ax.set_ylabel("Predviđeno evakuacijsko vrijeme (s)")
    ax.set_title("Surrogate površina — optimizacija konfiguracije izlaza")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "optimization_surrogate_surface.png"), dpi=150
    )
    plt.close()

    print(f"\nSvi rezultati u '{OUTPUT_DIR}/':")
    for fname in ["optimization_results.json", "optimization_surrogate_surface.png"]:
        print(f"  {fname}")


if __name__ == "__main__":
    main()