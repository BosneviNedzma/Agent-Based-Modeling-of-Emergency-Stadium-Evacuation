from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from src.sim.analysis import save_metrics
from src.sim.scenarios import (
    build_exits_symmetric,
    run_single_replicate,
    RANDOM_STATE,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_REPS_PER_POINT: int = 3   # replications per design point
OUTPUT_DIR: str = "outputs"


# ---------------------------------------------------------------------------
# Design matrix generation
# ---------------------------------------------------------------------------

def simulate_design_point(
    n_exits: int,
    exit_width: float,
    n_reps: int,
) -> float:
    """Run n_reps replications for one design point and return mean evac time.

    Args:
        n_exits: Number of exits (must be even, range 2-8).
        exit_width: Width per exit [m].
        n_reps: Number of Monte Carlo replications.

    Returns:
        Mean total evacuation time across replications [s].
    """
    exits = build_exits_symmetric(n_exits, exit_width)
    times = [
        run_single_replicate(exits, seed=RANDOM_STATE + i)["total_time"]
        for i in range(n_reps)
    ]
    return float(np.mean(times))


def generate_design_matrix(
    n_exits_range: tuple[int, int],
    width_range: tuple[float, float],
    n_width_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-factorial design over (n_exits, exit_width).

    Features are: n_exits, exit_width, and total_width (= n_exits * exit_width).
    The derived total_width feature captures the aggregate throughput capacity
    and typically dominates the Random Forest feature importance ranking.

    Args:
        n_exits_range: (min, max) number of exits (even values only).
        width_range: (min_width, max_width) per exit [m].
        n_width_steps: Number of width discretisation steps.

    Returns:
        X: Feature matrix of shape (n_points, 3).
        y: Response vector of mean evacuation times [s].
    """
    exit_counts = list(range(n_exits_range[0], n_exits_range[1] + 1, 2))
    widths = np.linspace(width_range[0], width_range[1], n_width_steps)
    total = len(exit_counts) * len(widths)

    X_rows, y_vals = [], []
    done = 0

    for n_ex in exit_counts:
        for w in widths:
            done += 1
            print(f"  Design point {done}/{total}: n_exits={n_ex}, width={w:.1f} m")
            t_mean = simulate_design_point(n_ex, w, N_REPS_PER_POINT)
            X_rows.append([n_ex, w, n_ex * w])
            y_vals.append(t_mean)

    return np.array(X_rows), np.array(y_vals)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate design matrix, train Random Forest, evaluate and save outputs."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Generiranje design matrice (parametarski pregled)...")
    print("=" * 60)

    X, y = generate_design_matrix(
        n_exits_range=(2, 8),
        width_range=(3.0, 9.0),
        n_width_steps=5,
    )

    print(f"\nDesign matrica: {X.shape}")
    print(f"Raspon evakuacijskog vremena: {y.min():.1f}s – {y.max():.1f}s\n")

    # ------------------------------------------------------------------
    # Train Random Forest with 5-fold cross-validation
    # ------------------------------------------------------------------
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_r2 = cross_val_score(rf, X, y, cv=kf, scoring="r2")
    cv_mae = -cross_val_score(rf, X, y, cv=kf, scoring="neg_mean_absolute_error")

    print("5-struka unakrsna validacija:")
    print(f"  R²  : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    print(f"  MAE : {cv_mae.mean():.2f}s ± {cv_mae.std():.2f}s")

    rf.fit(X, y)
    y_pred = rf.predict(X)
    r2_full = r2_score(y, y_pred)
    mae_full = mean_absolute_error(y, y_pred)

    print(f"\nFit na punom skupu podataka:")
    print(f"  R²  : {r2_full:.4f}")
    print(f"  MAE : {mae_full:.2f}s")

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    metamodel_metrics = {
        "cv_r2_mean": float(cv_r2.mean()),
        "cv_r2_std": float(cv_r2.std()),
        "cv_mae_mean": float(cv_mae.mean()),
        "cv_mae_std": float(cv_mae.std()),
        "full_r2": float(r2_full),
        "full_mae": float(mae_full),
        "feature_importances": {
            "n_exits": float(rf.feature_importances_[0]),
            "exit_width": float(rf.feature_importances_[1]),
            "total_width": float(rf.feature_importances_[2]),
        },
        "n_design_points": int(X.shape[0]),
        "n_reps_per_point": N_REPS_PER_POINT,
    }
    save_metrics(
        metamodel_metrics,
        os.path.join(OUTPUT_DIR, "metamodel_metrics.json"),
    )

    # ------------------------------------------------------------------
    # Plot 1: Predicted vs. Actual
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y, y_pred, alpha=0.7, edgecolors="k", linewidths=0.5)
    lims = [
        min(y.min(), y_pred.min()) - 5,
        max(y.max(), y_pred.max()) + 5,
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Idealna linija")
    ax.set_xlabel("Simulirano vrijeme evakuacije (s)")
    ax.set_ylabel("Predviđeno vrijeme evakuacije (s)")
    ax.set_title(
        f"Metamodel: predviđeno vs. stvarno\n"
        f"R²={r2_full:.3f}, MAE={mae_full:.1f}s"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "metamodel_predicted_vs_actual.png"), dpi=150
    )
    plt.close()

    # ------------------------------------------------------------------
    # Plot 2: Feature importance
    # ------------------------------------------------------------------
    features = ["Broj izlaza", "Širina izlaza (m)", "Ukupna širina (m)"]
    importances = rf.feature_importances_

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(features, importances, color="steelblue", edgecolor="k")
    ax.set_xlabel("Relativna važnost")
    ax.set_title("Random Forest — važnost parametara")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "metamodel_feature_importance.png"), dpi=150
    )
    plt.close()

    # ------------------------------------------------------------------
    # Plot 3: Design space (total_width vs n_exits, colour = evac time)
    # ------------------------------------------------------------------
    total_widths = X[:, 2]
    n_exits_vals = X[:, 0]

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        total_widths, n_exits_vals, c=y, cmap="RdYlGn_r",
        s=120, edgecolors="k", linewidths=0.5,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Srednje vrijeme evakuacije (s)")
    ax.set_xlabel("Ukupna širina izlaza (m)")
    ax.set_ylabel("Broj izlaza")
    ax.set_title("Prostor dizajna — vrijeme evakuacije")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "metamodel_design_space.png"), dpi=150
    )
    plt.close()

    print(f"\nSvi rezultati u '{OUTPUT_DIR}/':")
    for fname in [
        "metamodel_metrics.json",
        "metamodel_predicted_vs_actual.png",
        "metamodel_feature_importance.png",
        "metamodel_design_space.png",
    ]:
        print(f"  {fname}")


if __name__ == "__main__":
    main()