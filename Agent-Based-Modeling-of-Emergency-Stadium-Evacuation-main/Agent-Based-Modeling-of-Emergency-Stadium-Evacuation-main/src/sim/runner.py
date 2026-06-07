from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Tuple

from ..model.agent import Agent
from ..model.environment import Environment


class Simulation:

    def __init__(
        self,
        env: Environment,
        agents: List[Agent],
        dt: float = 0.05,
        max_steps: int = 12000,
        tau: float = 0.5,
        v_max: float = 3.0,
        seed: int = 42,
    ) -> None:
        if not agents:
            raise ValueError("Simulation zahtijeva barem jednog agenta.")
        if dt <= 0:
            raise ValueError(f"dt mora biti > 0, dobiveno {dt}.")

        self.env = env
        self.agents = agents
        self.dt = dt
        self.max_steps = max_steps
        self.tau = tau
        self.v_max = v_max
        self.time: float = 0.0
        self._rng = np.random.default_rng(seed)

        n = len(agents)
        self._pos: np.ndarray = np.array(
            [a.position for a in agents], dtype=float
        )
        self._vel: np.ndarray = np.array(
            [a.velocity for a in agents], dtype=float
        )
        self._v0: np.ndarray = np.array(
            [a.desired_speed for a in agents], dtype=float
        )
        self._radii: np.ndarray = np.array(
            [a.radius for a in agents], dtype=float
        )
        self._evacuated: np.ndarray = np.zeros(n, dtype=bool)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _desired_forces(self, active: np.ndarray) -> np.ndarray:
        """Compute desired forces for all active agents.

        Args:
            active: Boolean mask of non-evacuated agents.

        Returns:
            (N_active, 2) desired force array.
        """
        pos_active = self._pos[active]
        vel_active = self._vel[active]
        v0_active = self._v0[active]

        # Pass the shared rng for reproducible stochastic targeting
        targets = self.env.nearest_exit_targets(pos_active, rng=self._rng)
        e = targets - pos_active
        dist = np.linalg.norm(e, axis=1, keepdims=True)
        e_hat = np.where(dist > 1e-6, e / dist, 0.0)

        return (v0_active[:, np.newaxis] * e_hat - vel_active) / self.tau

    def _step(self) -> None:
        
        active = ~self._evacuated
        if not np.any(active):
            return

        pos_active = self._pos[active]
        vel_active = self._vel[active]
        radii_active = self._radii[active]

        f_des = self._desired_forces(active)
        f_wall = self.env.wall_forces(pos_active)
        f_agent = self.env.agent_agent_forces(pos_active, radii_active)
        f_net = f_des + f_wall + f_agent

        # Euler integration
        vel_new = vel_active + f_net * self.dt

        # Hard speed cap
        speed = np.linalg.norm(vel_new, axis=1, keepdims=True)
        vel_new = np.where(
            speed > self.v_max,
            vel_new * self.v_max / speed,
            vel_new,
        )

        pos_new = pos_active + vel_new * self.dt

        margin = radii_active
        pos_new[:, 0] = np.clip(
            pos_new[:, 0], margin, self.env.width - margin
        )
        pos_new[:, 1] = np.clip(
            pos_new[:, 1], margin, self.env.height - margin
        )

        self._vel[active] = vel_new
        self._pos[active] = pos_new

        just_evacuated = self.env.check_evacuations(pos_new)
        indices = np.where(active)[0]
        newly_out = indices[just_evacuated]
        self._evacuated[newly_out] = True
        self._vel[newly_out] = 0.0

    def run(
        self,
        position_sample_rate: int = 10,
    ) -> Dict[str, Any]:
       
        if position_sample_rate < 1:
            raise ValueError(
                f"position_sample_rate mora biti >= 1, "
                f"dobiveno {position_sample_rate}."
            )

        evac_times: Dict[int, float | None] = {
            a.id: None for a in self.agents
        }
        history: List[np.ndarray] = []
        prev_evacuated = self._evacuated.copy()
        step = 0

        for step in range(self.max_steps):
            self._step()
            self.time = (step + 1) * self.dt

            if step % position_sample_rate == 0:
                history.append(self._pos.copy())

            newly = self._evacuated & ~prev_evacuated
            for idx in np.where(newly)[0]:
                evac_times[self.agents[idx].id] = self.time
            prev_evacuated = self._evacuated.copy()

            if self._evacuated.all():
                break

        return {
            "evac_times": evac_times,
            "total_time": self.time,
            "positions": history,
            "steps": step + 1,
            "dt": self.dt,
            "position_sample_rate": position_sample_rate,
        }


def create_agents_grid(
    n: int,
    area: Tuple[float, float, float, float],
    seed: int = 42,
    desired_speed_mean: float = 1.34,
    desired_speed_std: float = 0.26,
) -> List[Agent]:
    
    if n < 1:
        raise ValueError(f"n mora biti >= 1, dobiveno {n}.")

    x1, y1, x2, y2 = area
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Neispravna oblast agenta: ({x1}, {y1}, {x2}, {y2}). "
            "x2 mora biti > x1 i y2 mora biti > y1."
        )

    rng = np.random.default_rng(seed)

    xs = rng.uniform(x1, x2, size=n)
    ys = rng.uniform(y1, y2, size=n)
    speeds = np.clip(
        rng.normal(desired_speed_mean, desired_speed_std, size=n),
        0.5,
        2.5,
    )
    radii = rng.uniform(0.25, 0.32, size=n)
    masses = np.clip(rng.normal(70.0, 10.0, size=n), 40.0, 120.0)

    return [
        Agent(
            id=i,
            position=np.array([xs[i], ys[i]], dtype=float),
            velocity=np.zeros(2, dtype=float),
            desired_speed=float(speeds[i]),
            radius=float(radii[i]),
            mass=float(masses[i]),
        )
        for i in range(n)
    ]