"""Simulation runner: initialize agents, run steps, collect metrics."""
from __future__ import annotations
import numpy as np
from typing import List, Tuple
from ..model.agent import Agent
from ..model.environment import Environment, Exit


class Simulation:
    def __init__(self, env: Environment, agents: List[Agent], dt: float = 0.05, max_steps: int = 20000):
        self.env = env
        self.agents = agents
        self.dt = dt
        self.max_steps = max_steps
        self.time = 0.0
        self.history_positions = []  # for density maps

    def run(self):
        step = 0
        evac_times = {a.id: None for a in self.agents}
        while step < self.max_steps:
            # compute forces and update
            positions = []
            for a in self.agents:
                if a.evacuated:
                    positions.append(a.position.copy())
                    continue
                # pick nearest exit target sample
                targets = [ex.sample_target() for ex in self.env.exits]
                # choose closest target
                dists = [np.linalg.norm(a.position - t) for t in targets]
                target = targets[int(np.argmin(dists))]
                f_des = a.desired_force(target)
                f_ia = self.env.agent_agent_force(a, self.agents)
                f_w = self.env.agent_wall_force(a)
                f = f_des + f_ia + f_w
                a.update_position(f, dt=self.dt)
                positions.append(a.position.copy())
            self.env.step_evacuations(self.agents)
            # record evacuation times
            for a in self.agents:
                if a.evacuated and evac_times[a.id] is None:
                    evac_times[a.id] = step * self.dt
            self.history_positions.append(np.vstack(positions))
            step += 1
            self.time += self.dt
            if all(a.evacuated for a in self.agents):
                break
        return {
            "evac_times": evac_times,
            "total_time": self.time,
            "positions": self.history_positions,
            "steps": step,
        }


def create_agents_grid(n: int, area: Tuple[float, float, float, float]) -> List[Agent]:
    """Create n agents uniformly inside rectangle (x1,y1,x2,y2).
    """
    x1, y1, x2, y2 = area
    xs = np.random.uniform(x1, x2, size=n)
    ys = np.random.uniform(y1, y2, size=n)
    agents = []
    for i in range(n):
        pos = np.array([xs[i], ys[i]], dtype=float)
        vel = np.zeros(2)
        a = Agent(id=i, position=pos, velocity=vel, desired_speed=np.random.normal(1.2, 0.2))
        agents.append(a)
    return agents
