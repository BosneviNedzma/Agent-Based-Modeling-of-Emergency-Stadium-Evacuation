"""Environment and social interactions (walls, exits, pairwise forces).
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple
from .agent import Agent


class Exit:
    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.p1 = np.array([x1, y1], dtype=float)
        self.p2 = np.array([x2, y2], dtype=float)

    def sample_target(self) -> np.ndarray:
        t = np.random.rand()
        return self.p1 * (1 - t) + self.p2 * t

    def contains(self, position: np.ndarray, threshold: float = 0.6) -> bool:
        # approximate: distance to segment
        v = self.p2 - self.p1
        w = position - self.p1
        c1 = np.dot(w, v)
        if c1 <= 0:
            proj = self.p1
        else:
            c2 = np.dot(v, v)
            if c2 <= c1:
                proj = self.p2
            else:
                b = c1 / c2
                proj = self.p1 + b * v
        return np.linalg.norm(position - proj) <= threshold


class Environment:
    """Stadium environment with exits and simple wall boundaries.
    Coordinates: rectangular area [0,width] x [0,height].
    """

    def __init__(self, width: float, height: float, exits: List[Exit]):
        self.width = width
        self.height = height
        self.exits = exits

    def agent_wall_force(self, agent: Agent, A: float = 10.0, B: float = 0.2) -> np.ndarray:
        # simple repulsive force from boundaries (4 walls at x=0,x=width,y=0,y=height)
        p = agent.position
        f = np.zeros(2)
        # left wall x=0
        d = p[0]
        if d < 2.0:
            f += A * np.exp(-d / B) * np.array([1.0, 0.0])
        # right wall
        d = self.width - p[0]
        if d < 2.0:
            f += A * np.exp(-d / B) * np.array([-1.0, 0.0])
        # bottom
        d = p[1]
        if d < 2.0:
            f += A * np.exp(-d / B) * np.array([0.0, 1.0])
        # top
        d = self.height - p[1]
        if d < 2.0:
            f += A * np.exp(-d / B) * np.array([0.0, -1.0])
        return f

    def agent_agent_force(self, agent: Agent, others: List[Agent], A: float = 3.0, B: float = 0.5) -> np.ndarray:
        # pairwise exponential repulsion (vectorized loop)
        f = np.zeros(2)
        for other in others:
            if other.id == agent.id or other.evacuated:
                continue
            dvec = agent.position - other.position
            dist = np.linalg.norm(dvec)
            if dist <= 0:
                continue
            n = dvec / dist
            # magnitude
            mag = A * np.exp((agent.radius + other.radius - dist) / B)
            f += mag * n
        return f

    def step_evacuations(self, agents: List[Agent]) -> None:
        # mark evacuated agents if within any exit
        for a in agents:
            if a.evacuated:
                continue
            for ex in self.exits:
                if ex.contains(a.position):
                    a.evacuated = True
                    a.velocity = np.zeros(2)
                    break

