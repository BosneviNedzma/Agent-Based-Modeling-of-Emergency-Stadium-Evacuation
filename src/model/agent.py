"""Agent model for social-force based pedestrian dynamics.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass
class Agent:
    """Represents a single pedestrian agent.

    Attributes:
        id: unique identifier
        position: 2D position as numpy array
        velocity: 2D velocity as numpy array
        desired_speed: preferred speed (m/s)
        radius: personal radius for collisions
        evacuated: whether agent has left through an exit
    """
    id: int
    position: np.ndarray
    velocity: np.ndarray
    desired_speed: float
    radius: float = 0.3
    evacuated: bool = False

    def desired_force(self, target: np.ndarray, tau: float = 0.5) -> np.ndarray:
        """Compute desired force to move toward target position.
        f_des = m*(v0*e - v)/tau, but we take m=1 for simplicity.
        """
        e = target - self.position
        dist = np.linalg.norm(e)
        if dist > 0:
            e = e / dist
        else:
            e = np.zeros(2)
        return (self.desired_speed * e - self.velocity) / tau

    def update_position(self, force: np.ndarray, dt: float = 0.05, v_max: float = 3.0) -> None:
        """Integrate motion with simple Euler step.
        Cap speed to v_max.
        """
        self.velocity += force * dt
        speed = np.linalg.norm(self.velocity)
        if speed > v_max:
            self.velocity = (self.velocity / speed) * v_max
        self.position += self.velocity * dt

