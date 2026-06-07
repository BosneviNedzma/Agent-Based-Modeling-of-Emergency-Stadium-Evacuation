from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Agent:

    id: int
    position: np.ndarray
    velocity: np.ndarray
    desired_speed: float
    radius: float = 0.3
    evacuated: bool = False
    mass: float = 70.0

    def __post_init__(self) -> None:
        
        if self.desired_speed <= 0:
            raise ValueError(
                f"Agent {self.id}: desired_speed mora biti > 0, "
                f"dobiveno {self.desired_speed}."
            )
        if self.radius <= 0:
            raise ValueError(
                f"Agent {self.id}: radius mora biti > 0, "
                f"dobiveno {self.radius}."
            )
        if self.mass <= 0:
            raise ValueError(
                f"Agent {self.id}: mass mora biti > 0, "
                f"dobiveno {self.mass}."
            )
        if self.position.shape != (2,):
            raise TypeError(
                f"Agent {self.id}: position mora biti 1D niz oblika (2,), "
                f"dobiveno {self.position.shape}."
            )
        if self.velocity.shape != (2,):
            raise TypeError(
                f"Agent {self.id}: velocity mora biti 1D niz oblika (2,), "
                f"dobiveno {self.velocity.shape}."
            )