from __future__ import annotations

import numpy as np
from typing import List


class Exit:

    def __init__(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.p1: np.ndarray = np.array([x1, y1], dtype=float)
        self.p2: np.ndarray = np.array([x2, y2], dtype=float)
        self._midpoint: np.ndarray = (self.p1 + self.p2) / 2.0
        self._length: float = float(np.linalg.norm(self.p2 - self.p1))

    @property
    def midpoint(self) -> np.ndarray:
        """Return the geometric centre of the exit segment."""
        return self._midpoint

    @property
    def length(self) -> float:
        """Return the length of the exit segment [m]."""
        return self._length

    def sample_target(self, rng: np.random.Generator | None = None) -> np.ndarray:
        t = (rng or np.random.default_rng()).random()
        return self.p1 * (1.0 - t) + self.p2 * t

    def distance_to(self, positions: np.ndarray) -> np.ndarray:
        
        v = self.p2 - self.p1
        w = positions - self.p1  # (N, 2)
        c1 = w @ v               # (N,)
        c2 = float(np.dot(v, v))
        # Guard against zero-length exit segment
        t = np.clip(c1 / (c2 + 1e-12), 0.0, 1.0)  # (N,)
        proj = self.p1 + t[:, np.newaxis] * v        # (N, 2)
        return np.linalg.norm(positions - proj, axis=1)  # (N,)

    def contains_batch(
        self,
        positions: np.ndarray,
        threshold: float = 0.6,
    ) -> np.ndarray:
    
        return self.distance_to(positions) <= threshold


class Environment:

    def __init__(
        self,
        width: float,
        height: float,
        exits: List[Exit],
        A_wall: float = 10.0,
        B_wall: float = 0.2,
        A_agent: float = 2000.0,
        B_agent: float = 0.08,
        wall_suppression_radius: float = 2.0,
    ) -> None:
        if not exits:
            raise ValueError("Environment mora imati barem jedan izlaz.")
        self.width = width
        self.height = height
        self.exits = exits
        self.A_wall = A_wall
        self.B_wall = B_wall
        self.A_agent = A_agent
        self.B_agent = B_agent
        self.wall_suppression_radius = wall_suppression_radius

    def wall_forces(self, positions: np.ndarray) -> np.ndarray:
        
        forces = np.zeros_like(positions)
        A, B = self.A_wall, self.B_wall

        forces[:, 0] += A * np.exp(-positions[:, 0] / B)
        forces[:, 0] -= A * np.exp(-(self.width - positions[:, 0]) / B)
        forces[:, 1] += A * np.exp(-positions[:, 1] / B)
        forces[:, 1] -= A * np.exp(-(self.height - positions[:, 1]) / B)

        if self.wall_suppression_radius > 0 and self.exits:
            suppression = np.ones(positions.shape[0])
            for ex in self.exits:
                dist = ex.distance_to(positions)
                local_sup = np.clip(
                    dist / self.wall_suppression_radius, 0.0, 1.0
                )
                suppression = np.minimum(suppression, local_sup)
            forces *= suppression[:, np.newaxis]

        return forces

    def agent_agent_forces(
        self,
        positions: np.ndarray,
        radii: np.ndarray,
        chunk_size: int = 500,
    ) -> np.ndarray:
        N = positions.shape[0]
        if N < 2:
            return np.zeros_like(positions)

        forces = np.zeros_like(positions)
        A, B = self.A_agent, self.B_agent

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_len = end - start

            diff = (
                positions[start:end, np.newaxis, :]
                - positions[np.newaxis, :, :]
            )
            dist = np.linalg.norm(diff, axis=2)  # (chunk, N)

            for local_i in range(chunk_len):
                dist[local_i, start + local_i] = 1e6

            r_sum = radii[start:end, np.newaxis] + radii[np.newaxis, :]  # (chunk, N)
            mag = A * np.exp((r_sum - dist) / B)  # (chunk, N)

            with np.errstate(invalid="ignore", divide="ignore"):
                n_hat = diff / dist[:, :, np.newaxis]  # (chunk, N, 2)
            n_hat = np.nan_to_num(n_hat)

            forces[start:end] = np.einsum("ij,ijk->ik", mag, n_hat)

        return forces

    def nearest_exit_targets(
        self,
        positions: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        
        _rng = rng or np.random.default_rng()
        exit_mids = np.array([ex.midpoint for ex in self.exits])  # (E, 2)

        dists = np.linalg.norm(
            positions[:, np.newaxis, :] - exit_mids[np.newaxis, :, :],
            axis=2,
        )  # (N, E)
        nearest_idx = np.argmin(dists, axis=1)  # (N,)

        targets = np.empty_like(positions)
        for i, ex_idx in enumerate(nearest_idx):
            targets[i] = self.exits[ex_idx].sample_target(rng=_rng)
        return targets

    def check_evacuations(
        self,
        positions: np.ndarray,
        threshold: float = 0.6,
    ) -> np.ndarray:
        
        reached = np.zeros(positions.shape[0], dtype=bool)
        for ex in self.exits:
            reached |= ex.contains_batch(positions, threshold)
        return reached