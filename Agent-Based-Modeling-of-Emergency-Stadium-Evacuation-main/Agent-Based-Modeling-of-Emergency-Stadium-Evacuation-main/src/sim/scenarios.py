from __future__ import annotations

import numpy as np
from typing import Dict, Any, List

from src.model.environment import Environment, Exit
from src.sim.runner import create_agents_grid, Simulation

# ---------------------------------------------------------------------------
# Module-level constants — single source of truth for all scripts
# ---------------------------------------------------------------------------
ARENA_W: float = 80.0
ARENA_H: float = 60.0
NUM_AGENTS: int = 2000
DT: float = 0.05
MAX_STEPS: int = 12000
WARMUP_FRACTION: float = 0.10
RANDOM_STATE: int = 42

AGENT_AREA: tuple[float, float, float, float] = (5.0, 5.0, 75.0, 55.0)


def build_exits_symmetric(
    n_exits: int,
    exit_width: float,
    arena_w: float = ARENA_W,
    arena_h: float = ARENA_H,
    y_margin: float = 0.15,
) -> List[Exit]:
    if n_exits < 2 or n_exits % 2 != 0:
        raise ValueError(
            f"n_exits mora biti pozitivan parni broj, dobiveno {n_exits}."
        )
    if exit_width <= 0:
        raise ValueError(
            f"exit_width mora biti > 0, dobiveno {exit_width}."
        )

    half = n_exits // 2
    centers = np.linspace(arena_h * y_margin, arena_h * (1.0 - y_margin), half)

    exits: List[Exit] = []
    for y_c in centers:
        y0 = max(0.0, y_c - exit_width / 2.0)
        y1 = min(arena_h, y_c + exit_width / 2.0)
        exits.append(Exit(0.0, y0, 0.0, y1))       # left wall
        exits.append(Exit(arena_w, y0, arena_w, y1))  # right wall
    return exits


def build_scenario_a(exit_width: float = 6.0) -> List[Exit]:
   
    y_c = ARENA_H / 2.0
    y0 = max(0.0, y_c - exit_width / 2.0)
    y1 = min(ARENA_H, y_c + exit_width / 2.0)
    return [
        Exit(0.0, y0, 0.0, y1),
        Exit(ARENA_W, y0, ARENA_W, y1),
    ]


def build_scenario_b(exit_width: float = 3.0) -> List[Exit]:
    
    centers = [ARENA_H * 0.25, ARENA_H * 0.75]
    exits: List[Exit] = []
    for y_c in centers:
        y0 = max(0.0, y_c - exit_width / 2.0)
        y1 = min(ARENA_H, y_c + exit_width / 2.0)
        exits.append(Exit(0.0, y0, 0.0, y1))
        exits.append(Exit(ARENA_W, y0, ARENA_W, y1))
    return exits

def apply_warmup(
    res: Dict[str, Any],
    warmup_fraction: float = WARMUP_FRACTION,
) -> Dict[str, Any]:
    
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError(
            f"warmup_fraction mora biti u [0, 1), dobiveno {warmup_fraction}."
        )
    positions = res.get("positions", [])
    n_warmup = int(len(positions) * warmup_fraction)
    return {
        **res,
        "positions": positions[n_warmup:],
        "warmup_steps": n_warmup,
        "warmup_time": n_warmup * res.get("dt", DT)
        * res.get("position_sample_rate", 1),
    }


def run_single_replicate(
    exits: List[Exit],
    seed: int = RANDOM_STATE,
    warmup_fraction: float = WARMUP_FRACTION,
    num_agents: int = NUM_AGENTS,
    position_sample_rate: int = 10,
) -> Dict[str, Any]:
  
    env = Environment(width=ARENA_W, height=ARENA_H, exits=exits)
    agents = create_agents_grid(num_agents, area=AGENT_AREA, seed=seed)
    sim = Simulation(env, agents, dt=DT, max_steps=MAX_STEPS, seed=seed)
    res = sim.run(position_sample_rate=position_sample_rate)
    return apply_warmup(res, warmup_fraction=warmup_fraction)