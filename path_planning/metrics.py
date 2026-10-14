from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .config import ExperimentConfig
from .planner import AStarPlanner, State


@dataclass
class PathMetrics:
    length_m: float
    course_change_rad: float
    heading_penalty_inv_m: float
    normalized_criterion: float

    def to_dict(self) -> dict:
        return asdict(self)


def _aggregate(
    lengths: list[float], changes: list[float], penalties: list[float], config: ExperimentConfig
) -> PathMetrics:
    length = sum(lengths)
    course_change = sum(changes)
    penalty = sum(penalties)
    alpha, beta, gamma = config.weights["Astar-L-Psi-K"]
    normalized = (
        alpha * length / config.length_scale_m
        + beta * course_change / math.pi
        + gamma * penalty / config.heading_penalty_scale_inv_m
    )
    return PathMetrics(length, course_change, penalty, normalized)


def state_path_metrics(
    states: list[State], planner: AStarPlanner, config: ExperimentConfig
) -> PathMetrics:
    lengths, changes, penalties = [], [], []
    for a, b in zip(states, states[1:]):
        length, dpsi, penalty = planner.edge_metrics(a, b)
        lengths.append(length)
        changes.append(dpsi)
        penalties.append(penalty)
    return _aggregate(lengths, changes, penalties, config)
