"""Smooth configuration paths with fixed endpoint positions and headings.

Each returned transition is checked for collisions and heading constraints.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import ExperimentConfig
from .geometry import Point3, WorldMap, angular_distance, distance, horizontal_course, wrap_angle
from .metrics import PathMetrics, _aggregate

Configuration = tuple[Point3, float]


def configuration_edge_valid(a: Configuration, b: Configuration, world: WorldMap,
                             config: ExperimentConfig) -> bool:
    p, heading_a = a
    q, heading_b = b
    length = distance(p, q)
    if length <= 1e-12:
        return False
    change = angular_distance(heading_a, heading_b)
    if change > length / config.heading_change_length_m + 1e-12:
        return False
    if config.heading_constraint_mode == "heading_aligned" and math.hypot(q[0] - p[0], q[1] - p[1]) > 1e-12:
        mean = heading_a + 0.5 * wrap_angle(heading_b - heading_a)
        if angular_distance(horizontal_course(p, q), mean) > 0.51 * config.heading_step_rad + 1e-12:
            return False
    return world.segment_free(p, q)


def configuration_path_metrics(path: list[Configuration], config: ExperimentConfig) -> PathMetrics:
    lengths, changes, penalties = [], [], []
    for (a, ha), (b, hb) in zip(path, path[1:]):
        length = distance(a, b)
        change = angular_distance(ha, hb)
        lengths.append(length)
        changes.append(change)
        penalties.append(change * change / length)
    return _aggregate(lengths, changes, penalties, config)


def shortcut_configurations(path: list[Configuration], world: WorldMap,
                            config: ExperimentConfig) -> list[Configuration]:
    """Greedily retain the furthest subsequent admissible configuration."""
    result = [path[0]]
    current = 0
    while current < len(path) - 1:
        target = len(path) - 1
        while target > current + 1 and not configuration_edge_valid(path[current], path[target], world, config):
            target -= 1
        result.append(path[target])
        current = target
    return result


def quintic_configurations(a: Configuration, b: Configuration, samples: int) -> list[Configuration]:
    """Quintic Hermite segment with prescribed endpoint horizontal courses.

    For t in [0, 1], endpoint second derivatives are zero. Endpoint derivative
    horizontal magnitude is the chord's horizontal length; vertical derivative
    equals its vertical displacement. Interior courses use derivative azimuths.
    A vertical segment interpolates heading along the shortest angular interval.
    """
    p, ha = a
    q, hb = b
    delta = tuple(q[i] - p[i] for i in range(3))
    horizontal_length = math.hypot(delta[0], delta[1])
    v0 = (horizontal_length * math.cos(ha), horizontal_length * math.sin(ha), delta[2])
    v1 = (horizontal_length * math.cos(hb), horizontal_length * math.sin(hb), delta[2])
    a3 = tuple(10 * delta[i] - 6 * v0[i] - 4 * v1[i] for i in range(3))
    a4 = tuple(-15 * delta[i] + 8 * v0[i] + 7 * v1[i] for i in range(3))
    a5 = tuple(6 * delta[i] - 3 * v0[i] - 3 * v1[i] for i in range(3))
    result = []
    for j in range(samples + 1):
        t = j / samples
        point = tuple(p[i] + v0[i] * t + a3[i] * t**3 + a4[i] * t**4 + a5[i] * t**5 for i in range(3))
        derivative = tuple(v0[i] + 3 * a3[i] * t**2 + 4 * a4[i] * t**3 + 5 * a5[i] * t**4 for i in range(3))
        heading = (math.atan2(derivative[1], derivative[0]) if horizontal_length > 1e-12
                   else ha + t * wrap_angle(hb - ha)) % (2 * math.pi)
        result.append((point, heading))
    result[0], result[-1] = a, b
    return result


@dataclass
class PostprocessResult:
    path: list[Configuration]
    retained_edges: int
    accepted_quintic_edges: int
    rejected_quintic_edges: int


def postprocess_configurations(path: list[Configuration], world: WorldMap,
                              config: ExperimentConfig, samples: int = 16) -> PostprocessResult:
    if len(path) < 2 or samples < 2:
        raise ValueError("A path needs two configurations and at least two samples")
    if not all(configuration_edge_valid(a, b, world, config) for a, b in zip(path, path[1:])):
        raise ValueError("Input configuration path is not admissible")
    shortened = shortcut_configurations(path, world, config)
    output = [shortened[0]]
    accepted = 0
    for a, b in zip(shortened, shortened[1:]):
        proposal = quintic_configurations(a, b, samples)
        if all(configuration_edge_valid(x, y, world, config) for x, y in zip(proposal, proposal[1:])):
            output.extend(proposal[1:])
            accepted += 1
        else:
            # Retain the admissible straight transition.
            output.append(b)
    return PostprocessResult(output, len(shortened) - 1, accepted, len(shortened) - 1 - accepted)
