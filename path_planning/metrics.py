from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .config import ExperimentConfig
from .geometry import Point3, WorldMap, angular_distance, distance, horizontal_course
from .planner import AStarPlanner, State


@dataclass
class PathMetrics:
    length_m: float
    course_change_rad: float
    curvature_integral_inv_m: float
    normalized_criterion: float

    def to_dict(self) -> dict:
        return asdict(self)


def _aggregate(
    lengths: list[float], changes: list[float], curvatures: list[float], config: ExperimentConfig
) -> PathMetrics:
    length = sum(lengths)
    course_change = sum(changes)
    curvature = sum(curvatures)
    alpha, beta, gamma = config.weights["Astar-L-Psi-K"]
    normalized = (
        alpha * length / config.length_scale_m
        + beta * course_change / math.pi
        + gamma * curvature / config.curvature_scale_inv_m
    )
    return PathMetrics(length, course_change, curvature, normalized)


def state_path_metrics(
    states: list[State], planner: AStarPlanner, config: ExperimentConfig
) -> PathMetrics:
    lengths, changes, curvatures = [], [], []
    for a, b in zip(states, states[1:]):
        length, dpsi, curvature = planner.edge_metrics(a, b)
        lengths.append(length)
        changes.append(dpsi)
        curvatures.append(curvature)
    return _aggregate(lengths, changes, curvatures, config)


def polyline_metrics(points: list[Point3], config: ExperimentConfig) -> PathMetrics:
    if len(points) < 2:
        return _aggregate([], [], [], config)
    lengths = [distance(a, b) for a, b in zip(points, points[1:])]
    headings: list[float] = []
    last = 0.0
    for a, b in zip(points, points[1:]):
        last = horizontal_course(a, b, last)
        headings.append(last)
    changes = [0.0]
    curvatures = [0.0]
    for i in range(1, len(headings)):
        dpsi = angular_distance(headings[i - 1], headings[i])
        local_length = max(1e-12, 0.5 * (lengths[i - 1] + lengths[i]))
        changes.append(dpsi)
        curvatures.append(dpsi * dpsi / local_length)
    return _aggregate(lengths, changes, curvatures, config)


def minimum_polyline_turn_radius(points: list[Point3]) -> float:
    """Legacy horizontal heading-distance proxy, not a 3-D turn radius.

    Retained for compatibility with the supplied archival results. The revised
    postprocessor in postprocess.py validates complete configurations instead.
    """
    if len(points) < 3:
        return math.inf
    lengths = [distance(a, b) for a, b in zip(points, points[1:])]
    headings = []
    last_heading = 0.0
    for a, b in zip(points, points[1:]):
        last_heading = horizontal_course(a, b, last_heading)
        headings.append(last_heading)
    radii = []
    for index in range(1, len(headings)):
        dpsi = angular_distance(headings[index - 1], headings[index])
        if dpsi <= 1e-12:
            continue
        local_length = 0.5 * (lengths[index - 1] + lengths[index])
        radii.append(local_length / dpsi)
    return min(radii, default=math.inf)


def _unit(a: Point3, b: Point3) -> Point3:
    d = distance(a, b)
    if d < 1e-12:
        return (0.0, 0.0, 0.0)
    return tuple((b[i] - a[i]) / d for i in range(3))


def _lerp(a: Point3, b: Point3, t: float) -> Point3:
    return tuple(a[i] + t * (b[i] - a[i]) for i in range(3))


def shortcut_path(points: list[Point3], world: WorldMap) -> list[Point3]:
    if len(points) <= 2:
        return list(points)
    result = [points[0]]
    current = 0
    while current < len(points) - 1:
        target = len(points) - 1
        while target > current + 1 and not world.segment_free(points[current], points[target], 12):
            target -= 1
        result.append(points[target])
        current = target
    return result


def _quintic_curve(
    p0: Point3, p1: Point3, v0: Point3, v1: Point3, samples: int
) -> list[Point3]:
    output = []
    delta = tuple(p1[i] - p0[i] for i in range(3))
    a3 = tuple(10 * delta[i] - 6 * v0[i] - 4 * v1[i] for i in range(3))
    a4 = tuple(-15 * delta[i] + 8 * v0[i] + 7 * v1[i] for i in range(3))
    a5 = tuple(6 * delta[i] - 3 * v0[i] - 3 * v1[i] for i in range(3))
    for index in range(samples + 1):
        t = index / samples
        output.append(
            tuple(
                p0[i] + v0[i] * t + a3[i] * t**3 + a4[i] * t**4 + a5[i] * t**5
                for i in range(3)
            )
        )
    return output


def smooth_path(points: list[Point3], world: WorldMap, config: ExperimentConfig) -> list[Point3]:
    points = shortcut_path(points, world)
    if len(points) <= 2:
        return points
    fraction = config.smoothing.corner_fraction
    samples = config.smoothing.samples_per_corner
    output: list[Point3] = [points[0]]
    for i in range(1, len(points) - 1):
        prev, corner, nxt = points[i - 1], points[i], points[i + 1]
        radius = fraction * min(distance(prev, corner), distance(corner, nxt))
        if radius < 1e-9:
            output.append(corner)
            continue
        entry = _lerp(corner, prev, radius / distance(corner, prev))
        exit_ = _lerp(corner, nxt, radius / distance(corner, nxt))
        u0, u1 = _unit(prev, corner), _unit(corner, nxt)
        chord = distance(entry, exit_)
        velocity_scale = 0.7 * chord
        v0 = tuple(velocity_scale * x for x in u0)
        v1 = tuple(velocity_scale * x for x in u1)
        curve = _quintic_curve(entry, exit_, v0, v1, samples)
        candidate = [output[-1], entry, *curve[1:]]
        if world.polyline_free(candidate, sample_count=5):
            if distance(output[-1], entry) > 1e-9:
                output.append(entry)
            output.extend(curve[1:])
        else:
            output.append(corner)
    if distance(output[-1], points[-1]) > 1e-9:
        output.append(points[-1])
    return output
