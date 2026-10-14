from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable


Point3 = tuple[float, float, float]


def distance(a: Point3, b: Point3) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angular_distance(a: float, b: float) -> float:
    return abs(wrap_angle(b - a))


def horizontal_course(a: Point3, b: Point3, fallback: float = 0.0) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) + abs(dy) < 1e-12:
        return fallback
    return math.atan2(dy, dx) % (2.0 * math.pi)


@dataclass(frozen=True)
class Box:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    def inflated(self, margin: float, bounds: Point3) -> "Box":
        return Box(
            max(0.0, self.xmin - margin),
            min(bounds[0], self.xmax + margin),
            max(0.0, self.ymin - margin),
            min(bounds[1], self.ymax + margin),
            max(0.0, self.zmin - margin),
            min(bounds[2], self.zmax + margin),
        )

    def contains(self, point: Point3) -> bool:
        x, y, z = point
        return (
            self.xmin <= x <= self.xmax
            and self.ymin <= y <= self.ymax
            and self.zmin <= z <= self.zmax
        )

    def intersects_segment(self, a: Point3, b: Point3) -> bool:
        """Return whether the closed segment intersects this closed box.

        The slab test is exact for an axis-aligned box and avoids tunnelling
        through an obstacle between sparsely sampled collision-check points.
        Touching the safety-inflated box boundary counts as a collision.
        """
        t_min, t_max = 0.0, 1.0
        bounds = (
            (self.xmin, self.xmax),
            (self.ymin, self.ymax),
            (self.zmin, self.zmax),
        )
        for coordinate_a, coordinate_b, (lower, upper) in zip(a, b, bounds):
            delta = coordinate_b - coordinate_a
            if abs(delta) < 1e-12:
                if coordinate_a < lower or coordinate_a > upper:
                    return False
                continue
            entry = (lower - coordinate_a) / delta
            exit_ = (upper - coordinate_a) / delta
            if entry > exit_:
                entry, exit_ = exit_, entry
            t_min = max(t_min, entry)
            t_max = min(t_max, exit_)
            if t_min > t_max:
                return False
        return True

    def overlaps(self, other: "Box", gap: float = 0.0) -> bool:
        return not (
            self.xmax + gap < other.xmin
            or other.xmax + gap < self.xmin
            or self.ymax + gap < other.ymin
            or other.ymax + gap < self.ymin
            or self.zmax + gap < other.zmin
            or other.zmax + gap < self.zmin
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WorldMap:
    name: str
    bounds: Point3
    obstacles: list[Box]
    safety_clearance: float

    def __post_init__(self):
        self.inflated_obstacles = [
            obstacle.inflated(self.safety_clearance, self.bounds)
            for obstacle in self.obstacles
        ]

    def within_bounds(self, point: Point3) -> bool:
        return all(0.0 <= value <= limit for value, limit in zip(point, self.bounds))

    def point_free(self, point: Point3) -> bool:
        return self.within_bounds(point) and not any(
            obstacle.contains(point) for obstacle in self.inflated_obstacles
        )

    def segment_free(self, a: Point3, b: Point3) -> bool:
        if not self.point_free(a) or not self.point_free(b):
            return False
        return not any(
            obstacle.intersects_segment(a, b) for obstacle in self.inflated_obstacles
        )

    def polyline_free(self, points: Iterable[Point3]) -> bool:
        points = list(points)
        return all(
            self.segment_free(points[i - 1], points[i])
            for i in range(1, len(points))
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "bounds": list(self.bounds),
            "safety_clearance": self.safety_clearance,
            "obstacles": [obstacle.to_dict() for obstacle in self.obstacles],
        }
