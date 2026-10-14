from __future__ import annotations

import random

from .config import ExperimentConfig
from .geometry import Box, WorldMap


def _simple_boxes() -> list[Box]:
    """Approximation of the five cuboids visible in Figure 1."""
    return [
        Box(25, 50, 165, 225, 0, 90),
        Box(80, 110, 90, 140, 0, 50),
        Box(145, 180, 15, 65, 0, 20),
        Box(200, 230, 15, 65, 0, 20),
        Box(210, 235, 110, 185, 0, 75),
    ]


def _random_nonoverlapping_boxes(
    count: int, bounds: tuple[float, float, float], seed: int, clearance: float
) -> list[Box]:
    rng = random.Random(seed)
    boxes: list[Box] = []
    step = 5
    attempts = 0
    while len(boxes) < count and attempts < 20000:
        attempts += 1
        sx = rng.choice([10, 15, 20, 25, 30])
        sy = rng.choice([10, 15, 20, 25, 30])
        sz = rng.choice([20, 30, 40, 50, 60, 70])
        xmin = rng.randrange(10, int(bounds[0] - sx - 5) + 1, step)
        ymin = rng.randrange(10, int(bounds[1] - sy - 5) + 1, step)
        candidate = Box(xmin, xmin + sx, ymin, ymin + sy, 0, min(sz, bounds[2]))
        if all(not candidate.overlaps(existing, gap=2 * clearance) for existing in boxes):
            boxes.append(candidate)
    if len(boxes) != count:
        raise RuntimeError(f"Could create only {len(boxes)} of {count} obstacles")
    return boxes


def _labyrinth_boxes(bounds: tuple[float, float, float]) -> list[Box]:
    """Deterministic 24-wall maze with alternating passages and heights."""
    boxes: list[Box] = []
    x_positions = [35, 70, 105, 140, 175, 210]
    for i, x in enumerate(x_positions):
        gap_center = 35 if i % 2 == 0 else 205
        height = 45 + 5 * (i % 4)
        if gap_center < 100:
            boxes.append(Box(x, x + 10, 0, gap_center - 15, 0, height))
            boxes.append(Box(x, x + 10, gap_center + 15, bounds[1], 0, height))
        else:
            boxes.append(Box(x, x + 10, 0, gap_center - 15, 0, height))
            boxes.append(Box(x, x + 10, gap_center + 15, bounds[1], 0, height))
    y_positions = [45, 85, 125, 165, 205, 235]
    for i, y in enumerate(y_positions):
        gap_center = 210 if i % 2 == 0 else 45
        height = 55 + 5 * (i % 3)
        if gap_center < 100:
            boxes.append(Box(0, gap_center - 15, y, y + 10, 0, height))
            boxes.append(Box(gap_center + 15, bounds[0], y, y + 10, 0, height))
        else:
            boxes.append(Box(0, gap_center - 15, y, y + 10, 0, height))
            boxes.append(Box(gap_center + 15, bounds[0], y, y + 10, 0, height))
    return boxes[:24]


def build_maps(config: ExperimentConfig) -> dict[str, WorldMap]:
    bounds = config.bounds_m
    clearance = config.safety_clearance_m
    return {
        "simple": WorldMap("simple", bounds, _simple_boxes(), clearance),
        "complex": WorldMap(
            "complex",
            bounds,
            _random_nonoverlapping_boxes(
                18, bounds, config.map_seeds["complex"], clearance
            ),
            clearance,
        ),
        "labyrinth": WorldMap(
            "labyrinth", bounds, _labyrinth_boxes(bounds), clearance
        ),
    }
