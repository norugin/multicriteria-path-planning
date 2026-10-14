from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SmoothingConfig:
    corner_fraction: float = 0.30
    samples_per_corner: int = 8


@dataclass(frozen=True)
class ExperimentConfig:
    bounds_m: tuple[float, float, float] = (250.0, 250.0, 100.0)
    grid_step_m: float = 5.0
    heading_count: int = 16
    # Legacy field name: r_psi, the discrete heading-change length scale.
    # It does not certify a physical/spatial turning radius.
    minimum_turn_radius_m: float = 15.0
    safety_clearance_m: float = 5.0
    length_scale_m: float = 5.0
    weights: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: {
            "Astar-L": (1.0, 0.0, 0.0),
            "Astar-L-Psi": (0.7, 0.3, 0.0),
            "Astar-L-K": (0.7, 0.0, 0.3),
            "Astar-L-Psi-K": (0.5, 0.3, 0.2),
        }
    )
    map_seeds: dict[str, int] = field(
        default_factory=lambda: {"simple": 1103, "complex": 2207, "labyrinth": 3301}
    )
    pair_seed: int = 20260919
    minimum_pair_distance_m: float = 150.0
    edge_samples: int = 3
    max_expansions: int = 500000
    kinematic_mode: str = "heading_aligned"
    spatial_neighbor_radius_cells: int = 2
    maximum_heading_step_change: int = 2
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)

    @property
    def heading_step_rad(self) -> float:
        return 2.0 * math.pi / self.heading_count

    @property
    def curvature_scale_inv_m(self) -> float:
        return self.length_scale_m / (self.minimum_turn_radius_m**2)

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return tuple(round(v / self.grid_step_m) + 1 for v in self.bounds_m)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["bounds_m"] = tuple(float(x) for x in data["bounds_m"])
    data["weights"] = {k: tuple(float(x) for x in v) for k, v in data["weights"].items()}
    data["smoothing"] = SmoothingConfig(**data.get("smoothing", {}))
    return ExperimentConfig(**data)
