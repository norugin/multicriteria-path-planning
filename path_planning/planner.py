from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass

from .config import ExperimentConfig
from .geometry import WorldMap, angular_distance, distance, horizontal_course, wrap_angle


State = tuple[int, int, int, int]


@dataclass
class PlanResult:
    status: str
    states: list[State]
    cost: float
    runtime_s: float
    expanded: int
    generated: int


class AStarPlanner:
    def __init__(self, world: WorldMap, config: ExperimentConfig):
        self.world = world
        self.config = config
        self.shape = config.grid_shape
        self.step = config.grid_step_m
        self.heading_step = config.heading_step_rad
        self._point_free_cache: dict[tuple[int, int, int], bool] = {}
        self._edge_free_cache: dict[tuple[tuple[int, int, int], tuple[int, int, int]], bool] = {}
        radius = config.spatial_neighbor_radius_cells
        self._offsets = [
            (dx, dy, dz)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            for dz in range(-radius, radius + 1)
            if (dx, dy, dz) != (0, 0, 0)
        ]
        self._templates_by_heading = self._build_transition_templates()

    def _build_transition_templates(self):
        """Precompute all position-independent admissible local transitions."""
        templates: list[list[tuple[int, int, int, int, tuple[float, float, float]]]] = []
        for heading_index in range(self.config.heading_count):
            source = (0, 0, 0, heading_index)
            heading_templates = []
            for dx, dy, dz in self._offsets:
                length = self.step * math.sqrt(dx * dx + dy * dy + dz * dz)
                max_heading_change = self.config.maximum_heading_step_change
                for dh in range(-max_heading_change, max_heading_change + 1):
                    next_heading = (heading_index + dh) % self.config.heading_count
                    target = (dx, dy, dz, next_heading)
                    dpsi = abs(dh) * self.heading_step
                    heading_penalty = 0.0 if dpsi == 0 else dpsi * dpsi / length
                    if dpsi > 0 and length / dpsi + 1e-12 < self.config.heading_change_length_m:
                        continue
                    if not self._heading_aligned(source, target):
                        continue
                    heading_templates.append(
                        (dx, dy, dz, next_heading, (length, dpsi, heading_penalty))
                    )
            templates.append(heading_templates)
        return templates

    def point(self, state: State) -> tuple[float, float, float]:
        return tuple(self.step * state[i] for i in range(3))

    def heading(self, state: State) -> float:
        return (state[3] % self.config.heading_count) * self.heading_step

    def state_valid(self, state: State) -> bool:
        if not all(0 <= state[i] < self.shape[i] for i in range(3)):
            return False
        if not 0 <= state[3] < self.config.heading_count:
            return False
        spatial = state[:3]
        value = self._point_free_cache.get(spatial)
        if value is None:
            value = self.world.point_free(self.point(state))
            self._point_free_cache[spatial] = value
        return value

    def _spatial_edge_free(self, a: State, b: State) -> bool:
        pa, pb = a[:3], b[:3]
        key = (pa, pb) if pa <= pb else (pb, pa)
        value = self._edge_free_cache.get(key)
        if value is None:
            value = self.world.segment_free(
                self.point(a), self.point(b)
            )
            self._edge_free_cache[key] = value
        return value

    def edge_metrics(self, a: State, b: State) -> tuple[float, float, float]:
        length = distance(self.point(a), self.point(b))
        dpsi = angular_distance(self.heading(a), self.heading(b))
        heading_penalty = 0.0 if dpsi == 0 else dpsi * dpsi / length
        return length, dpsi, heading_penalty

    def _heading_aligned(self, a: State, b: State) -> bool:
        if self.config.heading_constraint_mode != "heading_aligned":
            return True
        pa, pb = self.point(a), self.point(b)
        if abs(pb[0] - pa[0]) + abs(pb[1] - pa[1]) < 1e-12:
            return True
        chord = horizontal_course(pa, pb)
        dpsi_signed = wrap_angle(self.heading(b) - self.heading(a))
        mean_heading = (self.heading(a) + 0.5 * dpsi_signed) % (2.0 * math.pi)
        return angular_distance(chord, mean_heading) <= 0.51 * self.heading_step

    def neighbors(self, state: State):
        for dx, dy, dz, next_heading, metrics in self._templates_by_heading[state[3]]:
            spatial = (state[0] + dx, state[1] + dy, state[2] + dz)
            if not all(0 <= spatial[i] < self.shape[i] for i in range(3)):
                continue
            next_state = (*spatial, next_heading)
            if not self.state_valid(next_state):
                continue
            if not self._spatial_edge_free(state, next_state):
                continue
            yield next_state, metrics

    def edge_cost(self, edge: tuple[float, float, float], method: str) -> float:
        length, dpsi, heading_penalty = edge
        alpha, beta, gamma = self.config.weights[method]
        return (
            alpha * length / self.config.length_scale_m
            + beta * dpsi / math.pi
            + gamma * heading_penalty / self.config.heading_penalty_scale_inv_m
        )

    def heuristic(self, state: State, goal: State, method: str) -> float:
        alpha = self.config.weights[method][0]
        return alpha * distance(self.point(state), self.point(goal)) / self.config.length_scale_m

    def plan(self, start: State, goal: State, method: str) -> PlanResult:
        if method not in self.config.weights:
            raise KeyError(f"Unknown method: {method}")
        started = time.perf_counter()
        if not self.state_valid(start) or not self.state_valid(goal):
            return PlanResult("invalid_endpoint", [], math.inf, 0.0, 0, 0)

        counter = itertools.count()
        queue: list[tuple[float, int, State]] = []
        heapq.heappush(queue, (self.heuristic(start, goal, method), next(counter), start))
        g_score: dict[State, float] = {start: 0.0}
        parent: dict[State, State] = {}
        closed: set[State] = set()
        expanded = 0
        generated = 1

        while queue:
            _, _, current = heapq.heappop(queue)
            if current in closed:
                continue
            if current == goal:
                path = [current]
                while path[-1] in parent:
                    path.append(parent[path[-1]])
                path.reverse()
                return PlanResult(
                    "ok",
                    path,
                    g_score[current],
                    time.perf_counter() - started,
                    expanded,
                    generated,
                )
            closed.add(current)
            expanded += 1
            if expanded >= self.config.max_expansions:
                return PlanResult(
                    "expansion_limit",
                    [],
                    math.inf,
                    time.perf_counter() - started,
                    expanded,
                    generated,
                )

            current_g = g_score[current]
            for neighbor, edge in self.neighbors(current):
                if neighbor in closed:
                    continue
                tentative = current_g + self.edge_cost(edge, method)
                if tentative + 1e-12 >= g_score.get(neighbor, math.inf):
                    continue
                parent[neighbor] = current
                g_score[neighbor] = tentative
                priority = tentative + self.heuristic(neighbor, goal, method)
                heapq.heappush(queue, (priority, next(counter), neighbor))
                generated += 1

        return PlanResult(
            "no_path",
            [],
            math.inf,
            time.perf_counter() - started,
            expanded,
            generated,
        )
