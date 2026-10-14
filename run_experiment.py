from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from path_planning.config import ExperimentConfig, load_config
from path_planning.geometry import distance
from path_planning.maps import build_maps
from path_planning.metrics import state_path_metrics
from path_planning.planner import AStarPlanner, State
from path_planning.postprocess import configuration_path_metrics, postprocess_configurations
from path_planning.stats import pairwise_deltas, paired_comparisons, path_equivalence, summarize
from path_planning.visualize import render_projections


ALL_METHODS = [
    "Astar-L",
    "Astar-L-S",
    "Astar-L-Psi",
    "Astar-L-K",
    "Astar-L-Psi-K",
]

_WORKER_CONFIG: ExperimentConfig | None = None
_WORKER_WORLDS: dict = {}


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_pairs(
    planner: AStarPlanner, count: int, seed: int, minimum_distance: float
) -> list[tuple[State, State]]:
    rng = random.Random(seed)
    pairs: list[tuple[State, State]] = []
    used = set()
    attempts = 0
    while len(pairs) < count and attempts < count * 10000:
        attempts += 1
        start = (
            rng.randrange(planner.shape[0]),
            rng.randrange(planner.shape[1]),
            rng.randrange(planner.shape[2]),
            rng.randrange(planner.config.heading_count),
        )
        goal = (
            rng.randrange(planner.shape[0]),
            rng.randrange(planner.shape[1]),
            rng.randrange(planner.shape[2]),
            rng.randrange(planner.config.heading_count),
        )
        key = (start, goal)
        if key in used or not planner.state_valid(start) or not planner.state_valid(goal):
            continue
        if distance(planner.point(start), planner.point(goal)) < minimum_distance:
            continue
        used.add(key)
        pairs.append((start, goal))
    if len(pairs) != count:
        raise RuntimeError(f"Could generate only {len(pairs)} of {count} requested pairs")
    return pairs


def path_record(map_name: str, pair_id: int, method: str, points, states=None):
    return {
        "map": map_name,
        "pair_id": pair_id,
        "method": method,
        "points": [list(point) for point in points],
        "states": [list(state) for state in states] if states else None,
    }


def path_digest(points) -> str:
    payload = json.dumps([list(point) for point in points], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def initialize_worker(config: ExperimentConfig, worlds: dict) -> None:
    """Install immutable experiment inputs inside each worker process."""
    global _WORKER_CONFIG, _WORKER_WORLDS
    _WORKER_CONFIG = config
    _WORKER_WORLDS = worlds


def run_pair(task: tuple[str, int, State, State, list[str]]) -> dict:
    """Evaluate every selected method for one start/goal pair."""
    map_name, pair_id, start, goal, selected_methods = task
    if _WORKER_CONFIG is None or map_name not in _WORKER_WORLDS:
        raise RuntimeError("Worker was not initialized")

    config = _WORKER_CONFIG
    world = _WORKER_WORLDS[map_name]
    rows: list[dict] = []
    path_records: list[dict] = []
    example_paths: dict[str, list[tuple[float, float, float]]] = {}
    cache: dict[str, tuple] = {}

    for method in selected_methods:
        search_method = "Astar-L" if method == "Astar-L-S" else method
        if search_method not in cache:
            # A fresh planner gives every search method the same cold-cache
            # timing conditions. Astar-L-S deliberately reuses Astar-L search.
            method_planner = AStarPlanner(world, config)
            cache[search_method] = (
                method_planner.plan(start, goal, search_method),
                method_planner,
            )
        result, planner = cache[search_method]
        row = {
            "map": map_name,
            "pair_id": pair_id,
            "method": method,
            "status": result.status,
            "runtime_s": result.runtime_s,
            "expanded": result.expanded,
            "generated": result.generated,
            "search_cost": result.cost,
            "path_sha256": "",
            "length_m": math.nan,
            "course_change_rad": math.nan,
            "heading_penalty_inv_m": math.nan,
            "normalized_criterion": math.nan,
        }
        if result.status == "ok":
            states = result.states
            if method == "Astar-L-S":
                smoothing_started = time.perf_counter()
                original = [(planner.point(state), planner.heading(state)) for state in states]
                processed = postprocess_configurations(original, world, config, samples=16)
                points = [point for point, heading in processed.path]
                row["runtime_s"] = result.runtime_s + (
                    time.perf_counter() - smoothing_started
                )
                row["search_cost"] = cache["Astar-L"][0].cost
                metrics = configuration_path_metrics(processed.path, config)
                path_data = path_record(map_name, pair_id, method, points)
                path_data["headings"] = [heading for point, heading in processed.path]
            else:
                points = [planner.point(state) for state in states]
                metrics = state_path_metrics(states, planner, config)
                path_data = path_record(map_name, pair_id, method, points, states)
            row.update(metrics.to_dict())
            row["path_sha256"] = path_digest(points)
            path_records.append(path_data)
            if pair_id == 0 and method in {"Astar-L", "Astar-L-Psi-K"}:
                example_paths[method] = points
        rows.append(row)

    return {
        "map": map_name,
        "pair_id": pair_id,
        "rows": rows,
        "path_records": path_records,
        "example_paths": example_paths,
    }


def automatic_worker_count(task_count: int) -> int:
    """Use several cores without risking excessive A* memory pressure."""
    logical_cpus = os.cpu_count() or 1
    return max(1, min(task_count, 4, logical_cpus - 1 if logical_cpus > 1 else 1))


def report_progress(completed: int, total: int, started: float, result: dict) -> None:
    elapsed = time.perf_counter() - started
    eta = elapsed * (total - completed) / completed if completed else math.nan
    ok_count = sum(row["status"] == "ok" for row in result["rows"])
    print(
        f"[{completed:>3}/{total}] {100.0 * completed / total:5.1f}% | "
        f"{result['map']} pair={result['pair_id'] + 1} | "
        f"ok={ok_count}/{len(result['rows'])} | "
        f"elapsed={elapsed / 60:.1f} min | ETA={eta / 60:.1f} min",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Multi-criteria A* path-planning experiment")
    parser.add_argument("--config", default="default_config.json")
    parser.add_argument("--output", default="output/run")
    parser.add_argument("--pairs-per-map", type=int, default=100)
    parser.add_argument("--maps", default="simple,complex,labyrinth")
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--pair-seed", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes; 0 selects a conservative automatic value (up to 4)",
    )
    args = parser.parse_args()

    if args.workers < 0:
        raise SystemExit("--workers must be 0 or a positive integer")

    config = load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selected_maps = [x.strip() for x in args.maps.split(",") if x.strip()]
    selected_methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    invalid_methods = set(selected_methods) - set(ALL_METHODS)
    if invalid_methods:
        raise SystemExit(f"Unknown methods: {sorted(invalid_methods)}")

    maps = build_maps(config)
    missing_maps = set(selected_maps) - set(maps)
    if missing_maps:
        raise SystemExit(f"Unknown maps: {sorted(missing_maps)}")

    resolved = config.to_dict()
    resolved["pairs_per_map"] = args.pairs_per_map
    resolved["selected_maps"] = selected_maps
    resolved["selected_methods"] = selected_methods
    output.joinpath("maps.json").write_text(
        json.dumps({name: maps[name].to_dict() for name in selected_maps}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows: list[dict] = []
    pair_rows: list[dict] = []
    examples: dict[str, dict[str, list[tuple[float, float, float]]]] = {}
    path_records: list[dict] = []
    tasks: list[tuple[str, int, State, State, list[str]]] = []
    for map_index, map_name in enumerate(selected_maps):
        planner = AStarPlanner(maps[map_name], config)
        seed = (args.pair_seed if args.pair_seed is not None else config.pair_seed) + 1000 * map_index
        pairs = generate_pairs(planner, args.pairs_per_map, seed, config.minimum_pair_distance_m)
        for pair_id, (start, goal) in enumerate(pairs):
            pair_rows.append(
                {
                    "map": map_name,
                    "pair_id": pair_id,
                    "seed": seed,
                    "start_ix": start[0],
                    "start_iy": start[1],
                    "start_iz": start[2],
                    "start_heading_index": start[3],
                    "goal_ix": goal[0],
                    "goal_iy": goal[1],
                    "goal_iz": goal[2],
                    "goal_heading_index": goal[3],
                }
            )
            tasks.append((map_name, pair_id, start, goal, selected_methods))

    worker_count = args.workers or automatic_worker_count(len(tasks))
    worker_count = min(worker_count, len(tasks))
    resolved["worker_processes"] = worker_count
    resolved["worker_processes_requested"] = args.workers
    output.joinpath("resolved_config.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    selected_worlds = {name: maps[name] for name in selected_maps}
    run_started = time.perf_counter()
    print(
        f"Starting {len(tasks)} pair tasks ({len(tasks) * len(selected_methods)} method runs) "
        f"with {worker_count} worker process(es)",
        flush=True,
    )

    if worker_count == 1:
        initialize_worker(config, selected_worlds)
        results = []
        for completed, task in enumerate(tasks, start=1):
            result = run_pair(task)
            results.append(result)
            report_progress(completed, len(tasks), run_started, result)
    else:
        results = []
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=initialize_worker,
            initargs=(config, selected_worlds),
        ) as executor:
            futures = [executor.submit(run_pair, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                report_progress(completed, len(tasks), run_started, result)

    map_order = {name: index for index, name in enumerate(selected_maps)}
    method_order = {name: index for index, name in enumerate(selected_methods)}
    for result in results:
        rows.extend(result["rows"])
        path_records.extend(result["path_records"])
        if result["example_paths"]:
            examples[result["map"]] = result["example_paths"]
    rows.sort(key=lambda row: (map_order[row["map"]], row["pair_id"], method_order[row["method"]]))
    path_records.sort(
        key=lambda record: (
            map_order[record["map"]],
            record["pair_id"],
            method_order[record["method"]],
        )
    )
    with gzip.open(output / "paths.ndjson.gz", "wt", encoding="utf-8") as path_stream:
        for record in path_records:
            path_stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_csv(output / "pairs.csv", pair_rows)
    write_csv(output / "raw_runs.csv", rows)
    write_csv(output / "summary.csv", summarize(rows))
    comparison_baselines = [
        method
        for method in ("Astar-L", "Astar-L-Psi", "Astar-L-K")
        if method in selected_methods and "Astar-L-Psi-K" in selected_methods
    ]
    wilcoxon_rows = []
    delta_rows = []
    for baseline in comparison_baselines:
        wilcoxon_rows.extend(paired_comparisons(rows, baseline=baseline))
        delta_rows.extend(pairwise_deltas(rows, baseline=baseline))
    write_csv(output / "wilcoxon.csv", wilcoxon_rows)
    write_csv(output / "pairwise_deltas.csv", delta_rows)
    write_csv(output / "path_equivalence.csv", path_equivalence(rows))

    for map_name, paths in examples.items():
        if "Astar-L" in paths and "Astar-L-Psi-K" in paths:
            render_projections(
                maps[map_name],
                paths["Astar-L"],
                paths["Astar-L-Psi-K"],
                output / "figures" / f"{map_name}_pair_000.png",
            )

    wall_runtime_s = time.perf_counter() - run_started
    output.joinpath("run_metadata.json").write_text(
        json.dumps(
            {
                "worker_processes": worker_count,
                "pair_tasks": len(tasks),
                "method_runs": len(tasks) * len(selected_methods),
                "wall_runtime_s": wall_runtime_s,
                "python": __import__("sys").version,
                "platform": __import__("platform").platform(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Results written to {output.resolve()} in {wall_runtime_s / 60:.1f} min")


if __name__ == "__main__":
    main()
