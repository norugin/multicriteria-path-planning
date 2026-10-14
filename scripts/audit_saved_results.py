"""Audit path/CSV consistency and admissibility of the reviewed experiment."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_planning.config import load_config
from path_planning.geometry import Box, WorldMap
from path_planning.metrics import state_path_metrics
from path_planning.planner import AStarPlanner
from path_planning.postprocess import configuration_edge_valid, configuration_path_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", nargs="?", default="results/reviewed_300")
    args = p.parse_args()
    root = Path(args.results)
    config = load_config(root / "core_config.json")
    maps = json.loads((root / "maps.json").read_text())
    worlds = {name: WorldMap(name, tuple(m["bounds"]), [Box(**b) for b in m["obstacles"]], m["safety_clearance"])
              for name, m in maps.items()}
    planners = {name: AStarPlanner(world, config) for name, world in worlds.items()}
    with (root / "raw_runs.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    lookup = {(r["map"], int(r["pair_id"]), r["method"]): r for r in rows}
    with (root / "pairs.csv").open(encoding="utf-8-sig", newline="") as f:
        pairs = {(r["map"], int(r["pair_id"])): r for r in csv.DictReader(f)}
    issues = Counter()
    seen = set()
    with gzip.open(root / "paths.ndjson.gz", "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            key = (record["map"], record["pair_id"], record["method"])
            if key in seen:
                issues["duplicate_record"] += 1
            seen.add(key)
            row = lookup[key]
            planner = planners[record["map"]]
            points = [tuple(x) for x in record["points"]]
            digest = hashlib.sha256(json.dumps([list(x) for x in points], separators=(",", ":")).encode()).hexdigest()
            if digest != row["path_sha256"]:
                issues["path_digest"] += 1
            if record["states"]:
                states = [tuple(x) for x in record["states"]]
                path = [(planner.point(s), planner.heading(s)) for s in states]
                if [p for p, h in path] != points:
                    issues["state_position"] += 1
                for a, b in zip(states, states[1:]):
                    if b not in dict(planner.neighbors(a)):
                        issues["graph_transition"] += 1
                metrics = state_path_metrics(states, planner, config)
            else:
                path = list(zip(points, record["headings"]))
                metrics = configuration_path_metrics(path, config)
            pair = pairs[(record["map"], record["pair_id"])]
            for prefix, boundary in (("start", path[0]), ("goal", path[-1])):
                state = tuple(int(pair[f"{prefix}_{suffix}"]) for suffix in ("ix", "iy", "iz", "heading_index"))
                if boundary != (planner.point(state), planner.heading(state)):
                    issues["boundary_configuration"] += 1
            if not all(configuration_edge_valid(a, b, worlds[record["map"]], config)
                       for a, b in zip(path, path[1:])):
                issues["configuration_admissibility"] += 1
            for name, value in metrics.to_dict().items():
                if not math.isclose(value, float(row[name]), abs_tol=1e-9, rel_tol=1e-9):
                    issues["metric_" + name] += 1
    if len(seen) != len(rows):
        issues["record_count"] += 1
    report = {"rows": len(rows), "paths": len(seen), "issues": dict(issues), "passed": not issues}
    (root / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
