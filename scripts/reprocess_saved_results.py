"""Revise postprocessing and analysis without rerunning the four A* searches."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_planning.config import load_config
from path_planning.geometry import Box, WorldMap
from path_planning.planner import AStarPlanner
from path_planning.postprocess import configuration_path_metrics, postprocess_configurations
from path_planning.stats import paired_comparisons, path_equivalence
from summarize_article_results import article_comparisons, article_summary


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    i = int(position)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (position - i)


def paired_effects(rows, repetitions=10000, seed=20261014):
    """Percent change of paired means, stratified by the three fixed maps."""
    lookup = {(r["map"], int(r["pair_id"]), r["method"]): r for r in rows}
    keys = sorted({(r["map"], int(r["pair_id"])) for r in rows})
    comparisons = [(b, "Astar-L-Psi-K") for b in ("Astar-L", "Astar-L-Psi", "Astar-L-K")]
    comparisons += [("Astar-L", "Astar-L-S"), ("Astar-L-Psi-K", "Astar-L-S")]
    output = []
    for baseline, candidate in comparisons:
        for metric in ("length_m", "course_change_rad", "curvature_integral_inv_m", "normalized_criterion"):
            strata = []
            for name in sorted({k[0] for k in keys}):
                values = [(float(lookup[(name, i, baseline)][metric]), float(lookup[(name, i, candidate)][metric]))
                          for m, i in keys if m == name]
                strata.append(values)
            base_mean = statistics.fmean(a for values in strata for a, b in values)
            candidate_mean = statistics.fmean(b for values in strata for a, b in values)
            rng = random.Random(seed)
            simulated = []
            for _ in range(repetitions):
                sb = sc = 0.0
                for values in strata:
                    for j in rng.choices(range(len(values)), k=len(values)):
                        a, b = values[j]
                        sb += a
                        sc += b
                simulated.append(100 * (sb - sc) / sb)
            output.append({"baseline": baseline, "candidate": candidate, "metric": metric,
                           "n": len(keys), "baseline_mean": base_mean, "candidate_mean": candidate_mean,
                           "reduction_percent": 100 * (base_mean - candidate_mean) / base_mean,
                           "ci95_low_percent": percentile(simulated, .025),
                           "ci95_high_percent": percentile(simulated, .975),
                           "bootstrap_repetitions": repetitions, "bootstrap_seed": seed})
    return output


def holm(rows):
    finite = [(i, float(r["p_one_sided"])) for i, r in enumerate(rows) if math.isfinite(float(r["p_one_sided"]))]
    ordered = sorted(finite, key=lambda x: x[1])
    previous = 0.0
    for position, (index, value) in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - position) * value))
        rows[index]["p_holm"] = previous
    for r in rows:
        r.setdefault("p_holm", math.nan)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="results/full_300")
    parser.add_argument("--output", default="results/reviewed_300")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config_dict = json.loads((source / "resolved_config.json").read_text())
    config_path = output / "core_config.json"
    from dataclasses import fields
    from path_planning.config import ExperimentConfig
    core = {k: v for k, v in config_dict.items() if k in {f.name for f in fields(ExperimentConfig)}}
    config_path.write_text(json.dumps(core, indent=2), encoding="utf-8")
    config = load_config(config_path)
    map_dict = json.loads((source / "maps.json").read_text())
    worlds = {name: WorldMap(name, tuple(m["bounds"]), [Box(**b) for b in m["obstacles"]], m["safety_clearance"])
              for name, m in map_dict.items()}
    archived_rows = read_csv(source / "raw_runs.csv")
    rows = []
    for row in archived_rows:
        if row["method"] == "Astar-L-S":
            continue
        rows.append({**row, "search_runtime_s": row["runtime_s"], "postprocess_runtime_s": "",
                     "retained_edges": "", "accepted_quintic_edges": "", "rejected_quintic_edges": ""})
    lookup = {(r["map"], int(r["pair_id"]), r["method"]): r for r in archived_rows}
    planners = {name: AStarPlanner(world, config) for name, world in worlds.items()}
    records, diagnostics, sensitivity = [], [], []
    with gzip.open(source / "paths.ndjson.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["method"] == "Astar-L-S":
                continue
            records.append(record)
            if record["method"] != "Astar-L":
                continue
            path = [(tuple(point), state[3] * config.heading_step_rad)
                    for point, state in zip(record["points"], record["states"])]
            started = time.perf_counter()
            result = postprocess_configurations(path, worlds[record["map"]], config, args.samples)
            post_time = time.perf_counter() - started
            metrics = configuration_path_metrics(result.path, config).to_dict()
            points = [list(p) for p, h in result.path]
            headings = [h for p, h in result.path]
            digest = hashlib.sha256(json.dumps(points, separators=(",", ":")).encode()).hexdigest()
            base = lookup[(record["map"], record["pair_id"], "Astar-L")]
            # Archived search timings and current postprocessing timings come
            # from different execution environments; do not add them together.
            rows.append({**base, "method": "Astar-L-S", "status": "ok", "runtime_s": math.nan,
                         "path_sha256": digest, **metrics, "search_runtime_s": base["runtime_s"],
                         "postprocess_runtime_s": post_time, "retained_edges": result.retained_edges,
                         "accepted_quintic_edges": result.accepted_quintic_edges,
                         "rejected_quintic_edges": result.rejected_quintic_edges})
            records.append({"map": record["map"], "pair_id": record["pair_id"], "method": "Astar-L-S",
                            "points": points, "headings": headings, "states": None})
            diagnostics.append({"map": record["map"], "pair_id": record["pair_id"],
                                "input_edges": len(path) - 1, "retained_edges": result.retained_edges,
                                "accepted_quintic_edges": result.accepted_quintic_edges,
                                "rejected_quintic_edges": result.rejected_quintic_edges,
                                "endpoint_positions_preserved": result.path[0][0] == path[0][0] and result.path[-1][0] == path[-1][0],
                                "endpoint_courses_preserved": result.path[0][1] == path[0][1] and result.path[-1][1] == path[-1][1]})
            refined = postprocess_configurations(path, worlds[record["map"]], config, 2 * args.samples)
            fine = configuration_path_metrics(refined.path, config).to_dict()
            sensitivity.append({"map": record["map"], "pair_id": record["pair_id"],
                                **{f"coarse_{k}": v for k, v in metrics.items()},
                                **{f"fine_{k}": v for k, v in fine.items()},
                                "coarse_accepted_edges": result.accepted_quintic_edges,
                                "fine_accepted_edges": refined.accepted_quintic_edges})
    rows.sort(key=lambda r: (r["map"], int(r["pair_id"]), r["method"]))
    records.sort(key=lambda r: (r["map"], r["pair_id"], r["method"]))
    write_csv(output / "raw_runs.csv", rows)
    write_csv(output / "postprocess_diagnostics.csv", diagnostics)
    write_csv(output / "sampling_sensitivity.csv", sensitivity)
    write_csv(output / "article_summary.csv", article_summary(rows))
    write_csv(output / "article_comparisons.csv", holm(article_comparisons(rows)))
    map_comparisons = []
    for baseline in ("Astar-L", "Astar-L-Psi", "Astar-L-K"):
        map_comparisons.extend(paired_comparisons(rows, baseline))
    write_csv(output / "wilcoxon.csv", holm(map_comparisons))
    write_csv(output / "path_equivalence.csv", path_equivalence(rows))
    write_csv(output / "paired_effects_ci.csv", paired_effects(rows, args.bootstrap))
    with gzip.open(output / "paths.ndjson.gz", "wt", encoding="utf-8") as stream:
        for r in records:
            stream.write(json.dumps(r) + "\n")
    for name in ("maps.json", "pairs.csv", "resolved_config.json"):
        (output / name).write_bytes((source / name).read_bytes())
    metadata = {"source": str(source), "source_search_runs_retained": 1200,
                "postprocessing_runs_recomputed": 300, "quintic_samples": args.samples,
                "bootstrap_repetitions": args.bootstrap, "bootstrap_seed": 20261014,
                "python": sys.version, "platform": platform.platform(),
                "timing_note": "Archived search and current postprocessing timings are not pooled; total smoothed runtime is unavailable.",
                "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                  for name in ("raw_runs.csv", "paths.ndjson.gz", "maps.json", "pairs.csv")}}
    (output / "review_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "paths": len(records), "accepted_quintic_edges": sum(r["accepted_quintic_edges"] for r in diagnostics),
                      "paths_with_quintic_edges": sum(r["accepted_quintic_edges"] > 0 for r in diagnostics)}, indent=2))


if __name__ == "__main__":
    main()
