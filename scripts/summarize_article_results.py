from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from path_planning.stats import wilcoxon_greater


METRICS = (
    "length_m",
    "course_change_rad",
    "heading_penalty_inv_m",
    "normalized_criterion",
    "runtime_s",
)
CANDIDATE = "Astar-L-Psi-K"
BASELINES = ("Astar-L", "Astar-L-Psi", "Astar-L-K")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson_interval(successes: int, attempts: int) -> tuple[float, float]:
    if attempts == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / attempts
    denominator = 1.0 + z * z / attempts
    center = (p + z * z / (2.0 * attempts)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / attempts + z * z / (4.0 * attempts**2))
    half /= denominator
    return 100.0 * (center - half), 100.0 * (center + half)


def article_summary(rows: list[dict[str, str]]) -> list[dict]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    output = []
    for method in ("Astar-L", "Astar-L-S", "Astar-L-Psi", "Astar-L-K", CANDIDATE):
        group = grouped[method]
        successful = [row for row in group if row["status"] == "ok"]
        low, high = wilson_interval(len(successful), len(group))
        record = {
            "method": method,
            "attempts": len(group),
            "successes": len(successful),
            "success_rate_percent": 100.0 * len(successful) / len(group),
            "success_rate_ci95_low_percent": low,
            "success_rate_ci95_high_percent": high,
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in successful if math.isfinite(float(row[metric]))]
            record[f"{metric}_mean"] = statistics.fmean(values) if values else math.nan
            record[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else math.nan
        output.append(record)
    return output


def article_comparisons(rows: list[dict[str, str]]) -> list[dict]:
    lookup = {(row["map"], row["pair_id"], row["method"]): row for row in rows}
    pair_keys = sorted({(row["map"], row["pair_id"]) for row in rows})
    output = []
    for baseline in BASELINES:
        comparable = []
        for map_name, pair_id in pair_keys:
            base = lookup[(map_name, pair_id, baseline)]
            candidate = lookup[(map_name, pair_id, CANDIDATE)]
            if base["status"] == "ok" and candidate["status"] == "ok":
                comparable.append((base, candidate))

        identical = sum(
            base["path_sha256"] == candidate["path_sha256"]
            for base, candidate in comparable
        )
        for metric in ("course_change_rad", "heading_penalty_inv_m", "normalized_criterion"):
            differences = [float(base[metric]) - float(candidate[metric]) for base, candidate in comparable]
            result = wilcoxon_greater(differences)
            output.append(
                {
                    "baseline": baseline,
                    "candidate": CANDIDATE,
                    "metric": metric,
                    "comparable_paths": len(comparable),
                    "identical_paths": identical,
                    "identical_percent": 100.0 * identical / len(comparable),
                    **result,
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the aggregate tables used in the article")
    parser.add_argument("results", nargs="?", default="results/search_300")
    args = parser.parse_args()
    results = Path(args.results)
    rows = read_csv(results / "raw_runs.csv")
    write_csv(results / "article_summary.csv", article_summary(rows))
    write_csv(results / "article_comparisons.csv", article_comparisons(rows))
    print(f"Article tables written to {results.resolve()}")


if __name__ == "__main__":
    main()
