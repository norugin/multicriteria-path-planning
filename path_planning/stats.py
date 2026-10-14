from __future__ import annotations

import math
import statistics
from collections import defaultdict


METRICS = [
    "length_m",
    "course_change_rad",
    "heading_penalty_inv_m",
    "normalized_criterion",
    "runtime_s",
    "expanded",
]


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["map"], row["method"])].append(row)
    output = []
    for (map_name, method), group in sorted(groups.items()):
        successful = [row for row in group if row["status"] == "ok"]
        record = {
            "map": map_name,
            "method": method,
            "attempts": len(group),
            "successes": len(successful),
            "success_rate_percent": 100.0 * len(successful) / len(group) if group else 0.0,
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in successful]
            record[f"{metric}_mean"] = statistics.fmean(values) if values else math.nan
            record[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(record)
    return output


def _average_ranks(values: list[float]) -> tuple[list[float], float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    tie_correction = 0.0
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and abs(values[order[j]] - values[order[i]]) < 1e-12:
            j += 1
        average = 0.5 * ((i + 1) + j)
        for k in range(i, j):
            ranks[order[k]] = average
        tie = j - i
        tie_correction += tie**3 - tie
        i = j
    return ranks, tie_correction


def wilcoxon_greater(differences: list[float]) -> dict:
    """One-sided signed-rank test for median(differences) > 0.

    Uses an exact sign-enumeration distribution for n <= 25 and a normal
    approximation with tie and continuity corrections for larger samples.
    """
    differences = [float(d) for d in differences if abs(float(d)) > 1e-12]
    n = len(differences)
    if n == 0:
        return {
            "n": 0,
            "W_plus": 0.0,
            "z": math.nan,
            "p_one_sided": math.nan,
            "p_method": "undefined",
        }
    ranks, tie_correction = _average_ranks([abs(d) for d in differences])
    w_plus = sum(rank for rank, d in zip(ranks, differences) if d > 0)
    mean = n * (n + 1) / 4.0
    variance = (n * (n + 1) * (2 * n + 1) - 0.5 * tie_correction) / 24.0
    if variance <= 0:
        z = math.nan
        p_normal = math.nan
    else:
        correction = 0.5 if w_plus > mean else (-0.5 if w_plus < mean else 0.0)
        z = (w_plus - mean - correction) / math.sqrt(variance)
        p_normal = 0.5 * math.erfc(z / math.sqrt(2.0))

    if n <= 25:
        doubled_ranks = [int(round(2.0 * rank)) for rank in ranks]
        observed = int(round(2.0 * w_plus))
        counts = {0: 1}
        for rank in doubled_ranks:
            updated = dict(counts)
            for total, count in counts.items():
                updated[total + rank] = updated.get(total + rank, 0) + count
            counts = updated
        p = sum(count for total, count in counts.items() if total >= observed) / (2**n)
        p_method = "exact"
    else:
        p = p_normal
        p_method = "normal"
    return {
        "n": n,
        "W_plus": w_plus,
        "z": z,
        "p_one_sided": p,
        "p_method": p_method,
    }


def path_equivalence(rows: list[dict], candidate: str = "Astar-L-Psi-K") -> list[dict]:
    lookup = {(row["map"], row["pair_id"], row["method"]): row for row in rows}
    maps = sorted({row["map"] for row in rows})
    methods = sorted({row["method"] for row in rows if row["method"] != candidate})
    output = []
    for map_name in maps:
        pair_ids = sorted({row["pair_id"] for row in rows if row["map"] == map_name})
        for method in methods:
            comparable = 0
            identical = 0
            for pair_id in pair_ids:
                reference = lookup.get((map_name, pair_id, candidate))
                other = lookup.get((map_name, pair_id, method))
                if not reference or not other:
                    continue
                if reference["status"] != "ok" or other["status"] != "ok":
                    continue
                comparable += 1
                identical += reference.get("path_sha256") == other.get("path_sha256")
            output.append(
                {
                    "map": map_name,
                    "candidate": candidate,
                    "other_method": method,
                    "comparable_paths": comparable,
                    "identical_paths": identical,
                    "identical_percent": 100.0 * identical / comparable if comparable else math.nan,
                }
            )
    return output


def paired_comparisons(
    rows: list[dict], baseline: str = "Astar-L", candidate: str = "Astar-L-Psi-K"
) -> list[dict]:
    lookup = {(row["map"], row["pair_id"], row["method"]): row for row in rows}
    maps = sorted({row["map"] for row in rows})
    output = []
    for map_name in maps:
        pair_ids = sorted({row["pair_id"] for row in rows if row["map"] == map_name})
        for metric in [
            "course_change_rad",
            "heading_penalty_inv_m",
            "normalized_criterion",
            "length_m",
        ]:
            differences = []
            for pair_id in pair_ids:
                base = lookup.get((map_name, pair_id, baseline))
                cand = lookup.get((map_name, pair_id, candidate))
                if not base or not cand or base["status"] != "ok" or cand["status"] != "ok":
                    continue
                # Positive means that the candidate reduced a lower-is-better metric.
                differences.append(float(base[metric]) - float(cand[metric]))
            result = wilcoxon_greater(differences)
            output.append(
                {
                    "map": map_name,
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": metric,
                    **result,
                }
            )
    return output


def pairwise_deltas(
    rows: list[dict], baseline: str, candidate: str = "Astar-L-Psi-K"
) -> list[dict]:
    """Return pair-level deltas; positive metric deltas favor the candidate."""
    lookup = {(row["map"], row["pair_id"], row["method"]): row for row in rows}
    maps = sorted({row["map"] for row in rows})
    output = []
    metrics = [
        "length_m",
        "course_change_rad",
        "heading_penalty_inv_m",
        "normalized_criterion",
        "runtime_s",
        "expanded",
    ]
    for map_name in maps:
        pair_ids = sorted({row["pair_id"] for row in rows if row["map"] == map_name})
        for pair_id in pair_ids:
            base = lookup.get((map_name, pair_id, baseline))
            cand = lookup.get((map_name, pair_id, candidate))
            if not base or not cand:
                continue
            record = {
                "map": map_name,
                "pair_id": pair_id,
                "baseline": baseline,
                "candidate": candidate,
                "baseline_status": base["status"],
                "candidate_status": cand["status"],
                "paths_identical": "",
            }
            comparable = base["status"] == "ok" and cand["status"] == "ok"
            if comparable:
                record["paths_identical"] = (
                    base.get("path_sha256") == cand.get("path_sha256")
                )
            for metric in metrics:
                if comparable:
                    base_value = float(base[metric])
                    candidate_value = float(cand[metric])
                    delta = base_value - candidate_value
                    relative = 100.0 * delta / base_value if abs(base_value) > 1e-12 else math.nan
                else:
                    base_value = candidate_value = delta = relative = math.nan
                record[f"baseline_{metric}"] = base_value
                record[f"candidate_{metric}"] = candidate_value
                record[f"improvement_{metric}"] = delta
                record[f"improvement_{metric}_percent"] = relative
            output.append(record)
    return output
