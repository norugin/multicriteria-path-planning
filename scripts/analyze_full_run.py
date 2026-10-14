"""Analyze a new full run without changing its postprocessing or timings."""
from __future__ import annotations
import argparse
import json
from dataclasses import fields
from pathlib import Path
from reprocess_saved_results import holm, paired_effects, read_csv, write_csv
from summarize_article_results import article_comparisons, article_summary
from path_planning.config import ExperimentConfig
from path_planning.stats import paired_comparisons, path_equivalence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results")
    p.add_argument("--bootstrap", type=int, default=10000)
    args = p.parse_args()
    root = Path(args.results)
    rows = read_csv(root / "raw_runs.csv")
    config = json.loads((root / "resolved_config.json").read_text())
    core = {k: v for k, v in config.items() if k in {f.name for f in fields(ExperimentConfig)}}
    (root / "core_config.json").write_text(json.dumps(core, indent=2), encoding="utf-8")
    write_csv(root / "article_summary.csv", article_summary(rows))
    write_csv(root / "article_comparisons.csv", holm(article_comparisons(rows)))
    comparisons = []
    for baseline in ("Astar-L", "Astar-L-Psi", "Astar-L-K"):
        comparisons.extend(paired_comparisons(rows, baseline))
    write_csv(root / "wilcoxon.csv", holm(comparisons))
    write_csv(root / "paired_effects_ci.csv", paired_effects(rows, args.bootstrap))
    write_csv(root / "path_equivalence.csv", path_equivalence(rows))
    (root / "analysis_metadata.json").write_text(json.dumps({"bootstrap_repetitions": args.bootstrap,
        "bootstrap_seed": 20261014, "timings": "retained from this single run"}, indent=2), encoding="utf-8")
    print(f"Analyzed {len(rows)} rows without replacing any paths or runtimes")


if __name__ == "__main__":
    main()
