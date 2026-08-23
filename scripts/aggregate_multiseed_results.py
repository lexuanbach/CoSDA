#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath


DEFAULT_BASELINES = ["naive", "deita_style", "cosda_relaxed", "cosda_equal_budget"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CoSDA HF metrics across seed artifacts.")
    parser.add_argument("artifacts", nargs="+", help="Essential artifact .tar.gz files")
    parser.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--out-json")
    parser.add_argument("--out-csv")
    args = parser.parse_args()

    rows = []
    for artifact in args.artifacts:
        rows.extend(read_artifact(artifact))
    rows = [row for row in rows if row["baseline"] in set(args.baselines)]
    report = build_report(rows, args.baselines)
    print_report(report)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_csv:
        write_csv(args.out_csv, rows)


def read_artifact(path: str) -> list[dict]:
    rows = []
    with tarfile.open(path, "r:gz") as tf:
        for name in tf.getnames():
            parts = PurePosixPath(name).parts
            if len(parts) < 7 or parts[-1] != "test_metrics.json" or parts[-3] != "hf":
                continue
            run_name = parts[0]
            task = parts[1]
            language = parts[2]
            slug = parts[3]
            baseline = parts[-2]
            metrics = json.load(tf.extractfile(name))
            macro_f1 = metric_value(metrics)
            if macro_f1 is None:
                continue
            rows.append(
                {
                    "artifact": str(path),
                    "run_name": run_name,
                    "task": task,
                    "language": language,
                    "seed": parse_seed(slug),
                    "slug": slug,
                    "baseline": baseline,
                    "macro_f1": macro_f1,
                }
            )
    return rows


def metric_value(metrics: dict) -> float | None:
    for key in ("test_macro_f1", "eval_macro_f1", "macro_f1", "f1_macro"):
        if key in metrics:
            return float(metrics[key])
    return None


def parse_seed(slug: str) -> int | None:
    match = re.search(r"_s(\d+)", slug)
    return int(match.group(1)) if match else None


def build_report(rows: list[dict], baselines: list[str]) -> dict:
    summary = {
        "rows": rows,
        "overall": summarize_group(rows, ["baseline"], baselines),
        "by_task": summarize_group(rows, ["task", "baseline"], baselines),
        "by_cell": summarize_group(rows, ["task", "language", "baseline"], baselines),
        "wins_vs_naive": wins_vs_naive(rows, baselines),
        "coverage": coverage(rows, baselines),
    }
    return summary


def summarize_group(rows: list[dict], keys: list[str], baselines: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row["macro_f1"])
    out = []
    for group_key, values in sorted(grouped.items()):
        item = {key: value for key, value in zip(keys, group_key)}
        item.update(stats(values))
        out.append(item)
    return out


def stats(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) >= 2 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def wins_vs_naive(rows: list[dict], baselines: list[str]) -> dict:
    by_cell_seed = defaultdict(dict)
    for row in rows:
        by_cell_seed[(row["task"], row["language"], row["seed"])][row["baseline"]] = row["macro_f1"]
    out = {}
    for baseline in baselines:
        if baseline == "naive":
            continue
        wins = ties = losses = missing = 0
        deltas = []
        for values in by_cell_seed.values():
            if "naive" not in values or baseline not in values:
                missing += 1
                continue
            delta = values[baseline] - values["naive"]
            deltas.append(delta)
            if math.isclose(delta, 0.0, abs_tol=1e-12):
                ties += 1
            elif delta > 0:
                wins += 1
            else:
                losses += 1
        out[baseline] = {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "missing": missing,
            "mean_delta": statistics.mean(deltas) if deltas else None,
            "std_delta": statistics.stdev(deltas) if len(deltas) >= 2 else 0.0,
        }
    return out


def coverage(rows: list[dict], baselines: list[str]) -> dict:
    seeds = sorted({row["seed"] for row in rows})
    cells = sorted({(row["task"], row["language"]) for row in rows})
    counts = defaultdict(int)
    for row in rows:
        counts[row["baseline"]] += 1
    return {
        "seeds": seeds,
        "cell_count": len(cells),
        "expected_rows_per_baseline": len(seeds) * len(cells),
        "rows_per_baseline": {baseline: counts[baseline] for baseline in baselines},
    }


def write_csv(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["artifact", "run_name", "task", "language", "seed", "slug", "baseline", "macro_f1"],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_report(report: dict) -> None:
    print("Coverage:", json.dumps(report["coverage"], ensure_ascii=False))
    print("\nOverall mean/std macro-F1:")
    for row in sorted(report["overall"], key=lambda item: item["mean"], reverse=True):
        print(f"{row['baseline']:20s} mean={row['mean']:.4f} std={row['std']:.4f} n={row['n']}")
    print("\nWins vs naive:")
    for baseline, item in report["wins_vs_naive"].items():
        print(
            f"{baseline:20s} wins={item['wins']} ties={item['ties']} losses={item['losses']} "
            f"mean_delta={item['mean_delta']:+.4f}"
            if item["mean_delta"] is not None
            else f"{baseline:20s} missing"
        )


if __name__ == "__main__":
    main()
