#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


BASELINES = ["naive", "alpagasus_style", "deita_style", "cosda_equal_budget", "cosda_rerank_v2"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CoSDA rerank-v2 HF pilot results.")
    parser.add_argument("--run-root", required=True, help="Run directory, e.g. runs/aws_vllm_revised_select1_v2_pilot_20260521")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    metric_rows = read_metrics(run_root)
    quality_rows = read_selection_quality(run_root)
    write_csv(args.out_csv, metric_rows, quality_rows)
    write_markdown(args.out_md, run_root, metric_rows, quality_rows)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")


def read_metrics(run_root: Path) -> list[dict]:
    rows = []
    for path in sorted(run_root.glob("*/*/b*_m*_s*/hf/*/test_metrics.json")):
        rel = path.relative_to(run_root).parts
        task, language, slug, baseline = rel[0], rel[1], rel[2], rel[4]
        metrics = json.loads(path.read_text(encoding="utf-8"))
        value = metric_value(metrics)
        if value is None:
            continue
        rows.append(
            {
                "kind": "hf",
                "task": task,
                "language": language,
                "slug": slug,
                "baseline": baseline,
                "macro_f1": value,
                "path": str(path),
            }
        )
    return rows


def read_selection_quality(run_root: Path) -> list[dict]:
    rows = []
    for path in sorted(run_root.glob("*/*/b*_m*_s*/selected/*.jsonl")):
        rel = path.relative_to(run_root).parts
        task, language, slug, baseline = rel[0], rel[1], rel[2], path.stem
        selected = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows.append(
            {
                "kind": "selection",
                "task": task,
                "language": language,
                "slug": slug,
                "baseline": baseline,
                "selected_count": len(selected),
                "judge_label_correctness": mean(
                    [float((((row.get("metadata") or {}).get("llm_judge") or {}).get("label_correctness", 0.0))) for row in selected]
                ),
                "judge_quality_score": mean(
                    [float((((row.get("metadata") or {}).get("llm_judge") or {}).get("quality_score", 0.0))) for row in selected]
                ),
                "score_C": mean([float((row.get("scores") or {}).get("C", 0.0)) for row in selected]),
                "score_L": mean([float((row.get("scores") or {}).get("L", 0.0)) for row in selected]),
                "score_H": mean([float((row.get("scores") or {}).get("H", 0.0)) for row in selected]),
                "hard_reject_rate": mean([1.0 if row.get("hard_reject") else 0.0 for row in selected]),
                "path": str(path),
            }
        )
    return rows


def metric_value(metrics: dict) -> float | None:
    for key in ("test_macro_f1", "eval_macro_f1", "macro_f1", "f1_macro"):
        if key in metrics:
            return float(metrics[key])
    return None


def write_csv(path: str, metric_rows: list[dict], quality_rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kind",
        "task",
        "language",
        "slug",
        "baseline",
        "macro_f1",
        "selected_count",
        "judge_label_correctness",
        "judge_quality_score",
        "score_C",
        "score_L",
        "score_H",
        "hard_reject_rate",
        "path",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metric_rows + quality_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_markdown(path: str, run_root: Path, metric_rows: list[dict], quality_rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    by_cell = defaultdict(dict)
    for row in metric_rows:
        by_cell[(row["task"], row["language"], row["slug"])][row["baseline"]] = row["macro_f1"]
    quality_by_baseline = defaultdict(list)
    for row in quality_rows:
        quality_by_baseline[row["baseline"]].append(row)

    summary = summarize_metrics(metric_rows)
    wins = wins_vs_naive(by_cell)
    decision = decision_text(wins.get("cosda_rerank_v2", {}), summary)

    lines = [
        "# CoSDA Rerank V2 Pilot Analysis",
        "",
        f"Run root: `{run_root}`",
        "",
        "## Decision",
        "",
        decision,
        "",
        "## HF Macro-F1 Summary",
        "",
        "| Baseline | n | Mean | Std | Wins vs naive | Mean delta vs naive |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for baseline in BASELINES:
        item = summary.get(baseline, {"n": 0, "mean": 0.0, "std": 0.0})
        win = wins.get(baseline, {"wins": "-", "losses": "-", "mean_delta": 0.0})
        wins_text = "-" if baseline == "naive" else f"{win['wins']}/{win['wins'] + win['losses'] + win.get('ties', 0)}"
        delta = "-" if baseline == "naive" else f"{win['mean_delta']:+.4f}"
        lines.append(
            f"| {baseline} | {item['n']} | {item['mean']:.4f} | {item['std']:.4f} | {wins_text} | {delta} |"
        )

    lines.extend(["", "## Per-Cell HF Results", "", "| Cell | " + " | ".join(BASELINES) + " | V2 - naive |", "|---" + "|---:" * (len(BASELINES) + 1) + "|"])
    for key, values in sorted(by_cell.items()):
        cell = f"{key[0]}/{key[1]}"
        v2_delta = values.get("cosda_rerank_v2", float("nan")) - values.get("naive", float("nan"))
        score_cells = [format_float(values.get(baseline)) for baseline in BASELINES]
        lines.append(f"| {cell} | " + " | ".join(score_cells) + f" | {v2_delta:+.4f} |")

    lines.extend(
        [
            "",
            "## Selection Quality",
            "",
            "| Baseline | n | Label correctness | Quality | C | L | H | Hard reject |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for baseline in BASELINES:
        items = quality_by_baseline.get(baseline, [])
        if not items:
            lines.append(f"| {baseline} | 0 | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {baseline} | {len(items)} | "
            f"{mean([float(x['judge_label_correctness']) for x in items]):.3f} | "
            f"{mean([float(x['judge_quality_score']) for x in items]):.3f} | "
            f"{mean([float(x['score_C']) for x in items]):.3f} | "
            f"{mean([float(x['score_L']) for x in items]):.3f} | "
            f"{mean([float(x['score_H']) for x in items]):.3f} | "
            f"{mean([float(x['hard_reject_rate']) for x in items]):.3f} |"
        )

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_metrics(rows: list[dict]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["baseline"]].append(float(row["macro_f1"]))
    return {
        baseline: {
            "n": len(values),
            "mean": mean(values),
            "std": statistics.stdev(values) if len(values) >= 2 else 0.0,
        }
        for baseline, values in grouped.items()
    }


def wins_vs_naive(by_cell: dict) -> dict[str, dict]:
    out = {}
    for baseline in BASELINES:
        if baseline == "naive":
            continue
        wins = ties = losses = missing = 0
        deltas = []
        for values in by_cell.values():
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
            "mean_delta": mean(deltas),
        }
    return out


def decision_text(v2_wins: dict, summary: dict[str, dict]) -> str:
    total = v2_wins.get("wins", 0) + v2_wins.get("losses", 0) + v2_wins.get("ties", 0)
    wins = v2_wins.get("wins", 0)
    delta = v2_wins.get("mean_delta", 0.0)
    v2_mean = summary.get("cosda_rerank_v2", {}).get("mean", 0.0)
    naive_mean = summary.get("naive", {}).get("mean", 0.0)
    if total and wins >= 3 and delta > 0 and v2_mean > naive_mean:
        return (
            f"Story A is viable for the pilot: `cosda_rerank_v2` beats naive on {wins}/{total} cells "
            f"with mean delta {delta:+.4f}. Keep claims limited until true multi-seed evaluation."
        )
    return (
        f"Use Story B: `cosda_rerank_v2` does not clearly beat naive in this pilot "
        f"({wins}/{total}, mean delta {delta:+.4f}). Emphasize interpretable quality diagnostics "
        "and LLM-as-judge audit rather than downstream superiority."
    )


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def format_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
