#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

from cosda.config import load_run_config  # noqa: E402
from cosda.io_utils import write_jsonl  # noqa: E402
from cosda.selection import (  # noqa: E402
    _balanced_take,
    _eligible_pool,
    _passes_constraints,
    _rank_score,
)
from cosda.types import DataRecord  # noqa: E402


DEFAULT_BASELINES = [
    "naive",
    "alpagasus_style",
    "deita_style",
    "cosda_equal_budget",
    "cosda_rerank_v2",
]


SCORE_KEYS = ["U", "C", "D", "L", "H", "S"]
JUDGE_KEYS = [
    "label_correctness",
    "quality_score",
    "fluency",
    "cultural_plausibility",
    "leakage_suspicion",
    "counterfactual_validity",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local proxy comparison for budget-preserving CoSDA selectors."
    )
    parser.add_argument("--artifact", required=True, help="Essential artifact .tar.gz")
    parser.add_argument(
        "--config",
        default="configs/scenarios/aws_g6e4xlarge_vllm_revised_select1_v2.yaml",
        help="Run config used for audit thresholds and baseline definitions.",
    )
    parser.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-per-source", type=int, default=3)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument(
        "--materialize-dir",
        help="Optional directory to write simulated selected JSONL files for inspection.",
    )
    args = parser.parse_args()

    cfg = load_run_config(ROOT / args.config)
    cells = read_artifact(args.artifact)
    rows = []
    for cell_key, payload in sorted(cells.items()):
        task, language, slug = cell_key
        audit = payload.get("audit") or []
        gold_rows = payload.get("gold") or []
        gold = [DataRecord.from_json(row) for row in gold_rows]
        if not audit or not gold:
            continue
        if args.materialize_dir:
            cell_dir = Path(args.materialize_dir) / task / language / slug
            write_jsonl(cell_dir / "gold.jsonl", gold_rows)
            write_jsonl(cell_dir / "audit.jsonl", audit)
        existing_budget = existing_selection_budget(payload)
        selection_multiplier = cfg.selection_multiplier if cfg.selection_multiplier is not None else 1
        budget = existing_budget or len(gold) * selection_multiplier
        selections = {}
        for baseline in args.baselines:
            selected = simulate_selection(
                audit,
                gold,
                budget,
                baseline,
                cfg.audit,
                seed=args.seed,
                max_per_source=args.max_per_source,
            )
            selections[baseline] = selected
            if args.materialize_dir:
                out = (
                    Path(args.materialize_dir)
                    / task
                    / language
                    / slug
                    / "selected"
                    / f"{baseline}.jsonl"
                )
                write_jsonl(out, selected)

        anchors = {
            name: {row_key(row) for row in selections.get(name, [])}
            for name in ("naive", "alpagasus_style", "deita_style", "cosda_equal_budget")
        }
        for baseline, selected in selections.items():
            rows.append(
                summarize_selection(
                    task,
                    language,
                    slug,
                    baseline,
                    selected,
                    gold,
                    budget,
                    cfg.audit,
                    anchors,
                )
            )

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_csv, rows)
    write_markdown(args.out_md, args.artifact, args.config, rows)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")


def read_artifact(path: str) -> dict[tuple[str, str, str], dict]:
    cells: dict[tuple[str, str, str], dict] = defaultdict(dict)
    with tarfile.open(path, "r:gz") as tf:
        for name in tf.getnames():
            parts = PurePosixPath(name).parts
            if len(parts) < 5:
                continue
            task, language, slug = parts[1], parts[2], parts[3]
            key = (task, language, slug)
            if parts[-1] == "audit.jsonl":
                cells[key]["audit"] = read_jsonl(tf, name)
            elif parts[-1] == "gold.jsonl":
                cells[key]["gold"] = read_jsonl(tf, name)
            elif len(parts) >= 6 and parts[-2] == "selected" and parts[-1].endswith(".jsonl"):
                baseline = parts[-1].removesuffix(".jsonl")
                cells[key].setdefault("existing_selected_counts", {})[baseline] = len(read_jsonl(tf, name))
    return cells


def read_jsonl(tf: tarfile.TarFile, name: str) -> list[dict]:
    out = []
    f = tf.extractfile(name)
    if not f:
        return out
    for line in f:
        line = line.decode("utf-8").strip()
        if line:
            out.append(json.loads(line))
    return out


def existing_selection_budget(payload: dict) -> int | None:
    counts = payload.get("existing_selected_counts") or {}
    for baseline in ("naive", "cosda_equal_budget", "alpagasus_style"):
        value = counts.get(baseline)
        if value:
            return int(value)
    return None


def simulate_selection(
    rows: list[dict],
    gold: list[DataRecord],
    budget: int,
    baseline: str,
    audit_cfg,
    seed: int,
    max_per_source: int,
) -> list[dict]:
    if baseline == "naive":
        pool = rows[:]
        random.Random(seed).shuffle(pool)
        return pool[:budget]
    if baseline == "random":
        pool = rows[:]
        random.Random(seed + 7919).shuffle(pool)
        return pool[:budget]
    pool = _eligible_pool(rows, baseline, audit_cfg)
    ranked = sorted(pool, key=lambda row: _rank_score(row, baseline, audit_cfg), reverse=True)
    return _balanced_take(ranked, gold, budget, max_per_source=max_per_source)


def summarize_selection(
    task: str,
    language: str,
    slug: str,
    baseline: str,
    selected: list[dict],
    gold: list[DataRecord],
    budget: int,
    audit_cfg,
    anchors: dict[str, set[str]],
) -> dict:
    gold_counts = Counter(str(record.label) for record in gold if record.label is not None)
    selected_counts = Counter(str(row.get("label")) for row in selected if row.get("label") is not None)
    keys = {row_key(row) for row in selected}
    row = {
        "task": task,
        "language": language,
        "slug": slug,
        "baseline": baseline,
        "budget": budget,
        "selected_count": len(selected),
        "fill_rate": ratio(len(selected), budget),
        "label_tvd_vs_gold": label_tvd(selected_counts, gold_counts),
        "hard_reject_rate": mean([1.0 if item.get("hard_reject") else 0.0 for item in selected]),
        "strict_pass_rate": mean([1.0 if _passes_constraints(item, audit_cfg) else 0.0 for item in selected]),
        "proxy_score_mean": mean([_rank_score(item, "cosda_rerank_v2", audit_cfg) for item in selected]),
        "source_max": max(source_counts(selected).values() or [0]),
    }
    for key in SCORE_KEYS:
        row[f"score_{key}_mean"] = mean([float((item.get("scores") or {}).get(key, 0.0)) for item in selected])
    for key in JUDGE_KEYS:
        row[f"judge_{key}_mean"] = mean(
            [float((((item.get("metadata") or {}).get("llm_judge") or {}).get(key, 0.0))) for item in selected]
        )
    for anchor_name, anchor_keys in anchors.items():
        row[f"jaccard_vs_{anchor_name}"] = jaccard(keys, anchor_keys)
    return row


def row_key(row: dict) -> str:
    return json.dumps(
        {
            "label": row.get("label"),
            "text": row.get("text"),
            "source_seed_ids": row.get("source_seed_ids") or [],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def source_counts(rows: list[dict]) -> Counter:
    counts = Counter()
    for row in rows:
        for source_id in row.get("source_seed_ids") or []:
            counts[source_id] += 1
    return counts


def label_tvd(selected: Counter, gold: Counter) -> float:
    if not selected or not gold:
        return 0.0
    labels = set(selected) | set(gold)
    selected_total = sum(selected.values())
    gold_total = sum(gold.values())
    return 0.5 * sum(abs(selected[label] / selected_total - gold[label] / gold_total) for label in labels)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def ratio(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: str, artifact: str, config: str, rows: list[dict]) -> None:
    aggregate = aggregate_rows(rows)
    lines = [
        "# CoSDA Rerank V2 Local Proxy Report",
        "",
        f"Artifact: `{artifact}`",
        f"Config: `{config}`",
        "",
        "This is a local proxy check only. It does not claim downstream HF gains.",
        "",
        "## Aggregate Proxy Metrics",
        "",
        "| Baseline | n | Fill | Proxy | Label TVD | L | H | C | Judge Label | Judge Quality | Hard Reject | Jaccard vs naive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sorted(aggregate, key=lambda row: row["proxy_score_mean"], reverse=True):
        lines.append(
            "| {baseline} | {n} | {fill_rate:.3f} | {proxy_score_mean:.3f} | "
            "{label_tvd_vs_gold:.3f} | {score_L_mean:.3f} | {score_H_mean:.3f} | "
            "{score_C_mean:.3f} | {judge_label_correctness_mean:.3f} | "
            "{judge_quality_score_mean:.3f} | {hard_reject_rate:.3f} | "
            "{jaccard_vs_naive:.3f} |".format(**item)
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- Prefer selectors with fill near 1.0, low label TVD, low L/H, high C, and high judge label/quality.",
            "- `cosda_rerank_v2` is designed to preserve budget while still penalizing leakage, shortcut, and weak counterfactual validity.",
            "- Treat this as a go/no-go screen before GPU training, not as a paper result.",
            "",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["baseline"]].append(row)
    out = []
    numeric_keys = [
        "fill_rate",
        "label_tvd_vs_gold",
        "hard_reject_rate",
        "strict_pass_rate",
        "proxy_score_mean",
        "source_max",
        "jaccard_vs_naive",
        *[f"score_{key}_mean" for key in SCORE_KEYS],
        *[f"judge_{key}_mean" for key in JUDGE_KEYS],
    ]
    for baseline, items in grouped.items():
        row = {"baseline": baseline, "n": len(items)}
        for key in numeric_keys:
            row[key] = mean([float(item.get(key, 0.0)) for item in items])
        out.append(row)
    return out


if __name__ == "__main__":
    main()
