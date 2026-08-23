#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import tarfile
from pathlib import PurePosixPath


METRIC_KEYS = ("test_macro_f1", "eval_macro_f1", "macro_f1", "f1_macro")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find cells where two baselines selected identical examples but produced different HF metrics."
    )
    parser.add_argument("artifact", help="Essential artifact .tar.gz")
    parser.add_argument("--baseline-a", default="naive")
    parser.add_argument("--baseline-b", default="cosda_equal_budget")
    parser.add_argument("--tol", type=float, default=1e-9)
    args = parser.parse_args()

    selected, metrics = read_artifact(args.artifact, {args.baseline_a, args.baseline_b})
    flagged = []
    for task, language, slug in sorted({key[:3] for key in selected}):
        key_a = (task, language, slug, args.baseline_a)
        key_b = (task, language, slug, args.baseline_b)
        if key_a not in selected or key_b not in selected:
            continue
        multiset_a = selected[key_a]
        multiset_b = selected[key_b]
        set_a = set(multiset_a)
        set_b = set(multiset_b)
        set_union = set_a | set_b
        set_jaccard = len(set_a & set_b) / len(set_union) if set_union else 1.0
        intersection = sum((multiset_a & multiset_b).values())
        union = sum((multiset_a | multiset_b).values())
        multiset_jaccard = intersection / union if union else 1.0
        exact_multiset_equal = multiset_a == multiset_b
        metric_a = metrics.get(key_a)
        metric_b = metrics.get(key_b)
        delta = None if metric_a is None or metric_b is None else metric_b - metric_a
        row = {
            "task": task,
            "language": language,
            "slug": slug,
            "baseline_a": args.baseline_a,
            "baseline_b": args.baseline_b,
            "selected_a": sum(multiset_a.values()),
            "selected_b": sum(multiset_b.values()),
            "unique_a": len(set_a),
            "unique_b": len(set_b),
            "jaccard": multiset_jaccard,
            "set_jaccard": set_jaccard,
            "multiset_jaccard": multiset_jaccard,
            "exact_multiset_equal": exact_multiset_equal,
            "metric_a": metric_a,
            "metric_b": metric_b,
            "delta": delta,
            "flag": exact_multiset_equal and delta is not None and not math.isclose(delta, 0.0, abs_tol=args.tol),
        }
        flagged.append(row)

    print(json.dumps({"artifact": args.artifact, "rows": flagged}, indent=2, ensure_ascii=False))
    bad = [row for row in flagged if row["flag"]]
    if bad:
        raise SystemExit(2)


def read_artifact(path: str, baselines: set[str]) -> tuple[dict[tuple[str, str, str, str], Counter[str]], dict]:
    selected: dict[tuple[str, str, str, str], Counter[str]] = {}
    metrics = {}
    with tarfile.open(path, "r:gz") as tf:
        for member in tf.getmembers():
            parts = PurePosixPath(member.name).parts
            if len(parts) >= 6 and parts[-2] == "selected" and parts[-1].endswith(".jsonl"):
                baseline = parts[-1].removesuffix(".jsonl")
                if baseline not in baselines:
                    continue
                key = (parts[1], parts[2], parts[3], baseline)
                selected[key] = read_selected_payloads(tf, member)
            elif len(parts) >= 7 and parts[-3] == "hf" and parts[-1] == "test_metrics.json":
                baseline = parts[-2]
                if baseline not in baselines:
                    continue
                metrics[(parts[1], parts[2], parts[3], baseline)] = metric_value(json.load(tf.extractfile(member)))
    return selected, metrics


def read_selected_payloads(tf: tarfile.TarFile, member: tarfile.TarInfo) -> Counter[str]:
    payloads: Counter[str] = Counter()
    fileobj = tf.extractfile(member)
    if fileobj is None:
        return payloads
    for line in fileobj:
        row = json.loads(line)
        payloads[canonical_training_payload(row)] += 1
    return payloads


def canonical_training_payload(row: dict) -> str:
    raw_item = (row.get("metadata") or {}).get("raw_item") or {}
    return json.dumps(
        {
            "candidate_id": row.get("candidate_id", row.get("id", "")),
            "dataset_id": row.get("dataset_id"),
            "task": row.get("task"),
            "language": row.get("language"),
            "text": row.get("text", ""),
            "label": row.get("label"),
            "tokens": (row.get("metadata") or {}).get("tokens") or raw_item.get("tokens"),
            "tags": (row.get("metadata") or {}).get("tags") or raw_item.get("tags"),
            "summary": (row.get("metadata") or {}).get("summary") or raw_item.get("summary"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def metric_value(metrics: dict) -> float | None:
    for key in METRIC_KEYS:
        if key in metrics:
            return float(metrics[key])
    return None


if __name__ == "__main__":
    main()
