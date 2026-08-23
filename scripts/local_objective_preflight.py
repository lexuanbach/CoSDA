#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import tarfile
from collections import Counter, defaultdict
from pathlib import PurePosixPath


DEFAULT_THRESHOLDS = {"tau_l": 0.15, "tau_c": 0.60, "tau_h": 0.70}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local, dependency-light objective checks on a packaged CoSDA artifact."
    )
    parser.add_argument("--artifact", required=True, help="Path to an essential artifact .tar.gz")
    parser.add_argument("--out", help="Optional JSON report path")
    parser.add_argument("--tau-l", type=float, default=DEFAULT_THRESHOLDS["tau_l"])
    parser.add_argument("--tau-c", type=float, default=DEFAULT_THRESHOLDS["tau_c"])
    parser.add_argument("--tau-h", type=float, default=DEFAULT_THRESHOLDS["tau_h"])
    parser.add_argument("--relaxed-tau-c", type=float, default=0.45)
    args = parser.parse_args()

    report = analyze_artifact(args.artifact, args.tau_l, args.tau_c, args.tau_h, args.relaxed_tau_c)
    print_summary(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Wrote {args.out}")


def analyze_artifact(path: str, tau_l: float, tau_c: float, tau_h: float, relaxed_tau_c: float) -> dict:
    cells: dict[tuple[str, str, str], dict] = defaultdict(dict)
    with tarfile.open(path, "r:gz") as tf:
        for name in tf.getnames():
            parts = PurePosixPath(name).parts
            if len(parts) < 5:
                continue
            task, lang, slug = parts[1], parts[2], parts[3]
            key = (task, lang, slug)
            if parts[-1] == "audit.jsonl":
                cells[key]["audit"] = read_jsonl(tf, name)
            elif parts[-1] == "gold.jsonl":
                cells[key]["gold"] = read_jsonl(tf, name)
            elif len(parts) >= 6 and parts[-2] == "selected" and parts[-1].endswith(".jsonl"):
                baseline = parts[-1].removesuffix(".jsonl")
                cells[key].setdefault("selected_counts", {})[baseline] = len(read_jsonl(tf, name))
            elif len(parts) >= 7 and parts[-1] == "test_metrics.json" and parts[-3] == "hf":
                baseline = parts[-2]
                cells[key].setdefault("hf_metrics", {})[baseline] = json.load(tf.extractfile(name))

    cell_reports = []
    metric_rows = []
    for (task, lang, slug), payload in sorted(cells.items()):
        audit = payload.get("audit") or []
        gold = payload.get("gold") or []
        multiplier = parse_multiplier(slug)
        synthetic_budget = len(gold) * multiplier if gold else parse_budget(slug)
        gold_counts = Counter(str(row.get("label")) for row in gold if row.get("label") is not None)

        variants = {
            "cosda_hard": select_variant(audit, gold_counts, synthetic_budget, tau_l, tau_c, tau_h, "hard"),
            "cosda_relaxed": select_variant(
                audit, gold_counts, synthetic_budget, tau_l, min(tau_c, relaxed_tau_c), tau_h, "hard"
            ),
            "cosda_soft": select_variant(audit, gold_counts, synthetic_budget, tau_l, tau_c, tau_h, "soft"),
            "cosda_equal_budget": select_variant(audit, gold_counts, synthetic_budget, tau_l, tau_c, tau_h, "equal"),
        }

        hf = {
            baseline: metric_value(metrics)
            for baseline, metrics in (payload.get("hf_metrics") or {}).items()
            if metric_value(metrics) is not None
        }
        if hf:
            metric_rows.append({"task": task, "language": lang, "metrics": hf})

        hard_fill = ratio(len(variants["cosda_hard"]), synthetic_budget)
        gold_imbalance = ratio(max(gold_counts.values(), default=0), sum(gold_counts.values()))
        warnings = []
        if hard_fill < 0.5:
            warnings.append("hard_cosda_underfills_budget")
        if gold_imbalance > 0.6:
            warnings.append("gold_seed_label_imbalance")
        if hf:
            cosda = hf.get("cosda") or hf.get("cosda_hard")
            naive = hf.get("naive")
            if cosda is not None and naive is not None and cosda < naive:
                warnings.append("current_cosda_loses_to_naive")

        cell_reports.append(
            {
                "task": task,
                "language": lang,
                "slug": slug,
                "gold_count": len(gold),
                "synthetic_budget": synthetic_budget,
                "candidate_count": len(audit),
                "gold_label_counts": dict(gold_counts),
                "selected_counts_existing": payload.get("selected_counts") or {},
                "simulated_variant_counts": {name: len(rows) for name, rows in variants.items()},
                "simulated_variant_label_counts": {
                    name: dict(Counter(str(row.get("label")) for row in rows)) for name, rows in variants.items()
                },
                "hf_macro_f1": hf,
                "warnings": warnings,
            }
        )

    return {
        "artifact": path,
        "thresholds": {
            "tau_l": tau_l,
            "tau_c": tau_c,
            "tau_h": tau_h,
            "relaxed_tau_c": relaxed_tau_c,
        },
        "cells": cell_reports,
        "aggregate": aggregate(cell_reports, metric_rows),
    }


def read_jsonl(tf: tarfile.TarFile, name: str) -> list[dict]:
    rows = []
    f = tf.extractfile(name)
    if not f:
        return rows
    for line in f:
        line = line.decode("utf-8").strip()
        if line:
            rows.append(json.loads(line))
    return rows


def parse_multiplier(slug: str) -> int:
    match = re.search(r"_m(\d+)_", slug)
    return int(match.group(1)) if match else 1


def parse_budget(slug: str) -> int:
    match = re.search(r"b(\d+)_m(\d+)_", slug)
    if not match:
        return 0
    return int(match.group(1)) * int(match.group(2))


def select_variant(
    rows: list[dict],
    gold_counts: Counter,
    budget: int,
    tau_l: float,
    tau_c: float,
    tau_h: float,
    mode: str,
) -> list[dict]:
    if mode == "hard":
        pool = [row for row in rows if passes(row, tau_l, tau_c, tau_h, use_counterfactual=True)]
        ranked = sorted(pool, key=score_s, reverse=True)
    elif mode == "soft":
        pool = [row for row in rows if passes(row, tau_l, tau_c, tau_h, use_counterfactual=False)]
        ranked = sorted(pool, key=score_s, reverse=True)
    elif mode == "equal":
        ranked = sorted(rows, key=lambda row: (passes(row, tau_l, tau_c, tau_h, True), score_s(row)), reverse=True)
    else:
        raise ValueError(mode)
    return balanced_take(ranked, gold_counts, budget)


def passes(row: dict, tau_l: float, tau_c: float, tau_h: float, use_counterfactual: bool) -> bool:
    scores = row.get("scores") or {}
    leakage = float(scores.get("L", 0.0))
    shortcut = float(scores.get("H", 0.0))
    consistency = float(scores.get("C", 0.0))
    if leakage > tau_l or shortcut > tau_h:
        return False
    if use_counterfactual and consistency < tau_c:
        return False
    return True


def score_s(row: dict) -> float:
    return float((row.get("scores") or {}).get("S", 0.0))


def balanced_take(rows: list[dict], gold_counts: Counter, budget: int, max_per_source: int = 3) -> list[dict]:
    if budget <= 0:
        return []
    if not gold_counts:
        return source_capped(rows, budget, max_per_source)
    total_gold = sum(gold_counts.values())
    label_caps = {label: max(1, round(budget * count / total_gold)) for label, count in gold_counts.items()}
    diff = budget - sum(label_caps.values())
    for label, _ in gold_counts.most_common():
        if diff == 0:
            break
        label_caps[label] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1

    selected = []
    label_used = Counter()
    source_used = Counter()
    deferred = []
    for row in rows:
        label = str(row.get("label"))
        source_ids = row.get("source_seed_ids") or []
        if label_used[label] >= label_caps.get(label, budget):
            deferred.append(row)
            continue
        if any(source_used[src] >= max_per_source for src in source_ids):
            deferred.append(row)
            continue
        selected.append(row)
        label_used[label] += 1
        for src in source_ids:
            source_used[src] += 1
        if len(selected) >= budget:
            return selected
    if len(selected) < budget:
        selected.extend(source_capped(deferred, budget - len(selected), max_per_source, source_used))
    return selected[:budget]


def source_capped(rows: list[dict], budget: int, max_per_source: int, source_used: Counter | None = None) -> list[dict]:
    source_used = source_used or Counter()
    selected = []
    for row in rows:
        source_ids = row.get("source_seed_ids") or []
        if any(source_used[src] >= max_per_source for src in source_ids):
            continue
        selected.append(row)
        for src in source_ids:
            source_used[src] += 1
        if len(selected) >= budget:
            break
    return selected


def metric_value(metrics: dict) -> float | None:
    for key in ("test_macro_f1", "eval_macro_f1", "macro_f1", "f1_macro"):
        if key in metrics:
            return float(metrics[key])
    return None


def aggregate(cells: list[dict], metric_rows: list[dict]) -> dict:
    variant_counts = defaultdict(list)
    for cell in cells:
        budget = cell["synthetic_budget"]
        for name, count in cell["simulated_variant_counts"].items():
            variant_counts[name].append(ratio(count, budget))

    metric_summary = {}
    wins_vs_naive = {"wins": 0, "ties": 0, "losses": 0, "missing": 0}
    for row in metric_rows:
        metrics = row["metrics"]
        for baseline, value in metrics.items():
            metric_summary.setdefault(baseline, []).append(value)
        cosda = metrics.get("cosda") or metrics.get("cosda_hard")
        naive = metrics.get("naive")
        if cosda is None or naive is None:
            wins_vs_naive["missing"] += 1
        elif abs(cosda - naive) < 1e-12:
            wins_vs_naive["ties"] += 1
        elif cosda > naive:
            wins_vs_naive["wins"] += 1
        else:
            wins_vs_naive["losses"] += 1

    return {
        "cell_count": len(cells),
        "variant_fill_rate_mean": {k: statistics.mean(v) for k, v in variant_counts.items() if v},
        "variant_fill_rate_min": {k: min(v) for k, v in variant_counts.items() if v},
        "hf_macro_f1_mean": {k: statistics.mean(v) for k, v in sorted(metric_summary.items()) if v},
        "current_cosda_vs_naive": wins_vs_naive,
        "claim_guard_ready": wins_vs_naive["losses"] == 0 and wins_vs_naive["wins"] > 0,
        "warning_counts": dict(Counter(w for cell in cells for w in cell["warnings"])),
    }


def ratio(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def print_summary(report: dict) -> None:
    agg = report["aggregate"]
    print(f"Cells analyzed: {agg['cell_count']}")
    print("Current CoSDA vs naive:", agg["current_cosda_vs_naive"])
    print("Claim guard ready:", agg["claim_guard_ready"])
    print("Warning counts:", agg["warning_counts"])
    print("Mean fill rates:")
    for name, value in sorted(agg["variant_fill_rate_mean"].items()):
        print(f"  {name:20s} {value:.1%}")
    if agg["hf_macro_f1_mean"]:
        print("Mean HF macro-F1:")
        for name, value in sorted(agg["hf_macro_f1_mean"].items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {name:20s} {value:.4f}")


if __name__ == "__main__":
    main()
