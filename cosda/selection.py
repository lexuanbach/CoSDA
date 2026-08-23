from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

from .config import AuditConfig
from .io_utils import iter_jsonl, write_jsonl
from .types import DataRecord


BASELINES = {
    "cosda_hard",
    "cosda",
    "cosda_relaxed",
    "cosda_soft",
    "cosda_equal_budget",
    "cosda_rerank_v2",
    "naive",
    "random",
    "quality_only",
    "diversity_only",
    "deita_style",
    "alpagasus_style",
    "alpagasus_hardreject",
    "less_style",
}


def select_candidates(
    audit_path: str | Path,
    gold_path: str | Path,
    output_path: str | Path,
    budget: int,
    baseline: str,
    cfg: AuditConfig,
    seed: int = 13,
    max_per_source: int = 3,
) -> list[dict]:
    if baseline not in BASELINES:
        raise ValueError(f"Unknown selection baseline: {baseline}")
    rows = list(iter_jsonl(audit_path))
    gold = [DataRecord.from_json(row) for row in iter_jsonl(gold_path)]
    rng = random.Random(seed)
    if baseline == "naive":
        pool = rows[:]
        rng.shuffle(pool)
        selected = pool[:budget]
    elif baseline == "random":
        pool = rows[:]
        random.Random(seed + 7919).shuffle(pool)
        selected = pool[:budget]
    else:
        pool = _eligible_pool(rows, baseline, cfg)
        ranked = sorted(pool, key=lambda row: _rank_score(row, baseline, cfg), reverse=True)
        selected = _balanced_take(ranked, gold, budget, max_per_source=max_per_source)
    write_jsonl(output_path, selected)
    return selected


def _eligible_pool(rows: list[dict], baseline: str, cfg: AuditConfig) -> list[dict]:
    if baseline in {"cosda", "cosda_hard", "alpagasus_hardreject"}:
        # alpagasus_hardreject is the stress-test selector: apply the audit's strict
        # gate, then rank the survivors by the AlpaGasus judged-quality score. Note
        # that cosda_rerank_v2 does NOT use this gate (see _eligible_pool below and
        # _cosda_rerank_v2_score), so this measures the cost of a gate V2 omits
        # rather than removing a step from V2.
        return [row for row in rows if _passes_constraints(row, cfg, tau_c=cfg.tau_c)]
    if baseline == "cosda_relaxed":
        return [row for row in rows if _passes_constraints(row, cfg, tau_c=min(cfg.tau_c, 0.45))]
    if baseline == "cosda_soft":
        return [row for row in rows if _passes_constraints(row, cfg, use_counterfactual=False)]
    if baseline == "cosda_equal_budget":
        return rows
    if baseline == "cosda_rerank_v2":
        return rows
    if baseline in {"quality_only", "diversity_only", "deita_style", "alpagasus_style", "less_style"}:
        return rows
    return rows


def _rank_score(row: dict, baseline: str, cfg: AuditConfig) -> float:
    s = row.get("scores", {})
    if baseline in {"cosda", "cosda_hard", "cosda_relaxed", "cosda_soft"}:
        return float(s.get("S", 0.0))
    if baseline == "cosda_equal_budget":
        # Fill the requested training budget for a fair data-volume comparison:
        # accepted CoSDA rows first, then the least-risky rejected rows as backfill.
        eligible_bonus = 100.0 if not row.get("hard_reject") else 0.0
        return eligible_bonus + float(s.get("S", 0.0))
    if baseline == "cosda_rerank_v2":
        return _cosda_rerank_v2_score(row, cfg)
    if baseline == "quality_only":
        return float(s.get("U", 0.0))
    if baseline == "diversity_only":
        return float(s.get("D", 0.0))
    if baseline in {"alpagasus_style", "alpagasus_hardreject"}:
        judge = (row.get("metadata", {}).get("llm_judge") or {})
        return float(row.get("metadata", {}).get("judge_score", judge.get("quality_score", s.get("U", 0.0))))
    if baseline == "deita_style":
        complexity = min(1.0, len(row.get("text", "").split()) / 256.0)
        judge = (row.get("metadata", {}).get("llm_judge") or {})
        quality = float(judge.get("quality_score", s.get("U", 0.0)))
        return 0.45 * quality + 0.35 * float(s.get("D", 0.0)) + 0.20 * complexity
    if baseline == "less_style":
        return float(row.get("metadata", {}).get("influence_score", s.get("U", 0.0) + 0.25 * s.get("D", 0.0)))
    return float(s.get("S", 0.0))


def _cosda_rerank_v2_score(row: dict, cfg: AuditConfig) -> float:
    """Budget-preserving CoSDA score.

    The first CoSDA variants treated counterfactual audit mainly as a hard
    rejector. That preserved precision but often destroyed coverage. This score
    keeps all rows eligible, rewards LLM-judge quality signals, and applies
    graded penalties for leakage, shortcuts, and weak counterfactual validity.
    """
    s = row.get("scores") or {}
    judge = ((row.get("metadata") or {}).get("llm_judge") or {})
    u = _bounded(s.get("U", judge.get("quality_score", 0.0)))
    d = _bounded(s.get("D", 0.0))
    c = _bounded(s.get("C", judge.get("counterfactual_validity", 0.0)))
    l = _bounded(s.get("L", judge.get("leakage_suspicion", 0.0)))
    h = _bounded(s.get("H", 0.0))
    label_correctness = _bounded(judge.get("label_correctness", u))
    quality = _bounded(judge.get("quality_score", u))
    fluency = _bounded(judge.get("fluency", quality))
    cultural = _bounded(judge.get("cultural_plausibility", quality))
    leakage_suspicion = _bounded(judge.get("leakage_suspicion", l))

    quality_block = (
        1.25 * label_correctness
        + 0.85 * quality
        + 0.45 * fluency
        + 0.35 * cultural
        + 0.35 * u
    )
    audit_block = 0.65 * c + 0.45 * d
    risk_penalty = (
        1.40 * l
        + 1.00 * h
        + 1.10 * leakage_suspicion
        + 0.55 * max(0.0, cfg.tau_c - c)
        + 0.45 * max(0.0, l - cfg.tau_l)
        + 0.35 * max(0.0, h - cfg.tau_h)
    )
    strict_bonus = 0.25 if _passes_constraints(row, cfg, tau_c=cfg.tau_c) else 0.0
    relaxed_bonus = 0.10 if _passes_constraints(row, cfg, tau_c=min(cfg.tau_c, 0.45)) else 0.0
    return quality_block + audit_block + strict_bonus + relaxed_bonus - risk_penalty


def _bounded(value: object, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = default
    return max(0.0, min(1.0, x))


def _passes_constraints(
    row: dict,
    cfg: AuditConfig,
    tau_c: float | None = None,
    use_counterfactual: bool = True,
) -> bool:
    scores = row.get("scores") or {}
    leakage = float(scores.get("L", 0.0))
    shortcut = float(scores.get("H", 0.0))
    consistency = float(scores.get("C", 0.0))
    if leakage > cfg.tau_l or shortcut > cfg.tau_h:
        return False
    if use_counterfactual and consistency < (cfg.tau_c if tau_c is None else tau_c):
        return False
    return True


def _balanced_take(rows: list[dict], gold: list[DataRecord], budget: int, max_per_source: int) -> list[dict]:
    labels = [str(row.get("label")) for row in rows if row.get("label") is not None]
    if not labels or len(set(labels)) <= 1:
        return _source_capped(rows, budget, max_per_source)
    gold_counts = Counter(str(record.label) for record in gold if record.label is not None)
    if not gold_counts:
        return _source_capped(rows, budget, max_per_source)
    total_gold = sum(gold_counts.values())
    label_caps = {
        label: max(1, round(budget * count / total_gold))
        for label, count in gold_counts.items()
    }
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
        selected.extend(_source_capped(deferred, budget - len(selected), max_per_source, source_used))
    return selected[:budget]


def _source_capped(rows: list[dict], budget: int, max_per_source: int, source_used: Counter | None = None) -> list[dict]:
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
