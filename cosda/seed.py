from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from .data import CLASSIFICATION_TASKS, iter_records, load_manifest, dataset_entry
from .io_utils import write_jsonl
from .types import DataRecord


def sample_gold(
    manifest_path: str,
    dataset_id: str,
    language: str,
    budget: int,
    seed: int,
    output_path: str | Path,
    per_class_when_possible: bool = True,
    sampling_strategy: str | None = None,
) -> list[DataRecord]:
    manifest = load_manifest(manifest_path)
    entry = dataset_entry(manifest, dataset_id=dataset_id)
    records = list(iter_records(manifest, dataset_id, language, "train"))
    rng = random.Random(seed)
    strategy = sampling_strategy or ("per_class" if per_class_when_possible else "random")
    if strategy not in {"random", "balanced", "per_class"}:
        raise ValueError(f"Unknown seed sampling strategy: {strategy}")
    if entry["task"] in CLASSIFICATION_TASKS and strategy == "per_class":
        selected = _sample_per_class_classification(records, budget, rng)
    elif entry["task"] in CLASSIFICATION_TASKS and strategy == "balanced":
        selected = _sample_balanced_classification(records, budget, rng)
    else:
        selected = _sample_flat(records, budget, rng)
    write_jsonl(output_path, [record.to_json() for record in selected])
    return selected


def _sample_flat(records: list[DataRecord], budget: int, rng: random.Random) -> list[DataRecord]:
    shuffled = records[:]
    rng.shuffle(shuffled)
    return shuffled[: min(budget, len(shuffled))]


def _sample_per_class_classification(records: list[DataRecord], budget: int, rng: random.Random) -> list[DataRecord]:
    by_label: dict[str, list[DataRecord]] = defaultdict(list)
    for record in records:
        by_label[str(record.label)].append(record)
    if by_label and all(len(items) >= budget for items in by_label.values()):
        selected: list[DataRecord] = []
        for label in sorted(by_label):
            items = by_label[label][:]
            rng.shuffle(items)
            selected.extend(items[:budget])
        rng.shuffle(selected)
        return selected

    # Fallback: stratified per-cell budget, preserving the observed label prior.
    total = len(records)
    selected = []
    remaining = budget
    labels = sorted(by_label)
    for i, label in enumerate(labels):
        items = by_label[label][:]
        rng.shuffle(items)
        if i == len(labels) - 1:
            take = remaining
        else:
            take = round(budget * len(items) / total)
            take = max(1, min(take, remaining))
        selected.extend(items[: min(take, len(items))])
        remaining = max(0, budget - len(selected))
    if len(selected) < budget:
        chosen = {x.id for x in selected}
        rest = [x for x in records if x.id not in chosen]
        rng.shuffle(rest)
        selected.extend(rest[: budget - len(selected)])
    rng.shuffle(selected)
    return selected[:budget]


def _sample_balanced_classification(records: list[DataRecord], budget: int, rng: random.Random) -> list[DataRecord]:
    by_label: dict[str, list[DataRecord]] = defaultdict(list)
    for record in records:
        by_label[str(record.label)].append(record)
    labels = sorted(by_label)
    if not labels:
        return _sample_flat(records, budget, rng)

    shuffled_by_label = {}
    for label in labels:
        items = by_label[label][:]
        rng.shuffle(items)
        shuffled_by_label[label] = items

    selected: list[DataRecord] = []
    while len(selected) < min(budget, len(records)):
        progressed = False
        for label in labels:
            if len(selected) >= budget:
                break
            bucket = shuffled_by_label[label]
            if not bucket:
                continue
            selected.append(bucket.pop())
            progressed = True
        if not progressed:
            break
    rng.shuffle(selected)
    return selected[:budget]
