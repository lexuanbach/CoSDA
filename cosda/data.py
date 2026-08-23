from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator

from .io_utils import read_json, repo_root
from .types import DataRecord


CLASSIFICATION_TASKS = {"news_topic_classification", "sentiment_classification", "intent_slot_filling"}


def load_manifest(path: str | Path = "data/manifest/datasets.json") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    return read_json(p)


def dataset_entry(manifest: dict, dataset_id: str | None = None, task: str | None = None) -> dict:
    for entry in manifest["datasets"]:
        if dataset_id and entry["dataset_id"] != dataset_id:
            continue
        if task and entry["task"] != task:
            continue
        return entry
    raise KeyError(f"No dataset entry for dataset_id={dataset_id!r}, task={task!r}")


def labels_for_cell(manifest: dict, dataset_id: str, language: str) -> list[str]:
    entry = dataset_entry(manifest, dataset_id=dataset_id)
    labels = entry["cells"][language]["train"].get("labels") or {}
    return sorted(labels)


def iter_records(
    manifest: dict,
    dataset_id: str,
    language: str,
    split: str,
    root: Path | None = None,
) -> Iterator[DataRecord]:
    root = root or repo_root()
    entry = dataset_entry(manifest, dataset_id=dataset_id)
    task = entry["task"]
    cell = entry["cells"][language][split]
    path = root / cell["path"]
    if task == "news_topic_classification":
        yield from _read_news(path, dataset_id, task, language, split)
    elif task == "sentiment_classification":
        yield from _read_sentiment(path, dataset_id, task, language, split)
    elif task == "named_entity_recognition":
        yield from _read_ner(path, dataset_id, task, language, split)
    elif task == "intent_slot_filling":
        yield from _read_massive(path, dataset_id, task, language, split)
    elif task == "summarization":
        yield from _read_xlsum(path, dataset_id, task, language, split)
    else:
        raise ValueError(f"Unsupported task: {task}")


def _read_news(path: Path, dataset_id: str, task: str, language: str, split: str) -> Iterator[DataRecord]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for i, row in enumerate(reader):
            headline = row.get("headline", "")
            body = row.get("text", "")
            text = f"{headline}\n\n{body}".strip()
            yield DataRecord(
                id=f"{dataset_id}:{language}:{split}:{i}",
                dataset_id=dataset_id,
                task=task,
                language=language,
                split=split,
                text=text,
                label=row.get("category"),
                metadata={"url": row.get("url")},
            )


def _read_sentiment(path: Path, dataset_id: str, task: str, language: str, split: str) -> Iterator[DataRecord]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for i, row in enumerate(reader):
            yield DataRecord(
                id=f"{dataset_id}:{language}:{split}:{i}",
                dataset_id=dataset_id,
                task=task,
                language=language,
                split=split,
                text=row.get("tweet", ""),
                label=row.get("label"),
            )


def _read_ner(path: Path, dataset_id: str, task: str, language: str, split: str) -> Iterator[DataRecord]:
    sent_id = 0
    tokens: list[str] = []
    tags: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                if tokens:
                    yield DataRecord(
                        id=f"{dataset_id}:{language}:{split}:{sent_id}",
                        dataset_id=dataset_id,
                        task=task,
                        language=language,
                        split=split,
                        text=" ".join(tokens),
                        label="ner",
                        tokens=tokens,
                        tags=tags,
                    )
                    sent_id += 1
                    tokens, tags = [], []
                continue
            parts = line.split()
            if len(parts) >= 2:
                tokens.append(parts[0])
                tags.append(parts[-1])
    if tokens:
        yield DataRecord(
            id=f"{dataset_id}:{language}:{split}:{sent_id}",
            dataset_id=dataset_id,
            task=task,
            language=language,
            split=split,
            text=" ".join(tokens),
            label="ner",
            tokens=tokens,
            tags=tags,
        )


def _read_massive(path: Path, dataset_id: str, task: str, language: str, split: str) -> Iterator[DataRecord]:
    partition = "dev" if split == "dev" else split
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("partition") != partition:
                continue
            yield DataRecord(
                id=f"{dataset_id}:{language}:{split}:{row['id']}",
                dataset_id=dataset_id,
                task=task,
                language=language,
                split=split,
                text=row.get("utt", ""),
                label=row.get("intent"),
                metadata={
                    "locale": row.get("locale"),
                    "scenario": row.get("scenario"),
                    "annot_utt": row.get("annot_utt"),
                    "slot_method": row.get("slot_method", []),
                },
            )


def _read_xlsum(path: Path, dataset_id: str, task: str, language: str, split: str) -> Iterator[DataRecord]:
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            article_id = row.get("id", i)
            yield DataRecord(
                id=f"{dataset_id}:{language}:{split}:{article_id}",
                dataset_id=dataset_id,
                task=task,
                language=language,
                split=split,
                text=row.get("text", ""),
                label="summary",
                summary=row.get("summary"),
                metadata={"title": row.get("title"), "url": row.get("url")},
            )
