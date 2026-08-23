from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .data import CLASSIFICATION_TASKS, dataset_entry, iter_records, load_manifest
from .embeddings import cosine_matrix, make_embedder
from .io_utils import iter_jsonl, write_json
from .metrics import accuracy, macro_f1, micro_f1_tags, expected_calibration_error
from .types import DataRecord


class CentroidClassifier:
    def __init__(self, embedder_name: str = "hash"):
        self.embedder = make_embedder(embedder_name)
        self.labels: list[str] = []
        self.centroids = None

    def fit(self, records: list[DataRecord]) -> None:
        import numpy as np

        labelled = [r for r in records if r.label is not None]
        self.labels = sorted({str(r.label) for r in labelled})
        embeddings = self.embedder.encode([r.text for r in labelled])
        by_label = defaultdict(list)
        for emb, record in zip(embeddings, labelled):
            by_label[str(record.label)].append(emb)
        self.centroids = np.vstack([np.mean(by_label[label], axis=0) for label in self.labels])

    def predict_proba(self, records: list[DataRecord]) -> list[dict[str, float]]:
        import numpy as np

        if self.centroids is None or not self.labels:
            return [{} for _ in records]
        emb = self.embedder.encode([r.text for r in records])
        sims = cosine_matrix(emb, self.centroids)
        rows = []
        for row in sims:
            row = row - np.max(row)
            exp = np.exp(row)
            probs = exp / max(float(exp.sum()), 1e-12)
            rows.append({label: float(prob) for label, prob in zip(self.labels, probs)})
        return rows

    def predict(self, records: list[DataRecord]) -> list[str]:
        return [max(p, key=p.get) if p else "" for p in self.predict_proba(records)]


def load_selected_as_records(path: str | Path, split: str = "synthetic") -> list[DataRecord]:
    records = []
    for row in iter_jsonl(path):
        raw_item = (row.get("metadata") or {}).get("raw_item") or {}
        metadata = {"synthetic": True, "scores": row.get("scores", {}), **(row.get("metadata") or {})}
        records.append(
            DataRecord(
                id=row.get("candidate_id", row.get("id", "")),
                dataset_id=row["dataset_id"],
                task=row["task"],
                language=row["language"],
                split=split,
                text=row.get("text", ""),
                label=row.get("label"),
                tokens=(row.get("metadata") or {}).get("tokens") or raw_item.get("tokens"),
                tags=(row.get("metadata") or {}).get("tags") or raw_item.get("tags"),
                summary=(row.get("metadata") or {}).get("summary") or raw_item.get("summary"),
                metadata=metadata,
            )
        )
    return records


def evaluate_run(
    manifest_path: str,
    dataset_id: str,
    language: str,
    gold_path: str | Path,
    selected_path: str | Path | None,
    output_path: str | Path,
    embedder: str = "hash",
) -> dict:
    manifest = load_manifest(manifest_path)
    entry = dataset_entry(manifest, dataset_id=dataset_id)
    task = entry["task"]
    gold = [DataRecord.from_json(row) for row in iter_jsonl(gold_path)]
    synthetic = load_selected_as_records(selected_path) if selected_path else []
    train = gold + synthetic
    test = list(iter_records(manifest, dataset_id, language, "test"))

    if task in CLASSIFICATION_TASKS:
        clf = CentroidClassifier(embedder)
        clf.fit(train)
        pred = clf.predict(test)
        probs = clf.predict_proba(test)
        true = [str(r.label) for r in test]
        metrics = {
            "macro_f1": macro_f1(true, pred),
            "accuracy": accuracy(true, pred),
            "ece": expected_calibration_error(true, probs),
        }
    elif task == "named_entity_recognition":
        metrics = _evaluate_ner_memorization(train, test)
    elif task == "summarization":
        metrics = _evaluate_summarization_retrieval(train, test, embedder)
    else:
        raise ValueError(f"Unsupported task: {task}")

    result = {
        "dataset_id": dataset_id,
        "task": task,
        "language": language,
        "train_gold": len(gold),
        "train_synthetic": len(synthetic),
        "test_size": len(test),
        "evaluator": f"lightweight_{embedder}",
        "metrics": metrics,
    }
    write_json(output_path, result)
    return result


def _evaluate_ner_memorization(train: list[DataRecord], test: list[DataRecord]) -> dict:
    tag_by_token = defaultdict(Counter)
    for record in train:
        for token, tag in zip(record.tokens or record.text.split(), record.tags or []):
            tag_by_token[token.lower()][tag] += 1
    pred_tags = []
    gold_tags = []
    for record in test:
        pred = []
        for token in record.tokens or record.text.split():
            counts = tag_by_token[token.lower()]
            pred.append(counts.most_common(1)[0][0] if counts else "O")
        pred_tags.append(pred)
        gold_tags.append(record.tags or ["O"] * len(pred))
    return {"entity_micro_f1": micro_f1_tags(gold_tags, pred_tags)}


def _evaluate_summarization_retrieval(train: list[DataRecord], test: list[DataRecord], embedder_name: str) -> dict:
    embedder = make_embedder(embedder_name)
    train_texts = [r.text for r in train]
    train_summaries = [r.summary or r.metadata.get("summary") or "" for r in train]
    if not train_texts:
        return {"retrieval_overlap": 0.0}
    train_emb = embedder.encode(train_texts)
    test_emb = embedder.encode([r.text for r in test])
    sims = cosine_matrix(test_emb, train_emb)
    scores = []
    for row, record in zip(sims, test):
        idx = int(row.argmax())
        pred = set(train_summaries[idx].split())
        gold = set((record.summary or "").split())
        scores.append(len(pred & gold) / max(len(pred | gold), 1))
    return {"summary_jaccard": sum(scores) / max(len(scores), 1)}
