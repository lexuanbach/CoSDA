from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataRecord:
    id: str
    dataset_id: str
    task: str
    language: str
    split: str
    text: str
    label: str | None = None
    tokens: list[str] | None = None
    tags: list[str] | None = None
    summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "task": self.task,
            "language": self.language,
            "split": self.split,
            "text": self.text,
            "label": self.label,
            "tokens": self.tokens,
            "tags": self.tags,
            "summary": self.summary,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "DataRecord":
        return cls(
            id=str(row["id"]),
            dataset_id=row["dataset_id"],
            task=row["task"],
            language=row["language"],
            split=row["split"],
            text=row.get("text") or "",
            label=row.get("label"),
            tokens=row.get("tokens"),
            tags=row.get("tags"),
            summary=row.get("summary"),
            metadata=row.get("metadata") or {},
        )


@dataclass
class Candidate:
    candidate_id: str
    dataset_id: str
    task: str
    language: str
    text: str
    label: str | None
    source_seed_ids: list[str]
    generator: str
    generator_confidence: float
    prompt_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    counterfactual: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "dataset_id": self.dataset_id,
            "task": self.task,
            "language": self.language,
            "text": self.text,
            "label": self.label,
            "source_seed_ids": self.source_seed_ids,
            "generator": self.generator,
            "generator_confidence": self.generator_confidence,
            "prompt_hash": self.prompt_hash,
            "metadata": self.metadata,
            "counterfactual": self.counterfactual,
        }

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "Candidate":
        return cls(
            candidate_id=str(row["candidate_id"]),
            dataset_id=row["dataset_id"],
            task=row["task"],
            language=row["language"],
            text=row.get("text") or "",
            label=row.get("label"),
            source_seed_ids=[str(x) for x in row.get("source_seed_ids", [])],
            generator=row.get("generator", "unknown"),
            generator_confidence=float(row.get("generator_confidence", 0.5)),
            prompt_hash=row.get("prompt_hash", ""),
            metadata=row.get("metadata") or {},
            counterfactual=row.get("counterfactual"),
        )
