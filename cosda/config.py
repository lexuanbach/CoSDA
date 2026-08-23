from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AuditConfig:
    tau_l: float = 0.15
    tau_c: float = 0.60
    tau_h: float = 0.70
    alpha_u: float = 1.00
    alpha_c: float = 0.75
    alpha_d: float = 0.50
    alpha_l: float = 1.00
    alpha_h: float = 0.75
    ngram_n: int = 10
    minhash_n: int = 5
    minhash_threshold: float = 0.85
    embedding_leakage_threshold: float = 0.85
    embedder: str = "hash"
    embedding_dim: int = 768
    teacher: str = "centroid"
    neighbor_k: int = 5


@dataclass
class GenerationConfig:
    backend: str = "heuristic"
    model: str = "CohereLabs/aya-23-8B"
    endpoint_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 256
    candidates_per_seed: int = 1
    max_workers: int = 1
    request_timeout: int = 180
    request_retries: int = 3


@dataclass
class JudgeConfig:
    backend: str = "disabled"
    model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    endpoint_url: str | None = None
    api_key_env: str | None = None
    aws_region: str = "us-east-1"
    temperature: float = 0.0
    max_tokens: int = 512
    max_workers: int = 1
    request_timeout: int = 180
    request_retries: int = 3


@dataclass
class RunConfig:
    manifest_path: str = "data/manifest/datasets.json"
    output_dir: str = "runs"
    datasets: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    budgets: list[int] = field(default_factory=lambda: [64])
    multipliers: list[int] = field(default_factory=lambda: [3])
    selection_multiplier: int | None = None
    seeds: list[int] = field(default_factory=lambda: [13])
    baselines: list[str] = field(default_factory=lambda: ["cosda"])
    sample_per_class_when_possible: bool = False
    seed_sampling: str | None = None
    audit: AuditConfig = field(default_factory=AuditConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)


def _merge_dataclass(cls, data: dict[str, Any]):
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in data.items() if k in known})


def load_run_config(path: str | Path) -> RunConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    audit = _merge_dataclass(AuditConfig, data.get("audit", {}))
    generation = _merge_dataclass(GenerationConfig, data.get("generation", {}))
    judge = _merge_dataclass(JudgeConfig, data.get("judge", {}))
    known = {f.name for f in RunConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    values = {k: v for k, v in data.items() if k in known and k not in {"audit", "generation", "judge"}}
    values["audit"] = audit
    values["generation"] = generation
    values["judge"] = judge
    return RunConfig(**values)
