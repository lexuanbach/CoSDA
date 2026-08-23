from __future__ import annotations

import os
import random
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .config import GenerationConfig
from .data import labels_for_cell, load_manifest
from .http_utils import post_json_with_retries
from .io_utils import iter_jsonl, stable_hash, write_jsonl
from .text_utils import compact_text
from .types import Candidate


class CounterfactualBuilder:
    def build(self, candidate: Candidate, labels: list[str]) -> dict:
        raise NotImplementedError


class HeuristicCounterfactualBuilder(CounterfactualBuilder):
    def __init__(self, seed: int = 13):
        self.rng = random.Random(seed)

    def build(self, candidate: Candidate, labels: list[str]) -> dict:
        if labels and candidate.label in labels and len(labels) > 1:
            current = labels.index(candidate.label)
            new_label = labels[(current + 1) % len(labels)]
        elif labels:
            new_label = labels[0]
        else:
            new_label = candidate.label
        if candidate.task == "summarization":
            cf_text = candidate.text + "\n[COUNTERFACTUAL: salient fact changed]"
        elif candidate.task == "named_entity_recognition":
            cf_text = candidate.text + " CF"
        else:
            cf_text = f"{candidate.text}\nCounterfactual label cue: {new_label}."
        return {
            "text": cf_text,
            "label": new_label,
            "semantic_edit_score": 0.65 if new_label != candidate.label else 0.40,
            "method": "heuristic",
            "counterfactual_id": stable_hash(f"{candidate.candidate_id}:{cf_text}:{new_label}"),
        }


class EndpointCounterfactualBuilder(CounterfactualBuilder):
    def __init__(self, config: GenerationConfig):
        if not config.endpoint_url:
            raise ValueError("endpoint_url is required for endpoint counterfactual generation")
        self.config = config

    def build(self, candidate: Candidate, labels: list[str]) -> dict:
        text = compact_text(candidate.text, 2200)
        prompt = (
            f"Task: {candidate.task}\nLanguage: {candidate.language}\nAllowed labels: {labels}\n"
            f"Original label: {candidate.label}\nOriginal text excerpt:\n{text}\n\n"
            "Create a minimal counterfactual that changes one task-relevant variable and returns strict JSON: "
            "{\"text\": str, \"label\": str, \"semantic_edit_score\": float}."
        )
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "You create strict JSON counterfactual NLP examples."},
                {"role": "user", "content": prompt},
            ],
            "temperature": min(self.config.temperature, 0.4),
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_new_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env and os.getenv(self.config.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[self.config.api_key_env]}"
        from .judge import parse_json_object

        attempts = max(1, self.config.request_retries)
        last_error: Exception | None = None
        parsed = None
        raw_text = ""
        for attempt in range(attempts):
            try:
                data = post_json_with_retries(
                    self.config.endpoint_url,
                    payload,
                    headers=headers,
                    timeout=self.config.request_timeout,
                    retries=self.config.request_retries,
                )
                raw_text = data["choices"][0]["message"]["content"]
                parsed = parse_json_object(raw_text)
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(min(8, 2**attempt))
        if parsed is None:
            raise RuntimeError(
                f"Failed to parse counterfactual JSON for candidate {candidate.candidate_id}: {last_error}"
            ) from last_error
        return {
            "text": parsed.get("text", ""),
            "label": str(parsed.get("label", candidate.label)),
            "semantic_edit_score": float(parsed.get("semantic_edit_score", 0.5)),
            "method": self.config.model,
            "counterfactual_id": stable_hash(f"{candidate.candidate_id}:{raw_text}"),
        }


def make_counterfactual_builder(config: GenerationConfig, seed: int = 13) -> CounterfactualBuilder:
    if config.backend == "heuristic":
        return HeuristicCounterfactualBuilder(seed=seed)
    return EndpointCounterfactualBuilder(config)


def build_counterfactuals(
    manifest_path: str,
    candidates_path: str | Path,
    output_path: str | Path,
    config: GenerationConfig,
    random_seed: int,
    resume: bool = False,
) -> list[Candidate]:
    manifest = load_manifest(manifest_path)
    builder = make_counterfactual_builder(config, seed=random_seed)
    candidates = [Candidate.from_json(row) for row in iter_jsonl(candidates_path)]
    existing: list[Candidate] = []
    done = set()
    if resume and Path(output_path).exists():
        existing = [Candidate.from_json(row) for row in iter_jsonl(output_path)]
        done = {candidate.candidate_id for candidate in existing}
    rows = existing[:]
    pending = [candidate for candidate in candidates if candidate.candidate_id not in done]

    def one(candidate: Candidate) -> Candidate:
        labels = labels_for_cell(manifest, candidate.dataset_id, candidate.language)
        try:
            candidate.counterfactual = builder.build(candidate, labels)
        except Exception as exc:
            candidate.metadata = candidate.metadata or {}
            candidate.metadata["counterfactual_failed"] = {
                "error": str(exc)[:500],
                "policy": "retained_for_audit_with_zero_counterfactual_score",
            }
            candidate.counterfactual = failed_counterfactual(candidate, config, exc)
        return candidate

    if config.max_workers > 1 and len(pending) > 1:
        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            futures = {pool.submit(one, candidate): candidate.candidate_id for candidate in pending}
            for future in as_completed(futures):
                rows.append(future.result())
        failures = [row for row in rows if (row.metadata or {}).get("counterfactual_failed")]
        if len(failures) > max(5, int(0.10 * max(len(rows), 1))):
            rows.sort(key=lambda c: (c.source_seed_ids[0] if c.source_seed_ids else "", c.candidate_id))
            write_jsonl(output_path, [row.to_json() for row in rows])
            detail = "; ".join(
                f"{row.candidate_id}: {((row.metadata or {}).get('counterfactual_failed') or {}).get('error')}"
                for row in failures[:5]
            )
            raise RuntimeError(
                f"{len(failures)} counterfactuals failed after writing audit-marked output: {detail}"
            )
    else:
        for candidate in pending:
            rows.append(one(candidate))
    rows.sort(key=lambda c: (c.source_seed_ids[0] if c.source_seed_ids else "", c.candidate_id))
    write_jsonl(output_path, [row.to_json() for row in rows])
    return rows


def failed_counterfactual(candidate: Candidate, config: GenerationConfig, error: Exception) -> dict[str, Any]:
    return {
        "text": "",
        "label": candidate.label,
        "semantic_edit_score": 0.0,
        "method": config.model,
        "counterfactual_id": stable_hash(f"{candidate.candidate_id}:counterfactual_failed:{error}"),
        "failed": True,
        "error": str(error)[:500],
    }
