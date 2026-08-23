from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import GenerationConfig
from .data import labels_for_cell, load_manifest
from .http_utils import post_json_with_retries
from .io_utils import iter_jsonl, stable_hash, write_jsonl
from .text_utils import compact_text, text_was_compacted
from .types import Candidate, DataRecord


def load_seed_records(path: str | Path) -> list[DataRecord]:
    return [DataRecord.from_json(row) for row in iter_jsonl(path)]


class BaseGenerator:
    name = "base"

    def generate(self, seed: DataRecord, labels: list[str], n: int) -> list[Candidate]:
        raise NotImplementedError


class HeuristicGenerator(BaseGenerator):
    name = "heuristic"

    def __init__(self, seed: int = 13):
        self.rng = random.Random(seed)

    def generate(self, seed: DataRecord, labels: list[str], n: int) -> list[Candidate]:
        rows = []
        for i in range(n):
            label = seed.label
            text = self._variant_text(seed, i)
            metadata = {"variant": i, "seed_text_hash": stable_hash(seed.text)}
            if seed.task == "named_entity_recognition":
                metadata.update({"tokens": seed.tokens or text.split(), "tags": seed.tags or []})
            if seed.task == "summarization":
                metadata.update({"summary": seed.summary or ""})
            prompt = build_generation_prompt(seed, labels)
            candidate_id = stable_hash(f"{seed.id}:{i}:{text}:{label}")
            rows.append(
                Candidate(
                    candidate_id=candidate_id,
                    dataset_id=seed.dataset_id,
                    task=seed.task,
                    language=seed.language,
                    text=text,
                    label=label,
                    source_seed_ids=[seed.id],
                    generator=self.name,
                    generator_confidence=0.62,
                    prompt_hash=stable_hash(prompt),
                    metadata=metadata,
                )
            )
        return rows

    def _variant_text(self, seed: DataRecord, i: int) -> str:
        if seed.task == "summarization":
            summary = seed.summary or ""
            return f"{seed.text}\n\nSynthetic summary target: {summary}".strip()
        if seed.task == "named_entity_recognition" and seed.tokens:
            tokens = seed.tokens[:]
            if tokens:
                j = i % len(tokens)
                tokens[j] = tokens[j]
            return " ".join(tokens)
        suffix = "" if i == 0 else f" [variant {i}]"
        return f"{seed.text}{suffix}".strip()


class OpenAICompatibleGenerator(BaseGenerator):
    """Generator for local vLLM/SGLang/TGI OpenAI-compatible chat endpoints."""

    name = "openai_compatible"

    def __init__(self, config: GenerationConfig):
        if not config.endpoint_url:
            raise ValueError("endpoint_url is required for openai_compatible generation")
        self.config = config

    def generate(self, seed: DataRecord, labels: list[str], n: int) -> list[Candidate]:
        prompt = build_generation_prompt(seed, labels, n=n)
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env and os.getenv(self.config.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[self.config.api_key_env]}"

        rows: list[Candidate] = []
        seen_texts: set[tuple[str, str | None]] = set()
        attempts = max(1, self.config.request_retries)
        last_error: Exception | None = None
        saw_endpoint_response = False
        for attempt in range(attempts):
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": "You generate low-resource NLP training data as strict JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_new_tokens,
            }
            try:
                data = post_json_with_retries(
                    self.config.endpoint_url,
                    payload,
                    headers=headers,
                    timeout=self.config.request_timeout,
                    retries=self.config.request_retries,
                )
                saw_endpoint_response = True
                content = data["choices"][0]["message"]["content"]
                parsed = _parse_json_list(content)
                for item in parsed:
                    candidate = _candidate_from_generated_item(
                        item=item,
                        seed=seed,
                        prompt=prompt,
                        model=self.config.model,
                        raw_content=content,
                        index=len(rows),
                    )
                    key = (candidate.text.strip(), candidate.label)
                    if candidate.text.strip() and key not in seen_texts:
                        rows.append(candidate)
                        seen_texts.add(key)
                    if len(rows) >= n:
                        return rows[:n]
                last_error = ValueError(f"model returned {len(rows)}/{n} usable examples for seed {seed.id}")
            except Exception as exc:
                last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(8, 2**attempt))
        if rows:
            for candidate in rows:
                candidate.metadata["generation_shortfall"] = {
                    "requested": n,
                    "returned": len(rows),
                    "reason": str(last_error)[:300],
                }
            return rows
        if not saw_endpoint_response:
            raise RuntimeError(f"Failed to reach generation endpoint for seed {seed.id}: {last_error}") from last_error
        print(
            f"WARNING: skipping seed {seed.id}; no usable JSON examples after {attempts} attempts: {last_error}",
            file=sys.stderr,
        )
        return []


def _parse_json_list(text: str) -> list[dict[str, Any]]:
    text = _strip_markdown_json(text).strip()
    candidates = [text]
    for opener, closer in [("[", "]"), ("{", "}")]:
        payload = _extract_balanced_json(text, opener, closer)
        if payload and payload not in candidates:
            candidates.append(payload)
    errors: list[str] = []
    obj: Any | None = None
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            break
        except json.JSONDecodeError as strict_exc:
            try:
                obj = json.loads(candidate, strict=False)
                break
            except json.JSONDecodeError as exc:
                errors.append(f"{strict_exc.msg} at char {strict_exc.pos}; {exc.msg} at char {exc.pos}")
            continue
    if obj is None:
        raise ValueError(f"Generator did not return parseable JSON: {'; '.join(errors)}")
    if isinstance(obj, dict):
        obj = obj.get("examples") or [obj]
    if not isinstance(obj, list):
        raise ValueError("Generator did not return a JSON list")
    return [x for x in obj if isinstance(x, dict)]


def _strip_markdown_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    return match.group(1) if match else text


def _extract_balanced_json(text: str, opener: str, closer: str) -> str | None:
    start = text.find(opener)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _candidate_from_generated_item(
    item: dict[str, Any],
    seed: DataRecord,
    prompt: str,
    model: str,
    raw_content: str,
    index: int,
) -> Candidate:
    label = str(item.get("label", seed.label)) if item.get("label", seed.label) is not None else None
    text = item.get("text") or item.get("input") or ""
    confidence = float(item.get("confidence", 0.70))
    candidate_id = stable_hash(f"{seed.id}:{index}:{text}:{label}:{raw_content}")
    return Candidate(
        candidate_id=candidate_id,
        dataset_id=seed.dataset_id,
        task=seed.task,
        language=seed.language,
        text=text,
        label=label if seed.task != "summarization" else "summary",
        source_seed_ids=[seed.id],
        generator=model,
        generator_confidence=max(0.0, min(1.0, confidence)),
        prompt_hash=stable_hash(prompt),
        metadata={
            "raw_item": item,
            "summary": item.get("summary"),
            "tokens": item.get("tokens"),
            "tags": item.get("tags"),
            "seed_text_compacted_for_prompt": text_was_compacted(
                seed.text,
                prompt_text_limit(seed.task),
            ),
        },
    )


def build_generation_prompt(seed: DataRecord, labels: list[str], n: int = 1) -> str:
    label_text = ", ".join(labels) if labels else "task-specific output"
    if seed.task in {"news_topic_classification", "sentiment_classification", "intent_slot_filling"}:
        seed_excerpt = compact_text(seed.text, prompt_text_limit(seed.task))
        length_hint = (
            "Each generated text should be a concise news item of 60-120 words."
            if seed.task == "news_topic_classification"
            else "Each generated text should be concise and natural, usually under 60 words."
        )
        return (
            f"Task: {seed.task}\n"
            f"Language: {seed.language}\n"
            f"Allowed labels: {label_text}\n"
            f"Seed label: {seed.label}\n"
            f"Seed text excerpt:\n{seed_excerpt}\n\n"
            f"Generate {n} new examples in the same language and label. "
            f"{length_hint} "
            "Do not copy the seed text or held-out text. Do not translate an English template. "
            "Return valid JSON only, with no Markdown fences or commentary. "
            "Return JSON list: [{\"text\": str, \"label\": str, \"confidence\": float}]."
        )
    if seed.task == "named_entity_recognition":
        return (
            f"Task: named entity recognition\nLanguage: {seed.language}\n"
            f"Seed tokens: {seed.tokens}\nSeed BIO tags: {seed.tags}\n"
            f"Generate {n} plausible token sequences with BIO tags over PER, ORG, LOC, DATE. "
            "Return JSON list: [{\"text\": str, \"tokens\": [str], \"tags\": [str], \"confidence\": float}]."
        )
    return (
        f"Task: summarization\nLanguage: {seed.language}\n"
        f"Source article excerpt:\n{compact_text(seed.text, prompt_text_limit(seed.task))}\n"
        f"Seed summary:\n{compact_text(seed.summary, 900)}\n\n"
        f"Generate {n} source-summary training pairs in the same language. "
        "Return valid JSON only, with no Markdown fences or commentary. "
        "Return JSON list: [{\"text\": source, \"summary\": str, \"label\": \"summary\", \"confidence\": float}]."
    )


def prompt_text_limit(task: str) -> int:
    if task == "news_topic_classification":
        return 2200
    if task == "summarization":
        return 3200
    return 1200


def make_generator(config: GenerationConfig, seed: int = 13) -> BaseGenerator:
    if config.backend == "heuristic":
        return HeuristicGenerator(seed=seed)
    if config.backend in {"openai_compatible", "vllm", "sglang"}:
        return OpenAICompatibleGenerator(config)
    raise ValueError(f"Unsupported generation backend: {config.backend}")


def generate_candidates(
    manifest_path: str,
    seed_path: str | Path,
    output_path: str | Path,
    config: GenerationConfig,
    random_seed: int,
    limit: int | None = None,
    resume: bool = False,
) -> list[Candidate]:
    seeds = load_seed_records(seed_path)
    if limit:
        seeds = seeds[:limit]
    manifest = load_manifest(manifest_path)
    generator = make_generator(config, seed=random_seed)
    existing: list[Candidate] = []
    done_seed_ids: set[str] = set()
    if resume and Path(output_path).exists():
        existing = [Candidate.from_json(row) for row in iter_jsonl(output_path)]
        for candidate in existing:
            done_seed_ids.update(candidate.source_seed_ids)
    pending = [record for record in seeds if record.id not in done_seed_ids]
    rows: list[Candidate] = existing[:]

    def one(record: DataRecord) -> list[Candidate]:
        labels = labels_for_cell(manifest, record.dataset_id, record.language)
        return generator.generate(record, labels, config.candidates_per_seed)

    if config.max_workers > 1 and len(pending) > 1:
        errors: list[tuple[str, Exception]] = []
        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            futures = {pool.submit(one, record): record.id for record in pending}
            for future in as_completed(futures):
                try:
                    rows.extend(future.result())
                except Exception as exc:
                    errors.append((futures[future], exc))
        if errors:
            rows.sort(key=lambda c: (c.source_seed_ids[0] if c.source_seed_ids else "", c.candidate_id))
            write_jsonl(output_path, [row.to_json() for row in rows])
            detail = "; ".join(f"{seed_id}: {exc}" for seed_id, exc in errors[:5])
            raise RuntimeError(f"{len(errors)} generation workers failed after writing partial output: {detail}") from errors[0][1]
    else:
        for record in pending:
            rows.extend(one(record))
    rows.sort(key=lambda c: (c.source_seed_ids[0] if c.source_seed_ids else "", c.candidate_id))
    write_jsonl(output_path, [row.to_json() for row in rows])
    return rows
