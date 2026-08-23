from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import JudgeConfig
from .http_utils import post_json_with_retries
from .io_utils import iter_jsonl, stable_hash, write_jsonl
from .text_utils import compact_text
from .types import Candidate


class BaseJudge:
    name = "base"

    def score(self, candidate: Candidate) -> dict[str, Any]:
        raise NotImplementedError


class HeuristicJudge(BaseJudge):
    name = "heuristic_judge"

    def score(self, candidate: Candidate) -> dict[str, Any]:
        text = candidate.text or ""
        cf = candidate.counterfactual or {}
        length = len(text.split())
        quality = 0.75 if 4 <= length <= 256 else 0.45
        if "[variant" in text.lower():
            quality -= 0.15
        cf_valid = 1.0 if cf.get("text") and cf.get("label") != candidate.label else 0.35
        leakage = 0.15 if len(text) > 20 else 0.30
        return normalize_judgment(
            {
                "judge_model": self.name,
                "quality_score": quality,
                "label_correctness": quality,
                "fluency": min(1.0, quality + 0.10),
                "cultural_plausibility": 0.70,
                "leakage_suspicion": leakage,
                "counterfactual_validity": cf_valid,
                "rationale": "Heuristic local smoke-test judgment; not a paper metric.",
            }
        )


class OpenAICompatibleJudge(BaseJudge):
    name = "openai_compatible_judge"

    def __init__(self, config: JudgeConfig):
        if not config.endpoint_url:
            raise ValueError("judge.endpoint_url is required for openai_compatible judge")
        self.config = config

    def score(self, candidate: Candidate) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env and os.getenv(self.config.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[self.config.api_key_env]}"
        attempts = max(1, self.config.request_retries)
        last_error: Exception | None = None
        for attempt in range(attempts):
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": judge_system_prompt()},
                    {"role": "user", "content": judge_user_prompt(candidate)},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            try:
                data = post_json_with_retries(
                    self.config.endpoint_url,
                    payload,
                    headers=headers,
                    timeout=self.config.request_timeout,
                    retries=self.config.request_retries,
                )
                content = data["choices"][0]["message"]["content"]
                parsed = parse_json_object(content)
                parsed["judge_model"] = self.config.model
                return normalize_judgment(parsed)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(min(8, 2**attempt))
        raise RuntimeError(f"Failed to parse LLM judge JSON for candidate {candidate.candidate_id}: {last_error}") from last_error


class BedrockJudge(BaseJudge):
    name = "bedrock_judge"

    def __init__(self, config: JudgeConfig):
        self.config = config
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("Install boto3 or `pip install -r requirements-aws.txt` for Bedrock judge") from exc
        self.client = boto3.client("bedrock-runtime", region_name=config.aws_region)

    def score(self, candidate: Candidate) -> dict[str, Any]:
        # Claude Messages API format used by Anthropic models on Bedrock.
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "system": judge_system_prompt(),
            "messages": [{"role": "user", "content": [{"type": "text", "text": judge_user_prompt(candidate)}]}],
        }
        response = self.client.invoke_model(
            modelId=self.config.model,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        raw = json.loads(response["body"].read())
        content = "".join(part.get("text", "") for part in raw.get("content", []) if part.get("type") == "text")
        parsed = parse_json_object(content)
        parsed["judge_model"] = self.config.model
        return normalize_judgment(parsed)


def make_judge(config: JudgeConfig) -> BaseJudge | None:
    if config.backend in {"disabled", "none", ""}:
        return None
    if config.backend == "heuristic":
        return HeuristicJudge()
    if config.backend in {"openai_compatible", "vllm", "sglang"}:
        return OpenAICompatibleJudge(config)
    if config.backend == "bedrock":
        return BedrockJudge(config)
    raise ValueError(f"Unsupported judge backend: {config.backend}")


def judge_candidates(candidates_path: str | Path, output_path: str | Path, config: JudgeConfig, resume: bool = False) -> list[Candidate]:
    judge = make_judge(config)
    candidates = [Candidate.from_json(row) for row in iter_jsonl(candidates_path)]
    existing: list[Candidate] = []
    done = set()
    if resume and Path(output_path).exists():
        existing = [Candidate.from_json(row) for row in iter_jsonl(output_path)]
        done = {candidate.candidate_id for candidate in existing}
    rows = existing[:]
    pending = [candidate for candidate in candidates if candidate.candidate_id not in done]

    def one(candidate: Candidate) -> Candidate:
        if judge is not None:
            judgment = judge.score(candidate)
            candidate.metadata = candidate.metadata or {}
            candidate.metadata["llm_judge"] = judgment
            candidate.metadata["judge_score"] = judgment["quality_score"]
            candidate.metadata["cultural_plausibility"] = judgment["cultural_plausibility"]
            if candidate.counterfactual:
                candidate.counterfactual["judge_validity"] = judgment["counterfactual_validity"]
        return candidate

    if config.max_workers > 1 and len(pending) > 1:
        errors: list[tuple[str, Exception]] = []
        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            futures = {pool.submit(one, candidate): candidate.candidate_id for candidate in pending}
            for future in as_completed(futures):
                try:
                    rows.append(future.result())
                except Exception as exc:
                    errors.append((futures[future], exc))
        if errors:
            rows.sort(key=lambda c: (c.source_seed_ids[0] if c.source_seed_ids else "", c.candidate_id))
            write_jsonl(output_path, [row.to_json() for row in rows])
            detail = "; ".join(str(exc) for _, exc in errors[:5])
            raise RuntimeError(f"{len(errors)} judge workers failed after writing partial output: {detail}") from errors[0][1]
    else:
        for candidate in pending:
            rows.append(one(candidate))
    rows.sort(key=lambda c: (c.source_seed_ids[0] if c.source_seed_ids else "", c.candidate_id))
    write_jsonl(output_path, [row.to_json() for row in rows])
    return rows


def judge_system_prompt() -> str:
    return (
        "You are an expert evaluator for low-resource NLP synthetic data. "
        "Judge the candidate conservatively. Return strict JSON only, with numeric scores in [0,1]."
    )


def judge_user_prompt(candidate: Candidate) -> str:
    cf = candidate.counterfactual or {}
    candidate_text = compact_text(candidate.text, 2200)
    cf_text = compact_text(cf.get("text", ""), 1400)
    return (
        f"Task: {candidate.task}\n"
        f"Language code: {candidate.language}\n"
        f"Candidate label/output: {candidate.label}\n"
        f"Candidate text excerpt:\n{candidate_text}\n\n"
        f"Counterfactual expected label/output: {cf.get('label')}\n"
        f"Counterfactual text excerpt:\n{cf_text}\n\n"
        "Assess these fields:\n"
        "- label_correctness: candidate matches its label or structured output.\n"
        "- fluency: language is grammatical/natural.\n"
        "- cultural_plausibility: content is plausible for the target language/community; use 0.5 if unsure.\n"
        "- leakage_suspicion: likely copied from held-out data or a known article/benchmark item.\n"
        "- counterfactual_validity: counterfactual minimally changes a task-relevant variable and should change label/output.\n"
        "- quality_score: overall suitability for fine-tuning.\n\n"
        "Return exactly valid JSON only, with no Markdown fences or commentary:\n"
        "{\"label_correctness\": float, \"fluency\": float, \"cultural_plausibility\": float, "
        "\"leakage_suspicion\": float, \"counterfactual_validity\": float, "
        "\"quality_score\": float, \"rationale\": string}"
    )


def parse_json_object(text: str) -> dict[str, Any]:
    text = _strip_markdown_json(text).strip()
    candidates = [text]
    payload = _extract_balanced_json(text, "{", "}")
    if payload and payload not in candidates:
        candidates.append(payload)
    errors: list[str] = []
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as strict_exc:
            try:
                obj = json.loads(candidate, strict=False)
            except json.JSONDecodeError as exc:
                errors.append(f"{strict_exc.msg} at char {strict_exc.pos}; {exc.msg} at char {exc.pos}")
                continue
        if isinstance(obj, dict):
            return obj
        raise ValueError("LLM response was JSON but not an object")
    raise ValueError(f"LLM response was not parseable JSON object: {'; '.join(errors)}")


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


def normalize_judgment(row: dict[str, Any]) -> dict[str, Any]:
    out = {
        "judge_id": stable_hash(json.dumps(row, sort_keys=True, ensure_ascii=False)),
        "judge_model": row.get("judge_model", "unknown"),
        "label_correctness": clamp(row.get("label_correctness", 0.5)),
        "fluency": clamp(row.get("fluency", 0.5)),
        "cultural_plausibility": clamp(row.get("cultural_plausibility", 0.5)),
        "leakage_suspicion": clamp(row.get("leakage_suspicion", 0.5)),
        "counterfactual_validity": clamp(row.get("counterfactual_validity", 0.5)),
        "quality_score": clamp(row.get("quality_score", 0.5)),
        "rationale": str(row.get("rationale", ""))[:500],
    }
    return out


def clamp(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 0.5
    return max(0.0, min(1.0, v))
