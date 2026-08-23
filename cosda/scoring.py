from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import AuditConfig
from .data import iter_records, load_manifest
from .embeddings import cosine_matrix, make_embedder
from .io_utils import iter_jsonl, write_jsonl
from .types import Candidate, DataRecord


def token_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = re.findall(r"\w+|[^\w\s]", (text or "").lower(), flags=re.UNICODE)
    if len(toks) < n:
        return set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def softmax(scores: np.ndarray, temperature: float = 0.25) -> np.ndarray:
    scores = scores / max(temperature, 1e-6)
    scores = scores - np.max(scores)
    exp = np.exp(scores)
    return exp / max(float(exp.sum()), 1e-12)


class CentroidTeacher:
    def __init__(self, embedder):
        self.embedder = embedder
        self.labels: list[str] = []
        self.centroids: np.ndarray | None = None

    def fit(self, records: list[DataRecord]) -> None:
        labelled = [r for r in records if r.label is not None]
        self.labels = sorted({str(r.label) for r in labelled})
        if not self.labels:
            self.centroids = np.zeros((0, 1), dtype=np.float32)
            return
        embeddings = self.embedder.encode([r.text for r in labelled])
        by_label = defaultdict(list)
        for row, record in zip(embeddings, labelled):
            by_label[str(record.label)].append(row)
        self.centroids = np.vstack([np.mean(by_label[label], axis=0) for label in self.labels]).astype(np.float32)
        self.centroids = self.centroids / np.maximum(np.linalg.norm(self.centroids, axis=1, keepdims=True), 1e-12)

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        if self.centroids is None or len(self.labels) == 0:
            return [{} for _ in texts]
        emb = self.embedder.encode(texts)
        return self.predict_proba_embeddings(emb)

    def predict_proba_embeddings(self, embeddings: np.ndarray) -> list[dict[str, float]]:
        if self.centroids is None or len(self.labels) == 0:
            return [{} for _ in range(len(embeddings))]
        emb = embeddings
        sims = cosine_matrix(emb, self.centroids)
        rows = []
        for row in sims:
            probs = softmax(row)
            rows.append({label: float(prob) for label, prob in zip(self.labels, probs)})
        return rows

    def confidence_for(self, text: str, label: str | None) -> float:
        if label is None:
            return 0.5
        probs = self.predict_proba([text])[0]
        return float(probs.get(str(label), 0.0))

    def confidence_for_embedding(self, embedding: np.ndarray, label: str | None) -> float:
        if label is None:
            return 0.5
        probs = self.predict_proba_embeddings(embedding.reshape(1, -1))[0]
        return float(probs.get(str(label), 0.0))

    def predict(self, text: str) -> str | None:
        probs = self.predict_proba([text])[0]
        return max(probs, key=probs.get) if probs else None

    def predict_embedding(self, embedding: np.ndarray) -> str | None:
        probs = self.predict_proba_embeddings(embedding.reshape(1, -1))[0]
        return max(probs, key=probs.get) if probs else None


class LeakageIndex:
    def __init__(self, heldout_texts: list[str], embedder, cfg: AuditConfig):
        self.cfg = cfg
        self.exact = set()
        self.shingle_sets: list[set[tuple[str, ...]]] = []
        self.inverted: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for idx, text in enumerate(heldout_texts):
            self.exact.update(token_ngrams(text, cfg.ngram_n))
            shingles = token_ngrams(text, cfg.minhash_n)
            self.shingle_sets.append(shingles)
            for shingle in shingles:
                self.inverted[shingle].append(idx)
        self.heldout_embeddings = embedder.encode(heldout_texts) if heldout_texts else np.zeros((0, cfg.embedding_dim))
        self.embedder = embedder

    def score(self, text: str, embedding: np.ndarray | None = None) -> tuple[float, dict]:
        exact_hit = 1.0 if token_ngrams(text, self.cfg.ngram_n) & self.exact else 0.0
        shingles = token_ngrams(text, self.cfg.minhash_n)
        candidates = Counter()
        for shingle in shingles:
            for idx in self.inverted.get(shingle, []):
                candidates[idx] += 1
        max_jaccard = 0.0
        for idx, overlap in candidates.most_common(128):
            denom = len(shingles | self.shingle_sets[idx])
            if denom:
                max_jaccard = max(max_jaccard, overlap / denom)
        emb_score = 0.0
        if len(self.heldout_embeddings):
            emb = embedding.reshape(1, -1) if embedding is not None else self.embedder.encode([text])
            max_cosine = float(np.max(cosine_matrix(emb, self.heldout_embeddings)))
            # Convert raw cosine to a leakage risk. A low raw cosine should not trip
            # tau_l; only near-duplicate embedding similarity should.
            threshold = self.cfg.embedding_leakage_threshold
            emb_score = max(0.0, (max_cosine - threshold) / max(1.0 - threshold, 1e-6))
        leakage = max(exact_hit, max_jaccard, emb_score)
        return leakage, {"exact_ngram": exact_hit, "minhash_jaccard": max_jaccard, "embedding_risk": emb_score}


def artifact_features(text: str, label: str | None, all_labels: list[str]) -> dict[str, float]:
    toks = re.findall(r"\w+", text or "", flags=re.UNICODE)
    lower = (text or "").lower()
    label_hits = 0
    for item in all_labels:
        if item and item.lower() in lower:
            label_hits += 1
    uppercase = sum(1 for tok in toks if tok[:1].isupper())
    return {
        "length": float(len(toks)),
        "label_word_hit": float(label_hits > 0),
        "template_marker": float("[variant" in lower or "counterfactual label cue" in lower),
        "named_entity_proxy": min(1.0, uppercase / max(len(toks), 1)),
        "punct_ratio": sum(1 for c in text if c in "!?.,;:") / max(len(text), 1),
    }


class ShortcutScorer:
    def __init__(self, candidates: list[Candidate]):
        labels = sorted({str(c.label) for c in candidates if c.label is not None})
        self.labels = labels
        lengths = [artifact_features(c.text, c.label, labels)["length"] for c in candidates]
        self.mean_len = float(np.mean(lengths)) if lengths else 0.0
        self.std_len = float(np.std(lengths)) if lengths else 1.0

    def score(self, candidate: Candidate) -> tuple[float, dict]:
        feat = artifact_features(candidate.text, candidate.label, self.labels)
        len_z = abs(feat["length"] - self.mean_len) / max(self.std_len, 1.0)
        score = max(
            0.0,
            min(
                1.0,
                0.35 * feat["label_word_hit"]
                + 0.30 * feat["template_marker"]
                + 0.20 * min(1.0, len_z / 3.0)
                + 0.10 * feat["named_entity_proxy"]
                + 0.05 * min(1.0, feat["punct_ratio"] * 10),
            ),
        )
        return score, feat


@dataclass
class AuditContext:
    gold: list[DataRecord]
    dev: list[DataRecord]
    test: list[DataRecord]


class AuditScorer:
    def __init__(self, cfg: AuditConfig, context: AuditContext, candidates: list[Candidate]):
        self.cfg = cfg
        self.embedder = make_embedder(cfg.embedder, cfg.embedding_dim)
        self.teacher = CentroidTeacher(self.embedder)
        self.teacher.fit(context.gold)
        self.gold_embeddings = self.embedder.encode([r.text for r in context.gold])
        self.gold_labels = [str(r.label) for r in context.gold]
        self.leakage = LeakageIndex([r.text for r in context.dev + context.test], self.embedder, cfg)
        self.shortcut = ShortcutScorer(candidates)
        self.candidate_embeddings = {
            candidate.candidate_id: embedding
            for candidate, embedding in zip(candidates, self.embedder.encode([c.text for c in candidates]))
        }
        cf_texts = [((candidate.counterfactual or {}).get("text") or "") for candidate in candidates]
        self.counterfactual_embeddings = {
            candidate.candidate_id: embedding
            for candidate, embedding in zip(candidates, self.embedder.encode(cf_texts))
        }

    def utility(self, candidate: Candidate) -> tuple[float, dict]:
        judge = (candidate.metadata or {}).get("llm_judge") or {}
        gen = float(judge.get("quality_score", candidate.generator_confidence))
        teacher = self.teacher.confidence_for_embedding(self.candidate_embeddings[candidate.candidate_id], candidate.label)
        neighbor = self._neighbor_vote(candidate)
        utility = (gen + teacher + neighbor) / 3.0
        return utility, {"generator_or_judge_confidence": gen, "teacher_confidence": teacher, "neighbor_vote": neighbor}

    def _neighbor_vote(self, candidate: Candidate) -> float:
        if not len(self.gold_embeddings) or candidate.label is None:
            return 0.5
        emb = self.candidate_embeddings[candidate.candidate_id].reshape(1, -1)
        sims = cosine_matrix(emb, self.gold_embeddings)[0]
        k = min(self.cfg.neighbor_k, len(sims))
        top = np.argsort(-sims)[:k]
        hits = sum(1 for idx in top if self.gold_labels[idx] == str(candidate.label))
        return hits / max(k, 1)

    def diversity(self, candidate: Candidate) -> float:
        if not len(self.gold_embeddings):
            return 1.0
        emb = self.candidate_embeddings[candidate.candidate_id].reshape(1, -1)
        value = float(1.0 - np.max(cosine_matrix(emb, self.gold_embeddings)))
        return max(0.0, min(1.0, value))

    def consistency(self, candidate: Candidate) -> tuple[float, dict]:
        cf = candidate.counterfactual or {}
        cf_text = cf.get("text", "")
        cf_label = cf.get("label")
        cf_embedding = self.counterfactual_embeddings[candidate.candidate_id]
        teacher_probs = self.teacher.predict_proba_embeddings(cf_embedding.reshape(1, -1))[0]
        teacher_pred = self.teacher.predict_embedding(cf_embedding)
        teacher_expected_confidence = float(teacher_probs.get(str(cf_label), 0.0)) if cf_label is not None else 0.5
        semantic = float(cf.get("judge_validity", cf.get("semantic_edit_score", 0.0)))
        has_llm_judge = "judge_validity" in cf or bool((candidate.metadata or {}).get("llm_judge"))
        if has_llm_judge:
            # With LLM-as-judge audit, counterfactual_validity is the primary
            # consistency signal. The centroid teacher is intentionally kept as
            # a soft sanity check because it is noisy at small gold budgets.
            consistency = semantic * (0.75 + 0.25 * teacher_expected_confidence)
            mode = "llm_judge_primary"
        else:
            consistency = semantic * teacher_expected_confidence
            mode = "teacher_soft"
        return consistency, {
            "teacher_pred": teacher_pred,
            "expected_label": cf_label,
            "teacher_expected_confidence": teacher_expected_confidence,
            "llm_counterfactual_validity": semantic if has_llm_judge else None,
            "semantic_edit_score": semantic,
            "mode": mode,
        }

    def score_one(self, candidate: Candidate) -> dict:
        u, u_parts = self.utility(candidate)
        l, l_parts = self.leakage.score(candidate.text, self.candidate_embeddings[candidate.candidate_id])
        h, h_parts = self.shortcut.score(candidate)
        d = self.diversity(candidate)
        c, c_parts = self.consistency(candidate)
        soft = (
            self.cfg.alpha_u * u
            + self.cfg.alpha_c * c
            + self.cfg.alpha_d * d
            - self.cfg.alpha_l * l
            - self.cfg.alpha_h * h
        )
        hard_reject = l > self.cfg.tau_l or c < self.cfg.tau_c or h > self.cfg.tau_h
        reason = []
        if l > self.cfg.tau_l:
            reason.append("leakage")
        if c < self.cfg.tau_c:
            reason.append("counterfactual")
        if h > self.cfg.tau_h:
            reason.append("shortcut")
        return {
            **candidate.to_json(),
            "scores": {"U": u, "L": l, "H": h, "D": d, "C": c, "S": soft},
            "score_parts": {"utility": u_parts, "leakage": l_parts, "shortcut": h_parts, "consistency": c_parts},
            "hard_reject": hard_reject,
            "reject_reasons": reason,
        }


def audit_candidates(
    manifest_path: str,
    dataset_id: str,
    language: str,
    gold_path: str | Path,
    candidates_path: str | Path,
    output_path: str | Path,
    cfg: AuditConfig,
) -> list[dict]:
    manifest = load_manifest(manifest_path)
    gold = [DataRecord.from_json(row) for row in iter_jsonl(gold_path)]
    dev = list(iter_records(manifest, dataset_id, language, "dev"))
    test = list(iter_records(manifest, dataset_id, language, "test"))
    candidates = [Candidate.from_json(row) for row in iter_jsonl(candidates_path)]
    scorer = AuditScorer(cfg, AuditContext(gold=gold, dev=dev, test=test), candidates)
    rows = [scorer.score_one(candidate) for candidate in candidates]
    write_jsonl(output_path, rows)
    return rows
