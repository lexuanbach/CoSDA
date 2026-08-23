from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import yaml

from .config import load_run_config


def expand_cells(config_path: str | Path, manifest: dict) -> Iterator[dict]:
    cfg = load_run_config(config_path)
    datasets = set(cfg.datasets)
    tasks = set(cfg.tasks)
    languages = set(cfg.languages)
    for entry in manifest["datasets"]:
        if datasets and entry["dataset_id"] not in datasets:
            continue
        if tasks and entry["task"] not in tasks:
            continue
        for language in entry["cells"]:
            if languages and language not in languages:
                continue
            for budget in cfg.budgets:
                for multiplier in cfg.multipliers:
                    for seed in cfg.seeds:
                        yield {
                            "dataset_id": entry["dataset_id"],
                            "task": entry["task"],
                            "language": language,
                            "budget": budget,
                            "multiplier": multiplier,
                            "selection_multiplier": cfg.selection_multiplier
                            if cfg.selection_multiplier is not None
                            else multiplier,
                            "seed": seed,
                        }


def render_commands(config_path: str | Path, manifest: dict, run_name: str) -> list[str]:
    cfg = load_run_config(config_path)
    commands = []
    for cell in expand_cells(config_path, manifest):
        slug = (
            f"{run_name}/{cell['task']}/{cell['language']}/"
            f"b{cell['budget']}_m{cell['multiplier']}_s{cell['seed']}"
        )
        base = f"{cfg.output_dir}/{slug}"
        common = (
            f"--manifest {cfg.manifest_path} --dataset-id {cell['dataset_id']} "
            f"--language {cell['language']} --budget {cell['budget']} --seed {cell['seed']}"
        )
        seed_cmd = f"cosda make-seeds {common} --out {base}/gold.jsonl"
        if cfg.seed_sampling:
            seed_cmd += f" --sampling-strategy {cfg.seed_sampling}"
        elif not cfg.sample_per_class_when_possible:
            seed_cmd += " --per-cell"
        commands.extend(
            [
                seed_cmd,
                (
                    f"cosda generate --config {config_path} --seed-file {base}/gold.jsonl "
                    f"--out {base}/candidates.jsonl --candidates-per-seed {cell['multiplier']} --resume"
                ),
                (
                    f"cosda counterfactuals --config {config_path} --candidates {base}/candidates.jsonl "
                    f"--out {base}/candidates_cf.jsonl --resume"
                ),
            ]
        )
        candidate_for_audit = f"{base}/candidates_cf.jsonl"
        if cfg.judge.backend not in {"disabled", "none", ""}:
            candidate_for_audit = f"{base}/candidates_judged.jsonl"
            commands.append(
                f"cosda judge --config {config_path} --candidates {base}/candidates_cf.jsonl "
                f"--out {candidate_for_audit} --resume"
            )
        commands.append(
            f"cosda audit --config {config_path} {common} --gold {base}/gold.jsonl "
            f"--candidates {candidate_for_audit} --out {base}/audit.jsonl"
        )
        baselines = cfg.baselines or ["cosda"]
        for baseline in baselines:
            if baseline == "gold_only":
                commands.append(
                    f"cosda evaluate --manifest {cfg.manifest_path} --dataset-id {cell['dataset_id']} "
                    f"--language {cell['language']} --gold {base}/gold.jsonl "
                    f"--out {base}/results/gold_only.json"
                )
                continue
            commands.extend(
                [
                    (
                        f"cosda select --config {config_path} --audit {base}/audit.jsonl --gold {base}/gold.jsonl "
                        f"--out {base}/selected/{baseline}.jsonl "
                        f"--budget $(( $(wc -l < {base}/gold.jsonl) * {cell['selection_multiplier']} )) "
                        f"--baseline {baseline} --seed {cell['seed']}"
                    ),
                    (
                        f"cosda evaluate --manifest {cfg.manifest_path} --dataset-id {cell['dataset_id']} "
                        f"--language {cell['language']} --gold {base}/gold.jsonl "
                        f"--selected {base}/selected/{baseline}.jsonl --out {base}/results/{baseline}.json"
                    ),
                ]
            )
    return commands


def render_training_commands(
    config_path: str | Path,
    manifest: dict,
    run_name: str,
    classification_model: str = "FacebookAI/xlm-roberta-base",
    summarization_model: str = "google/mt5-base",
    deterministic: bool = False,
) -> list[str]:
    cfg = load_run_config(config_path)
    commands = []
    for cell in expand_cells(config_path, manifest):
        slug = (
            f"{run_name}/{cell['task']}/{cell['language']}/"
            f"b{cell['budget']}_m{cell['multiplier']}_s{cell['seed']}"
        )
        base = f"{cfg.output_dir}/{slug}"
        model = summarization_model if cell["task"] == "summarization" else classification_model
        batch_size = 4 if cell["task"] == "summarization" else 16
        max_length = 512 if cell["task"] == "summarization" else 256
        for baseline in cfg.baselines or ["cosda"]:
            selected = "" if baseline == "gold_only" else f" --selected {base}/selected/{baseline}.jsonl"
            deterministic_flag = " --deterministic" if deterministic else ""
            commands.append(
                f"cosda train-hf --manifest {cfg.manifest_path} --dataset-id {cell['dataset_id']} "
                f"--language {cell['language']} --gold {base}/gold.jsonl{selected} "
                f"--out-dir {base}/hf/{baseline} --model {model} --epochs 3 "
                f"--batch-size {batch_size} --max-length {max_length} --seed {cell['seed']}{deterministic_flag}"
            )
    return commands


def summarize_plan(config_path: str | Path, manifest: dict) -> dict:
    cfg = load_run_config(config_path)
    cells = list(expand_cells(config_path, manifest))
    total_gold = 0
    total_candidates = 0
    total_selected = 0
    for cell in cells:
        entry = next(item for item in manifest["datasets"] if item["dataset_id"] == cell["dataset_id"])
        train_labels = entry["cells"][cell["language"]]["train"].get("labels") or {}
        if cfg.seed_sampling == "per_class" and train_labels:
            gold_count = sum(min(cell["budget"], count) for count in train_labels.values())
        elif cfg.seed_sampling in {"balanced", "random"}:
            gold_count = min(cell["budget"], entry["cells"][cell["language"]]["train"]["rows"])
        elif cfg.sample_per_class_when_possible and train_labels:
            gold_count = sum(min(cell["budget"], count) for count in train_labels.values())
        else:
            gold_count = min(cell["budget"], entry["cells"][cell["language"]]["train"]["rows"])
        total_gold += gold_count
        total_candidates += gold_count * cell["multiplier"]
        total_selected += gold_count * cell["selection_multiplier"]
    selection_multiplier = cfg.selection_multiplier
    non_gold_baselines = [b for b in (cfg.baselines or ["cosda"]) if b != "gold_only"]
    return {
        "config": str(config_path),
        "cells": len(cells),
        "budgets": cfg.budgets,
        "multipliers": cfg.multipliers,
        "selection_multiplier": selection_multiplier,
        "seeds": cfg.seeds,
        "sample_per_class_when_possible": cfg.sample_per_class_when_possible,
        "seed_sampling": cfg.seed_sampling,
        "estimated_gold_records": total_gold,
        "estimated_generated_candidates": total_candidates,
        "estimated_selected_candidates_per_baseline": total_selected,
        "estimated_counterfactual_requests": total_candidates,
        "estimated_judge_requests": total_candidates if cfg.judge.backend not in {"disabled", "none", ""} else 0,
        "selection_baselines": cfg.baselines,
        "estimated_lightweight_eval_runs": len(cells) * len(cfg.baselines or ["cosda"]),
        "estimated_hf_training_runs": len(cells) * len(cfg.baselines or ["cosda"]),
        "non_gold_selection_outputs": len(cells) * len(non_gold_baselines),
    }
