#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosda.config import load_run_config  # noqa: E402
from cosda.data import load_manifest  # noqa: E402
from cosda.scenario import expand_cells, summarize_plan  # noqa: E402


PROFILES = {
    "g6e_1gpu": {"gpu_jobs": 1, "xnlp_min": (6, 12), "sum_min": (25, 60), "audit_overhead_min": (5, 20)},
    "p4d_8gpu": {"gpu_jobs": 8, "xnlp_min": (4, 9), "sum_min": (18, 45), "audit_overhead_min": (8, 30)},
    "p5_8gpu": {"gpu_jobs": 8, "xnlp_min": (2, 6), "sum_min": (10, 30), "audit_overhead_min": (6, 25)},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate CoSDA AWS runtime before launching paid jobs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="p4d_8gpu")
    parser.add_argument("--generation-rps", type=float, default=4.0, help="Effective successful generator requests/sec.")
    parser.add_argument("--counterfactual-rps", type=float, default=4.0, help="Effective successful counterfactual requests/sec.")
    parser.add_argument("--judge-rps", type=float, default=2.0, help="Effective successful LLM-as-judge requests/sec after rate limits.")
    parser.add_argument("--gpu-jobs", type=int, help="Override concurrent train-hf jobs.")
    args = parser.parse_args()

    cfg = load_run_config(ROOT / args.config)
    manifest = load_manifest(cfg.manifest_path)
    plan = summarize_plan(ROOT / args.config, manifest)
    profile = dict(PROFILES[args.profile])
    if args.gpu_jobs:
        profile["gpu_jobs"] = args.gpu_jobs

    cells = list(expand_cells(ROOT / args.config, manifest))
    baselines = cfg.baselines or ["cosda"]
    task_runs = {"summarization": 0, "xnlp": 0}
    for cell in cells:
        runs = len(baselines)
        if cell["task"] == "summarization":
            task_runs["summarization"] += runs
        else:
            task_runs["xnlp"] += runs

    gen_h = plan["estimated_generated_candidates"] / max(args.generation_rps, 1e-6) / 3600
    cf_h = plan["estimated_counterfactual_requests"] / max(args.counterfactual_rps, 1e-6) / 3600
    judge_h = plan["estimated_judge_requests"] / max(args.judge_rps, 1e-6) / 3600
    xnlp_min = profile["xnlp_min"]
    sum_min = profile["sum_min"]
    jobs = max(1, int(profile["gpu_jobs"]))
    train_low_h = (task_runs["xnlp"] * xnlp_min[0] + task_runs["summarization"] * sum_min[0]) / jobs / 60
    train_high_h = (task_runs["xnlp"] * xnlp_min[1] + task_runs["summarization"] * sum_min[1]) / jobs / 60
    overhead_low_h, overhead_high_h = [x / 60 for x in profile["audit_overhead_min"]]

    report = {
        "config": args.config,
        "profile": args.profile,
        "plan": plan,
        "assumptions": {
            "generation_rps": args.generation_rps,
            "counterfactual_rps": args.counterfactual_rps,
            "judge_rps": args.judge_rps,
            "parallel_training_jobs": jobs,
            "xnlp_minutes_per_run": xnlp_min,
            "summarization_minutes_per_run": sum_min,
            "audit_embedding_overhead_minutes": profile["audit_overhead_min"],
        },
        "task_training_runs": task_runs,
        "stage_hours": {
            "generation": round(gen_h, 2),
            "counterfactual_generation": round(cf_h, 2),
            "llm_judge": round(judge_h, 2),
            "audit_selection_lightweight_eval": [round(overhead_low_h, 2), round(overhead_high_h, 2)],
            "hf_training_eval": [round(train_low_h, 2), round(train_high_h, 2)],
        },
        "total_wall_clock_hours_estimate": [
            round(gen_h + cf_h + judge_h + overhead_low_h + train_low_h, 2),
            round(gen_h + cf_h + judge_h + overhead_high_h + train_high_h, 2),
        ],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
