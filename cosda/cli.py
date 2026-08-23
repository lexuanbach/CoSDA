from __future__ import annotations

import argparse
import json
import signal
from dataclasses import replace
from pathlib import Path

from .config import load_run_config
from .counterfactual import build_counterfactuals
from .evaluate import evaluate_run
from .generation import generate_candidates
from .io_utils import read_json, write_jsonl
from .scoring import audit_candidates
from .seed import sample_gold
from .selection import BASELINES, select_candidates


def main(argv: list[str] | None = None) -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    parser = argparse.ArgumentParser(prog="cosda")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-seeds")
    add_common_cell_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--per-cell", action="store_true", help="Disable per-class sampling for classification tasks.")
    p.add_argument(
        "--sampling-strategy",
        choices=["random", "balanced", "per_class"],
        help="Gold seed sampling mode. balanced keeps the total budget and rotates labels.",
    )

    p = sub.add_parser("generate")
    p.add_argument("--config", required=True)
    p.add_argument("--seed-file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--candidates-per-seed", type=int)
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("counterfactuals")
    p.add_argument("--config", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("judge")
    p.add_argument("--config", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("audit")
    add_common_cell_args(p)
    p.add_argument("--config", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("select")
    p.add_argument("--config", required=True)
    p.add_argument("--audit", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--baseline", choices=sorted(BASELINES), default="cosda")
    p.add_argument("--seed", type=int, default=13)

    p = sub.add_parser("evaluate")
    p.add_argument("--manifest", default="data/manifest/datasets.json")
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--language", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--selected")
    p.add_argument("--out", required=True)
    p.add_argument("--embedder", default="hash")

    p = sub.add_parser("train-hf")
    p.add_argument("--manifest", default="data/manifest/datasets.json")
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--language", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--selected")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--keep-checkpoints", action="store_true")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument(
        "--deterministic",
        action="store_true",
        default=None,
        help="Use slower deterministic HF settings for reproducibility checks.",
    )

    p = sub.add_parser("plan-scenario")
    p.add_argument("--config", required=True)
    p.add_argument("--run-name", default="scenario")
    p.add_argument("--out")

    p = sub.add_parser("plan-training")
    p.add_argument("--config", required=True)
    p.add_argument("--run-name", default="scenario")
    p.add_argument("--classification-model", default="FacebookAI/xlm-roberta-base")
    p.add_argument("--summarization-model", default="google/mt5-base")
    p.add_argument("--deterministic", action="store_true", help="Add --deterministic to every train-hf command.")
    p.add_argument("--out")

    p = sub.add_parser("plan-summary")
    p.add_argument("--config", required=True)

    p = sub.add_parser("claim-ledger")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", default="results/claim_ledger.csv")

    args = parser.parse_args(argv)

    if args.cmd == "make-seeds":
        sample_gold(
            args.manifest,
            args.dataset_id,
            args.language,
            args.budget,
            args.seed,
            args.out,
            per_class_when_possible=not args.per_cell,
            sampling_strategy=args.sampling_strategy,
        )
    elif args.cmd == "generate":
        cfg = load_run_config(args.config)
        gen_cfg = cfg.generation
        if args.candidates_per_seed is not None:
            gen_cfg = replace(gen_cfg, candidates_per_seed=args.candidates_per_seed)
        generate_candidates(cfg.manifest_path, args.seed_file, args.out, gen_cfg, args.seed, limit=args.limit, resume=args.resume)
    elif args.cmd == "counterfactuals":
        cfg = load_run_config(args.config)
        build_counterfactuals(cfg.manifest_path, args.candidates, args.out, cfg.generation, args.seed, resume=args.resume)
    elif args.cmd == "judge":
        cfg = load_run_config(args.config)
        from .judge import judge_candidates

        judge_candidates(args.candidates, args.out, cfg.judge, resume=args.resume)
    elif args.cmd == "audit":
        cfg = load_run_config(args.config)
        audit_candidates(args.manifest, args.dataset_id, args.language, args.gold, args.candidates, args.out, cfg.audit)
    elif args.cmd == "select":
        cfg = load_run_config(args.config)
        select_candidates(args.audit, args.gold, args.out, args.budget, args.baseline, cfg.audit, args.seed)
    elif args.cmd == "evaluate":
        evaluate_run(args.manifest, args.dataset_id, args.language, args.gold, args.selected, args.out, args.embedder)
    elif args.cmd == "train-hf":
        from .hf_training import train_hf

        train_hf(
            args.manifest,
            args.dataset_id,
            args.language,
            args.gold,
            args.selected,
            args.out_dir,
            args.model,
            args.epochs,
            args.batch_size,
            args.lr,
            args.max_length,
            args.keep_checkpoints,
            args.seed,
            args.deterministic,
        )
    elif args.cmd == "plan-scenario":
        from .data import load_manifest
        from .scenario import render_commands

        cfg = load_run_config(args.config)
        manifest = load_manifest(cfg.manifest_path)
        commands = render_commands(args.config, manifest, args.run_name)
        if args.out:
            write_jsonl(args.out, [{"cmd": cmd} for cmd in commands])
        else:
            print(shell_header())
            for cmd in commands:
                print(shell_command(cmd))
    elif args.cmd == "plan-training":
        from .data import load_manifest
        from .scenario import render_training_commands

        cfg = load_run_config(args.config)
        manifest = load_manifest(cfg.manifest_path)
        commands = render_training_commands(
            args.config,
            manifest,
            args.run_name,
            classification_model=args.classification_model,
            summarization_model=args.summarization_model,
            deterministic=args.deterministic,
        )
        if args.out:
            write_jsonl(args.out, [{"cmd": cmd} for cmd in commands])
        else:
            print(shell_header())
            for cmd in commands:
                print(shell_command(cmd))
    elif args.cmd == "plan-summary":
        from .data import load_manifest
        from .scenario import summarize_plan

        cfg = load_run_config(args.config)
        manifest = load_manifest(cfg.manifest_path)
        print(json.dumps(summarize_plan(args.config, manifest), indent=2, ensure_ascii=False))
    elif args.cmd == "claim-ledger":
        from .ledger import build_claim_ledger

        build_claim_ledger(args.runs_dir, args.out)


def add_common_cell_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", default="data/manifest/datasets.json")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, default=13)


def shell_header() -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'COSDA="${COSDA:-$(command -v cosda || true)}"',
            'if [ -z "$COSDA" ] && [ -x ".venv/bin/cosda" ]; then COSDA=".venv/bin/cosda"; fi',
            'if [ -z "$COSDA" ]; then echo "cosda command not found; activate .venv or set COSDA=/path/to/cosda" >&2; exit 1; fi',
        ]
    )


def shell_command(cmd: str) -> str:
    if cmd.startswith("cosda "):
        return '"$COSDA" ' + cmd[len("cosda ") :]
    return cmd


if __name__ == "__main__":
    main()
