#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosda.config import load_run_config  # noqa: E402
from cosda.data import load_manifest  # noqa: E402
from cosda.scenario import expand_cells  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate materialized CoSDA run files before HF replay or after HF replay.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=["replay-inputs", "hf-results"], default="replay-inputs")
    parser.add_argument("--min-selected", type=int, default=1)
    args = parser.parse_args()

    cfg = load_run_config(ROOT / args.config)
    manifest = load_manifest(ROOT / cfg.manifest_path)
    report = {
        "config": args.config,
        "run_name": args.run_name,
        "mode": args.mode,
        "checks": [],
        "ok": True,
    }

    for cell in expand_cells(ROOT / args.config, manifest):
        slug = f"{args.run_name}/{cell['task']}/{cell['language']}/b{cell['budget']}_m{cell['multiplier']}_s{cell['seed']}"
        base = Path(cfg.output_dir) / slug
        baselines = cfg.baselines or ["cosda"]

        if args.mode == "replay-inputs":
            add_jsonl_check(report, base / "gold.jsonl", "gold", minimum=1, cell=cell)
            for baseline in baselines:
                if baseline == "gold_only":
                    continue
                add_jsonl_check(
                    report,
                    base / "selected" / f"{baseline}.jsonl",
                    f"selected:{baseline}",
                    minimum=args.min_selected,
                    cell=cell,
                )
        elif args.mode == "hf-results":
            for baseline in baselines:
                add_json_check(report, base / "hf" / baseline / "test_metrics.json", f"hf:{baseline}", cell=cell)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["ok"] else 1)


def add_jsonl_check(report: dict, path: Path, name: str, minimum: int, cell: dict) -> None:
    exists = path.exists()
    rows = count_lines(path) if exists else 0
    ok = exists and rows >= minimum
    report["checks"].append(check_row(path, name, ok, cell, rows=rows, minimum=minimum))
    report["ok"] = report["ok"] and ok


def add_json_check(report: dict, path: Path, name: str, cell: dict) -> None:
    exists = path.exists()
    ok = False
    detail = None
    if exists:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            ok = bool(metric_value(obj) is not None)
            detail = {"metric": metric_value(obj)}
        except Exception as exc:
            detail = {"error": repr(exc)}
    report["checks"].append(check_row(path, name, ok, cell, detail=detail))
    report["ok"] = report["ok"] and ok


def check_row(path: Path, name: str, ok: bool, cell: dict, **extra) -> dict:
    return {
        "ok": ok,
        "name": name,
        "path": str(path),
        "dataset_id": cell["dataset_id"],
        "task": cell["task"],
        "language": cell["language"],
        "budget": cell["budget"],
        "multiplier": cell["multiplier"],
        "seed": cell["seed"],
        **extra,
    }


def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for line in f if line.strip())


def metric_value(metrics: dict) -> float | None:
    for key in ("test_macro_f1", "eval_macro_f1", "macro_f1", "f1_macro", "test_entity_f1", "entity_f1"):
        if key in metrics:
            return float(metrics[key])
    return None


if __name__ == "__main__":
    main()
