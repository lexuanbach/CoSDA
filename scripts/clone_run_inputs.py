#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


INPUT_FILENAMES = {
    "gold.jsonl",
    "candidates.jsonl",
    "candidates_cf.jsonl",
    "candidates_judged.jsonl",
    "audit.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clone reusable CoSDA generated inputs from one run tree to another."
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--target-run", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = args.runs_dir / args.source_run
    target_root = args.runs_dir / args.target_run
    if not source_root.exists():
        raise SystemExit(f"source run not found: {source_root}")
    if source_root.resolve() == target_root.resolve():
        raise SystemExit("source and target run must be different")

    copied = 0
    skipped = 0
    for source_path in source_root.rglob("*"):
        if not source_path.is_file() or source_path.name not in INPUT_FILENAMES:
            continue
        relative = source_path.relative_to(source_root)
        target_path = target_root / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and not args.overwrite:
            skipped += 1
            continue
        shutil.copy2(source_path, target_path)
        copied += 1

    print(
        {
            "source_run": args.source_run,
            "target_run": args.target_run,
            "copied": copied,
            "skipped": skipped,
            "target_root": str(target_root),
        }
    )
    if copied == 0 and skipped == 0:
        raise SystemExit("no reusable input files were found")


if __name__ == "__main__":
    main()
