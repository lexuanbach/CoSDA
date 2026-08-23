#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosda.ledger import build_claim_ledger  # noqa: E402


INCLUDE_SUFFIXES = {".json", ".jsonl", ".csv", ".log", ".txt", ".yaml", ".yml", ".md", ".sh"}
EXCLUDE_SUFFIXES = {".bin", ".safetensors", ".pt", ".pth", ".ckpt", ".arrow"}
EXCLUDE_PART_PREFIXES = ("checkpoint-",)
LOG_DIR_NAMES = ("cell_queue_logs", "queue_logs")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_skip(path: Path, root: Path, max_file_mib: float) -> str | None:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    for part in rel.parts:
        if any(part.startswith(prefix) for prefix in EXCLUDE_PART_PREFIXES):
            return "checkpoint"
    if path.suffix in EXCLUDE_SUFFIXES:
        return f"large_model_artifact:{path.suffix}"
    size_mib = path.stat().st_size / 1024**2
    if size_mib > max_file_mib:
        return f"over_size_limit:{size_mib:.1f}MiB"
    if path.suffix not in INCLUDE_SUFFIXES:
        return f"suffix_not_included:{path.suffix or '<none>'}"
    return None


def iter_run_files(run_root: Path, max_file_mib: float) -> tuple[list[Path], list[dict]]:
    files: list[Path] = []
    skipped: list[dict] = []
    for path in sorted(run_root.rglob("*")):
        if not path.is_file():
            continue
        reason = should_skip(path, run_root, max_file_mib)
        if reason is None:
            files.append(path)
        else:
            skipped.append({"path": str(path), "reason": reason})
    return files, skipped


def iter_log_files(runs_dir: Path, max_file_mib: float) -> tuple[list[tuple[Path, Path]], list[dict]]:
    files: list[tuple[Path, Path]] = []
    skipped: list[dict] = []
    for dirname in LOG_DIR_NAMES:
        log_root = runs_dir / dirname
        if not log_root.exists():
            continue
        for path in sorted(log_root.rglob("*")):
            if not path.is_file():
                continue
            reason = should_skip(path, log_root, max_file_mib)
            if reason is None:
                files.append((path, Path("logs") / dirname / path.relative_to(log_root)))
            else:
                skipped.append({"path": str(path), "reason": reason})
    return files, skipped


def add_file(tar: tarfile.TarFile, source: Path, arcname: Path, manifest_files: list[dict]) -> None:
    info = {
        "archive_path": str(arcname),
        "source_path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": sha256(source),
    }
    manifest_files.append(info)
    tar.add(source, arcname=str(arcname))


def main() -> None:
    parser = argparse.ArgumentParser(description="Package essential CoSDA run artifacts without bulky checkpoints.")
    parser.add_argument("--runs-dir", default=os.environ.get("COSDA_RUNS", "runs"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out-dir", default=os.environ.get("COSDA_ARTIFACTS", "artifacts"))
    parser.add_argument("--config", help="Scenario YAML to include in the archive.")
    parser.add_argument("--max-file-mib", type=float, default=128.0)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    run_root = runs_dir / args.run_name
    if not run_root.exists():
        raise SystemExit(f"Run directory not found: {run_root}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle_path = out_dir / f"{args.run_name}_essential_{timestamp}.tar.gz"
    ledger_path = out_dir / f"{args.run_name}_claim_ledger_{timestamp}.csv"
    manifest_path = out_dir / f"{args.run_name}_artifact_manifest_{timestamp}.json"

    build_claim_ledger(run_root, ledger_path)

    run_files, skipped = iter_run_files(run_root, args.max_file_mib)
    log_files, skipped_logs = iter_log_files(runs_dir, args.max_file_mib)
    skipped.extend(skipped_logs)

    manifest_files: list[dict] = []
    manifest = {
        "run_name": args.run_name,
        "created_at_utc": timestamp,
        "runs_dir": str(runs_dir),
        "run_root": str(run_root),
        "max_file_mib": args.max_file_mib,
        "files": manifest_files,
        "skipped": skipped,
    }

    with tarfile.open(bundle_path, "w:gz") as tar:
        for path in run_files:
            add_file(tar, path, Path(args.run_name) / path.relative_to(run_root), manifest_files)
        for source, arcname in log_files:
            add_file(tar, source, arcname, manifest_files)
        add_file(tar, ledger_path, Path("metadata") / ledger_path.name, manifest_files)
        if args.config:
            config_path = Path(args.config)
            if config_path.exists():
                add_file(tar, config_path, Path("metadata") / config_path.name, manifest_files)
        if Path("scratch.env").exists():
            add_file(tar, Path("scratch.env"), Path("metadata") / "scratch.env", manifest_files)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        add_file(tar, manifest_path, Path("metadata") / manifest_path.name, manifest_files)

    print(
        json.dumps(
            {
                "bundle": str(bundle_path),
                "manifest": str(manifest_path),
                "claim_ledger": str(ledger_path),
                "included_files": len(manifest_files),
                "skipped_files": len(skipped),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
