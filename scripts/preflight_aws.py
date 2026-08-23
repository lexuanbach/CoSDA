#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosda.config import load_run_config  # noqa: E402
from cosda.data import load_manifest  # noqa: E402
from cosda.scenario import summarize_plan  # noqa: E402


def run(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return 127, "not found"
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def check_import(name: str) -> dict:
    code = f"import {name}; print(getattr({name}, '__version__', 'ok'))"
    rc, out = run([sys.executable, "-c", code])
    return {"ok": rc == 0, "detail": out[:500]}


def check_endpoint(url: str | None) -> dict:
    if not url:
        return {"ok": False, "detail": "not configured"}
    base = url
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    models_url = base.rstrip("/") + "/models"
    try:
        response = requests.get(models_url, timeout=10)
        return {"ok": response.ok, "status": response.status_code, "detail": response.text[:500]}
    except Exception as exc:
        return {"ok": False, "detail": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scenarios/aws_classification.yaml")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-imports", action="store_true")
    parser.add_argument("--check-endpoints", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=200.0)
    args = parser.parse_args()

    cfg = load_run_config(ROOT / args.config)
    report: dict = {"config": args.config, "checks": {}}

    manifest_path = ROOT / cfg.manifest_path
    report["checks"]["manifest"] = {"ok": manifest_path.exists(), "path": str(manifest_path)}
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        cells = sum(len(entry["cells"]) for entry in manifest["datasets"])
        report["checks"]["manifest"]["cells"] = cells
        report["plan_summary"] = summarize_plan(ROOT / args.config, manifest)

    missing_paths = []
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        for entry in manifest["datasets"]:
            for splits in entry["cells"].values():
                for split in splits.values():
                    path = ROOT / split["path"]
                    if not path.exists():
                        missing_paths.append(str(path))
    report["checks"]["raw_data"] = {"ok": not missing_paths, "missing": missing_paths[:20], "missing_count": len(missing_paths)}

    usage = shutil.disk_usage(ROOT)
    report["checks"]["disk"] = {
        "ok": usage.free > args.min_free_gib * 1024**3,
        "free_gib": round(usage.free / 1024**3, 2),
        "required_free_gib": args.min_free_gib,
    }

    rc, nvidia = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"])
    report["checks"]["gpu"] = {"ok": rc == 0, "detail": nvidia[:2000]}
    if args.require_gpu and rc != 0:
        report["checks"]["gpu"]["required_failed"] = True

    imports = ["torch", "transformers", "accelerate", "sklearn", "sentence_transformers", "boto3"]
    report["checks"]["imports"] = {name: check_import(name) for name in imports}
    report["checks"]["imports_ok"] = {
        "ok": all(item.get("ok", False) for item in report["checks"]["imports"].values()),
        "packages": imports,
    }

    if args.check_endpoints:
        report["checks"]["generation_endpoint"] = check_endpoint(cfg.generation.endpoint_url)
        report["checks"]["judge_endpoint"] = check_endpoint(cfg.judge.endpoint_url)

    required_checks = ["manifest", "raw_data", "disk"]
    if args.require_gpu:
        required_checks.append("gpu")
    if args.require_imports:
        required_checks.append("imports_ok")
    if args.check_endpoints and cfg.generation.backend == "openai_compatible":
        required_checks.append("generation_endpoint")
    if args.check_endpoints and cfg.judge.backend == "openai_compatible":
        required_checks.append("judge_endpoint")
    ok = all(report["checks"][key].get("ok", False) for key in required_checks)
    report["ok"] = ok
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
