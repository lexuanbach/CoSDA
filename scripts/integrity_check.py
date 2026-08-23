#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_PATHS = ["README.md", "docs", "configs", "cosda", "scripts"]
FORBIDDEN = [
    (re.compile(r"\bhuman audit\b", re.I), "Use 'LLM-as-judge audit' or 'automatic audit' unless a real human study exists."),
    (re.compile(r"\bKrippendorff", re.I), "Do not report Krippendorff alpha without human annotations."),
    (re.compile(r"\bCohen'?s?\s+kappa\b|\bκ\b", re.I), "Do not report kappa without human annotations."),
    (re.compile(r"\bnative[- ]speaker validation\b", re.I), "Do not claim native-speaker validation without annotators."),
    (re.compile(r"\bannotator agreement\b", re.I), "Do not report annotator agreement without annotators."),
]
ALLOWLIST = {
    "README.md",
    "docs/scientific_integrity.md",
    "docs/evaluation_scenarios.md",
    "docs/evaluation_feasibility.md",
    "docs/aws_runbook.md",
    "scripts/integrity_check.py",
}


def iter_files(paths: list[str]) -> list[Path]:
    files = []
    for item in paths:
        path = Path(item)
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
        else:
            files.extend(p for p in path.rglob("*") if p.is_file())
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=DEFAULT_PATHS)
    parser.add_argument("--strict", action="store_true", help="Also scan protocol docs that mention forbidden phrases as examples.")
    args = parser.parse_args()

    violations = []
    for path in iter_files(args.paths):
        rel = path.as_posix()
        if not args.strict and rel in ALLOWLIST:
            continue
        if path.suffix in {".pyc", ".pdf", ".png", ".jpg", ".jpeg"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern, message in FORBIDDEN:
                if pattern.search(line):
                    violations.append((rel, lineno, line.strip(), message))

    if violations:
        for rel, lineno, line, message in violations:
            print(f"{rel}:{lineno}: {line}\n  -> {message}")
        raise SystemExit(1)
    print("Integrity check passed.")


if __name__ == "__main__":
    main()
