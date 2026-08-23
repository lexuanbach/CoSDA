#!/usr/bin/env python3
"""Verify every released result file against its digest in PROVENANCE.md.

Exits non-zero on any missing or mismatched entry, so a stale manifest fails
the reproduction run instead of silently misleading a reader.
"""
import hashlib, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
prov = (ROOT / "PROVENANCE.md").read_text()

problems = []
checked = 0
for pat in ("results/**/*", "runs/**/*.jsonl"):
    for p in sorted(ROOT.glob(pat)):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        blob = p.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()[:16]
        m = re.search(re.escape(rel) + r"` \| (\d+) \| `([0-9a-f]{16})", prov)
        checked += 1
        if not m:
            problems.append(f"missing from PROVENANCE.md: {rel}")
        elif int(m.group(1)) != len(blob) or m.group(2) != digest:
            problems.append(f"stale digest: {rel} (recorded {m.group(1)}B/{m.group(2)}, actual {len(blob)}B/{digest})")

for line in problems:
    print(f"  {line}", file=sys.stderr)
print(f"checked {checked} released files; {len(problems)} problem(s)")
sys.exit(1 if problems else 0)
