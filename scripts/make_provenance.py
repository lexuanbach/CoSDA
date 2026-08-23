#!/usr/bin/env python3
"""Regenerate PROVENANCE.md digests for every released result file."""
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
lines = ["# Provenance\n",
"Digests of every released result file, so a reader can confirm the files behind the\npaper's numbers are the ones shipped here. Regenerate with `scripts/make_provenance.py`.\n",
"| file | bytes | sha256 (first 16) |", "|---|---:|---|"]
for p in sorted(list(ROOT.glob("results/**/*")) + list(ROOT.glob("runs/**/*.jsonl"))):
    if p.is_file():
        b = p.read_bytes()
        lines.append(f"| `{p.relative_to(ROOT)}` | {len(b)} | `{hashlib.sha256(b).hexdigest()[:16]}` |")
(ROOT / "PROVENANCE.md").write_text("\n".join(lines) + "\n")
print("wrote PROVENANCE.md")
