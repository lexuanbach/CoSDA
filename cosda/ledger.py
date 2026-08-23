from __future__ import annotations

import json
from pathlib import Path

from .io_utils import ensure_parent


def build_claim_ledger(runs_dir: str | Path, output_csv: str | Path) -> list[dict]:
    root = Path(runs_dir)
    rows = []
    for result_path in sorted(root.glob("**/*.json")):
        if "results" not in result_path.parts and result_path.name != "test_metrics.json":
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metrics = result.get("metrics") or result
        for metric, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            rows.append(
                {
                    "claim_id": f"{result_path.parent.name}:{metric}",
                    "metric": metric,
                    "value": value,
                    "result_file": str(result_path),
                    "anchor": "",
                }
            )
    out = ensure_parent(output_csv)
    with out.open("w", encoding="utf-8") as handle:
        handle.write("claim_id,metric,value,result_file,anchor\n")
        for row in rows:
            handle.write(
                f"{csv_escape(row['claim_id'])},{csv_escape(row['metric'])},{row['value']},"
                f"{csv_escape(row['result_file'])},{csv_escape(row['anchor'])}\n"
            )
    return rows


def csv_escape(value: str) -> str:
    value = str(value)
    if any(ch in value for ch in [",", '"', "\n"]):
        return '"' + value.replace('"', '""') + '"'
    return value
