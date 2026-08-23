#!/usr/bin/env python3
"""Collect local machine facts for CoSDA feasibility checks."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"


def run(cmd: list[str]) -> dict:
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return {"available": False, "stdout": "", "stderr": "not found", "returncode": 127}
    return {
        "available": True,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "returncode": proc.returncode,
    }


def first_line(text: str) -> str | None:
    return text.splitlines()[0] if text else None


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    commands = {
        "uname": ["uname", "-a"],
        "sw_vers": ["sw_vers"],
        "cpu_brand": ["sysctl", "-n", "machdep.cpu.brand_string"],
        "cpu_logical": ["sysctl", "-n", "hw.logicalcpu"],
        "cpu_physical": ["sysctl", "-n", "hw.physicalcpu"],
        "memory_bytes": ["sysctl", "-n", "hw.memsize"],
        "df_project": ["df", "-h", str(ROOT)],
        "python": [str(ROOT / ".venv" / "bin" / "python"), "--version"],
        "pip": [str(ROOT / ".venv" / "bin" / "python"), "-m", "pip", "--version"],
    }
    raw = {name: run(cmd) for name, cmd in commands.items()}

    tools = {
        name: shutil.which(name)
        for name in ["git", "git-lfs", "curl", "wget", "uv", "conda", "mamba", "nvidia-smi", "ollama"]
    }

    torch_probe = run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            "-c",
            (
                "import json\n"
                "try:\n"
                " import torch\n"
                " print(json.dumps({'installed': True, 'version': torch.__version__, "
                "'mps_available': torch.backends.mps.is_available(), "
                "'mps_built': torch.backends.mps.is_built()}))\n"
                "except Exception as e:\n"
                " print(json.dumps({'installed': False, 'error': repr(e)}))\n"
            ),
        ]
    )
    try:
        torch = json.loads(torch_probe["stdout"])
    except json.JSONDecodeError:
        torch = {"installed": False, "error": torch_probe}

    memory_bytes = None
    if raw["memory_bytes"]["stdout"].isdigit():
        memory_bytes = int(raw["memory_bytes"]["stdout"])

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "uname": raw["uname"]["stdout"],
            "macos": raw["sw_vers"]["stdout"],
        },
        "hardware": {
            "cpu": first_line(raw["cpu_brand"]["stdout"]),
            "physical_cores": int(raw["cpu_physical"]["stdout"] or 0),
            "logical_cores": int(raw["cpu_logical"]["stdout"] or 0),
            "memory_bytes": memory_bytes,
            "memory_gib": round(memory_bytes / (1024**3), 2) if memory_bytes else None,
            "display": raw["system_profiler_display"]["stdout"],
        },
        "storage": {"df_project": raw["df_project"]["stdout"]},
        "tools": tools,
        "python_env": {
            "python": raw["python"]["stdout"] or raw["python"]["stderr"],
            "pip": raw["pip"]["stdout"] or raw["pip"]["stderr"],
            "torch": torch,
        },
    }

    out = REPORT_DIR / "environment.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
