#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


HEADER_PREFIXES = ("#!", "set ", "COSDA=", "if ")


def load_lines(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


def split_cells(lines: list[str]) -> list[list[str]]:
    commands = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    commands = [line for line in commands if not line.startswith(HEADER_PREFIXES)]
    cells: list[list[str]] = []
    current: list[str] = []
    for command in commands:
        if " make-seeds " in command and current:
            cells.append(current)
            current = []
        current.append(command)
    if current:
        cells.append(current)
    return cells


def keep_for_phase(command: str, phase: str) -> bool:
    if " make-seeds " in command and phase in {"generation", "judge"}:
        return True
    if phase == "generation":
        return " generate " in command or " counterfactuals " in command
    if phase == "judge":
        return any(token in command for token in (" judge ", " audit ", " select ", " evaluate "))
    if phase == "selection":
        return " select " in command or " evaluate " in command
    raise ValueError(f"unknown phase: {phase}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract dependency-preserving phase commands from a CoSDA scenario shell file.")
    parser.add_argument("scenario_file", type=Path)
    parser.add_argument("--phase", choices=["generation", "judge", "selection"], required=True)
    args = parser.parse_args()

    print("#!/usr/bin/env bash")
    print("set -euo pipefail")
    print('COSDA="${COSDA:-$(command -v cosda || true)}"')
    print('if [ -z "$COSDA" ] && [ -x ".venv/bin/cosda" ]; then COSDA=".venv/bin/cosda"; fi')
    print('if [ -z "$COSDA" ]; then echo "cosda command not found; activate .venv or set COSDA=/path/to/cosda" >&2; exit 1; fi')
    for cell in split_cells(load_lines(args.scenario_file)):
        kept = [command for command in cell if keep_for_phase(command, args.phase)]
        if len(kept) > 1:
            for command in kept:
                print(command)


if __name__ == "__main__":
    main()
