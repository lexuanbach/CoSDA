#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    spec = yaml.safe_load((ROOT / "configs/scenarios/sensitivity.yaml").read_text())
    base = yaml.safe_load((ROOT / spec["base_config"]).read_text())
    out_dir = ROOT / "configs/generated/sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    for param, values in spec["parameters"].items():
        for value in values:
            cfg = copy.deepcopy(base)
            cfg.setdefault("audit", {})[param] = value
            path = out_dir / f"{param}_{str(value).replace('.', 'p')}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            print(path)


if __name__ == "__main__":
    main()
