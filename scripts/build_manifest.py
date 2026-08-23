#!/usr/bin/env python3
"""Build a compact dataset manifest for CoSDA inputs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_HF = ROOT / "data" / "raw" / "hf"
EXTRACTED = ROOT / "data" / "raw" / "extracted"
MANIFEST_DIR = ROOT / "data" / "manifest"


def count_tsv(path: Path, label_col: str) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    rows = 0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows += 1
            labels[row[label_col]] += 1
    return {"rows": rows, "labels": dict(sorted(labels.items()))}


def count_jsonl(path: Path, label_key: str | None = None) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            if label_key:
                labels[str(json.loads(line)[label_key])] += 1
    result: dict[str, Any] = {"rows": rows}
    if label_key:
        result["labels"] = dict(sorted(labels.items()))
    return result


def count_conll_sentences(path: Path) -> dict[str, Any]:
    sentences = 0
    tokens = 0
    entities: Counter[str] = Counter()
    active = False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                if active:
                    sentences += 1
                    active = False
                continue
            active = True
            parts = line.split()
            if len(parts) >= 2:
                tokens += 1
                tag = parts[-1]
                if tag != "O":
                    entities[tag] += 1
    if active:
        sentences += 1
    return {"rows": sentences, "tokens": tokens, "labels": dict(sorted(entities.items()))}


def classification_entry(dataset_id: str, task: str, repo_dir: Path, langs: list[str], label_col: str) -> dict:
    splits = ["train", "dev", "test"]
    cells = {}
    for lang in langs:
        cells[lang] = {}
        for split in splits:
            path = repo_dir / "data" / lang / f"{split}.tsv"
            cells[lang][split] = {
                "path": str(path.relative_to(ROOT)),
                **count_tsv(path, label_col),
            }
    return {"dataset_id": dataset_id, "task": task, "cells": cells}


def xlsum_entry() -> dict:
    lang_map = {"amh": "amharic", "hau": "hausa", "swa": "swahili", "yor": "yoruba"}
    split_map = {"train": "train", "dev": "val", "test": "test"}
    cells = {}
    for code, lang in lang_map.items():
        cells[code] = {}
        for split, file_split in split_map.items():
            path = EXTRACTED / "xlsum" / lang / f"{lang}_{file_split}.jsonl"
            cells[code][split] = {
                "path": str(path.relative_to(ROOT)),
                **count_jsonl(path),
            }
    return {"dataset_id": "csebuetnlp/xlsum", "task": "summarization", "cells": cells}


def ner2_entry() -> dict:
    cells = {}
    for lang in ["hau", "swa", "yor"]:
        cells[lang] = {}
        for split in ["train", "dev", "test"]:
            path = EXTRACTED / "masakhaner2" / lang / f"{split}.txt"
            cells[lang][split] = {
                "path": str(path.relative_to(ROOT)),
                **count_conll_sentences(path),
            }
    return {
        "dataset_id": "masakhane/masakhaner2",
        "task": "named_entity_recognition",
        "cells": cells,
        "notes": ["MasakhaNER2 source does not include Amharic; only hau/swa/yor are prepared for the CoSDA language set."],
    }


def massive_entry() -> dict:
    lang_map = {"amh": "am-ET", "swa": "sw-KE"}
    split_map = {"train": "train", "dev": "dev", "test": "test"}
    cells = {}
    for code, locale in lang_map.items():
        source = EXTRACTED / "massive" / f"{locale}.jsonl"
        by_partition = {split: Counter() for split in split_map}
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                partition = row["partition"]
                if partition in by_partition:
                    by_partition[partition][str(row["intent"])] += 1
        cells[code] = {}
        for split, partition in split_map.items():
            cells[code][split] = {
                "path": str(source.relative_to(ROOT)),
                "rows": sum(by_partition[partition].values()),
                "labels": dict(sorted(by_partition[partition].items())),
            }
    return {"dataset_id": "qanastek/MASSIVE", "task": "intent_slot_filling", "cells": cells}


def main() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "datasets": [
            classification_entry(
                "masakhane/masakhanews",
                "news_topic_classification",
                RAW_HF / "masakhane__masakhanews",
                ["amh", "hau", "swa", "yor"],
                "category",
            ),
            classification_entry(
                "masakhane/afrisenti",
                "sentiment_classification",
                RAW_HF / "masakhane__afrisenti",
                ["amh", "hau", "swa", "yor"],
                "label",
            ),
            ner2_entry(),
            massive_entry(),
            xlsum_entry(),
        ],
        "known_spec_mismatches": [
            "CoSDA.pdf claims MasakhaNER 2.0 covers Amharic, but the current MasakhaNER2 source configs do not include amh.",
            "CoSDA.pdf mentions 16 task-language cells, but the listed tasks/languages imply 18 cells before the MasakhaNER2 Amharic mismatch.",
        ],
    }
    out = MANIFEST_DIR / "datasets.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
