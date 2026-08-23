#!/usr/bin/env python3
"""Download/extract the raw datasets needed by the CoSDA spec."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import requests
from huggingface_hub import snapshot_download
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
RAW_HF = ROOT / "data" / "raw" / "hf"
RAW_EXTERNAL = ROOT / "data" / "raw" / "external"
EXTRACTED = ROOT / "data" / "raw" / "extracted"

LANGS = ["amh", "hau", "swa", "yor"]
XLSUM_LANGS = ["amharic", "hausa", "swahili", "yoruba"]
MASSIVE_LANGS = ["am-ET", "sw-KE"]
NER2_LANGS = ["hau", "swa", "yor"]

NER2_BASE = "https://github.com/masakhane-io/masakhane-ner/raw/main/MasakhaNER2.0/data"
MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"


def download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0"))
        with dest.open("wb") as handle, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    bar.update(len(chunk))


def download_hf_sources() -> None:
    RAW_HF.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            "masakhane/masakhanews",
            [f"data/{lang}/*" for lang in LANGS] + ["README.md"],
        ),
        (
            "masakhane/afrisenti",
            [f"data/{lang}/*" for lang in LANGS] + ["README.md", "data/README.txt"],
        ),
        (
            "csebuetnlp/xlsum",
            [f"data/{lang}_XLSum_v2.0.tar.bz2" for lang in XLSUM_LANGS] + ["README.md"],
        ),
        ("masakhane/masakhaner2", ["README.md", "masakhaner2.py"]),
        ("qanastek/MASSIVE", ["README.md", "MASSIVE.py", "LICENSE", "CITATION.cff"]),
    ]
    for repo, patterns in jobs:
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            allow_patterns=patterns,
            local_dir=RAW_HF / repo.replace("/", "__"),
        )


def extract_xlsum() -> None:
    out_root = EXTRACTED / "xlsum"
    out_root.mkdir(parents=True, exist_ok=True)
    for lang in XLSUM_LANGS:
        archive = RAW_HF / "csebuetnlp__xlsum" / "data" / f"{lang}_XLSum_v2.0.tar.bz2"
        target = out_root / lang
        target.mkdir(parents=True, exist_ok=True)
        expected = [target / f"{lang}_{split}.jsonl" for split in ["train", "val", "test"]]
        if all(path.exists() for path in expected):
            continue
        with tarfile.open(archive, "r:bz2") as tar:
            safe_extract(tar, target)


def download_ner2() -> None:
    out_root = EXTRACTED / "masakhaner2"
    for lang in NER2_LANGS:
        for split in ["train", "dev", "test"]:
            url = f"{NER2_BASE}/{lang}/{split}.txt"
            download_url(url, out_root / lang / f"{split}.txt")


def extract_massive() -> None:
    archive = RAW_EXTERNAL / "massive" / "amazon-massive-dataset-1.0.tar.gz"
    download_url(MASSIVE_URL, archive)
    out_root = EXTRACTED / "massive"
    out_root.mkdir(parents=True, exist_ok=True)
    wanted = {f"1.0/data/{lang}.jsonl": out_root / f"{lang}.jsonl" for lang in MASSIVE_LANGS}
    if all(path.exists() for path in wanted.values()):
        return
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = member.name.lstrip("./")
            if name not in wanted:
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            dest = wanted[name]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source.read())


def safe_extract(tar: tarfile.TarFile, target: Path) -> None:
    target = target.resolve()
    for member in tar.getmembers():
        dest = (target / member.name).resolve()
        if not str(dest).startswith(str(target)):
            raise RuntimeError(f"Unsafe path in archive: {member.name}")
    tar.extractall(target)


def main() -> None:
    download_hf_sources()
    extract_xlsum()
    download_ner2()
    extract_massive()
    status = {
        "hf_root": str(RAW_HF),
        "external_root": str(RAW_EXTERNAL),
        "extracted_root": str(EXTRACTED),
        "xlsum_languages": XLSUM_LANGS,
        "massive_languages": MASSIVE_LANGS,
        "masakhaner2_languages": NER2_LANGS,
    }
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
