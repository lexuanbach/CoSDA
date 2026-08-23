#!/usr/bin/env python3
"""Restore the gold seed text that this release deliberately does not redistribute.

The gold sets are 64 records per cell drawn from MasakhaNEWS and AfriSenti. We ship
their record ids and a SHA-256 of the original text, but not the text itself, because
MasakhaNEWS reproduces BBC/VOA article bodies and AfriSenti reproduces tweet text, and
neither dataset's licence grants redistribution of that upstream content.

This script re-fetches the text from the source datasets and verifies each record
against the shipped hash, so you end up with a byte-identical gold set.

    pip install datasets
    python3 scripts/rehydrate_gold.py
"""
import glob, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("need `pip install datasets`", file=sys.stderr)
        return 2

    cache, restored, mismatched = {}, 0, 0
    for f in sorted(glob.glob(str(ROOT / "runs/select1_seed13/*/*/b64_m3_s13/gold.jsonl"))):
        rows = [json.loads(l) for l in open(f)]
        before_mismatched = mismatched
        out = []
        for r in rows:
            key = (r["dataset_id"], r["language"], r.get("split", "train"))
            if key not in cache:
                ds = load_dataset(r["dataset_id"], r["language"], split=key[2])
                cache[key] = {str(i): ex for i, ex in enumerate(ds)}
            src = cache[key].get(str(r["id"]).split(":")[-1])
            if src is None:
                print(f"  no source row for {r['id']}", file=sys.stderr)
                out.append(json.dumps(r, ensure_ascii=False)); continue
            # MasakhaNEWS text is headline + blank line + body (see cosda/data.py);
            # AfriSenti is the raw tweet. Getting this wrong silently corrupts the file.
            if r["dataset_id"].endswith("masakhanews"):
                text = f"{src.get('headline','')}\n\n{src.get('text','')}".strip()
            else:
                text = src.get("tweet", "") or ""
            got = hashlib.sha256(text.encode()).hexdigest()[:16]
            if r.get("text_sha256") and got != r["text_sha256"]:
                mismatched += 1
                print(f"  HASH MISMATCH for {r['id']}", file=sys.stderr)
            else:
                r["text"] = text
                restored += 1
            out.append(json.dumps(r, ensure_ascii=False))
        if mismatched == before_mismatched:
            Path(f).write_text("\n".join(out) + "\n")
        else:
            print(f"  not writing {f}: {mismatched - before_mismatched} mismatches", file=sys.stderr)

    print(f"restored {restored} gold records; {mismatched} hash mismatches")
    return 1 if mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
