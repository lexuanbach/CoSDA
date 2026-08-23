#!/usr/bin/env python3
"""Select top-k by LESS influence, then fine-tune and evaluate, matching the replay.

Selection is plain global top-k with no diversity, dedup, or balance constraint
(LESS Sec. 4.2; write_selected_data.py). Downstream training mirrors the CoSDA
replay: gold seeds + retained synthetic, XLM-R base, 3 epochs, bs 16, lr 2e-5.
"""
import argparse, json, os
from pathlib import Path

import numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent   # the artifact root
RAW = Path(os.environ.get("COSDA_RAW", ROOT / "data/raw/hf"))
POOLS = Path(os.environ.get("COSDA_POOLS", ROOT / "runs/select1_seed13"))
MODEL = "FacebookAI/xlm-roberta-base"
EPOCHS, BATCH, LR, MAXLEN = 3, 16, 2e-5, 256


def device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


def load_split(task, lang, split, labels):
    if task.startswith("news"):
        df = pd.read_csv(RAW / "masakhane__masakhanews/data" / lang / f"{split}.tsv", sep="\t")
        txt = (df["headline"].fillna("") + " " + df["text"].fillna("")).tolist()
        lab = df["category"].astype(str).tolist()
    else:
        df = pd.read_csv(RAW / "masakhane__afrisenti/data" / lang / f"{split}.tsv", sep="\t")
        txt, lab = df["tweet"].fillna("").tolist(), df["label"].astype(str).tolist()
    return [(t, l) for t, l in zip(txt, lab) if l in labels and isinstance(t, str) and t.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task"); ap.add_argument("lang")
    ap.add_argument("--seed", type=int, default=13); ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--invert", action="store_true", help="select bottom-k (sign diagnostic)")
    ap.add_argument("--balanced", action="store_true",
                    help="apply the replay's label quotas to the LESS ranking, so LESS's "
                         "ranking is tested separately from its unconstrained top-k")
    ap.add_argument("--selector", default="less",
                    help="'less' uses scores/, anything else reads the replay's selected/<name>.jsonl")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dv = device()

    cell = POOLS / a.task / a.lang / "b64_m3_s13"
    cands = {json.loads(l)["candidate_id"]: json.loads(l) for l in open(cell / "audit.jsonl")}
    gold = [json.loads(l) for l in open(cell / "gold.jsonl")]
    labels = sorted({str(g["label"]) for g in gold}); l2i = {l: i for i, l in enumerate(labels)}

    if a.selector == "less":
        scored = json.load(open(f"scores/{a.task}_{a.lang}.json"))
        scored.sort(key=lambda r: (r["influence_score"] if a.invert else -r["influence_score"]))
        ranked = [cands[r["candidate_id"]] for r in scored if r["candidate_id"] in cands]
        if a.balanced:
            # same rule the replay applies to every other selector: per-label quotas
            # rounded to the gold prior, overflow deferred and used as backfill.
            from collections import Counter
            gc = Counter(str(g["label"]) for g in gold); tot = sum(gc.values())
            caps = {l: max(1, round(a.budget * n / tot)) for l, n in gc.items()}
            d = a.budget - sum(caps.values())
            for l, _ in gc.most_common():
                if d == 0: break
                caps[l] += 1 if d > 0 else -1; d += -1 if d > 0 else 1
            used = Counter(); picked = []; deferred = []
            for r in ranked:
                lab = str(r["label"])
                if used[lab] < caps.get(lab, a.budget): picked.append(r); used[lab] += 1
                else: deferred.append(r)
                if len(picked) >= a.budget: break
            picked += deferred[: max(0, a.budget - len(picked))]
        else:
            picked = ranked[: a.budget]
        Path("selected").mkdir(exist_ok=True)
        with open(f"selected/{a.task}_{a.lang}.jsonl", "w") as f:
            for r in picked: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        picked = [json.loads(l) for l in open(cell / "selected" / f"{a.selector}.jsonl")]

    train = [(g["text"], str(g["label"])) for g in gold] + [(c["text"], str(c["label"])) for c in picked]
    test = load_split(a.task, a.lang, "test", set(labels))

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(labels)).to(dv)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    def enc(ts): return tok(ts, truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dv)

    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(train))
        for i in range(0, len(train), BATCH):
            idx = perm[i:i + BATCH]
            xb = [train[j][0] for j in idx]; yb = torch.tensor([l2i[train[j][1]] for j in idx]).to(dv)
            loss = model(**enc(xb), labels=yb).loss
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval(); preds, gts = [], []
    with torch.no_grad():
        for i in range(0, len(test), 32):
            ch = test[i:i + 32]
            logits = model(**enc([t for t, _ in ch])).logits
            preds += logits.argmax(-1).cpu().tolist(); gts += [l2i[l] for _, l in ch]
    f1 = f1_score(gts, preds, average="macro")
    print(json.dumps({"selector": a.selector, "task": a.task, "lang": a.lang, "seed": a.seed,
                      "n_selected": len(picked), "n_train": len(train),
                      "n_test": len(test), "macro_f1": f1}))


if __name__ == "__main__":
    main()
