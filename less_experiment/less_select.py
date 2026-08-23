#!/usr/bin/env python3
"""LESS (Xia et al., ICML 2024) data selection, implemented for the CoSDA replay.

Follows Definition 3.1 and Sec. 4.1-4.2 of the paper:

  Stage 1  Warmup-train a selection model on the candidate pool with LoRA and
           AdamW for N=4 epochs, checkpointing model + optimizer state each epoch
           (Sec. 4.1 Step 1; D_warmup = D is the paper's best setting, Table 5).
  Stage 2  Per-example gradients at batch size 1 over the trainable (LoRA + head)
           parameters (Sec. 4.1 Step 1; Appendix E).
  Stage 3  Adam preconditioning in FULL parameter space, using the checkpoint's
           stored moments plus the candidate's own gradient (Sec. 3.1):
               m' = b1*m + (1-b1)*g ;  v' = b2*v + (1-b2)*g^2 ;  G = m'/sqrt(v'+eps)
  Stage 4  Validation side is the PLAIN gradient at the same checkpoint
           (Definition 3.1; the reference repo forces --gradient_type sgd).
  Score    Sum over checkpoints of avg-epoch-LR-weighted cosine, then global top-k.

We compute Definition 3.1 exactly in R^P and omit the random projection, which the
paper introduces purely for efficiency (Sec. 4.1 Step 2); P is ~0.9M here.
"""
import argparse, json, math, os, os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent   # the artifact root
RAW = Path(os.environ.get("COSDA_RAW", ROOT / "data/raw/hf"))
POOLS = Path(os.environ.get("COSDA_POOLS", ROOT / "runs/select1_seed13"))
MODEL = "FacebookAI/xlm-roberta-base"

LORA_RANK, LORA_ALPHA = 8, 32
EPOCHS, LR, BATCH, MAXLEN = 4, 2e-5, 4, 256
B1, B2, EPS = 0.9, 0.999, 1e-8


def device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


class LoRALinear(nn.Module):
    def __init__(self, base, r=LORA_RANK, alpha=LORA_ALPHA):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad_(False)
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + ((x @ self.A.T) @ self.B.T) * self.scale


def add_lora(model):
    names = [n for n, _ in model.named_modules()
             if n.endswith("attention.self.query") or n.endswith("attention.self.value")]
    for n in names:
        parent, leaf = model.get_submodule(n.rsplit(".", 1)[0]), n.rsplit(".", 1)[1]
        setattr(parent, leaf, LoRALinear(getattr(parent, leaf)))


def load_dev(task, lang, labels):
    if task.startswith("news"):
        df = pd.read_csv(RAW / "masakhane__masakhanews/data" / lang / "dev.tsv", sep="\t")
        txt = (df["headline"].fillna("") + " " + df["text"].fillna("")).tolist()
        lab = df["category"].astype(str).tolist()
    else:
        df = pd.read_csv(RAW / "masakhane__afrisenti/data" / lang / "dev.tsv", sep="\t")
        txt, lab = df["tweet"].fillna("").tolist(), df["label"].astype(str).tolist()
    return [(t, l) for t, l in zip(txt, lab) if l in labels and isinstance(t, str) and t.strip()]


def flat(params, grads=True):
    out = []
    for p in params:
        g = p.grad if grads else p
        out.append((g if g is not None else torch.zeros_like(p)).detach().reshape(-1).float())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task"); ap.add_argument("lang")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--dev-n", type=int, default=256)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev_t = device()

    cell = POOLS / a.task / a.lang / "b64_m3_s13"
    cands = [json.loads(l) for l in open(cell / "audit.jsonl")]
    gold = [json.loads(l) for l in open(cell / "gold.jsonl")]
    labels = sorted({str(g["label"]) for g in gold})
    l2i = {l: i for i, l in enumerate(labels)}

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(labels))
    for p in model.parameters(): p.requires_grad_(False)   # freeze the backbone first
    add_lora(model)                                        # LoRA A/B are the trainable adapters
    for p in model.classifier.parameters(): p.requires_grad_(True)
    model.to(dev_t)
    params = [p for p in model.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)

    def enc(texts):
        return tok(texts, truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev_t)

    # Stage 1: warm up on the candidate pool (D_warmup = D), N=4 epochs
    opt = torch.optim.AdamW(params, lr=LR, betas=(B1, B2), eps=EPS, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.1, EPOCHS * max(1, len(cands) // BATCH))
    cx = [c["text"] for c in cands]
    cy = torch.tensor([l2i.get(str(c["label"]), 0) for c in cands])
    devset = load_dev(a.task, a.lang, set(labels))[: a.dev_n]

    ckpts = []   # (avg_lr_this_epoch, {name: (m, v)}, model state)
    steps_total = 0
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(cx)); lrs = []
        for i in range(0, len(cx), BATCH):
            idx = perm[i:i + BATCH]
            loss = model(**enc([cx[j] for j in idx]), labels=cy[idx].to(dev_t)).loss
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            lrs.append(sched.get_last_lr()[0]); steps_total += 1
        mv = []
        for p in params:
            st = opt.state.get(p, {})
            mv.append((st.get("exp_avg", torch.zeros_like(p)).detach().reshape(-1).float().clone(),
                       st.get("exp_avg_sq", torch.zeros_like(p)).detach().reshape(-1).float().clone()))
        ckpts.append((float(np.mean(lrs)), mv, {k: v.detach().clone() for k, v in model.state_dict().items()}))
    m_all = torch.cat([m for m, _ in ckpts[-1][1]]); v_all = torch.cat([v for _, v in ckpts[-1][1]])
    print(f"  P={P:,} trainable | {steps_total} optimizer steps | "
          f"exp_avg_sq nonzero frac {float((v_all>0).float().mean()):.3f} "
          f"(degenerate if ~0 -> Gamma collapses to SignGD)", flush=True)

    # Stages 2-4, accumulated over checkpoints
    w = np.array([c[0] for c in ckpts]); w = w / w.sum()
    total = torch.zeros(len(cands))
    for ci, (lr_i, mv, state) in enumerate(ckpts):
        model.load_state_dict(state)
        m_t = torch.cat([m for m, _ in mv]); v_t = torch.cat([v for _, v in mv])

        # validation side: PLAIN gradient (Definition 3.1)
        model.eval(); opt.zero_grad()
        for i in range(0, len(devset), 8):
            ch = devset[i:i + 8]
            y = torch.tensor([l2i[l] for _, l in ch]).to(dev_t)
            (model(**enc([t for t, _ in ch]), labels=y).loss * len(ch) / len(devset)).backward()
        gval = flat(params).cpu(); gval = gval / (gval.norm() + EPS)

        # candidate side: Adam-preconditioned, batch size 1
        sims = torch.zeros(len(cands))
        for k, c in enumerate(cands):
            opt.zero_grad()
            y = torch.tensor([l2i.get(str(c["label"]), 0)]).to(dev_t)
            model(**enc([c["text"]]), labels=y).loss.backward()
            g = flat(params).cpu()
            mp = B1 * m_t.cpu() + (1 - B1) * g
            vp = B2 * v_t.cpu() + (1 - B2) * g * g
            G = mp / torch.sqrt(vp + EPS)
            G = G / (G.norm() + EPS)
            sims[k] = torch.dot(G, gval)
        total += float(w[ci]) * sims
        print(f"  ckpt {ci+1}/{EPOCHS} (w={w[ci]:.3f}) done", flush=True)

    scores = total.tolist()
    out = [{"candidate_id": c["candidate_id"], "influence_score": s} for c, s in zip(cands, scores)]
    Path("scores").mkdir(exist_ok=True)
    json.dump(out, open(f"scores/{a.task}_{a.lang}.json", "w"), indent=1)
    print(f"  {a.task}/{a.lang}: {len(out)} scored, range [{min(scores):+.4f}, {max(scores):+.4f}]")


if __name__ == "__main__":
    main()
