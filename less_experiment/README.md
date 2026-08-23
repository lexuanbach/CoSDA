# Real LESS baseline

A faithful implementation of LESS (Xia et al., ICML 2024) for the CoSDA replay,
replacing the `U+0.25D` local control that the earlier package shipped under the
name "LESS-style".

## What is implemented

Definition 3.1 and Sec. 4.1-4.2 of the paper:

| Stage | Paper | Here |
|---|---|---|
| Warmup | LoRA + AdamW on `D_warmup ⊂ D`, N=4 epochs, checkpoint each epoch | same; `D_warmup = D` (the pool), which Table 5 reports as the best setting |
| Gradient features | per-example loss gradient over LoRA params, batch size 1 | same, over LoRA (rank 8, q/v of all 12 layers) + classification head |
| Adam preconditioning | `Γ = m'/√(v'+ε)`, `m' = β₁m+(1−β₁)g`, `v' = β₂v+(1−β₂)g²`, elementwise, in full parameter space | same, using each checkpoint's stored `exp_avg`/`exp_avg_sq` |
| Projection | Rademacher `Π`, d=8192 | **omitted**: P≈888K here, so Definition 3.1 is computed exactly. The projection is a pure efficiency approximation (Sec. 4.1 Step 2). |
| Validation side | plain gradient, NOT preconditioned | same |
| Score | LR-weighted sum over N checkpoints of cosine, per-example L2 normalised | same |
| Selection | plain global top-k, no diversity/dedup/balance | same |

## Deviations, disclosed

- Encoder-only XLM-RoBERTa base with a sequence-classification head rather than a
  decoder LLM. Definition 3.1 is architecture-agnostic; the paper itself runs a
  Pythia-14M selection model.
- The classification head is randomly initialised, so it is trainable and enters
  the gradient feature alongside the LoRA parameters.
- Validation gradients use up to 128 dev examples rather than a few-shot set;
  more validation data only tightens the influence estimate.
- One subtask (`m = 1`), so the `max_j` over subtasks collapses to the plain mean,
  which the paper explicitly permits.

## Fidelity check

The dominant risk at this scale is too few optimizer steps, which leaves
`exp_avg_sq` at its initialisation and silently collapses `Γ` toward SignGD — the
paper's weakest ablation (Table 9). With batch size 4 on ~190 candidates we take
192 steps per run and the reported non-zero fraction of `exp_avg_sq` is 1.000.

## Reproduce

```bash
python3 less_select.py <task> <lang>      # stages 1-4, writes scores/
python3 less_finetune.py <task> <lang>    # top-k selection + downstream fine-tune
./run_all.sh && ./run_ft.sh               # all eight cells
./run_comparators.sh                      # the same loop for U+D, naive, AlpaGasus
```

`less_finetune.py --selector <name>` runs any of the replay's existing pools
through the identical training loop, which is what makes the comparison valid:
the LESS number is not commensurable with the main-replay or extended-replay
tables, since those used different pipelines.
