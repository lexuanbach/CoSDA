# CoSDA — When Audit Quality Fails to Predict Downstream Utility

Artifact for the EMNLP 2026 Findings paper *"When Audit Quality Fails to Predict
Downstream Utility: A Controlled Study of Synthetic-Data Selectors for Low-Resource
African Classification."*

The paper asks whether a synthetic-data selector that scores well under audit metrics
also helps a downstream model learn. In a matched-budget replay over 8 task-language
cells it does not: audit rankings and downstream rankings diverge.

## Reproduce in one command

```bash
./reproduce.sh
```

Needs only `python3` — no GPU and no network. It recomputes every aggregate in the
paper from the committed per-run result files and regenerates the claim ledger. The
committed reference outputs are `results/extended_replay/ANALYSIS_OUTPUT.txt` and
`results/extended_replay/judge_agreement_output.txt`, so a re-run can be diffed
directly against them.

## What is here

```
cosda/                        audit + selection library (channels, judge, selectors, HF training)
scripts/                      pipeline drivers, extended-replay runners, analysis, ledger generator
less_experiment/              a faithful LESS (Xia et al. 2024) implementation and its results
configs/                      scenario configs used for the replays
runs/select1_seed13/          THE SEED-13 REPLAY INPUTS AND OUTPUTS, per task-language cell:
                                audit.jsonl      per-candidate audit records (generated text, U, L, H, D,
                                                 C, S, hard_reject, reject_reasons, counterfactual, judge)
                                gold.jsonl       the 64 gold seed ids (text withheld, see below)
                                selected/*.jsonl each selector's retained pool
results/main_replay/          claim ledger, per-cell rows, generated paper tables
results/extended_replay/      224 extra fine-tunes, multi-judge scores, ablation selection record
```

`runs/select1_seed13/` holds 8 cells x (1 audit file + 1 gold file + 7 selector pools).
These are the pools every number in the paper is computed over.

## Where each number comes from

`results/main_replay/claim_ledger.csv` maps ~100 reported values to the file and
computation that produces each one. Regenerate it with
`python3 scripts/make_claim_ledger.py`. Summary:

| Paper element | File |
|---|---|
| Table 1, Table 8 (audit channels) | `results/extended_replay/audit_aggregates.json` |
| Table 1 Macro-F1, Sec. 4.3 statistics | `results/main_replay/aws_vllm_revised_select1_v2_full_20260521_analysis_rows.csv` |
| Table 2 (head-to-head) | same main-replay CSV + `audit_aggregates.json` |
| Table 3 (per-cell Macro-F1) | same main-replay CSV |
| Table 4 (3 seeds + AfroXLMR) | `results/extended_replay/{grid_results,results_multiseed,results_afroxlmr}.json` |
| Table 5 (multi-judge LC) | `results/extended_replay/judge2_{qwen14b,phi35}.json` |
| Table 6, Figure 4 (AlpaGasus+HR ablation) | the three result JSONs + `ablation_alpagasus_hardreject_selection.md` for pool sizes |
| Diversity column, `C_i` pass rates | `runs/select1_seed13/*/*/b64_m3_s13/audit.jsonl` |
| Real LESS comparison (Table 5) | `less_experiment/RESULTS.json`, `results_*_seed13.jsonl` |
| Judge vs. human labels | `results/extended_replay/judge_human_validation.txt` |
| Per-selector retained pools | `runs/select1_seed13/*/*/b64_m3_s13/selected/` |

### Authoritative source files

These are the files treated as authoritative for the reported claims. Every reported
number should be recoverable from them or from a deterministic aggregation of them.

| Artifact | Use |
|---|---|
| `results/main_replay/cosda_full_v2_paper_summary_20260521.csv` | Human-readable summary of the final CoSDA-supported claims, including the audit/downstream mismatch and its scope conditions. |
| `results/main_replay/aws_vllm_revised_select1_v2_full_20260521_analysis_rows.csv` | Authoritative row-level replay table: selectors, task-language cells, audit channels, and deterministic downstream Macro-F1. A 4-cell pilot CSV ships alongside it; `scripts/analyze_main_replay.py` selects the 8-cell file. |
| `results/main_replay/cosda_paper_tables_20260521.tex` | Generated table source used to cross-check the downstream and audit summary tables. |
| `results/extended_replay/{grid_results,results_multiseed,results_afroxlmr}.json` | Per-run downstream Macro-F1 for the extended replay (224 fine-tunes): seven selectors x eight cells at seeds 13/21/42 on XLM-R, and at seed 13 on AfroXLMR. |
| `results/extended_replay/judge2_{qwen14b,phi35}.json` | Per-candidate label-correctness scores from the two additional judges (1,523 candidates each). |
| `results/extended_replay/ablation_alpagasus_hardreject_selection.md` | Selection-stage record for AlpaGasus+HR: per-cell hard-reject-pass pool size, retained size, and overlap with AlpaGasus's pool. |

## Two replay pipelines, reported separately

The **main replay** is the seed-13 deterministic trace behind Table 1. The **extended
replay** re-fits the *same* seed-13 retained pools at seeds 13/21/42 and on a second
backbone, on different hardware. Audit channels are identical across the two by
construction, and only the downstream fit changes. Absolute Macro-F1 differs slightly
between them, so the paper reports them separately rather than pooling, and reads the
extension for rank behaviour and seed spread.

`reproduce.sh` covers the **extended replay only**, so the seed-13 Macro-F1 column it
prints is the extension's, not Table 1's. Table 1's Macro-F1 column is a deterministic
aggregation of `results/main_replay/aws_vllm_revised_select1_v2_full_20260521_analysis_rows.csv`,
which also ships here.

## Setup for re-running the pipeline itself

```bash
pip install -r requirements.txt
```

The full replay additionally needs `pip install torch transformers accelerate
sentence-transformers scikit-learn seqeval` (every `aws_*` scenario config sets
`embedder: labse`).

Benchmarks are MasakhaNEWS (`masakhane/masakhanews`) and AfriSenti
(`masakhane/afrisenti`) from the Hugging Face Hub. Generation used
Qwen2.5-14B-Instruct, judging used Qwen2.5-32B-Instruct-AWQ, and downstream training
used XLM-RoBERTa base and AfroXLMR base. A full downstream replay costs roughly one
GPU-week per seed. `reproduce.sh` does **not** re-run any of that; it recomputes the
reported aggregates from committed results.

## Source text is not redistributed

We ship the **generated** candidates in full, because they are this paper's own output.
We do **not** ship the source-corpus text they were conditioned on. `gold.jsonl` carries
each seed's dataset id and a SHA-256 of its text but not the text, and the `raw_item`
field has been removed from every candidate record (a `raw_item_sha256` pointer remains).

The reason is upstream rights, not the dataset licences. MasakhaNEWS reproduces article
bodies from BBC, VOA and other outlets, and AfriSenti reproduces tweet text. Neither
Masakhane nor we hold those rights, and X's terms cover tweet text separately.

To restore a byte-identical gold set from the public datasets:

```bash
pip install datasets
python3 scripts/rehydrate_gold.py     # verifies every record against the shipped hash
```

Nothing in `reproduce.sh` needs the gold text; it is required only to re-run training.

## Known scope limits

- No native speaker of the four target languages reviewed the generated text. Every
  audit number here is a machine measurement. See the paper's Limitations section.
- `AlpaGasus+HR` retains fewer than 64 examples in two cells (31 in news/amh, 50 in
  news/hau), so it is not matched-budget there. `ablation_alpagasus_hardreject_selection.md`
  records the exact pool sizes.
- Generated examples are marked as generated in the released records.

## License

Code (`cosda/`, `scripts/`, `configs/`) is Apache-2.0, see `LICENSE`. The released data
carries terms inherited from the source datasets: news-derived candidates are CC BY-NC 4.0
and sentiment-derived candidates are CC BY 4.0. `DATA_LICENSE.md` explains why, and what
rights neither dataset licence grants.

## Citation

```bibtex
@inproceedings{le2026cosda,
  title     = {When Audit Quality Fails to Predict Downstream Utility: A Controlled
               Study of Synthetic-Data Selectors for Low-Resource African-Language Classification},
  author    = {Le, Xuan-Bach and Tran-Truong, Phat T. and Ha Xuan, Son},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```
