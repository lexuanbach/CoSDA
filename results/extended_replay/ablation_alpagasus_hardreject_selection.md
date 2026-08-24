# AlpaGasus + CoSDA-V2 hard-reject ablation — selection stage (seed 13)

Test of the underfill hypothesis: apply V2's hard-reject gate, then rank survivors by AlpaGasus judged quality. This measures the gate's total effect and does not separate pool size from pool composition.
Selection is CPU-only on the cached seed-13 audit records. Pool/retained/overlap are recomputable from runs/select1_seed13/*/*/b64_m3_s13/{audit.jsonl,selected/*.jsonl}; the downstream Macro-F1 for these pools is in results/extended_replay/{grid_results,results_multiseed}.json (paper Table 6).

| cell | HR-pass pool | AG+HR retained | budget-starved? | overlap w/ AlpaGasus |
|------|-------------:|---------------:|:---------------:|---------------------:|
| news_topic/amh | 31 | 31 | YES | 19/64 |
| news_topic/hau | 50 | 50 | YES | 29/64 |
| news_topic/swa | 81 | 64 | no | 42/64 |
| news_topic/yor | 87 | 64 | no | 53/64 |
| sentiment/amh | 93 | 64 | no | 55/64 |
| sentiment/hau | 153 | 64 | no | 63/64 |
| sentiment/swa | 131 | 64 | no | 63/64 |
| sentiment/yor | 138 | 64 | no | 62/64 |

**Finding:** in the two cells with the smallest audit-passing pools (news/amh, news/hau),
the gate leaves only 31 and 50 usable candidates, so AlpaGasus+hard-reject cannot fill the
64-example budget. Paper Table 6 reports the downstream effect: the aggregate deficit is
carried by news/amh alone, while news/hau gains +0.026.
