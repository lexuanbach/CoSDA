#!/usr/bin/env python3
"""Regenerate the camera-ready claim ledger from the released result files.

Every row maps a number that appears in the paper to the file and computation
that produces it. Run from anywhere:  python3 scripts/make_claim_ledger.py
"""
import csv, json, os
from pathlib import Path
from statistics import mean, pstdev
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
RES  = Path(os.environ.get("COSDA_RESULTS", ROOT / "results" / "extended_replay"))
MAIN = ROOT / "results" / "main_replay"
OUT  = MAIN / "claim_ledger.csv"

audit = json.load(open(RES / "audit_aggregates.json"))
grid  = json.load(open(RES / "grid_results.json"))
ms    = json.load(open(RES / "results_multiseed.json"))
afro  = json.load(open(RES / "results_afroxlmr.json"))
rows  = grid + ms

NAME = {"naive":"Naive","alpagasus_style":"AlpaGasus","deita_style":"DEITA",
        "cosda_equal_budget":"CoSDA-EQ","cosda_rerank_v2":"CoSDA-V2",
        "alpagasus_hardreject":"AlpaGasus+HR","less_style":"U+D control"}
SEEDS = (13, 21, 42)

def by_seed(rs):
    d = defaultdict(lambda: defaultdict(list))
    for r in rs:
        if r.get("macro_f1") is not None:
            d[r["seed"]][r["baseline"]].append(r["macro_f1"])
    return d

led = []
def add(cid, where, metric, value, src, how):
    led.append(dict(claim_id=cid, paper_location=where, metric=metric,
                    value=f"{value:.4f}" if isinstance(value, float) else value,
                    result_file=src, computation=how))

# --- Table 1: audit channels (main replay, seed 13) ---
IN_T1 = {"naive","alpagasus_style","deita_style","cosda_equal_budget","cosda_rerank_v2"}
for b, a in audit.items():
    for ch in ("LC","Q","C","D","H","L","HR"):
        where = ("Table 1 / Table 10" if b in IN_T1
                 else ("Table 3 (LC col)" if ch == "LC" else "not reported in the paper"))
        if ch not in a: continue
        add(f"tab1:{b}:{ch}", where, ch, a[ch],
            "results/extended_replay/audit_aggregates.json",
            f"mean of {ch} over the 8 retained pools of selector {b}")

# --- Table 3: 3-seed XLM-R means and AfroXLMR ---
d = by_seed(rows)
for b in NAME:
    per = [mean(d[s][b]) for s in SEEDS]
    add(f"tab2:{b}:f1_3seed", "Table 3 (XLM-R col)", "macro_f1_mean", mean(per),
        "results/extended_replay/{grid_results,results_multiseed}.json",
        "mean over 8 cells per seed, then mean over seeds 13/21/42")
    add(f"tab2:{b}:f1_3seed_sd", "Table 3 (XLM-R col)", "macro_f1_sd", pstdev(per),
        "results/extended_replay/{grid_results,results_multiseed}.json",
        "population sd of the three per-seed means")
af = defaultdict(list)
for r in afro:
    if r.get("macro_f1") is not None: af[r["baseline"]].append(r["macro_f1"])
for b, v in af.items():
    add(f"tab2:{b}:afroxlmr", "Table 3 (AfroXLMR col)", "macro_f1_mean", mean(v),
        "results/extended_replay/results_afroxlmr.json", "mean over 8 cells at seed 13")

# --- Table 6 / Figure 3: AlpaGasus+HR ablation ---
# Recomputed from the shipped audit records rather than hard-coded, so a stale
# literal cannot silently move a paper number.
import glob
HRPOOL = {}
for _f in glob.glob(str(ROOT / "runs" / "select1_seed13" / "*" / "*" / "b64_m3_s13" / "audit.jsonl")):
    _parts = _f.split("/"); _task, _lang = _parts[-4][:4], _parts[-3]
    HRPOOL[(_task, _lang)] = sum(1 for _l in open(_f) if not json.loads(_l).get("hard_reject"))
cell = defaultdict(list)
for r in rows:
    if r.get("macro_f1") is not None:
        cell[(r["task"][:4], r["lang"], r["baseline"])].append(r["macro_f1"])
deltas = {}
for k, pool in HRPOOL.items():
    ag = mean(cell[(k[0],k[1],"alpagasus_style")]); hr = mean(cell[(k[0],k[1],"alpagasus_hardreject")])
    deltas[k] = hr - ag
    add(f"tab4:{k[0]}/{k[1]}:pool", "Table 6 (HR pool)", "n_surviving", pool,
        "results/extended_replay/ablation_alpagasus_hardreject_selection.md", "hard-reject-pass pool size")
    add(f"tab4:{k[0]}/{k[1]}:AG", "Table 6", "macro_f1", ag,
        "results/extended_replay/{grid_results,results_multiseed}.json", "mean over seeds 13/21/42")
    add(f"tab4:{k[0]}/{k[1]}:AGHR", "Table 6", "macro_f1", hr,
        "results/extended_replay/{grid_results,results_multiseed}.json", "mean over seeds 13/21/42")
    add(f"tab4:{k[0]}/{k[1]}:delta", "Table 6 / Figure 3", "delta_macro_f1", hr - ag,
        "results/extended_replay/{grid_results,results_multiseed}.json", "AG+HR minus AG")
add("tab4:mean:delta", "Table 6 (mean row) / abstract", "delta_macro_f1", mean(deltas.values()),
    "results/extended_replay/{grid_results,results_multiseed}.json", "mean of the 8 per-cell deltas")
add("analysis:delta_excl_newsamh", "Section 5.1", "delta_macro_f1",
    mean([v for k,v in deltas.items() if k != ("news","amh")]),
    "results/extended_replay/{grid_results,results_multiseed}.json", "mean delta over the 7 non-amh-news cells")
add("analysis:delta_fullpool", "Section 5.1", "delta_macro_f1",
    mean([v for k,v in deltas.items() if HRPOOL[k] >= 64]),
    "results/extended_replay/{grid_results,results_multiseed}.json", "mean delta over the 6 unstarved cells")

# --- Table 5: multi-judge LC ---
for tag, fn in (("qwen14b","judge2_qwen14b.json"), ("phi35","judge2_phi35.json")):
    add(f"tab3:{tag}:n", "Table 5", "n_candidates_scored",
        len([v for v in json.load(open(RES / fn)).values() if v is not None]),
        f"results/extended_replay/{fn}", "candidates re-scored by this judge")

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["claim_id","paper_location","metric","value","result_file","computation"])
    w.writeheader(); w.writerows(led)
print(f"wrote {len(led)} ledger rows -> {OUT.relative_to(ROOT)}")
