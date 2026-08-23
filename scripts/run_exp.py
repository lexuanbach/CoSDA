import os
import json, subprocess, os, sys
from pathlib import Path
# usage: run_exp.py <tag> <model> <seeds_csv> [baselines_csv]
TAG   = sys.argv[1]
MODEL = sys.argv[2]
SEEDS = [int(s) for s in sys.argv[3].split(",")]
BASELINES = (sys.argv[4].split(",") if len(sys.argv) > 4 else
             ["naive","alpagasus_style","deita_style","cosda_equal_budget",
              "cosda_rerank_v2","alpagasus_hardreject","less_style"])
ROOT = Path(__file__).resolve().parent.parent
BASE = Path(os.environ.get("COSDA_RUNS", ROOT / "runs" / "select1_seed13"))
TASKS = {"news_topic_classification": "masakhane/masakhanews",
         "sentiment_classification": "masakhane/afrisenti"}
LANGS = ["amh","hau","swa","yor"]
env = dict(os.environ, HF_HOME="/workspace/hf", TOKENIZERS_PARALLELISM="false",
           PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python")
results, outjson = [], f"/workspace/results_{TAG}.json"
for task, ds in TASKS.items():
    for lang in LANGS:
        cell = BASE / task / lang / "b64_m3_s13"
        gold = cell / "gold.jsonl"
        for bl in BASELINES:
            sel = cell / "selected" / f"{bl}.jsonl"
            if not sel.exists():
                print("MISSING", task, lang, bl, flush=True); continue
            n_sel = sum(1 for _ in open(sel))
            for seed in SEEDS:
                out = Path(f"/workspace/out_{TAG}/{task}_{lang}_{bl}_s{seed}")
                cmd = ["python3","-m","cosda.cli","train-hf",
                       "--manifest",str(ROOT/"data"/"manifest"/"datasets.json"),
                       "--dataset-id",ds,"--language",lang,
                       "--gold",str(gold),"--selected",str(sel),
                       "--out-dir",str(out),"--model",MODEL,
                       "--deterministic","--seed",str(seed)]
                r = subprocess.run(cmd, env=env, capture_output=True, text=True)
                mf = None; mp = out / "test_metrics.json"
                if mp.exists(): mf = json.load(open(mp)).get("test_macro_f1")
                else: print("FAIL", task, lang, bl, seed, r.stderr[-200:], flush=True)
                results.append({"tag":TAG,"model":MODEL,"task":task,"lang":lang,
                                "baseline":bl,"seed":seed,"n_selected":n_sel,"macro_f1":mf})
                print(f"[{TAG}] {task[:4]}/{lang} {bl:22s} s{seed} n={n_sel:3d} f1={mf}", flush=True)
                json.dump(results, open(outjson,"w"), indent=2)
print(f"DONE {TAG} {len(results)} runs", flush=True)
