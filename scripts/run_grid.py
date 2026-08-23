import os
import json, subprocess, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
BASE = Path(os.environ.get("COSDA_RUNS", ROOT / "runs" / "select1_seed13"))
TASKS = {"news_topic_classification": "masakhane/masakhanews",
         "sentiment_classification": "masakhane/afrisenti"}
LANGS = ["amh", "hau", "swa", "yor"]
BASELINES = ["naive", "alpagasus_style", "deita_style", "cosda_equal_budget",
             "cosda_rerank_v2", "alpagasus_hardreject", "less_style"]
SEEDS = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [13]
env = dict(os.environ, HF_HOME="/workspace/hf", TOKENIZERS_PARALLELISM="false",
           PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python")
results = []
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
                out = Path(f"/workspace/out/{task}_{lang}_{bl}_s{seed}")
                cmd = ["python3", "-m", "cosda.cli", "train-hf",
                       "--manifest", str(ROOT/"data"/"manifest"/"datasets.json"),
                       "--dataset-id", ds, "--language", lang,
                       "--gold", str(gold), "--selected", str(sel),
                       "--out-dir", str(out), "--deterministic", "--seed", str(seed)]
                r = subprocess.run(cmd, env=env, capture_output=True, text=True)
                mf = None
                mp = out / "test_metrics.json"
                if mp.exists():
                    mf = json.load(open(mp)).get("test_macro_f1")
                else:
                    print("FAIL", task, lang, bl, seed, r.stderr[-300:], flush=True)
                results.append({"task": task, "lang": lang, "baseline": bl,
                                "seed": seed, "n_selected": n_sel, "macro_f1": mf})
                print(f"{task[:4]}/{lang} {bl:22s} s{seed} n={n_sel:3d} f1={mf}", flush=True)
                json.dump(results, open("/workspace/grid_results.json", "w"), indent=2)
print("DONE", len(results), "runs", flush=True)
