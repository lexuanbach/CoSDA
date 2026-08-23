import json, sys, os
from pathlib import Path
from statistics import mean, pstdev

# Resolve results relative to this file so the script runs from a clean checkout.
ROOT = Path(__file__).resolve().parent.parent
RES  = Path(os.environ.get("COSDA_RESULTS", ROOT / "results" / "extended_replay"))

def spearman(a, b):
    # rank-based Spearman over small lists
    def ranks(x):
        idx=sorted(range(len(x)), key=lambda i:x[i])
        r=[0]*len(x)
        i=0
        while i<len(x):
            j=i
            while j+1<len(x) and x[idx[j+1]]==x[idx[i]]: j+=1
            avg=(i+j)/2+1
            for k in range(i,j+1): r[idx[k]]=avg
            i=j+1
        return r
    ra,rb=ranks(a),ranks(b)
    n=len(a); mra=mean(ra); mrb=mean(rb)
    num=sum((ra[i]-mra)*(rb[i]-mrb) for i in range(n))
    den=(sum((ra[i]-mra)**2 for i in range(n))*sum((rb[i]-mrb)**2 for i in range(n)))**0.5
    return num/den if den else 0.0

audit=json.load(open(RES / "audit_aggregates.json"))
BLS=["naive","alpagasus_style","deita_style","cosda_equal_budget","cosda_rerank_v2","alpagasus_hardreject","less_style"]
NAME={"naive":"Naive","alpagasus_style":"AlpaGasus","deita_style":"DEITA","cosda_equal_budget":"CoSDA-EQ","cosda_rerank_v2":"CoSDA-V2","alpagasus_hardreject":"AlpaGasus+HR","less_style":"U+D control"}

def load(fp):
    try: return json.load(open(fp))
    except: return []

def by_selector_seed(rows):
    d={}
    for r in rows:
        if r.get("macro_f1") is None: continue
        d.setdefault((r["baseline"],r["seed"]),[]).append(r["macro_f1"])
    return d  # list of per-cell f1 (8 cells)

def report(tag, rows, seeds):
    print(f"\n===== {tag} =====")
    d=by_selector_seed(rows)
    # mean over cells, then over seeds
    means={}
    for bl in BLS:
        per_seed=[mean(d[(bl,s)]) for s in seeds if (bl,s) in d and d[(bl,s)]]
        if per_seed: means[bl]=(mean(per_seed), pstdev(per_seed) if len(per_seed)>1 else 0.0)
    print(f"{'selector':14s} {'LC':>6} {'meanF1':>7} {'±sd':>6}")
    lc_order=[]; f1_order=[]
    for bl in BLS:
        if bl not in means: continue
        lc=audit[bl]['LC']; f1,sd=means[bl]
        lc_order.append((bl,lc)); f1_order.append((bl,f1))
        print(f"{NAME[bl]:14s} {lc:.3f} {f1:.3f} {sd:.3f}")
    # spearman LC vs F1 across selectors
    common=[bl for bl in BLS if bl in means]
    rho=spearman([audit[bl]['LC'] for bl in common],[means[bl][0] for bl in common])
    print(f"Spearman(LC,F1) across {len(common)} selectors = {rho:+.3f}")
    print("LC order:", " > ".join(NAME[b] for b,_ in sorted(lc_order,key=lambda x:-x[1])))
    print("F1 order:", " > ".join(NAME[b] for b,_ in sorted(f1_order,key=lambda x:-x[1])))
    # ablation
    if "alpagasus_style" in means and "alpagasus_hardreject" in means:
        print(f"ABLATION: AlpaGasus F1={means['alpagasus_style'][0]:.3f}  AlpaGasus+HR F1={means['alpagasus_hardreject'][0]:.3f}  delta={means['alpagasus_hardreject'][0]-means['alpagasus_style'][0]:+.3f}")
    return means

# per-cell ablation detail for XLM-R seed 13
def per_cell_ablation(rows):
    print("\n--- per-cell: AlpaGasus vs AlpaGasus+HR (xlm-r s13) ---")
    d={}
    for r in rows:
        if r["seed"]!=13: continue
        d[(r["task"][:4],r["lang"],r["baseline"])]=(r["macro_f1"],r["n_selected"])
    print(f"{'cell':12s} {'AG':>7} {'AG+HR':>7} {'nHR':>4} {'delta':>7}")
    for t in ("news","sent"):
        for lang in ("amh","hau","swa","yor"):
            ag=d.get((t,lang,"alpagasus_style")); hr=d.get((t,lang,"alpagasus_hardreject"))
            if ag and hr:
                print(f"{t}/{lang:8s} {ag[0]:.3f} {hr[0]:.3f} {hr[1]:>4} {hr[0]-ag[0]:+.3f}")

grid=load(RES / "grid_results.json")
ms=load(RES / "results_multiseed.json")
afro=load(RES / "results_afroxlmr.json")

report("XLM-R seed 13", grid, [13])
per_cell_ablation(grid)
allseed=grid+ms
if ms: report("XLM-R multi-seed (13,21,42)", allseed, [13,21,42])
if afro: report("AfroXLMR seed 13", afro, [13])

def within_cell_and_perseed(rows, seeds):
    print("\n--- within-cell & per-seed Spearman(LC,F1) ---")
    # group f1 by (task,lang,seed) -> {baseline:f1}
    for s in seeds:
        if not any(r["seed"]==s and r["macro_f1"] is not None for r in rows):
            continue
        cellmap={}
        for r in rows:
            if r["seed"]!=s or r["macro_f1"] is None: continue
            cellmap.setdefault((r["task"],r["lang"]),{})[r["baseline"]]=r["macro_f1"]
        wc=[]
        for cell,bm in cellmap.items():
            common=[bl for bl in BLS if bl in bm]
            if len(common)<3: continue
            rho=spearman([audit[bl]['LC'] for bl in common],[bm[bl] for bl in common])
            wc.append(rho)
        # selector-level for this seed
        selmap={}
        for r in rows:
            if r["seed"]!=s or r["macro_f1"] is None: continue
            selmap.setdefault(r["baseline"],[]).append(r["macro_f1"])
        common=[bl for bl in BLS if bl in selmap]
        if len(common)<3: 
            continue
        selrho=spearman([audit[bl]['LC'] for bl in common],[mean(selmap[bl]) for bl in common])
        print(f"seed {s}: selector-level rho={selrho:+.3f} | within-cell mean rho={mean(wc):+.3f} median={sorted(wc)[len(wc)//2]:+.3f} (n={len(wc)} cells)")

grid=load(RES / "grid_results.json"); ms=load(RES / "results_multiseed.json"); afro=load(RES / "results_afroxlmr.json")
within_cell_and_perseed(grid+ms, [13,21,42])
if afro:
    print("\n[AfroXLMR]"); within_cell_and_perseed(afro,[13])

def instability_summary(rows, seeds, label):
    print(f"\n===== INSTABILITY SUMMARY: {label} =====")
    sel_rhos=[]; wc_means=[]
    for s in seeds:
        if not any(r["seed"]==s and r["macro_f1"] is not None for r in rows): continue
        selmap={}; cellmap={}
        for r in rows:
            if r["seed"]!=s or r["macro_f1"] is None: continue
            selmap.setdefault(r["baseline"],[]).append(r["macro_f1"])
            cellmap.setdefault((r["task"],r["lang"]),{})[r["baseline"]]=r["macro_f1"]
        common=[bl for bl in BLS if bl in selmap]
        if len(common)>=3:
            sel_rhos.append(spearman([audit[bl]['LC'] for bl in common],[mean(selmap[bl]) for bl in common]))
        wc=[]
        for cell,bm in cellmap.items():
            c=[bl for bl in BLS if bl in bm]
            if len(c)>=3: wc.append(spearman([audit[bl]['LC'] for bl in c],[bm[bl] for bl in c]))
        if wc: wc_means.append(mean(wc))
    if sel_rhos:
        print(f"selector-level rho across seeds: min={min(sel_rhos):+.3f} max={max(sel_rhos):+.3f} mean={mean(sel_rhos):+.3f} (n={len(sel_rhos)} seeds)")
    if wc_means:
        print(f"within-cell mean rho across seeds: min={min(wc_means):+.3f} max={max(wc_means):+.3f} mean={mean(wc_means):+.3f}")
    # per-selector mean +- sd across seeds
    print(f"{'selector':14s} {'LC':>6}  meanF1+-sd (over seeds)")
    for bl in BLS:
        per=[mean([r["macro_f1"] for r in rows if r["baseline"]==bl and r["seed"]==s and r["macro_f1"] is not None]) 
             for s in seeds if any(r["baseline"]==bl and r["seed"]==s and r["macro_f1"] is not None for r in rows)]
        if per: print(f"{NAME[bl]:14s} {audit[bl]['LC']:.3f}  {mean(per):.3f} +- {(pstdev(per) if len(per)>1 else 0):.3f}")

ms=load(RES / "results_multiseed.json"); grid=load(RES / "grid_results.json")
instability_summary(grid+ms, [13,21,42], "XLM-R across seeds 13/21/42")
