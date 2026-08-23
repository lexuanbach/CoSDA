import json, sys
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent.parent
from statistics import mean
BASE=Path(os.environ.get("COSDA_RUNS", ROOT / "runs" / "select1_seed13"))
D=Path(os.environ.get("COSDA_RESULTS", ROOT / "results" / "extended_replay"))
TASKS=["news_topic_classification","sentiment_classification"]; LANGS=["amh","hau","swa","yor"]
BLS=["naive","alpagasus_style","deita_style","cosda_equal_budget","cosda_rerank_v2","alpagasus_hardreject","less_style"]
NAME={"naive":"Naive","alpagasus_style":"AlpaGasus","deita_style":"DEITA","cosda_equal_budget":"CoSDA-EQ","cosda_rerank_v2":"CoSDA-V2","alpagasus_hardreject":"AlpaGasus+HR","less_style":"U+D control"}
def spearman(a,b):
    def rk(x):
        idx=sorted(range(len(x)),key=lambda i:x[i]); r=[0]*len(x); i=0
        while i<len(x):
            j=i
            while j+1<len(x) and x[idx[j+1]]==x[idx[i]]: j+=1
            for k in range(i,j+1): r[idx[k]]=(i+j)/2+1
            i=j+1
        return r
    ra,rb=rk(a),rk(b); n=len(a); ma=mean(ra); mb=mean(rb)
    num=sum((ra[i]-ma)*(rb[i]-mb) for i in range(n))
    den=(sum((ra[i]-ma)**2 for i in range(n))*sum((rb[i]-mb)**2 for i in range(n)))**.5
    return num/den if den else 0.0
def pearson(a,b):
    n=len(a);ma=mean(a);mb=mean(b)
    num=sum((a[i]-ma)*(b[i]-mb) for i in range(n))
    den=(sum((a[i]-ma)**2 for i in range(n))*sum((b[i]-mb)**2 for i in range(n)))**.5
    return num/den if den else 0.0
# judge1 LC per candidate from audit records
lc1={}
for t in TASKS:
    for lang in LANGS:
        for r in (json.loads(l) for l in open(BASE/t/lang/"b64_m3_s13"/"audit.jsonl")):
            lj=(r.get("metadata",{}).get("llm_judge") or {})
            if lj.get("label_correctness") is not None: lc1[r["candidate_id"]]=lj["label_correctness"]
def per_selector_lc(lcmap):
    out={}
    for bl in BLS:
        cell_means=[]
        for t in TASKS:
            for lang in LANGS:
                p=BASE/t/lang/"b64_m3_s13"/"selected"/f"{bl}.jsonl"
                if not p.exists(): continue
                vals=[]
                for r in (json.loads(l) for l in open(p)):
                    v=lcmap.get(r["candidate_id"])
                    if v is not None: vals.append(v)
                if vals: cell_means.append(mean(vals))
        # macro average: mean per cell, then across cells. Matches the paper's stated
        # convention ("averaged over 8 cells"); only AlpaGasus+HR has unequal cells.
        out[bl]=mean(cell_means) if cell_means else None
    return out
judges={"Qwen32B (orig)":lc1}
for tag,fn in [("Qwen14B","judge2_qwen14b.json"),("Phi-3.5-mini","judge2_phi35.json")]:
    f=D/fn
    if f.exists():
        j=json.load(open(f)); j={k:v for k,v in j.items() if v is not None}
        if j: judges[tag]=j
sel_lc={name:per_selector_lc(m) for name,m in judges.items()}
print(f"{'selector':14s} " + " ".join(f"{n:>14s}" for n in judges))
for bl in BLS:
    print(f"{NAME[bl]:14s} " + " ".join(f"{(sel_lc[n][bl] if sel_lc[n][bl] is not None else float('nan')):14.3f}" for n in judges))
print()
base=[sel_lc["Qwen32B (orig)"][bl] for bl in BLS]
for n in judges:
    if n=="Qwen32B (orig)": continue
    other=[sel_lc[n][bl] for bl in BLS]
    print(f"Judge agreement (orig vs {n}): Spearman={spearman(base,other):+.3f}  Pearson={pearson(base,other):+.3f}")
    order=lambda d:[NAME[b] for b in sorted(BLS,key=lambda x:-d[x])]
    print(f"   orig LC order: {' > '.join(order(sel_lc['Qwen32B (orig)']))}")
    print(f"   {n} LC order: {' > '.join(order(sel_lc[n]))}")
