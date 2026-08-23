#!/usr/bin/env python3
"""Recompute the main-replay statistics reported in Table 1 and Sec. 4.2-4.3.

Reads only results/main_replay/*_analysis_rows.csv, which ships here. Bootstrap
intervals are computed by EXACT enumeration over the discrete resampling
distribution (C(2n-1,n) multisets), so there is no seed and no Monte-Carlo drift.
"""
import csv, glob, itertools, os
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
# Prefer the full 8-cell replay over the 4-cell pilot that ships alongside it.
_c = sorted(glob.glob(str(ROOT / "results/main_replay/*_full_*_analysis_rows.csv"))) \
     or sorted(glob.glob(str(ROOT / "results/main_replay/*_analysis_rows.csv")))
CSV = os.environ.get("COSDA_MAIN_CSV", _c[0] if _c else None)

NAME = {"naive": "Naive", "alpagasus_style": "AlpaGasus", "deita_style": "DEITA",
        "cosda_equal_budget": "CoSDA-EQ", "cosda_rerank_v2": "CoSDA-V2"}
ORDER = ["naive", "alpagasus_style", "deita_style", "cosda_equal_budget", "cosda_rerank_v2"]
# Degeneracy rule, stated in code: a cell is near-degenerate when the spread of
# Macro-F1 across the five main selectors is at most DEGEN_MAX_SPREAD. The eight
# cells separate cleanly at this threshold (0.005, 0.009, 0.063 | 0.087, 0.156, ...).
DEGEN_MAX_SPREAD = 0.07


def spearman(a, b):
    def ranks(x):
        idx = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x); i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[idx[j + 1]] == x[idx[i]]: j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1): r[idx[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b); n = len(a); ma, mb = mean(ra), mean(rb)
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    den = (sum((x - ma) ** 2 for x in ra) * sum((x - mb) ** 2 for x in rb)) ** 0.5
    return num / den if den else 0.0


def exact_bootstrap_ci(vals, lo=0.025, hi=0.975):
    """Exact percentile CI of the mean under the nonparametric bootstrap.

    Enumerates every resampling multiset and weights it by its multinomial
    multiplicity n!/prod(c_i!), which is what an ordered-draw bootstrap samples.
    Weighting matters: an unweighted enumeration is a different distribution and
    gives materially wider intervals.
    """
    from collections import Counter
    from math import factorial
    n = len(vals)
    pts = []
    for idx in itertools.combinations_with_replacement(range(n), n):
        c = Counter(idx)
        w = factorial(n)
        for k in c.values():
            w //= factorial(k)
        pts.append((sum(vals[i] for i in idx) / n, w))
    pts.sort()
    total = sum(w for _, w in pts)
    out = []
    for q in (lo, hi):
        acc, target = 0, q * total
        for val, w in pts:
            acc += w
            if acc >= target:
                out.append(val); break
    return out[0], out[1]


def main():
    if not CSV:
        print("no main-replay CSV found"); return 1
    f1 = defaultdict(dict)    # (task,lang) -> selector -> macro_f1
    lc = defaultdict(dict)    # (task,lang) -> selector -> per-cell judged label correctness
    for r in csv.DictReader(open(CSV)):
        cell = (r["task"], r["language"])
        if r["kind"] == "hf" and r["macro_f1"]:
            f1[cell][r["baseline"]] = float(r["macro_f1"])
        elif r["kind"] == "selection" and r["judge_label_correctness"]:
            lc[cell][r["baseline"]] = float(r["judge_label_correctness"])
    cells = sorted(f1)
    audit = {}
    import json
    agg = ROOT / "results/extended_replay/audit_aggregates.json"
    if agg.exists(): audit = json.load(open(agg))

    print(f"cells: {len(cells)}\n")
    print(f"{'selector':12s} {'mean F1':>8s} {'LC':>7s}")
    means = {}
    for b in ORDER:
        v = [f1[c][b] for c in cells if b in f1[c]]
        means[b] = mean(v)
        _lc = audit.get(b, {}).get("LC")
        print(f"  {NAME[b]:10s} {means[b]:8.3f} {('%.3f' % _lc) if _lc else '     -':>7s}")

    if audit:
        common = [b for b in ORDER if b in audit]
        print(f"\nselector-level Spearman(LC, F1)      = {spearman([audit[b]['LC'] for b in common], [means[b] for b in common]):+.2f}")
        print(f"selector-level Spearman(hard-rej, F1) = {spearman([audit[b]['HR'] for b in common], [means[b] for b in common]):+.2f}")
        print(f"selector-level Spearman(C, F1)        = {spearman([audit[b]['C'] for b in common], [means[b] for b in common]):+.2f}")

        # Judge-free channels: D, L, H use no judge field. C and the residual reject
        # rate are judge-assisted (C scales the judge's counterfactual-validity rating,
        # and 758 of 759 gate failures are counterfactual), so they are excluded here.
        # "Direction-normalised" means each channel is oriented so higher = better
        # (D as-is, L and H negated) before equal-weight averaging of the raw values.
        print("\njudge-free channels vs mean Macro-F1:")
        for ch in ("D", "L", "H"):
            if ch in audit[common[0]]:
                print(f"  {ch}  {spearman([audit[b][ch] for b in common], [means[b] for b in common]):+.2f}")
        comp = [audit[b]["D"] - audit[b]["L"] - audit[b]["H"] for b in common]
        print(f"  equal-weight direction-normalised composite(D,L,H)  {spearman(comp, [means[b] for b in common]):+.2f}")

        wc = []
        print("\nwithin-cell Spearman(LC, F1)   [per-cell LC, not the global mean]:")
        for c in cells:
            common = [b for b in ORDER if b in f1[c] and b in lc.get(c, {})]
            rho = spearman([lc[c][b] for b in common], [f1[c][b] for b in common])
            wc.append(rho)
            print(f"  {c[0][:4]}/{c[1]}  {rho:+.2f}")
        ws = sorted(wc); n = len(ws)
        med = ws[n // 2] if n % 2 else (ws[n // 2 - 1] + ws[n // 2]) / 2
        print(f"  mean {mean(wc):+.2f}   median {med:+.2f}")

    # degeneracy
    degen = []
    for c in cells:
        vals = [f1[c][b] for b in ORDER if b in f1[c]]
        if vals and (max(vals) - min(vals)) <= DEGEN_MAX_SPREAD: degen.append(c)
    print(f"\nnear-degenerate cells (selector spread <= {DEGEN_MAX_SPREAD}): "
          f"{', '.join(f'{c[0][:4]}/{c[1]}' for c in degen)}")

    nd = [c for c in cells if c not in degen]
    print("non-degenerate subset means:")
    for b in ORDER:
        v = [f1[c][b] for c in nd if b in f1[c]]
        if v: print(f"  {NAME[b]:10s} {mean(v):.3f}")

    # deltas vs naive + exact CIs + leave-one-cell-out
    for b in ("cosda_rerank_v2", "alpagasus_style"):
        d = [f1[c][b] - f1[c]["naive"] for c in cells if b in f1[c] and "naive" in f1[c]]
        lo, hi = exact_bootstrap_ci(d)
        print(f"\n{NAME[b]} - Naive : mean {mean(d):+.3f}  exact 95% CI [{lo:+.3f}, {hi:+.3f}]")
        dn = [f1[c][b] - f1[c]["naive"] for c in nd if b in f1[c] and "naive" in f1[c]]
        if dn:
            lo2, hi2 = exact_bootstrap_ci(dn)
            print(f"{' '*len(NAME[b])}   non-degenerate: mean {mean(dn):+.3f}  CI [{lo2:+.3f}, {hi2:+.3f}]")
        if b == "cosda_rerank_v2":
            print("  leave-one-cell-out mean delta:")
            for i, c in enumerate(cells):
                rest = d[:i] + d[i+1:]
                print(f"    drop {c[0][:4]}/{c[1]:4s} -> {mean(rest):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
