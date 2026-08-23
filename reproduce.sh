#!/usr/bin/env bash
# One-command check that the released EXTENDED-REPLAY results reproduce the paper's
# Tables 2-4 and Table 1's audit columns. Table 1's Macro-F1 column comes from the
# main replay (results/main_replay/) and is not recomputed here.
# Requires only python3 (no GPU, no network) — it recomputes every aggregate
# from the committed per-run result files.
set -euo pipefail
cd "$(dirname "$0")"
echo "== 1/5  selector means, per-seed and within-cell Spearman, ablation =="
python3 scripts/analyze.py
echo
echo "== 2/5  multi-judge agreement =="
python3 scripts/judge_corr.py
echo
echo "== 3/5  regenerate the claim ledger =="
python3 scripts/make_claim_ledger.py
echo
echo "== 4/5  main-replay statistics (Table 1, Sec. 4.2-4.3) =="
python3 scripts/analyze_main_replay.py
echo
echo "== 5/5  verify released files against the committed PROVENANCE.md =="
# Deliberately does NOT regenerate first: the point is to detect drift between
# the committed manifest and the committed files.
python3 scripts/check_provenance.py
echo
echo "Done. Compare against results/extended_replay/ANALYSIS_OUTPUT.txt and"
echo "results/extended_replay/judge_agreement_output.txt and
results/main_replay/MAIN_REPLAY_OUTPUT.txt, which are the committed"
echo "outputs of steps 1 and 2."
