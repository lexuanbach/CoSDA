#!/usr/bin/env bash
# Run the comparators LESS must be judged against, through the IDENTICAL loop:
#   less_style       - the U+D control LESS replaces
#   naive            - the floor
#   alpagasus_style  - the downstream leader in the replay
set -u
out=results_comparators_seed13.jsonl; : > "$out"
for sel in less_style naive alpagasus_style; do
  for task in news_topic_classification sentiment_classification; do
    for lang in amh hau swa yor; do
      echo "== $sel $task/$lang" >&2
      python3 less_finetune.py "$task" "$lang" --seed 13 --selector "$sel" 2>/dev/null | tail -1 >> "$out"
    done
  done
done
echo "DONE $(wc -l < $out)/24"
