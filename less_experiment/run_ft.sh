#!/usr/bin/env bash
set -u
out=results_less_seed13.jsonl; : > "$out"
for task in news_topic_classification sentiment_classification; do
  for lang in amh hau swa yor; do
    echo "== ft $task/$lang" >&2
    python3 less_finetune.py "$task" "$lang" --seed 13 2>/dev/null | tail -1 >> "$out"
  done
done
echo "DONE $(wc -l < $out)/8"
