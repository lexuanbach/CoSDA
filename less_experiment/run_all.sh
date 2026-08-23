#!/usr/bin/env bash
set -u
for task in news_topic_classification sentiment_classification; do
  for lang in amh hau swa yor; do
    f="scores/${task}_${lang}.json"
    if [ -f "$f" ]; then echo "== skip $task/$lang (done)"; continue; fi
    echo "== $task/$lang"
    python3 less_select.py "$task" "$lang" --dev-n 128 2>&1 | grep -E "^  (P=|ckpt|news|sent)"
  done
done
echo "ALL CELLS DONE: $(ls scores/*.json 2>/dev/null | wc -l)/8"
