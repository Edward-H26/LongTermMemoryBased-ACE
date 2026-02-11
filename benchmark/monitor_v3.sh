#!/bin/bash
# Monitor baseline_v3 and ace_v3 inference progress.
# Run in a separate terminal after: python -m benchmark.run_v3
# With --clear-results (default), counts start at 0. With --no-clear-results, counts include prior runs.

cd "$(dirname "$0")/.." || exit 1
RESULT_DIR="benchmark/results"

while true; do
  b=$([ -f "$RESULT_DIR/baseline_v3.jsonl" ] && wc -l < "$RESULT_DIR/baseline_v3.jsonl" || echo 0)
  a=$([ -f "$RESULT_DIR/ace_v3.jsonl" ] && wc -l < "$RESULT_DIR/ace_v3.jsonl" || echo 0)
  elapsed=""
  for f in "$RESULT_DIR/baseline_v3.jsonl" "$RESULT_DIR/ace_v3.jsonl"; do
    if [ -f "$f" ]; then
      start_ts=$(stat -f %m "$f" 2>/dev/null || stat -c %Y "$f" 2>/dev/null)
      if [ -n "$start_ts" ]; then
        elapsed=" [elapsed: $(($(date +%s) - start_ts))s]"
        break
      fi
    fi
  done
  printf "%s%s Baseline: %s/200, ACE: %s/200\n" "$(date '+%H:%M:%S')" "$elapsed" "$b" "$a"
  if [ "$b" -ge 200 ] && [ "$a" -ge 200 ]; then break; fi
  sleep 60
done
echo "Inference complete."
