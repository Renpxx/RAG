#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIMIT="${LIMIT:-10}"
MODELS=(
  "gpt-5-nano"
  "ark-doubao-seed-1.6-flash-250715pt-5-mini-2025-08-07"
  "gpt-4o-mini"
)

slugify() {
  local input="$1"
  local slug
  slug="$(echo "$input" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '-' | sed 's/^-//;s/-$//')"
  if [[ -z "$slug" ]]; then
    slug="model"
  fi
  echo "$slug"
}

for model in "${MODELS[@]}"; do
  slug="$(slugify "$model")"
  echo "Running evaluation for ${model}..."
  python "$ROOT_DIR/test/run_question_evals.py" \
    --limit "$LIMIT" \
    --chat-model "$model" \
    --output-json "$ROOT_DIR/test/rag_vs_baseline_results_${slug}.json" \
    --comparison-image "$ROOT_DIR/test/rag_vs_baseline_comparison_${slug}.png" \
    "$@"
done

echo "All evaluations finished."
