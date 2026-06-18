#!/usr/bin/env bash
# run_baselines.sh — run several MODELS through the same correctness harness and compare.
# Each model runs RAW (no augmentation) over the multistructure analysis grid, multi-seed.
# Produces the per-model completion-vs-correctness table (the paper's backbone).
#
#   MODELS="qwen72b" bash run_baselines.sh                 # local only (free)
#   MODELS="qwen72b claude gpt" SEEDS=3 bash run_baselines.sh   # add API models (set keys first)
#
# Set keys before adding API models:  export ANTHROPIC_API_KEY=...  OPENROUTER_API_KEY=...
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BENCH="${BENCH:-$HOME/SciVisAgentBench}"
SEEDS="${SEEDS:-3}"
MODELS="${MODELS:-qwen72b}"

cd "$REPO" || exit 1
for m in $MODELS; do
  cfg="$HERE/config_model_${m}.json"
  [ -f "$cfg" ] || { echo "!! missing $cfg"; continue; }
  echo ""
  echo "==================== MODEL: $m ===================="
  python integrations/scivisagentbench/run_multistructure.py --bench "$BENCH" \
    --config "$cfg" --tag "model_${m}" --seeds "$SEEDS"
done

echo ""
echo "==================== COMPLETION vs CORRECTNESS ===================="
TAGS=""
for m in $MODELS; do TAGS="${TAGS:+$TAGS,}model_${m}"; done
python integrations/scivisagentbench/compare_models.py --tags "$TAGS"
