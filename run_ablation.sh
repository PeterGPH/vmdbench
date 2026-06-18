#!/usr/bin/env bash
# run_ablation.sh — 2x2 RAG x Wiki correctness ablation on the molecular_vis cases.
#
# Runs all four arms (none / rag / wiki / both) through the REAL VMD bridge, then
# scores each against oracle gold with score_correctness.py and prints a side-by-side
# table with compare_modes.py. Single seed (pass an arg to repeat — see SEED note).
#
#   bash integrations/scivisagentbench/run_ablation.sh
#
# Requirements: the vLLM server reachable at the configs' base_url (SSH tunnel up),
# and for the +rag arms a built docs index (~/.vmdai/docs_index) or that arm is a no-op.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BENCH="${BENCH:-$HOME/SciVisAgentBench}"
TASKS="$BENCH/SciVisAgentBench-tasks/molecular_vis"
YAML="$BENCH/benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml"
MODEL_TAG="Qwen-Qwen2.5-72B-Instruct-AWQ"   # how the runner slugifies the model name
ARMS="${ARMS:-none rag wiki both}"
SEEDS="${SEEDS:-3}"   # repeats per arm; the local model is stochastic, so >1 de-noises
# RUN_TAG namespaces every sweep: each (arm,seed) becomes a unique agent_mode, so a
# re-run NEVER overwrites a previous sweep's answer files. Override to group/replay a run.
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
ARCHIVE="$REPO/test_results/ablation_scoreboards"

echo "RUN_TAG=$RUN_TAG  ARMS='$ARMS'  SEEDS=$SEEDS"
cd "$REPO" || exit 1

for seed in $(seq 1 "$SEEDS"); do
  for arm in $ARMS; do
    cfg="$HERE/config_arm_${arm}.json"
    [ -f "$cfg" ] || { echo "!! missing config: $cfg"; continue; }
    exp="${RUN_TAG}_${arm}_s${seed}"
    echo ""
    echo "==================== arm=$arm seed=$seed  (mode ..._${exp}) ===================="
    # fresh wiki per (arm, seed) so neither arms nor seeds cross-contaminate.
    wroot="$HOME/.vmdai/bench_wiki/${exp}"
    rm -rf "$wroot" 2>/dev/null
    # --experiment-number sets the result-dir suffix (the config field is ignored);
    # VMD_AI_WIKI_ROOT gives this run its own wiki (the adapter reads the env first).
    VMD_AI_WIKI_ROOT="$wroot" \
      python integrations/scivisagentbench/run_vmd_ai.py --bench "$BENCH" \
        --config "$cfg" --yaml "$YAML" --cases "$TASKS" --no-eval \
        --experiment-number "$exp"
  done
done

echo ""
echo "==================== SCOREBOARD (mean over $SEEDS seeds) ===================="
mkdir -p "$ARCHIVE"
python integrations/scivisagentbench/aggregate_ablation.py --tasks "$TASKS" \
  --model-tag "$MODEL_TAG" --arms "$(echo $ARMS | tr ' ' ',')" --seeds "$SEEDS" \
  --run-tag "$RUN_TAG" | tee "$ARCHIVE/scoreboard_${RUN_TAG}.txt"
echo ""
echo "scoreboard archived -> $ARCHIVE/scoreboard_${RUN_TAG}.txt"
echo "re-score later with: --run-tag $RUN_TAG"
