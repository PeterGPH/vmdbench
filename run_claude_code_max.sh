#!/usr/bin/env bash
# run_claude_code_max.sh — run SciVisAgentBench's built-in `claude_code` agent
# billed to your Claude **Max subscription** instead of the pay-per-token API.
#
# WHY THIS WORKS
#   The claude_code agent shells out to the `claude` CLI. Claude Code picks its
#   credential in this order (first match wins):
#       1. cloud provider (Bedrock/Vertex/Foundry)
#       2. ANTHROPIC_AUTH_TOKEN        <- API billing
#       3. ANTHROPIC_API_KEY           <- API billing
#       4. apiKeyHelper
#       5. CLAUDE_CODE_OAUTH_TOKEN     <- your subscription (headless token)
#       6. interactive subscription login (`claude` / `/login`)
#   So to ride the subscription we UNSET 2 & 3 and rely on 5 or 6.
#
# ONE-TIME SETUP
#   1. Subscription token for headless runs:
#          claude setup-token        # authorize with your Max account; copy the token
#          export CLAUDE_CODE_OAUTH_TOKEN=<paste>   # add to your shell rc to persist
#      (If you instead ran `claude` once interactively and did /login, that also
#       works and you can skip the token — this script will still unset the keys.)
#   2. Keep OPENAI_API_KEY set if you want the gpt-4o vision judge (scoring).
#
# USAGE
#   bash run_claude_code_max.sh                 # probe: 1 case, agent only (no judge), cheapest
#   bash run_claude_code_max.sh --all           # all cases in the YAML, agent only
#   bash run_claude_code_max.sh --all --with-eval   # full sweep + gpt-4o judge
#   CASE=case_3 bash run_claude_code_max.sh
#   YAML_REL=benchmark/eval_cases/topology/topology_cases.yaml CASES_REL=SciVisAgentBench-tasks/topology \
#       bash run_claude_code_max.sh --all
#
# HEADS-UP (timing): until 2026-06-15 headless `claude -p` draws from your flat-rate
# Max limits. From 2026-06-15 it moves to the Agent-SDK credit pool billed at API
# rates — after that this stops being cheaper than the API for automated sweeps.

set -euo pipefail

# ---- where the benchmark lives (has benchmark/ and SciVisAgentBench-tasks/) ----
BENCH_DIR="${BENCH_DIR:-$HOME/SciVisAgentBench}"
CONFIG="${CONFIG:-benchmark/configs/claude_code/config.json}"
YAML_REL="${YAML_REL:-benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml}"
CASES_REL="${CASES_REL:-SciVisAgentBench-tasks/molecular_vis}"
CASE="${CASE:-case_1}"
EVAL_MODEL="${EVAL_MODEL:-gpt-4o}"
EXP="${EXP:-max_exp1}"

ALL=0; WITH_EVAL=0; SKIP_CHECK=0
for a in "$@"; do case "$a" in
  --all)        ALL=1 ;;
  --with-eval)  WITH_EVAL=1 ;;
  --skip-check) SKIP_CHECK=1 ;;
  --case=*)     CASE="${a#*=}" ;;
  -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
  *) echo "unknown arg: $a" >&2; exit 2 ;;
esac; done

if [ -t 1 ]; then G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; B=$'\e[1m'; N=$'\e[0m'; else G=; R=; Y=; B=; N=; fi
say(){ printf '%s\n' "$*"; }
die(){ printf '%s[FAIL]%s %s\n' "$R" "$N" "$*" >&2; exit 1; }

# ---- 1) force subscription billing: drop API-key credentials for this process ----
if [ -n "${ANTHROPIC_API_KEY:-}" ] || [ -n "${ANTHROPIC_AUTH_TOKEN:-}" ]; then
  say "${Y}[info]${N} Unsetting ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN for this run so"
  say "       Claude Code uses your subscription, not the API. (Your shell is untouched.)"
fi
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN

if [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
  export CLAUDE_CODE_OAUTH_TOKEN
  say "${G}[ok]${N} Using CLAUDE_CODE_OAUTH_TOKEN (subscription, headless)."
else
  say "${Y}[info]${N} CLAUDE_CODE_OAUTH_TOKEN not set — relying on an interactive 'claude' login."
  say "       For unattended/headless runs, generate one:  claude setup-token"
fi

# ---- 2) sanity: CLI present + judge key (only if scoring) ----
command -v claude >/dev/null 2>&1 || die "'claude' CLI not on PATH. Install it or set claude_code_path in $CONFIG."
[ -d "$BENCH_DIR/benchmark" ] || die "BENCH_DIR='$BENCH_DIR' has no benchmark/ — set BENCH_DIR to your SciVisAgentBench root."
if [ "$WITH_EVAL" = 1 ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  die "--with-eval needs OPENAI_API_KEY for the '$EVAL_MODEL' judge (judge is separate from the agent)."
fi

# ---- 3) confirm auth is really the subscription, cheaply, before a big sweep ----
if [ "$SKIP_CHECK" = 0 ]; then
  say "${B}Auth probe${N} (1 tiny prompt; should NOT appear on console.anthropic.com API usage)..."
  if printf '' | claude --print "reply with exactly: READY" >/tmp/_cc_probe 2>&1; then
    say "${G}[ok]${N} claude responded: $(tr -d '\n' </tmp/_cc_probe | cut -c1-40)"
  else
    say "${R}[probe failed]${N} $(cat /tmp/_cc_probe)"
    die "Auth probe failed. Run 'claude setup-token' (or 'claude' then /login) and retry, or pass --skip-check."
  fi
fi

# ---- 4) run the benchmark on the HOST (NOT docker, so host creds are used) ----
cd "$BENCH_DIR"
ARGS=(--agent claude_code --config "$CONFIG" --yaml "$YAML_REL" --cases "$CASES_REL" --experiment-number "$EXP")
[ "$ALL" = 1 ] || ARGS+=(--case "$CASE")
if [ "$WITH_EVAL" = 1 ]; then ARGS+=(--eval-model "$EVAL_MODEL"); else ARGS+=(--exe-only); fi

say ""
say "${B}Running:${N} python -m benchmark.evaluation_framework.run_evaluation ${ARGS[*]}"
say "${B}Mode:${N} $([ "$ALL" = 1 ] && echo 'ALL cases' || echo "single case ($CASE)") | $([ "$WITH_EVAL" = 1 ] && echo 'with judge' || echo 'agent only (no judge)')"
say ""
python -m benchmark.evaluation_framework.run_evaluation "${ARGS[@]}"

say ""
say "${G}Done.${N} Verify billing: open console.anthropic.com → Usage. If API usage did NOT"
say "increase, the run was covered by your subscription. (Reminder: flat-rate headless ends 2026-06-15.)"
