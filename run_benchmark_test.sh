#!/usr/bin/env bash
# run_benchmark_test.sh — step-by-step test of the vmd_ai -> SciVisAgentBench
# integration. Walks the ladder rung by rung and STOPS at the first failure
# with a precise message. Safe to re-run.
#
#   Rung 0  preflight     (pip-capable python + VMD)             free
#   Rung 1  bridge smoke  (smoke_bridge.py, native VMD)          free, no API
#   Rung 2  setup         (clone + tiny deps + tasks)            free, network
#   Rung 3  registration  (run_vmd_ai.py --list shows vmd_ai)    free
#   Rung 4  one agent run (run_vmd_ai.py --case .. --no-eval)    ~1 API call
#   Rung 5  one scored case (adds the LLM judge)                 COSTS $  [opt-in]
#
# It uses our lightweight launcher (run_vmd_ai.py), which sidesteps the
# benchmark's heavy CLI — so the only Python deps are pyyaml + your runtime.
#
# Usage:
#   bash run_benchmark_test.sh                # rungs 0-4
#   bash run_benchmark_test.sh --skip-setup   # reuse an existing clone
#   bash run_benchmark_test.sh --with-eval    # also run rung 5 (paid)
#   bash run_benchmark_test.sh --all          # run ALL 10 cases (drops --case)
#   bash run_benchmark_test.sh --all --with-eval   # full scored sweep (slow, paid)
#   CASE=case_2 bash run_benchmark_test.sh
#   PY=/opt/anaconda3/bin/python bash run_benchmark_test.sh   # force interpreter

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${BENCH_DIR:-$HOME/SciVisAgentBench}"
TASKS_DIR="${TASKS_DIR:-$BENCH_DIR/SciVisAgentBench-tasks}"
SRC_CONFIG="${SRC_CONFIG:-$SCRIPT_DIR/config_anthropic.json}"
VMD_AI_RUNTIME_PATH="${VMD_AI_RUNTIME_PATH:-$(cd "$SCRIPT_DIR/../../runtime" 2>/dev/null && pwd)}"
VMD_AI_VMD_BIN="${VMD_AI_VMD_BIN:-/Applications/VMD.app/Contents/vmd/vmd_MACOSXARM64}"
YAML_REL="${YAML_REL:-benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml}"
CASE="${CASE:-case_1}"
EVAL_MODEL="${EVAL_MODEL:-gpt-4o}"

SKIP_SETUP=0; WITH_EVAL=0; ALL=0; FULL=0
for a in "$@"; do case "$a" in
  --skip-setup) SKIP_SETUP=1 ;;
  --with-eval)  WITH_EVAL=1 ;;
  --all)        ALL=1 ;;
  --full)       FULL=1; ALL=1 ;;   # all 13 molecular_vis cases incl. the vision workflows
  --case=*)     CASE="${a#*=}" ;;
  -h|--help)    sed -n '2,22p' "$0"; exit 0 ;;
  *) echo "unknown arg: $a"; exit 2 ;;
esac; done

# --full -> the complete 13-case set (10 text tasks + 3 vision workflows), image-graded.
if [ "$FULL" = 1 ]; then YAML_REL="benchmark/eval_cases/molecular_vis/eval_analysis_all.yaml"; fi

export VMD_AI_VMD_BIN VMD_AI_RUNTIME_PATH BENCH_DIR

# Whether to run one case or the whole set.
if [ "$ALL" = 1 ]; then CASE_ARGS=(); RUN_LABEL="all 10 cases"; else CASE_ARGS=(--case "$CASE"); RUN_LABEL="$CASE"; fi

if [ -t 1 ]; then G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; B=$'\e[1m'; N=$'\e[0m'; else G=; R=; Y=; B=; N=; fi
hr(){ printf '%s\n' "------------------------------------------------------------"; }
rung(){ echo; hr; echo "${B}RUNG $1 — $2${N}"; hr; }
ok(){ echo "  ${G}[PASS]${N} $*"; }
warn(){ echo "  ${Y}[WARN]${N} $*"; }
die(){ echo "  ${R}[FAIL]${N} $*"; echo; echo "${R}Stopped at this rung.${N} Fix the above, then re-run."; exit 1; }
have(){ command -v "$1" >/dev/null 2>&1; }
_report_outputs(){
  if [ "$ALL" = 1 ]; then
    local n; n=$(find "$TASKS_DIR/molecular_vis" -path "*/results/$AGENT_MODE/*" -type f 2>/dev/null | wc -l | tr -d ' ')
    ok "produced $n output file(s) across cases (*/results/$AGENT_MODE/)"
    find "$TASKS_DIR/molecular_vis" -path "*/results/$AGENT_MODE/*" -type f 2>/dev/null | sed 's/^/      /' | head -12
  else
    local res="$TASKS_DIR/molecular_vis/$CASE/results/$AGENT_MODE"
    if compgen -G "$res/*" >/dev/null 2>&1; then
      ok "agent wrote output to $res"; ls -1 "$res" | sed 's/^/      /'
    else
      warn "no files under $res (agent ran, but answer file didn't land where the judge looks — verify #2)"
      find "$TASKS_DIR/molecular_vis/$CASE" -name '*.txt' 2>/dev/null | sed 's/^/      /' | tail -5
    fi
  fi
}

# pick a python that actually has pip (your `python3` is MacPorts and has none)
pick_py(){
  local c
  for c in "${PY:-}" python python3 python3.11 python3.10 \
           "$HOME/anaconda3/bin/python" "$HOME/miniconda3/bin/python" /opt/anaconda3/bin/python; do
    [ -z "$c" ] && continue
    if command -v "$c" >/dev/null 2>&1 && "$c" -m pip --version >/dev/null 2>&1; then echo "$c"; return 0; fi
  done
  return 1
}

# ============================================================================
rung 0 "preflight"
PY="$(pick_py)" || die "no python with pip found. Activate your conda env or set PY=/opt/anaconda3/bin/python"
ok "python: $("$PY" --version 2>&1) at $("$PY" -c 'import sys;print(sys.executable)')"
ok "pip:    $("$PY" -m pip --version 2>&1 | cut -d' ' -f1-2)"
[ -n "$VMD_AI_RUNTIME_PATH" ] && [ -d "$VMD_AI_RUNTIME_PATH/vmd_ai_runtime" ] \
  && ok "vmd_ai runtime: $VMD_AI_RUNTIME_PATH" \
  || die "vmd_ai runtime not found (need .../runtime containing vmd_ai_runtime/). Set VMD_AI_RUNTIME_PATH."
{ [ -x "$VMD_AI_VMD_BIN" ] && ok "VMD: $VMD_AI_VMD_BIN"; } || { have vmd && ok "VMD on PATH"; } || die "no VMD binary; set VMD_AI_VMD_BIN."

read -r AGENT_MODE EXP < <("$PY" - "$SRC_CONFIG" <<'PY'
import json,sys
c=json.load(open(sys.argv[1])); m=str(c.get("model","")).replace("/","-")
exp=c.get("experiment_number","exp1")
print(f'{c.get("agent_name","vmd_ai")}_{m}_{exp}', exp)
PY
)
echo "  agent_mode = ${B}$AGENT_MODE${N}"

# ============================================================================
rung 1 "bridge smoke (native VMD, no API)"
( cd "$SCRIPT_DIR" && "$PY" smoke_bridge.py ) || die "bridge smoke failed — see output above."
ok "bridge drives your VMD end-to-end"

# ============================================================================
rung 2 "setup: clone + tiny deps + tasks"
if [ "$SKIP_SETUP" = 1 ]; then warn "--skip-setup: assuming $BENCH_DIR is ready"; else
  if [ ! -d "$BENCH_DIR/.git" ]; then
    have git || die "git not found"
    echo "  cloning SciVisAgentBench…"; git clone --depth 1 https://github.com/KuangshiAi/SciVisAgentBench "$BENCH_DIR" || die "git clone failed"
  fi
  ok "benchmark: $BENCH_DIR"
  echo "  installing minimal deps (pyyaml tiktoken huggingface_hub pillow)…"
  "$PY" -m pip install -q pyyaml tiktoken huggingface_hub pillow || warn "pip install had issues"
  if [ ! -d "$TASKS_DIR/molecular_vis" ]; then
    echo "  downloading tasks (HuggingFace)…"
    "$PY" - "$TASKS_DIR" <<'PY' || die "task download failed"
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id="SciVisAgentBench/SciVisAgentBench-tasks",
                  repo_type="dataset", local_dir=sys.argv[1])
PY
  fi
  [ -d "$TASKS_DIR/molecular_vis" ] && ok "tasks: $TASKS_DIR/molecular_vis" || die "molecular_vis tasks missing under $TASKS_DIR"
fi

# The judge imports the openai (and anthropic) clients even for a Claude eval
# model, and --skip-setup skips the block above — so ensure them whenever we
# score, regardless of --skip-setup.
if [ "$WITH_EVAL" = 1 ] && ! "$PY" -c "import openai, anthropic" >/dev/null 2>&1; then
  echo "  installing judge deps (openai anthropic httpx)…"
  "$PY" -m pip install -q openai anthropic httpx || warn "judge deps install had issues"
fi

# --full needs the image-metric stack (vision rubrics) and the workflow data on disk.
if [ "$FULL" = 1 ]; then
  if ! "$PY" -c "import torch, lpips, skimage, cv2" >/dev/null 2>&1; then
    echo "  installing image-metric deps (torch torchvision lpips scikit-image opencv-python)… [large, one-time]"
    "$PY" -m pip install -q torch torchvision lpips scikit-image opencv-python "numpy<2" || warn "image-metric deps install had issues"
    "$PY" -c "import numpy" >/dev/null 2>&1 && "$PY" -c "import numpy,sys; sys.exit(0 if numpy.__version__[0]=='1' else 1)" || \
      { echo "  pinning numpy<2 (skimage ABI)…"; "$PY" -m pip install -q "numpy<2"; }
  fi
  miss=""
  for d in ras-raf-membrane curved-membrane trajectory-inspection; do
    [ -d "$TASKS_DIR/molecular_vis/$d/data" ] || miss="$miss $d"
  done
  if [ -n "$miss" ]; then
    die "workflow data missing under $TASKS_DIR/molecular_vis (need:$miss). Re-download the dataset incl. LFS (158MB trajectory):
       hf download SciVisAgentBench/SciVisAgentBench-tasks --repo-type dataset --local-dir $TASKS_DIR"
  fi
  ok "workflow data present (ras-raf-membrane, curved-membrane, trajectory-inspection)"
fi

# ============================================================================
rung 3 "registration (lightweight launcher)"
cd "$BENCH_DIR"
if "$PY" "$SCRIPT_DIR/run_vmd_ai.py" --list --bench "$BENCH_DIR" 2>&1 | tee /tmp/vmdai_list.txt | grep -q "vmd_ai"; then
  ok "vmd_ai registers and the framework imports (no paraview/mcp needed)"
else
  sed 's/^/    /' /tmp/vmdai_list.txt; die "vmd_ai did not register — import error above."
fi

# ============================================================================
rung 4 "agent run, no judge (--no-eval)"
if [ "$WITH_EVAL" = 1 ]; then
  warn "skipping the no-eval run — rung 5 runs the agent + judge in one pass (--with-eval)."
elif [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  warn "ANTHROPIC_API_KEY not set — skipping the live agent run."
  echo "  export ANTHROPIC_API_KEY=sk-ant-... then: bash run_benchmark_test.sh --skip-setup"
else
  echo "  running $RUN_LABEL (no judge cost)…"
  "$PY" "$SCRIPT_DIR/run_vmd_ai.py" \
      --bench "$BENCH_DIR" --config "$SRC_CONFIG" \
      --yaml "$BENCH_DIR/$YAML_REL" --cases "$TASKS_DIR/molecular_vis" \
      "${CASE_ARGS[@]}" --no-eval --experiment-number "$EXP" \
    || die "agent run errored — see traceback above."
  _report_outputs
fi

# ============================================================================
if [ "$WITH_EVAL" = 1 ]; then
  rung 5 "scored run: agent + LLM judge ($RUN_LABEL — COSTS money)"
  KEY="${OPENAI_API_KEY:-}"; case "$EVAL_MODEL" in *claude*|*anthropic*) KEY="${ANTHROPIC_API_KEY:-}";; esac
  if [ -z "$KEY" ]; then
    warn "no API key for eval model '$EVAL_MODEL' — skipping (set OPENAI_API_KEY, or EVAL_MODEL=claude-... with ANTHROPIC_API_KEY)."
  else
    echo "  running $RUN_LABEL with judge=$EVAL_MODEL…"
    "$PY" "$SCRIPT_DIR/run_vmd_ai.py" \
        --bench "$BENCH_DIR" --config "$SRC_CONFIG" \
        --yaml "$BENCH_DIR/$YAML_REL" --cases "$TASKS_DIR/molecular_vis" \
        "${CASE_ARGS[@]}" --eval-model "$EVAL_MODEL" --experiment-number "$EXP" \
      || die "scored run errored — see above."
    ok "scored — see */evaluation_results/$AGENT_MODE/ under $TASKS_DIR/molecular_vis/"
  fi
fi

echo; hr; echo "${G}${B}Done.${N} Reached the last rung without a hard failure."
echo "Next: drop --case to run all 10 molecular_vis cases."; hr
