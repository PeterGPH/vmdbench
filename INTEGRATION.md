# Plugging vmd_ai into SciVisAgentBench

Goal: get **your** vmd_ai/ChatVMD agent a score on SciVisAgentBench's
`molecular_vis` track, by driving your existing `ClaudeToolLoop` through a
headless VMD instead of the Tk panel.

## How it works (the one coupling point)

`ClaudeToolLoop.run()` executes every VMD tool call through a single call:

```python
tool_bridge.execute_tool(session_id=..., tool_name=..., tool_input=..., ...)
    -> {"ok": bool, "output": str, "error": str}
```

So the adapter changes **nothing** in your runtime. It just passes a different
`tool_bridge` — a headless one that executes the Tcl without the Tk panel and
captures `puts` output. Two backends ship; pick with `vmd_backend` in the config:

- **`subprocess` (default)** — drives your native `VMD.app` as a persistent
  `vmd -dispdev text` process. No install, and real `TachyonInternal` snapshots.
- **`vmd_python`** — in-process `vmd.evaltcl` (needs vmd-python, which has no
  Apple-Silicon build — Linux/Docker or a Rosetta osx-64 env only).

```
SciVisAgentBench run_evaluation
  └─ VmdAiAgent.run_task(question, task_config)        # vmd_ai_agent.py
        └─ ClaudeToolLoop.run(prompt=question,
                              tool_bridge=<backend>)    # your runtime, unchanged
                 └─ run_vmd_command / capture_vmd_snapshot
                        └─ vmd -dispdev text (subprocess)   # subprocess_vmd_bridge.py
```

## Files

| File | Goes to | Purpose |
|---|---|---|
| `vmd_ai_agent.py` | `benchmark/evaluation_framework/agents/` | the `@register_agent("vmd_ai")` adapter |
| `subprocess_vmd_bridge.py` | same dir | **default** backend — drives native VMD.app |
| `headless_vmd_bridge.py` | same dir | optional backend — in-process vmd-python |
| `config_anthropic.json` | `benchmark/configs/vmd_ai/` | provider/model/runtime-path/backend |
| `smoke_bridge.py` | (kept in this folder) | no-API test of the bridge — run this first |

## Setup

1. **VMD** — no new install: the default backend drives your existing VMD.app.
   Make sure `vmd` is on PATH, or set the binary explicitly:
   ```bash
   export VMD_AI_VMD_BIN=/Applications/VMD.app/Contents/vmd/vmd_MACOSXARM64
   ```
   (molecular_vis is licorice / CPK / coloring / analysis — no NewCartoon — so
   the STRIDE hang doesn't apply here. vmd-python is *not* on PyPI and has no
   Apple-Silicon build, which is why the default avoids it.)

2. **Get the benchmark + tasks** (per the repo README):
   ```bash
   git clone https://github.com/KuangshiAi/SciVisAgentBench
   pip install huggingface_hub && hf download SciVisAgentBench/SciVisAgentBench-tasks \
     --repo-type dataset --local-dir SciVisAgentBench/SciVisAgentBench-tasks
   ```

3. **Drop in the two modules** and register the agent — add this line to
   `benchmark/evaluation_framework/agents/__init__.py`:
   ```python
   from .vmd_ai_agent import VmdAiAgent
   ```

4. **Place the config** at `benchmark/configs/vmd_ai/config_anthropic.json`, set
   `vmd_ai_runtime_path` to `<your repo>/vmd_ai/runtime`, and export your key:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   ```

## Run

Start with **one case** to shake out paths before spending on the full set:

```bash
python -m benchmark.evaluation_framework.run_evaluation \
    --agent vmd_ai \
    --config benchmark/configs/vmd_ai/config_anthropic.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --eval-model gpt-5.2 \
    --experiment-number exp1
```

(The README's `actions/basic_actions.yaml` path is stale — the real
molecular_vis YAMLs are `eval_analysis_tasks.yaml`, `eval_analysis_workflows.yaml`,
`eval_analysis_all.yaml` at the `molecular_vis/` root.)

## How molecular_vis is graded (so the score is real)

These cases are **LLM-rubric over a file the agent writes**, not image
similarity. Each `question` tells the agent to load `data/1CRN.cif`, do the
visualization/analysis, and **write its answer to**
`case_N/results/{agent_mode}/answers_*.txt`; an LLM judge (`--eval-model`) reads
that file against the rubric. Your agent writes that file through
`run_vmd_command` Tcl (`set f [open ... w]; puts $f ...; close $f`) — no extra
tool needed.

`{agent_mode}` resolves to `vmd_ai_<model>_<experiment_number>` (e.g.
`vmd_ai_claude-sonnet-4-5_exp1`), computed identically by the framework and the
adapter, so paths line up.

## Three things to verify on the first case (the real risk lives here)

1. **stdout capture** — confirm the model sees `puts` output. Run `smoke_bridge.py`
   first; it checks exactly this (the subprocess sentinel + `catch` protocol).
   If `PROT=5 WAT=1` comes back, capture works.
2. **answer-file location** — the adapter `chdir`s to `task_config["working_dir"]`
   so `data/1CRN.cif` and `results/...` resolve. If the judge reports "file not
   found," check where the agent actually wrote vs the `rs-file` path in the YAML
   and adjust the `chdir` target (working_dir vs case_dir vs cases-root).
3. **VMD binary resolves** — the bridge auto-finds `/Applications/VMD.app/...`; if
   not, set `VMD_AI_VMD_BIN` (or `vmd_bin` in the config). Snapshots use real
   `TachyonInternal` on this backend, so they render proper images.

## Honest caveats

- **Tokens are estimated.** `ClaudeToolLoop.run()` doesn't surface provider token
  counts, so the efficiency metric uses a tiktoken estimate (±20%). Wire real
  counts later if the efficiency dimension matters.
- **No rate-limit backoff in your loop.** The `rate_limits` block is set to your
  30k-input-tokens/min tier, but your runtime has no 429 retry — keep runs small,
  or add backoff in `claude_loop.py` first.
- **RAG/wiki are off** (`docs_search=None`, `wiki_store=None`) for a clean
  baseline. Flip them on in `setup()` to A/B their effect on the same tasks —
  which is exactly the experiment your own benchmark is built around.
- **Pin one engine for reported numbers.** The default standardizes on your
  native VMD.app (`subprocess`). If you later want numbers directly comparable to
  SciVisAgentBench's published GMX-VMD runs, switch `vmd_backend` to `vmd_python`
  in Docker/Linux and keep it fixed.
```
