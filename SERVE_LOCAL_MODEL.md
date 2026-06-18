# Run the benchmark against a self-hosted model (no per-token cost)

Split of work: **VMD + the agent loop stay on your Mac**; only **LLM inference moves
to the 80 GB GPU server**. The agent calls the server's OpenAI-compatible API for each
turn; `run_vmd_command` still executes in your local VMD.

```
  your Mac                         GPU server (80 GB)
 ┌───────────────────────┐  HTTP  ┌──────────────────────────┐
 │ run_vmd_ai.py + VMD   │ ─────▶ │ vLLM  /v1/chat/completions│  (4-bit model)
 │ (subprocess bridge)   │ ◀───── │ + tool-call parser        │
 └───────────────────────┘        └──────────────────────────┘
```

---

## A. Server — serve a 4-bit model with tool calling (vLLM)

### A1. Prereqs
NVIDIA 80 GB GPU (A100/H100), recent driver + CUDA 12.x, Python 3.10+.
```bash
pip install vllm            # or use the vllm/vllm-openai Docker image
```

### A2. Pick a 4-bit (AWQ) model so it fits in 80 GB
| Model | parser flag | notes |
|---|---|---|
| `Qwen/Qwen2.5-72B-Instruct-AWQ` | `--tool-call-parser hermes` | strong tool use; recommended start |
| `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4` | `--tool-call-parser llama3_json` | well-documented on vLLM |
| (newer Qwen3 / Kimi / GLM / DeepSeek AWQ builds) | model-specific | same idea, check the model card for the parser |

4-bit ≈ 0.5 byte/param → ~35–50 GB weights, leaving ~30 GB for the KV cache.

### A3. Launch (OpenAI-compatible API + tool calling)
```bash
export VLLM_API_KEY=sk-local-pick-anything     # or omit --api-key to require none
# SHARED box: pin to ONE idle GPU (nvidia-smi -> pick a 0%-util card), cap memory,
# and bind to localhost (reach it via SSH tunnel) so you don't disturb other users.
CUDA_VISIBLE_DEVICES=4 vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \
    --quantization awq \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --host 127.0.0.1 --port 8000 --api-key "$VLLM_API_KEY"
```
A 4-bit 72B (~45 GB) fits on one 80 GB A100; no tensor-parallel needed. (If you ever
want a bigger/higher-precision model AND ≥2 GPUs are idle: `CUDA_VISIBLE_DEVICES=4,5
--tensor-parallel-size 2`.)
`--enable-auto-tool-choice` + the **matching** `--tool-call-parser` are mandatory — without
them the model never emits `tool_calls` and your agent will do nothing.

Docker alternative:
```bash
docker run --gpus all --ipc=host -p 8000:8000 vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-72B-Instruct-AWQ --quantization awq \
    --max-model-len 16384 --enable-auto-tool-choice --tool-call-parser hermes
```

### A4. Verify on the server
```bash
curl -s http://localhost:8000/v1/models -H "Authorization: Bearer $VLLM_API_KEY"
```

### Lighter alternative — Ollama
Your runtime already speaks Ollama natively. `ollama pull qwen2.5:72b` then serve; set the
config `provider: "ollama"` and `base_url: "http://server:11434"`. Easier, but vLLM is faster
and its tool-calling is more reliable for an agentic benchmark.

---

## B. Reach the server from your Mac
- Same network: use `http://<server-ip>:8000/v1` in the config.
- Or SSH-tunnel (simplest + secure):
  ```bash
  ssh -L 8000:localhost:8000 you@server      # leave open; then use http://localhost:8000/v1
  curl -s http://localhost:8000/v1/models -H "Authorization: Bearer $VLLM_API_KEY"
  ```

---

## C. Point the benchmark at it (config only — runtime now supports a base URL)
Edit `config_local.json` (already created): set `model`, `base_url`, and `api_key` to match
your server. Then run the same harness:
```bash
python integrations/scivisagentbench/run_vmd_ai.py \
    --bench ~/SciVisAgentBench \
    --config integrations/scivisagentbench/config_local.json \
    --yaml  ~/SciVisAgentBench/benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml \
    --cases ~/SciVisAgentBench/SciVisAgentBench-tasks/molecular_vis \
    --no-eval
```
(Or `run_benchmark_test.sh` with `SRC_CONFIG=.../config_local.json`.) The runtime change:
`_stream_openrouter` now reads `VMD_AI_OPENAI_BASE_URL` (set automatically from the config's
`base_url`), defaulting to OpenRouter when unset — so nothing else changes.

---

## Gotchas
- **Parser must match the model** (`hermes` for Qwen, `llama3_json` for Llama) or you get text,
  not tool calls — the #1 "the local model won't act" failure.
- **Context**: agentic runs grow; `--max-model-len 16384` is a sane start. Bigger context = more
  KV-cache VRAM, so raise it only if you hit context-limit errors.
- **Cloud GPU** if you don't own one: RunPod / Lambda / Vast rent an H100-80GB ~$2–3/hr — flat
  rate, no per-token, which is the whole point.
- **Expect a quality drop** vs Claude on Tcl + tool-calling. That gap is exactly the number your
  harness measures — and your RunRecorder transcripts are the fine-tuning data to close it.
