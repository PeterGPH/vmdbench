# Reproducing SciVisAgentBench paper numbers (molecular_vis)

Goal: re-run *their* agents through *their* pipeline to land near a leaderboard
row, e.g. Claude-Code+Claude-Sonnet-4.5 = 61.47 or GMX-VMD-MCP+Claude = 60.23.

The pipeline is a **3-step split** (this is the part our own run skipped):

1. **EXECUTE** the agent in a sandbox against a **GS-free** copy (so it can't read
   the answers).
2. **EVALUATE** on the host, where the ground-truth `GS/` images + metric stack live
   (`--eval-only --eval-model gpt-5.2`).
3. **REPORT** on the host → the table with the 8 columns.

Run everything from the repo root (`SciVisAgentBench-main/`). The task data must
be at `SciVisAgentBench-tasks/` next to `benchmark/` (download once with
`huggingface-cli download SciVisAgentBench/SciVisAgentBench-tasks --repo-type dataset --local-dir SciVisAgentBench-tasks`).

Three fidelity requirements (or it won't match the paper):
- **Judge = `gpt-5.2`** → needs an `OPENAI_API_KEY`. A Claude judge gives different numbers.
- **GS-isolation** during execution (Docker path does this; host-direct does not).
- **Seeds**: the `±std` means 3 runs (`exp1/exp2/exp3`) aggregated. One run = a point.

You will NOT hit the exact numbers (judge nondeterminism, model-snapshot drift,
seed variance). "Reproduce" = same ballpark, which validates the setup.

---

## Path A — Claude-Code via Docker (faithful; recommended)

Reproduces the **Claude-Code** rows. Uses the model you already have (Claude).

### A0. Container engine on Apple Silicon (the image is linux/amd64)
Run an **arm64 VM with Rosetta** and let Docker translate the amd64 image. Do NOT
pass `--arch x86_64` — that forces a full x86_64 VM under qemu (slow, and needs a
separate `lima-additional-guestagents` package → "guest agent binary not found").
```bash
brew install colima docker docker-buildx
colima delete -f 2>/dev/null
colima start --vm-type vz --vz-rosetta --cpu 4 --memory 8 --disk 80
docker run --rm --platform linux/amd64 alpine uname -m   # -> x86_64 (via Rosetta) = good
```
Fallback if `vz`/Rosetta is unavailable (macOS < 13):
```bash
brew install lima-additional-guestagents
colima delete -f && colima start --arch x86_64 --cpu 4 --memory 8 --disk 80   # qemu, slow but works
```

### A1. Build the images (one time, ~30–45 min, several GB)
```bash
./docker/build.sh           # base: ParaView/napari/VTK/TTK/MDAnalysis/torch/lpips/Mesa
./docker/build_claude.sh    # + Claude Code CLI
```

### A2. Host evaluation environment (the EVALUATE step runs on the host)
Install the metric stack into a host python env (your conda base is fine):
```bash
python -m pip install -r requirements.txt   # torch, lpips, scikit-image, opencv, openai, anthropic, …
```

### A3. Keys
```bash
export ANTHROPIC_API_KEY=sk-ant-...   # the agent (forwarded into the container)
export OPENAI_API_KEY=sk-...          # the gpt-5.2 judge (host side)
```

### A4. Run molecular_vis — full set (tasks + workflows), per seed
```bash
EXP=exp1   # then repeat the whole block with exp2, exp3 for the ±std

for Y in eval_analysis_tasks eval_analysis_workflows; do
  # (1) execute in the sandbox
  ./docker/run_eval_in_docker.sh --agent claude_code \
      --config benchmark/configs/claude_code/config.json \
      --yaml benchmark/eval_cases/molecular_vis/$Y.yaml \
      --cases SciVisAgentBench-tasks/molecular_vis --experiment-number $EXP

  # (2) evaluate on the host (gpt-5.2 judge + image metrics)
  python -m benchmark.run_claude_code_eval --agent claude_code \
      --config benchmark/configs/claude_code/config.json \
      --yaml benchmark/eval_cases/molecular_vis/$Y.yaml \
      --cases SciVisAgentBench-tasks/molecular_vis \
      --eval-model gpt-5.2 --eval-only --experiment-number $EXP
done

# (3) report
python -m benchmark.evaluation_reporter.run_reporter --agent claude_code \
    --config benchmark/configs/claude_code/config.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_all.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --test-results test_results/molecular_vis/claude_code_claude-sonnet-4-5_$EXP/ \
    --output eval_reports/molecular_vis/claude_code_claude-sonnet-4-5_$EXP \
    --agent-mode claude_code_claude-sonnet-4-5_$EXP --no-browser --static-only
```
The composite/completion/metric columns are in the generated `report.html` and the
per-case `evaluation_results/*.json`.

---

## Path B — host-direct, no Docker (lighter; "indicative" not official)

Same agent, no container. Skips GS-isolation, so treat numbers as indicative.
Needs the `claude` CLI installed+authed and a host viz env:
```bash
conda create -n scivis_bench -c conda-forge python=3.10 paraview numpy scipy matplotlib
conda activate scivis_bench && python -m pip install napari -r requirements.txt
# VMD must be on PATH for the molecular tasks; OPENAI_API_KEY for the judge.
```
Then replace step (1) above with a direct host run (no `--exe-only`/Docker):
```bash
python benchmark/run_claude_code_eval.py --agent claude_code \
    --config benchmark/configs/claude_code/config.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_all.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --eval-model gpt-5.2 --experiment-number exp1
```
(`run_claude_code_eval.py` does execute+evaluate+save in one host process.)

---

## Path C — GMX-VMD-MCP (the true VMD peer; the 60.23 row)

Closest agent to your `vmd_ai` (both drive VMD), so the best baseline to put
beside yours — but the heaviest setup. Per the repo README:
```bash
conda create -n gmx_vmd_mcp python=3.10 && conda activate gmx_vmd_mcp
python -m pip install -r requirements.txt
cd src/gmx_vmd_mcp && python -m pip install -r requirements.txt && python -m pip install -e . && cd ../..
# Prereqs: GROMACS and VMD on PATH; create src/gmx_vmd_mcp/config.json with their paths.
```
Then run (no Docker; the agent launches the MCP server itself):
```bash
python -m benchmark.evaluation_framework.run_evaluation --agent gmx_vmd_mcp \
    --config benchmark/configs/gmx_vmd_mcp/config_anthropic.json \
    --yaml benchmark/eval_cases/molecular_vis/eval_analysis_all.yaml \
    --cases SciVisAgentBench-tasks/molecular_vis \
    --eval-model gpt-5.2 --experiment-number exp1
```

---

## Gotchas

- The 158 MB trajectory (`trajectory-inspection_3to5us.xtc`) must be pulled by the
  dataset download (it's git-LFS). Check it exists before the workflow run.
- Colima needs disk headroom (`--disk 80`); the base image is several GB.
- Headless GL in the container uses Mesa/Xvfb; under amd64 emulation it is slow but works.
- Without `OPENAI_API_KEY` the gpt-5.2 judge is skipped and you only get completion, not scores.
- `docker-credential-desktop ... not found` (leftover from a past Docker Desktop):
  strip the stale helper — remove the `"credsStore"` key from `~/.docker/config.json`
  (public image pulls need no auth).
