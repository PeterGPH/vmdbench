# vmdbench

A **correctness-oriented** benchmark for LLM agents that drive [VMD](https://www.ks.uiuc.edu/Research/vmd/) by emitting Tcl.

Most agent benchmarks score **task completion** — did the agent produce an answer or render a scene? For a scientific instrument that is the wrong target: a confidently reported wrong number is worse than a refusal, and a visualization that "looks done" but contains zero representations is a silent failure. vmdbench grades every answer against an **independently computed oracle value**, so it measures whether the agent is *right*, not merely whether it *finished*.

## Headline result

Evaluated on Qwen2.5-72B-Instruct (AWQ) as a VMD agent:

| | completion | correctness | gap |
|---|---|---|---|
| baseline (raw agent) | ~65% | ~56% | the agent confidently answers wrong |
| + injection, Tcl validation, semantic tools | **97%** | **97%** | **~0** — when it answers, it is right |

A held-out check on operations the semantic tools *don't* cover shows the honest bound: correctness there is **58% → 67%**, so most of the 97% is tool coverage, with a smaller, idiom-specific gain from reference injection rather than a general capability jump. Oracle gold is cross-validated against MDAnalysis + Biopython (19/20 metrics agree).

## The tasks

**Analysis suite** — 5 structures × 5 scalar metrics, each with a per-structure oracle:

- Structures: `1CRN` (327 atoms), `1UBQ` (660), `1HSG` (1,686, has a ligand), `1HCK` (2,510), `1QMZ` (9,889, multi-chain).
- Metrics: radius of gyration, atom count, residue count, first–last Cα distance, SASA.

**Held-out suite** — 6 operations *outside* the semantic-tool coverage (agent must write raw Tcl): hydrogen bonds, a 3-atom angle, a backbone dihedral, center of mass, bounding-box extent, glycine count.

**Visualization** — scene-state assertions (does the requested representation exist, with the right style/coloring; was a non-empty image written) instead of "did an image appear."

## Repository layout

The flow is: **oracle computes truth → runner asks the agent → scorer compares → comparator tabulates.**

**Oracles (ground truth, run in VMD independently of the agent)**
- `gold_oracle.tcl` — gold for the original 1CRN tasks.
- `gold_oracle_multi.tcl` — parametric gold for the 5-structure analysis suite.
- `gold_oracle_heldout.tcl` — gold for the 6 held-out operations.

**Runners (drive the agent, collect answers)**
- `run_multistructure.py` — main analysis suite; writes `summary.json`.
- `run_heldout.py` — held-out generalization suite.
- `run_vmd_ai.py` — standalone single-task launcher (smoke/debug).

**Agent adapter + VMD plumbing**
- `vmd_ai_agent.py` — adapts the agent to the harness; picks model/provider and toggles the three interventions.
- `subprocess_vmd_bridge.py` — drives a real VMD.app via a pty; implements Tcl pre-validation and the semantic `vmd_measure` / `vmd_represent` tools.
- `headless_vmd_bridge.py`, `retrieval_bridge.py`, `smoke_bridge.py` — headless bridge, RAG retrieval helper, connectivity smoke test.

**Scorers + aggregators (turn runs into tables)**
- `score_correctness.py`, `score_viz.py` — score analysis answers and visualization scene state.
- `compare_models.py` — the completion-vs-correctness table.
- `compare_heldout.py` — the none/inject/improved held-out decomposition.
- `compare_modes.py`, `aggregate_ablation.py` — aggregate the RAG/wiki/inject ablation arms.
- `validate_gold.py` — cross-checks oracle gold against MDAnalysis + Biopython.

**Configs + drivers**
- `config_arm_*.json` — ablation arms (none, inject, rag, wiki, …).
- `config_model_*.json` — model/provider selection (Qwen / Claude / GPT).
- `run_*.sh` — convenience sweeps.

## Running it

Requires a local VMD.app and an OpenAI-compatible model endpoint (e.g. vLLM serving Qwen, or an API key for Claude/GPT).

```bash
# baseline arm, 3 seeds
python run_multistructure.py --config config_arm_none.json   --tag none     --seeds 3
# improved arm (injection + validation + semantic tools)
python run_multistructure.py --config config_local.json      --tag improved --seeds 3
# the headline table
python compare_models.py --tags none,improved

# held-out generalization check
python run_heldout.py --config config_arm_none.json --tag heldout_none     --seeds 3
python run_heldout.py --config config_local.json    --tag heldout_improved --seeds 3
python compare_heldout.py --tags heldout_none,heldout_improved
```

## Limitations

Small scale (25 analysis cells, all single-frame scalars — no trajectories yet); oracle tolerances are hand-set; scoring is currently lenient (picks the closest float in the answer); results are reported for one model (cross-model baselines in progress).
