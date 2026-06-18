# Beyond 1CRN — multi-structure correctness testing

Generalizes the correctness test from one structure (1CRN) to five, with portable,
structure-agnostic metrics, in **both** frameworks: the SciVisAgentBench correctness
harness (where the agent arms live) and the durable `vmdbench` cards.

## Structures (in `fixtures_multi/`)

| file | PDB | size |
|---|---|---|
| 1CRN.cif | crambin | 327 atoms |
| 1UBQ.pdb | ubiquitin | 602 |
| 1HSG.pdb | HIV protease | 1 514 |
| 1HCK.pdb | CDK2 kinase | 2 370 |
| 1QMZ.pdb | CDK2/cyclin | 9 036 |

Add a structure by dropping a `.pdb`/`.cif` into `fixtures_multi/` — it is picked up
automatically. No per-structure code: the metrics below compute on any protein.

## Portable metrics (gold from `gold_oracle_multi.tcl`)

`rgyr` (radius of gyration), `natoms`, `nresidues`, `ca_dist` (first↔last CA), `sasa`
(`measure sasa 1.4`). Each is computed by the oracle with no hard-coded residue ids, so
gold regenerates for any structure. (1CRN's old case_9 hard-coded "resid 5" — that's
what made it structure-specific; these don't.)

## A — run the agent arms across all structures

On the Mac, with the vLLM server + SSH tunnel up. One run per arm; compare the summaries.

```bash
cd ~/Documents/GitHub/PyMolAI/vmd_ai
P=integrations/scivisagentbench

python $P/run_multistructure.py --bench ~/SciVisAgentBench --config $P/config_arm_none.json   --tag none
python $P/run_multistructure.py --bench ~/SciVisAgentBench --config $P/config_arm_inject.json --tag inject
```

Each run computes gold per structure, drives the agent over the 5 structures × 5 metrics,
scores every answer against the oracle, and prints a per-metric pass-rate table plus an
overall correctness count. Results + `summary.json` land in
`test_results/multistructure/<tag>/`. The headline comparison: does `inject` beat `none`
on correctness across *all five* structures (confirming the 1CRN finding generalizes), or
only on some?

To test another arm, point `--config` at `config_arm_autorag.json`, `_smartinject.json`, etc.

## B — durable vmdbench cards

`vmdbench/tasks/analysis/rg_{1crn,1ubq,1hck}_001.yaml` (+ matching oracles) check the
radius of gyration with a `scalar_within` correctness band. First confirm/tighten each
band against its oracle (prints the exact value):

```bash
cd ~/Documents/GitHub/PyMolAI/vmd_ai
python -m vmdbench.cli score-oracle vmdbench/tasks/analysis/rg_1ubq_001.yaml vmdbench/oracles/rg_1ubq_001.tcl --timeout 120
python -m vmdbench.cli score-oracle vmdbench/tasks/analysis/rg_1hck_001.yaml vmdbench/oracles/rg_1hck_001.tcl --timeout 120
```

Read the printed `measure_rgyr`, then tighten the card's `min/max` to value ± 0.5 for a
strict check. Add more structures/metrics by copying a card+oracle pair.

## Notes
- `gold_oracle_multi.tcl` uses the `protein` selection for Rg/SASA (excludes waters/ligands),
  which is the standard convention and predictable across structures.
- `ca_dist` is "first vs last CA atom by atom order" — well-defined for multi-chain
  structures too; the agent prompt states the same definition so agent and oracle agree.
- `contacts8`/`sasa` are method-sensitive; the agent is told to use `measure sasa 1.4` so it
  matches the oracle. `sasa` tolerance is intentionally a little loose.
