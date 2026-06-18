#!/usr/bin/env python
"""
run_vmd_ai.py — lightweight launcher to evaluate the vmd_ai agent on
SciVisAgentBench *without* importing the heavy CLI.

Why: SciVisAgentBench's own `run_evaluation` imports all prebuilt agents
(ParaView/napari/MCP) at module load, several of which need conda-only packages
(paraview, mcp). That import wall blocks a base pip env. This launcher imports
*only* our agent and drives the framework's `UnifiedTestRunner` directly, which
needs just pyyaml + (for the judge) an LLM client. So you can test your agent
without installing the whole scivis stack.

Usage (point --bench at your clone, or set BENCH_DIR):
    python run_vmd_ai.py --list
    python run_vmd_ai.py --config config_anthropic.json \
        --yaml <bench>/benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml \
        --cases <tasks>/molecular_vis --case case_1 --no-eval --experiment-number exp1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _wire_paths(bench: Path) -> None:
    # 1) the framework package lives under <bench>/benchmark/
    sys.path.insert(0, str(bench / "benchmark"))
    # 2) our agent + bridges live next to this launcher (imported in place)
    sys.path.insert(0, str(SCRIPT_DIR))


def _parse():
    p = argparse.ArgumentParser(description="Evaluate vmd_ai on SciVisAgentBench (lightweight).")
    p.add_argument("--bench", default=os.environ.get("BENCH_DIR", str(Path.home() / "SciVisAgentBench")))
    p.add_argument("--config", default=str(SCRIPT_DIR / "config_anthropic.json"))
    p.add_argument("--yaml")
    p.add_argument("--cases")
    p.add_argument("--case", nargs="+", help="one or more case names (default: all)")
    p.add_argument("--no-eval", action="store_true", help="skip the LLM judge (agent only)")
    p.add_argument("--eval-model", default="gpt-4o")
    p.add_argument("--experiment-number", "--exp", default="exp1", dest="experiment_number")
    p.add_argument("--list", action="store_true", help="list registered agents and exit")
    p.add_argument("--list-cases", action="store_true")
    return p.parse_args()


async def _amain() -> int:
    args = _parse()
    bench = Path(args.bench).expanduser().resolve()
    _wire_paths(bench)

    # Importing our module runs @register_agent("vmd_ai"). No prebuilt agents.
    import vmd_ai_agent  # noqa: F401
    from evaluation_framework import get_agent, list_agents, UnifiedTestRunner

    if args.list:
        print("Registered agents:", ", ".join(sorted(list_agents())) or "(none)")
        return 0

    for req in ("yaml", "cases"):
        if not getattr(args, req):
            print(f"Error: --{req} is required (unless --list)"); return 2

    with open(args.config) as fh:
        config = json.load(fh)
    config["experiment_number"] = args.experiment_number

    agent = get_agent("vmd_ai")(config)
    print(f"agent_mode = {agent.agent_mode}")

    eml = args.eval_model.lower()
    eval_key = (os.getenv("ANTHROPIC_API_KEY") if ("claude" in eml or "anthropic" in eml)
                else os.getenv("OPENAI_API_KEY"))

    runner = UnifiedTestRunner(
        agent=agent, yaml_path=args.yaml, cases_dir=args.cases,
        eval_model=args.eval_model, openai_api_key=eval_key, config=config,
    )
    cases = runner.load_yaml_test_cases()
    if args.list_cases:
        print("Cases:", ", ".join(c.case_name for c in cases)); return 0

    selected = [c for c in cases if (not args.case or c.case_name in args.case)]
    if not selected:
        print("No matching cases. Available:", [c.case_name for c in cases]); return 1

    do_eval = (not args.no_eval) and bool(eval_key)
    if not args.no_eval and not eval_key:
        print("(no judge key found -> running agent only; set OPENAI_API_KEY/ANTHROPIC_API_KEY for scoring)")

    await agent.setup()
    all_ok = True
    try:
        for tc in selected:
            res = await runner.run_single_test_case(tc, save_result=False)
            if do_eval and res.get("status") == "completed":
                res["evaluation"] = await runner.run_evaluation(tc)
            await runner.save_centralized_result(tc, res)
            status = res.get("status")
            score = (res.get("evaluation", {}) or {}).get("scores", {}).get("percentage")
            print(f"  case {tc.case_name}: {status}" + (f"  score={score:.0f}%" if score is not None else ""))
            if status != "completed":
                err = res.get("error") or "(no error text; check the saved result JSON)"
                print("  --- failure detail ---")
                print("  " + str(err).replace("\n", "\n  ")[:3000])
                meta = res.get("metadata") or {}
                if meta.get("partial_response"):
                    print("  partial_response:", str(meta["partial_response"])[:400])
            all_ok = all_ok and status == "completed"
    finally:
        await agent.teardown()
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain()))
