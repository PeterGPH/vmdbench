#!/usr/bin/env python3
"""score_viz.py — visualization correctness via vmdbench scene-state assertions.

The SciVisAgentBench viz cases (case_1/2/4/6) are scored only by a yes/no text rubric
("does it show licorice? yes"). This replays the agent's captured .tcl transcript for each
through vmdbench's headless env and checks what it ACTUALLY built — the representation
style / coloring / selection — the visual analog of scalar_within. It surfaces cases the
rubric passes but the scene does not.

Prereq: re-run those cases with the current adapter so the .tcl transcripts exist, e.g.
  run_vmd_ai.py ... --case case_1 case_2 case_4 case_6 --experiment-number viz_demo

  python score_viz.py --tcl-dir ~/SciVisAgentBench/SciVisAgentBench-tasks/molecular_vis
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

# SciVisAgentBench case .tcl  ->  (vmdbench viz card, short description of expected scene)
MAP = {
    "case_1": ("viz_licorice_1crn_001",      "show protein as Licorice"),
    "case_2": ("viz_color_element_1crn_001",  "color by Element"),
    "case_4": ("viz_color_charge_1crn_001",   "color by Charge"),
    "case_6": ("viz_aromatic_1crn_001",       "select resname PHE TYR TRP"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcl-dir", required=True, help="dir with the agent's case_N.tcl transcripts")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--timeout", type=int, default=120)
    a = ap.parse_args()
    repo = Path(os.path.expanduser(a.repo))
    tcl_dir = Path(os.path.expanduser(a.tcl_dir))
    cards = repo / "vmdbench" / "tasks" / "viz"

    print(f"{'case':7} {'expected scene':30} {'rubric':7} {'scene-state':12} detail")
    print("-" * 96)
    n_rubric = n_scene = 0
    for case, (card_id, desc) in MAP.items():
        tcl = tcl_dir / f"{case}.tcl"
        card = cards / f"{card_id}.yaml"
        n_rubric += 1  # the text rubric always passes (agent wrote "yes")
        if not tcl.exists():
            print(f"{case:7} {desc:30} {'yes':7} {'(no .tcl)':12} re-run this case with the current adapter")
            continue
        try:
            out = subprocess.run(
                [sys.executable, "-m", "vmdbench.cli", "score-oracle",
                 str(card), str(tcl), "--timeout", str(a.timeout)],
                cwd=str(repo), capture_output=True, text=True, timeout=a.timeout + 60)
            payload = json.loads(out.stdout)
        except Exception as exc:  # noqa: BLE001
            print(f"{case:7} {desc:30} {'yes':7} {'ERROR':12} {str(exc)[:50]}")
            continue
        v = payload.get("verify", {})
        gate = bool(v.get("gate"))
        n_scene += int(gate)
        rep = next((r for r in v.get("required", []) if r["kind"] == "representation_exists"), None)
        detail = ""
        if rep and not rep["passed"]:
            detail = f"built instead: {rep.get('observed')}"
        print(f"{case:7} {desc:30} {'yes':7} {('PASS' if gate else 'FAIL'):12} {detail}")
    print("-" * 96)
    print(f"=> rubric passes {n_rubric}/{len(MAP)} (always 'yes'); scene-state passes {n_scene}/{len(MAP)}")
    print("rubric = SciVisAgentBench text check; scene-state = what vmdbench verified was actually built.")


if __name__ == "__main__":
    main()
