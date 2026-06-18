#!/usr/bin/env python
"""
smoke_bridge.py — isolate-and-test the riskiest part of the integration:
the headless VMD bridge. NO API key, NO SciVisAgentBench checkout needed.

Uses your *native* VMD.app via subprocess (the default backend). It needs only
a working VMD binary — which you already have. If `vmd` isn't on PATH, point at
it explicitly:

    export VMD_AI_VMD_BIN=/Applications/VMD.app/Contents/vmd/vmd_MACOSXARM64
    python smoke_bridge.py

It verifies the three things that decide whether the adapter can work at all:
  1. the VMD subprocess starts and survives the sentinel handshake.
  2. multi-line Tcl + atom selections behave.
  3. `puts` output is captured back into result["output"]  <-- the #1 risk
     (the agent reads molinfo / $sel num via puts; broken capture => blank
      tool results => every task silently fails).
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from subprocess_vmd_bridge import SubprocessVmdBridge

# A minimal but valid PDB: 5 protein atoms (ALA) + 1 water. Enough to exercise
# loading, selections, and puts-capture without any external file.
MINI_PDB = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.988  -0.773   1.199  1.00  0.00           C
HETATM    6  O   HOH W   1       5.000   5.000   0.000  1.00  0.00           O
END
"""


def _check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -> {detail}" if detail else ""))
    return ok


def main() -> int:
    pdb = Path(tempfile.gettempdir()) / "vmdai_smoke.pdb"
    pdb.write_text(MINI_PDB)

    bridge = SubprocessVmdBridge()
    all_ok = True
    try:
        # 1) start + reset (this also proves the binary resolves)
        try:
            bridge.reset()
            all_ok &= _check("VMD subprocess starts + resets", True)
        except Exception as exc:
            _check("VMD subprocess starts + resets", False, repr(exc))
            print("\nSet VMD_AI_VMD_BIN to your vmd_MACOSXARM64 and retry.")
            return 1

        # 2) load + multi-line Tcl + selection + puts-capture
        cmd = (
            f'mol new "{pdb}" waitfor all\n'
            'set p [atomselect top "protein"]\n'
            'set w [atomselect top "water"]\n'
            'puts "PROT=[$p num] WAT=[$w num]"\n'
            '$p delete; $w delete'
        )
        r = bridge.execute_tool(tool_name="run_vmd_command", tool_input={"command": cmd})
        print("  run_vmd_command result.output:\n    " + r.get("output", "").replace("\n", "\n    "))
        all_ok &= _check("run_vmd_command ok", r.get("ok") is True, r.get("error", ""))
        all_ok &= _check("puts output captured (THE key risk)", "PROT=" in r.get("output", ""),
                         "expected 'PROT=5 WAT=1' in output")
        all_ok &= _check("selection counts correct", "PROT=5 WAT=1" in r.get("output", ""))

        # 3) a deliberately bad command should come back ok=False with a message
        bad = bridge.execute_tool(tool_name="run_vmd_command",
                                  tool_input={"command": "atomselect top notarealkeyword and"})
        all_ok &= _check("Tcl errors are reported (not silently ok)", bad.get("ok") is False,
                         (bad.get("error", "") or "")[:60])

        # 4) snapshot — real TachyonInternal render with native VMD
        s = bridge.execute_tool(tool_name="capture_vmd_snapshot", tool_input={"purpose": "smoke"})
        all_ok &= _check("capture_vmd_snapshot returns ok", s.get("ok") is True,
                         "real image" if s.get("image_b64") else "no image (check render)")
    finally:
        bridge.close()

    print("\n" + ("ALL GOOD — the bridge works against your VMD; proceed up the ladder."
                  if all_ok else "SOMETHING FAILED — see INTEGRATION.md §'verify' notes."))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
