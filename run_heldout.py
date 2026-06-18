#!/usr/bin/env python3
"""run_heldout.py — correctness on tasks OUTSIDE the semantic-tool coverage.

vmd_measure covers rgyr/sasa/natoms/nresidues/ca_dist/rmsd_self. These six metrics are
deliberately NOT covered (hbonds, a 3-atom angle, a dihedral, center-of-mass, bounding-box
extent, a filtered GLY count), so the agent must write raw Tcl. Running none vs the improved
config here answers the key rigor question: do #1-#3 generalize, or did the semantic tools
just solve the exact benchmark metrics?

  python run_heldout.py --bench ~/SciVisAgentBench --config config_arm_none.json --tag heldout_none --seeds 3
  python run_heldout.py --bench ~/SciVisAgentBench --config config_local.json    --tag heldout_improved --seeds 3
"""
import argparse, asyncio, glob, json, os, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORACLE = HERE / "gold_oracle_heldout.tcl"
DEFAULT_VMD = "/Applications/VMD.app/Contents/vmd/vmd_MACOSXARM64"
FLOAT = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

# metric -> (prompt phrase, abs tolerance, scored). None of these maps to a vmd_measure metric.
METRICS = {
    "hbonds":     ("the number of hydrogen bonds within the protein, computed with Tcl "
                   "`measure hbonds 3.0 20 [atomselect top protein]` — the count is the length of the FIRST returned list", 2.0, True),
    "angle123":   ("the angle in degrees formed by the alpha-carbon (CA) atoms of residues 1, 2 and 3 "
                   "(measure angle on the three CA atom indices)", 1.0, True),
    "phi5":       ("the phi backbone dihedral of residue 5 in degrees (atoms: C of residue 4, then N, CA, C of residue 5)", 2.0, True),
    "com_x":      ("the x-coordinate of the protein's center of mass in Angstroms (measure center ... weight mass)", 0.1, True),
    "max_extent": ("the largest dimension (max of the x/y/z spans) of the protein's bounding box in Angstroms, from `measure minmax`", 0.2, True),
    "n_gly":      ("the number of glycine (GLY) residues in the protein", 0.5, True),
}
STRUCTURES = ["1CRN.cif", "1UBQ.pdb"]   # small + held-out; both have GLY


def compute_gold(vmd, structure):
    env = dict(os.environ, GOLD_STRUCT=structure)
    try:
        out = subprocess.run([vmd, "-dispdev", "text", "-e", str(ORACLE)],
                             env=env, capture_output=True, text=True, timeout=300).stdout
    except Exception as e:
        return {}, f"VMD failed: {e}"
    g = {}
    for line in out.splitlines():
        m = re.match(r"\s*GOLD\s+(\S+)\s+(\S+)", line)
        if m:
            try: g[m.group(1)] = float(m.group(2))
            except ValueError: pass
    return g, (None if g else "no GOLD lines")


def build_prompt(struct_path, phrase, answer_path):
    return ("You are controlling VMD headlessly through the run_vmd_command tool (Tcl).\n"
            f'1. Load the structure: mol new "{struct_path}"\n'
            f"2. Compute {phrase}.\n"
            "3. Write ONLY that single numeric value (digits only) to this exact file:\n"
            f'   set f [open "{answer_path}" w]; puts $f $value; close $f\n'
            "Create the directory with 'file mkdir' if needed. Finish in as few commands as possible.")


async def run_arm(args):
    bench = Path(os.path.expanduser(args.bench)).resolve()
    sys.path.insert(0, str(bench / "benchmark"))
    sys.path.insert(0, str(HERE))
    import vmd_ai_agent  # noqa: F401
    from evaluation_framework import get_agent
    config = json.load(open(args.config))

    sdir = os.path.expanduser(args.structures_dir)
    structures = [os.path.join(sdir, s) for s in STRUCTURES if os.path.exists(os.path.join(sdir, s))]
    outdir = Path(os.path.expanduser(args.out_dir)) / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    gold = {}
    for s in structures:
        name = Path(s).stem.upper()
        g, err = compute_gold(os.path.expanduser(args.vmd), s)
        gold[name] = g
        print(f"[gold] {name}: " + (", ".join(f"{k}={v}" for k, v in sorted(g.items())) if g else f"(FAILED: {err})"))

    agent = get_agent("vmd_ai")(config)
    await agent.setup()
    results, detail = {}, {}
    try:
        for seed in range(1, max(1, args.seeds) + 1):
            for s in structures:
                name = Path(s).stem.upper()
                for mkey, (phrase, tol, scored) in METRICS.items():
                    ans = str(outdir / f"{name}__{mkey}__s{seed}.txt")
                    if os.path.exists(ans):
                        os.remove(ans)
                    tcfg = {"working_dir": str(outdir), "case_dir": str(outdir),
                            "case_name": f"{name}_{mkey}_s{seed}", "timeout": args.timeout}
                    try:
                        await agent.run_task(build_prompt(os.path.abspath(s), phrase, ans), tcfg)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  [s{seed}] {name}/{mkey}: {exc}")
                    g = gold.get(name, {}).get(mkey)
                    val = None
                    if os.path.exists(ans):
                        fs = [float(x) for x in FLOAT.findall(open(ans, errors="replace").read())]
                        if fs and g is not None:
                            val = min(fs, key=lambda x: abs(x - g))
                    ok = (val is not None and g is not None and abs(val - g) <= tol) if scored else None
                    results.setdefault((name, mkey), []).append(ok)
                    detail[f"{name}__{mkey}__s{seed}"] = {"gold": g, "agent": val, "ok": ok}
                    print(f"  [s{seed}] {name:6} {mkey:11} gold={g} agent={val} -> "
                          + ("PASS" if ok else ("FAIL" if ok is False else "-")))
    finally:
        await agent.teardown()

    names = [Path(s).stem.upper() for s in structures]
    print(f"\n=== HELD-OUT CORRECTNESS (pass-rate over {args.seeds} seeds) — arm: {args.tag} ===")
    print(f"  {'metric':12} " + " ".join(f"{n:>8}" for n in names) + "    total")
    gp = gt = 0
    for m in METRICS:
        cells, mp, mt = [], 0, 0
        for n in names:
            oks = [x for x in results.get((n, m), []) if x is not None]
            p, t = sum(1 for x in oks if x), len(oks)
            cells.append(f"{p}/{t}" if t else "-"); mp += p; mt += t
        gp += mp; gt += mt
        print(f"  {m:12} " + " ".join(f"{c:>8}" for c in cells) + f"    {mp}/{mt}")
    print(f"  => overall: {gp}/{gt}")
    json.dump(detail, open(outdir / "summary.json", "w"), default=str, indent=2)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=os.environ.get("BENCH_DIR", str(Path.home() / "SciVisAgentBench")))
    ap.add_argument("--config", required=True)
    ap.add_argument("--structures-dir", default=str(HERE / "fixtures_multi"))
    ap.add_argument("--vmd", default=DEFAULT_VMD)
    ap.add_argument("--out-dir", default=str(HERE.parent.parent / "test_results" / "heldout"))
    ap.add_argument("--tag", default="heldout")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=1)
    raise SystemExit(asyncio.run(run_arm(ap.parse_args())))


if __name__ == "__main__":
    main()
