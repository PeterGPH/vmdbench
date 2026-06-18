#!/usr/bin/env python3
"""run_multistructure.py — generalize the correctness test across many structures.

Drives the VmdAiAgent (the SAME arms as the SciVisAgentBench ablation — none / inject /
autorag / …) over a structure × portable-metric grid, captures each numeric answer, and
scores it against gold computed by gold_oracle_multi.tcl. The metrics (Rg, atom / residue
counts, CA–CA first/last distance, SASA) compute on ANY protein, so adding a structure is
just dropping a .pdb/.cif into the fixtures dir.

Run on the Mac (vLLM server + tunnel up), once per arm, then compare summaries:
  python run_multistructure.py --bench ~/SciVisAgentBench --config config_arm_none.json   --tag none
  python run_multistructure.py --bench ~/SciVisAgentBench --config config_arm_inject.json --tag inject
"""
import argparse, asyncio, glob, json, os, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORACLE = HERE / "gold_oracle_multi.tcl"
DEFAULT_VMD = "/Applications/VMD.app/Contents/vmd/vmd_MACOSXARM64"
FLOAT = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

# metric key -> (prompt phrase, absolute tolerance, scored?)
METRICS = {
    "rgyr":      ("the radius of gyration of the protein, in Angstroms", 0.2, True),
    "natoms":    ("the total number of atoms loaded from the structure", 0.5, True),
    "nresidues": ("the total number of amino-acid residues in the protein across ALL chains "
                  "(residue numbers repeat between chains, so count VMD's unique 'residue' attribute, "
                  "i.e. llength [lsort -unique [$sel get residue]]), as one integer", 0.5, True),
    "ca_dist":   ("the distance in Angstroms between the FIRST and the LAST alpha-carbon "
                  "(CA) atom of the protein, in atom order", 0.2, True),
    "sasa":      ("the solvent-accessible surface area of the protein in square Angstroms, "
                  "using Tcl 'measure sasa 1.4'", 2.0, True),
}


def compute_gold(vmd, structure):
    env = dict(os.environ, GOLD_STRUCT=structure)
    try:
        out = subprocess.run([vmd, "-dispdev", "text", "-e", str(ORACLE)],
                             env=env, capture_output=True, text=True, timeout=300).stdout
    except Exception as e:
        return {}, f"VMD failed: {e}"
    gold = {}
    for line in out.splitlines():
        m = re.match(r"\s*GOLD\s+(\S+)\s+(\S+)", line)
        if m:
            try: gold[m.group(1)] = float(m.group(2))
            except ValueError: pass
    return gold, (None if gold else "no GOLD lines (check VMD path / structure)")


def build_prompt(struct_path, phrase, answer_path):
    return (
        "You are controlling VMD headlessly through the run_vmd_command tool (Tcl).\n"
        f'1. Load the structure: mol new "{struct_path}"\n'
        f"2. Compute {phrase}.\n"
        "3. Write ONLY that single numeric value (digits only, no extra words) to this exact file:\n"
        f'   set f [open "{answer_path}" w]; puts $f $value; close $f\n'
        "Create the directory first with 'file mkdir' if needed. Finish in as few commands as possible."
    )


async def run_arm(args):
    bench = Path(os.path.expanduser(args.bench)).resolve()
    sys.path.insert(0, str(bench / "benchmark"))   # evaluation_framework
    sys.path.insert(0, str(HERE))                  # vmd_ai_agent + bridges
    import vmd_ai_agent  # noqa: F401  (registers @register_agent("vmd_ai"))
    from evaluation_framework import get_agent

    with open(args.config) as fh:
        config = json.load(fh)

    structures = sorted(glob.glob(os.path.join(args.structures_dir, "*.pdb"))
                        + glob.glob(os.path.join(args.structures_dir, "*.cif")))
    if not structures:
        print(f"No structures in {args.structures_dir}")
        return 1

    outdir = Path(os.path.expanduser(args.out_dir)) / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- gold per structure (VMD oracle) ----
    gold = {}
    for s in structures:
        name = Path(s).stem.upper()
        g, err = compute_gold(os.path.expanduser(args.vmd), s)
        gold[name] = g
        print(f"[gold] {name}: " + (", ".join(f"{k}={v}" for k, v in sorted(g.items()))
                                    if g else f"(FAILED: {err})"))

    agent = get_agent("vmd_ai")(config)
    await agent.setup()

    seeds = max(1, int(args.seeds))
    results = {}   # (name, mkey) -> list of (ok|None) across seeds
    detail = {}    # name__mkey__sN -> {gold, agent, ok}
    try:
        for seed in range(1, seeds + 1):
            for s in structures:
                name = Path(s).stem.upper()
                for mkey, (phrase, tol, scored) in METRICS.items():
                    ans_path = str(outdir / f"{name}__{mkey}__s{seed}.txt")
                    if os.path.exists(ans_path):
                        os.remove(ans_path)
                    prompt = build_prompt(os.path.abspath(s), phrase, ans_path)
                    tcfg = {"working_dir": str(outdir), "case_dir": str(outdir),
                            "case_name": f"{name}_{mkey}_s{seed}", "timeout": args.timeout}
                    try:
                        await agent.run_task(prompt, tcfg)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  [s{seed}] {name}/{mkey}: run error: {exc}")
                    g = gold.get(name, {}).get(mkey)
                    val = None
                    if os.path.exists(ans_path):
                        fs = [float(x) for x in FLOAT.findall(open(ans_path, errors="replace").read())]
                        if fs and g is not None:
                            val = min(fs, key=lambda x: abs(x - g))
                    ok = (val is not None and g is not None and abs(val - g) <= tol) if scored else None
                    results.setdefault((name, mkey), []).append(ok)
                    detail[f"{name}__{mkey}__s{seed}"] = {"gold": g, "agent": val, "ok": ok}
                    print(f"  [s{seed}] {name:6} {mkey:10} gold={g} agent={val} -> "
                          + ("PASS" if ok else ("FAIL" if ok is False else "-")))
    finally:
        await agent.teardown()

    # ---- report: pass-rate over seeds ----
    names = [Path(s).stem.upper() for s in structures]
    print(f"\n=== CORRECTNESS across structures (pass-rate over {seeds} seed(s)) — arm: {args.tag} ===")
    print(f"  {'metric':10} " + " ".join(f"{n:>8}" for n in names) + "    total")
    grand_pass = grand_tot = 0
    for m, (_, _, sc) in METRICS.items():
        if not sc:
            continue
        cells, mp, mt = [], 0, 0
        for n in names:
            oks = [x for x in results.get((n, m), []) if x is not None]
            p, t = sum(1 for x in oks if x), len(oks)
            cells.append(f"{p}/{t}" if t else "-")
            mp += p
            mt += t
        grand_pass += mp
        grand_tot += mt
        print(f"  {m:10} " + " ".join(f"{c:>8}" for c in cells) + f"    {mp}/{mt}")
    print(f"  => overall: {grand_pass}/{grand_tot} checks passed (structures x metrics x seeds)")
    json.dump(detail, open(outdir / "summary.json", "w"), indent=2, default=str)
    print(f"  saved -> {outdir / 'summary.json'}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=os.environ.get("BENCH_DIR", str(Path.home() / "SciVisAgentBench")))
    ap.add_argument("--config", required=True)
    ap.add_argument("--structures-dir", default=str(HERE / "fixtures_multi"))
    ap.add_argument("--vmd", default=DEFAULT_VMD)
    ap.add_argument("--out-dir", default=str(HERE.parent.parent / "test_results" / "multistructure"))
    ap.add_argument("--tag", default="none")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--seeds", type=int, default=1, help="repeats per task; >1 de-noises no-answers")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run_arm(args)))


if __name__ == "__main__":
    main()
