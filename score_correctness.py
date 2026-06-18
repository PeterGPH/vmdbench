#!/usr/bin/env python3
"""score_correctness.py — correctness-vs-gold scorer for SciVisAgentBench molecular_vis runs.

The official text rubric checks *existence* ("did the agent write 'yes'?"). This scorer
checks *correctness*: it reads the actual numbers the agent wrote (Rg, RMSD, CA-CA
distance, phi/psi, contacts) and compares them to VMD-computed gold. A wrong-but-confident
answer (e.g. Rg = 329 Å when the truth is 9.67) passes the rubric but fails here.

Two independent axes are reported:
  * RELIABILITY  — did the agent finish the case? (from the run JSONs)
  * CORRECTNESS  — are the written numbers right? (numeric cases only; 1-6 are visual)

Usage (run on the Mac, where VMD and the answer files live):
  python score_correctness.py \
    --tasks ~/SciVisAgentBench/SciVisAgentBench-tasks/molecular_vis \
    --mode  vmd_ai_Qwen-Qwen2.5-72B-Instruct-AWQ_exp1
"""
import argparse, glob, json, os, re, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.join(HERE, "gold_oracle.tcl")
DEFAULT_VMD = "/Applications/VMD.app/Contents/vmd/vmd_MACOSXARM64"
DEFAULT_RESULTS = os.path.join(HERE, "..", "..", "test_results", "molecular_vis")

# Independently confirmed for 1CRN; used only if the VMD oracle can't run.
KNOWN_GOLD = {"rgyr": 9.666, "rmsd_self": 0.0}

# case -> [(gold_key, tolerance, unit)]. tolerance None => report only (definition-sensitive).
CASE_METRICS = {
    7:  [("rmsd_self", 0.05, "A")],
    8:  [("rgyr", 0.15, "A")],
    9:  [("dist_ca1_ca10", 0.10, "A"), ("phi_resid5", 2.0, "deg"), ("psi_resid5", 2.0, "deg")],
    10: [("contacts_8", None, "pairs")],
}
VISUAL_CASES = (1, 2, 3, 4, 5, 6)
FLOAT = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def compute_gold(vmd, structure):
    if not (vmd and os.path.exists(vmd) and os.path.exists(ORACLE)
            and structure and os.path.exists(structure)):
        return {}, "VMD / oracle / structure not found"
    env = dict(os.environ, GOLD_STRUCT=structure)
    try:
        out = subprocess.run([vmd, "-dispdev", "text", "-e", ORACLE],
                             env=env, capture_output=True, text=True, timeout=180).stdout
    except Exception as e:
        return {}, f"VMD run failed: {e}"
    gold = {}
    for line in out.splitlines():
        m = re.match(r"\s*GOLD\s+(\S+)\s+(\S+)", line)
        if m:
            try: gold[m.group(1)] = float(m.group(2))
            except ValueError: pass
    return gold, (None if gold else "oracle produced no GOLD lines")


def latest_status(results_dir, mode):
    status, by_case = {}, {}
    d = os.path.join(results_dir, mode)
    if not os.path.isdir(d):
        return status
    for f in glob.glob(os.path.join(d, "case_*_result_*.json")):
        m = re.search(r"case_(\d+)_result_(\d+)\.json", os.path.basename(f))
        if m:
            by_case.setdefault(int(m.group(1)), []).append((int(m.group(2)), f))
    for c, lst in by_case.items():
        f = sorted(lst)[-1][1]
        try: j = json.load(open(f))
        except Exception: continue
        status[c] = (j.get("status", "?"),
                     j.get("duration_seconds") or (j.get("metadata") or {}).get("duration"))
    return status


def answer_floats(tasks, case, mode):
    d = os.path.join(tasks, f"case_{case}", "results", mode)
    texts = []
    for f in sorted(glob.glob(os.path.join(d, "*.txt"))):
        try: texts.append(open(f, errors="replace").read())
        except Exception: pass
    raw = "\n".join(texts).strip()
    return [float(x) for x in FLOAT.findall(raw)], raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="path to .../SciVisAgentBench-tasks/molecular_vis")
    ap.add_argument("--mode", required=True, help="agent_mode dir name")
    ap.add_argument("--results", default=DEFAULT_RESULTS, help="test_results/molecular_vis dir")
    ap.add_argument("--vmd", default=DEFAULT_VMD)
    ap.add_argument("--structure", default="", help="defaults to <tasks>/data/1CRN.cif")
    args = ap.parse_args()

    tasks = os.path.expanduser(args.tasks)
    results = os.path.expanduser(args.results)
    structure = os.path.expanduser(args.structure) or os.path.join(tasks, "data", "1CRN.cif")

    gold, gerr = compute_gold(os.path.expanduser(args.vmd), structure)
    for k, v in KNOWN_GOLD.items():
        gold.setdefault(k, v)
    print(f"mode: {args.mode}")
    if gerr:
        print(f"[gold] {gerr} -> using fallback gold where available")
    print(f"[gold] " + ", ".join(f"{k}={v}" for k, v in sorted(gold.items())) + "\n")

    status = latest_status(results, args.mode)

    # ---------------- reliability ----------------
    done = sum(1 for c in range(1, 11) if status.get(c, ("",))[0] == "completed")
    print("=== RELIABILITY (did the agent finish?) ===")
    print(f"  {'case':<5}{'status':<12}{'dur(s)':<7}")
    for c in range(1, 11):
        st, du = status.get(c, ("(no result)", None))
        print(f"  {c:<5}{st:<12}{(round(du) if du else '-'):<7}")
    print(f"  => {done}/10 completed\n")

    # ---------------- correctness ----------------
    print("=== CORRECTNESS vs gold (numeric cases) ===")
    n_scored = n_pass = 0
    for c in sorted(CASE_METRICS):
        floats, _ = answer_floats(tasks, c, args.mode)
        st = status.get(c, ("(no result)",))[0]
        print(f"  case_{c} [{st}]  answer numbers: {floats if floats else '(none)'}")
        for key, tol, unit in CASE_METRICS[c]:
            g = gold.get(key)
            if g is None:
                print(f"      {key:<16} gold=?           -> SKIP (no gold)")
                continue
            if not floats:
                print(f"      {key:<16} gold={g:<9} agent=(none)   -> FAIL (no answer)")
                if tol is not None: n_scored += 1
                continue
            best = min(floats, key=lambda x: abs(x - g))
            if tol is None:
                print(f"      {key:<16} gold={g:<9} agent~{best:<9} d={best-g:+.3f} {unit}  [report only]")
                continue
            n_scored += 1
            ok = abs(best - g) <= tol
            n_pass += int(ok)
            print(f"      {key:<16} gold={g:<9} agent={best:<9} d={best-g:+.3f} {unit}  tol=+/-{tol}  -> {'PASS' if ok else 'FAIL'}")
        print()
    print(f"  => correctness: {n_pass}/{n_scored} numeric checks within tolerance\n")

    # ---------------- visual cases ----------------
    print("=== VISUAL CASES 1-6 (correctness needs the image; agent's written claim) ===")
    for c in VISUAL_CASES:
        _, raw = answer_floats(tasks, c, args.mode)
        st = status.get(c, ("(no result)",))[0]
        claim = (raw.replace("\n", " ")[:50] or "(no answer file)") if raw is not None else "(no answer file)"
        print(f"  case_{c} [{st}]: {claim}")


if __name__ == "__main__":
    main()
