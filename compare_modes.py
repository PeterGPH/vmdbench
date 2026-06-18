#!/usr/bin/env python3
"""compare_modes.py — side-by-side reliability + correctness across agent_modes (A/B arms).

Computes oracle gold once, then for each mode reports completion count and, per numeric
case, PASS/FAIL vs gold. Columns are the arm suffixes (none/rag/wiki/both), so the effect
of RAG and Wiki on *correctness* (not idioms) reads straight down the table.

  python compare_modes.py \
    --tasks ~/SciVisAgentBench/SciVisAgentBench-tasks/molecular_vis \
    --modes vmd_ai_Qwen-Qwen2.5-72B-Instruct-AWQ_none,...,_rag,_wiki,_both
"""
import argparse, os
import score_correctness as S   # reuse gold/oracle + extraction logic


def summarize(tasks, mode, results, gold):
    status = S.latest_status(results, mode)
    completed = sum(1 for c in range(1, 11) if status.get(c, ("",))[0] == "completed")
    rows, npass, nscored = {}, 0, 0
    for c in sorted(S.CASE_METRICS):
        floats, _ = S.answer_floats(tasks, c, mode)
        for key, tol, _unit in S.CASE_METRICS[c]:
            g = gold.get(key)
            label = f"case{c}:{key}"
            if g is None:
                rows[label] = "?"
                continue
            if tol is None:  # report-only metric
                best = min(floats, key=lambda x: abs(x - g)) if floats else None
                rows[label] = (f"{best:g}" if best is not None else "-") + " (rep)"
                continue
            nscored += 1
            if not floats:
                rows[label] = "FAIL(no ans)"
                continue
            best = min(floats, key=lambda x: abs(x - g))
            ok = abs(best - g) <= tol
            npass += int(ok)
            rows[label] = f"{'PASS' if ok else 'FAIL'}({best:g})"
    return {"completed": completed, "npass": npass, "nscored": nscored, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--modes", required=True, help="comma-separated agent_mode names")
    ap.add_argument("--results", default=S.DEFAULT_RESULTS)
    ap.add_argument("--vmd", default=S.DEFAULT_VMD)
    ap.add_argument("--structure", default="")
    a = ap.parse_args()

    tasks = os.path.expanduser(a.tasks)
    results = os.path.expanduser(a.results)
    structure = os.path.expanduser(a.structure) or os.path.join(tasks, "data", "1CRN.cif")

    gold, gerr = S.compute_gold(os.path.expanduser(a.vmd), structure)
    for k, v in S.KNOWN_GOLD.items():
        gold.setdefault(k, v)
    if gerr:
        print(f"[gold] {gerr} -> fallback gold where available")
    print("gold: " + ", ".join(f"{k}={v}" for k, v in sorted(gold.items())) + "\n")

    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    data = {m: summarize(tasks, m, results, gold) for m in modes}
    arms = [m.split("_")[-1] for m in modes]   # none/rag/wiki/both

    cw = 15
    head = f"{'metric':<22}" + "".join(f"{a_:>{cw}}" for a_ in arms)
    print(head)
    print("-" * len(head))
    print(f"{'reliability /10':<22}" + "".join(f"{data[m]['completed']:>{cw}}" for m in modes))
    print(f"{'correctness':<22}" + "".join(f"{str(data[m]['npass']) + '/' + str(data[m]['nscored']):>{cw}}" for m in modes))
    rows = sorted({k for m in modes for k in data[m]["rows"]})
    for r in rows:
        print(f"{r:<22}" + "".join(f"{data[m]['rows'].get(r, '-'):>{cw}}" for m in modes))


if __name__ == "__main__":
    main()
