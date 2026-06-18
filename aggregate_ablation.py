#!/usr/bin/env python3
"""aggregate_ablation.py — per-arm mean +/- std and per-case pass-rate across seeds.

Reads result modes named  <model_tag>_<arm>_s<seed>  (produced by run_ablation.sh with
SEEDS>1), groups by arm, and reports:
  * reliability (completed/10)        mean +/- std across seeds
  * correctness (numeric checks)       mean +/- std across seeds
  * per numeric metric: pass-rate across seeds  (e.g. case9:dist 2/3 -> exposes flaky cases)

The pass-rate row is the point: a flaky case (case_9's brace typo) shows up as e.g. 2/3,
so it can't masquerade as an arm effect the way it does at a single seed.

  python aggregate_ablation.py --tasks ~/.../molecular_vis \
      --model-tag Qwen-Qwen2.5-72B-Instruct-AWQ --arms none,rag,wiki,both --seeds 3
"""
import argparse, os, statistics as st
import score_correctness as S   # reuse gold/oracle + extraction


def score_mode(tasks, mode, results, gold):
    status = S.latest_status(results, mode)
    completed = sum(1 for c in range(1, 11) if status.get(c, ("",))[0] == "completed")
    present = bool(status)
    npass = ntot = 0
    per = {}
    for c in sorted(S.CASE_METRICS):
        floats, _ = S.answer_floats(tasks, c, mode)
        for key, tol, _u in S.CASE_METRICS[c]:
            g = gold.get(key)
            if g is None or tol is None:   # skip report-only / no-gold metrics
                continue
            ntot += 1
            ok = bool(floats) and abs(min(floats, key=lambda x: abs(x - g)) - g) <= tol
            per[f"case{c}:{key}"] = ok
            npass += int(ok)
    return present, completed, npass, ntot, per


def ms(xs):
    if not xs:
        return "-"
    m = sum(xs) / len(xs)
    s = st.pstdev(xs) if len(xs) > 1 else 0.0
    return f"{m:.1f}±{s:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--model-tag", default="Qwen-Qwen2.5-72B-Instruct-AWQ")
    ap.add_argument("--arms", default="none,rag,wiki,both")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--run-tag", default="", help="sweep namespace from run_ablation.sh (timestamp)")
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
        print(f"[gold] {gerr}")

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    tag = f"{a.run_tag}_" if a.run_tag else ""   # sweep namespace prefix, if any
    agg, metrics, ntot_seen = {}, [], 0
    for arm in arms:
        rel, cor, per_acc, n = [], [], {}, 0
        for s in range(1, a.seeds + 1):
            # accept seeded (_s1); fall back to a bare arm mode only when untagged + seeds==1
            for mode in (f"vmd_ai_{a.model_tag}_{tag}{arm}_s{s}",
                         f"vmd_ai_{a.model_tag}_{arm}" if (a.seeds == 1 and not tag) else None):
                if not mode:
                    continue
                present, comp, npass, ntot, per = score_mode(tasks, mode, results, gold)
                if not present:
                    continue
                n += 1
                rel.append(comp)
                cor.append(npass)
                ntot_seen = max(ntot_seen, ntot)
                for k, v in per.items():
                    per_acc[k] = per_acc.get(k, 0) + int(v)
                    if k not in metrics:
                        metrics.append(k)
                break
        agg[arm] = {"n": n, "rel": rel, "cor": cor, "per": per_acc}

    cw = 14
    print(f"\ngold: rgyr={gold.get('rgyr')}  (seeds requested: {a.seeds})\n")
    head = f"{'metric':<22}" + "".join(f"{arm:>{cw}}" for arm in arms)
    print(head)
    print("-" * len(head))
    print(f"{'seeds found':<22}" + "".join(f"{agg[arm]['n']:>{cw}}" for arm in arms))
    print(f"{'reliability /10':<22}" + "".join(f"{ms(agg[arm]['rel']):>{cw}}" for arm in arms))
    print(f"{f'correctness /{ntot_seen}':<22}" + "".join(f"{ms(agg[arm]['cor']):>{cw}}" for arm in arms))
    print("\nper-metric pass-rate across seeds (flaky cases show < n/n):")
    for mname in sorted(metrics):
        row = f"{mname:<22}"
        for arm in arms:
            n = agg[arm]["n"]
            p = agg[arm]["per"].get(mname, 0)
            row += f"{(f'{p}/{n}' if n else '-'):>{cw}}"
        print(row)


if __name__ == "__main__":
    main()
