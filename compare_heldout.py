#!/usr/bin/env python3
"""compare_heldout.py — per-metric pass-rate across held-out arms (none / inject / improved).

Isolates where the gains come from: none=raw, inject=cheatsheet only (no semantic tools),
improved=inject + semantic tools + bigger timeouts. Comparing inject vs improved on tool-
UNCOVERED metrics shows whether the semantic tools help, do nothing, or hurt off their coverage.

  python compare_heldout.py --tags heldout_none,heldout_inject,heldout_improved
"""
import argparse, collections, json, os
from pathlib import Path

METRICS = ["hbonds", "angle123", "phi5", "com_x", "max_extent", "n_gly"]


def perm(tag, root):
    f = os.path.join(root, tag, "summary.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    agg = collections.defaultdict(lambda: [0, 0])      # metric -> [pass, total]
    for k, v in d.items():
        metric = k.split("__")[1]
        oks = v if isinstance(v, list) else [v.get("ok")]   # legacy list or {ok,...}
        for x in oks:
            if x is None:
                continue
            agg[metric][1] += 1
            agg[metric][0] += int(bool(x))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--results-root", default=str(Path(__file__).resolve().parents[2] / "test_results" / "heldout"))
    a = ap.parse_args()
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    aggs = {t: perm(t, a.results_root) for t in tags}

    short = [t.replace("heldout_", "") for t in tags]
    print(f"{'metric':12}" + "".join(f"{s:>14}" for s in short))
    print("-" * (12 + 14 * len(tags)))
    tot = {t: [0, 0] for t in tags}
    for m in METRICS:
        row = f"{m:12}"
        for t in tags:
            ag = aggs[t]
            if ag is None:
                row += f"{'(no run)':>14}"
                continue
            p, n = ag.get(m, [0, 0])
            tot[t][0] += p
            tot[t][1] += n
            row += f"{f'{p}/{n}':>14}"
        print(row)
    print("-" * (12 + 14 * len(tags)))
    print(f"{'TOTAL':12}" + "".join(f"{f'{tot[t][0]}/{tot[t][1]}':>14}" for t in tags))
    print("\nnone=raw · inject=cheatsheet only · improved=inject+semantic tools+timeouts. "
          "inject vs improved isolates the tools' effect on uncovered ops.")


if __name__ == "__main__":
    main()
