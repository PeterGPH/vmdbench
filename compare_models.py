#!/usr/bin/env python3
"""compare_models.py — completion vs correctness per model (the flagship result).

Reads each model's run_multistructure summary.json and reports, per model:
  completion%  = fraction of scored tasks the agent produced ANY answer for (a plausible value)
  correctness% = fraction within the oracle tolerance
The gap between them is the paper's headline: completion/outcome scores overstate correctness.

  python compare_models.py --tags model_qwen72b,model_claude,model_gpt
"""
import argparse, json, os
from pathlib import Path


def load(tag, root):
    f = os.path.join(root, tag, "summary.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    vals = list(d.values())
    if vals and isinstance(vals[0], dict):           # {gold, agent, ok} per seed
        scored = [v for v in vals if v.get("ok") is not None]
        answered = sum(1 for v in scored if v.get("agent") is not None)
        correct = sum(1 for v in scored if v.get("ok"))
        return {"scored": len(scored), "answered": answered, "correct": correct}
    # legacy {key: [ok,...]} — correctness recoverable, completion not
    flat = [x for lst in vals for x in (lst if isinstance(lst, list) else [lst])]
    scored = [x for x in flat if x is not None]
    return {"scored": len(scored), "answered": None, "correct": sum(1 for x in scored if x)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma-separated run tags, e.g. model_qwen72b,model_claude")
    ap.add_argument("--results-root",
                    default=str(Path(__file__).resolve().parents[2] / "test_results" / "multistructure"))
    a = ap.parse_args()
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]

    print(f"{'model':24}{'completion':>12}{'correctness':>13}{'gap':>8}")
    print("-" * 57)
    for t in tags:
        s = load(t, a.results_root)
        if not s or not s["scored"]:
            print(f"{t:24}{'(no summary)':>12}")
            continue
        corr = 100 * s["correct"] / s["scored"]
        name = t[len("model_"):] if t.startswith("model_") else t
        if s["answered"] is None:
            print(f"{name:24}{'n/a':>12}{corr:>12.0f}%{'n/a':>8}")
        else:
            comp = 100 * s["answered"] / s["scored"]
            print(f"{name:24}{comp:>11.0f}%{corr:>12.0f}%{comp - corr:>7.0f}")
    print("\ncompletion = produced an answer (what a completion/rubric benchmark rewards);")
    print("correctness = within oracle tolerance; gap = how much completion overstates correctness.")


if __name__ == "__main__":
    main()
