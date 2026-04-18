#!/usr/bin/env python3
"""
All patterns × model × condition.

Answers: which patterns' model-level differences are prompt-induced (only show up
under specific system prompts) vs model-inherent (show up in A_empty too)?

Covers all 92 patterns (flat list, no source filter).

Output:
  analysis/patterns_by_condition.csv
    (pattern, model, condition, n_hit, n_total, rate)
"""

import csv
from collections import defaultdict

from _common import load_records, load_patterns, ANALYSIS_DIR, MODELS_MAIN, CONDITIONS


def main():
    records = load_records()
    patterns = load_patterns()
    print(f"Loaded {len(records)} records, {len(patterns)} patterns")

    # counts[(pattern, model, condition)] = [hits, total]
    counts = defaultdict(lambda: [0, 0])
    for r in records:
        m, c = r["model"], r["condition"]
        reply = r["reply"]
        for name, (pat, _) in patterns.items():
            counts[(name, m, c)][1] += 1
            if pat.search(reply):
                counts[(name, m, c)][0] += 1

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out = ANALYSIS_DIR / "patterns_by_condition.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pattern", "model", "condition", "n_hit", "n_total", "rate"])
        for name in patterns:
            for m in MODELS_MAIN:
                for c in CONDITIONS:
                    hit, total = counts[(name, m, c)]
                    rate = hit / total if total else 0
                    w.writerow([name, m, c, hit, total, f"{rate:.4f}"])
    print(f"→ {out}")

    # Highlight: 4.6 vs 4.7 deltas where the gap is > 0.01 per condition
    print("\n=== 4.7 vs 4.6 deltas per condition (|Δ| ≥ 0.010) ===")
    print(f"{'pattern':<32} {'cond':<12} {'4.6':>6} {'4.7':>6} {'Δ':>7} {'5.4':>6}")
    for name in patterns:
        for c in CONDITIONS:
            h46, t46 = counts[(name, "claude-opus-4-6", c)]
            h47, t47 = counts[(name, "claude-opus-4-7", c)]
            h54, t54 = counts[(name, "gpt-5.4", c)]
            r46 = h46 / t46 if t46 else 0
            r47 = h47 / t47 if t47 else 0
            r54 = h54 / t54 if t54 else 0
            delta = r47 - r46
            if abs(delta) >= 0.010:
                arrow = " ↑" if delta > 0 else " ↓"
                print(f"{name:<32} {c:<12} {r46:>6.3f} {r47:>6.3f} {delta:>+7.3f}{arrow} {r54:>6.3f}")


if __name__ == "__main__":
    main()
