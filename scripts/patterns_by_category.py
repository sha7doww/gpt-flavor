#!/usr/bin/env python3
"""
Gap 2 — 5 key patterns × model × seed category.

Answers: is the 4.6→4.7 drift global across all 13 seed categories, or
concentrated in some (emotional? analysis? refusal?)?

Key patterns (hand-picked from analysis results):
  - 加粗           (markdown bold — dropped sharply in 4.7)
  - 不是_是        (signature reversal)
  - 接住           (viral "catch you" umbrella)
  - 给你X          (offer-style, most significant drift +2.1pp)
  - 如果你愿意     (gpt-5.4 top log-odds招牌)

Output:
  analysis/patterns_by_category.csv
  analysis/figures/drift_by_category.png  (heatmap of 4.7−4.6 deltas)
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

from _common import load_records, load_patterns, ANALYSIS_DIR, MODELS_MAIN

FIG = ANALYSIS_DIR / "figures"

KEY_PATTERNS = ["加粗", "不是_是", "接住", "给你X", "如果你愿意"]


def configure_font():
    for p in [
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf",
    ]:
        if Path(p).exists():
            fm.fontManager.addfont(p)
            name = fm.FontProperties(fname=p).get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            return


def main():
    records = load_records()
    patterns = load_patterns()  # all yamls
    patterns = {k: v for k, v in patterns.items() if k in KEY_PATTERNS}
    if len(patterns) != len(KEY_PATTERNS):
        missing = set(KEY_PATTERNS) - patterns.keys()
        print(f"WARN missing patterns: {missing}")

    categories = sorted({r["category"] for r in records})
    print(f"{len(records)} records, {len(patterns)} patterns, "
          f"{len(categories)} categories, {len(MODELS_MAIN)} models")

    # counts[(pattern, model, category)] = [hits, total]
    counts = defaultdict(lambda: [0, 0])
    for r in records:
        m, cat = r["model"], r["category"]
        reply = r["reply"]
        for name, (pat, _) in patterns.items():
            counts[(name, m, cat)][1] += 1
            if pat.search(reply):
                counts[(name, m, cat)][0] += 1

    # write CSV: pattern, model, category, n_hit, n_total, rate
    out = ANALYSIS_DIR / "patterns_by_category.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pattern", "model", "category", "n_hit", "n_total", "rate"])
        for name in KEY_PATTERNS:
            for m in MODELS_MAIN:
                for cat in categories:
                    hit, total = counts[(name, m, cat)]
                    rate = hit / total if total else 0
                    w.writerow([name, m, cat, hit, total, f"{rate:.4f}"])
    print(f"→ {out}")

    # 4.7 − 4.6 delta heatmap: patterns × categories
    configure_font()
    FIG.mkdir(parents=True, exist_ok=True)
    delta_mat = np.zeros((len(KEY_PATTERNS), len(categories)))
    for i, name in enumerate(KEY_PATTERNS):
        if name not in patterns:
            continue
        for j, cat in enumerate(categories):
            h46, t46 = counts[(name, "claude-opus-4-6", cat)]
            h47, t47 = counts[(name, "claude-opus-4-7", cat)]
            r46 = h46 / t46 if t46 else 0
            r47 = h47 / t47 if t47 else 0
            delta_mat[i, j] = r47 - r46

    # annotate with pattern desc (English) for readability
    pattern_labels = [f"{k} — {patterns[k][1]['desc']}" if k in patterns else k
                      for k in KEY_PATTERNS]

    fig, ax = plt.subplots(figsize=(max(10, len(categories) * 0.8 + 3),
                                    max(4, len(KEY_PATTERNS) * 0.6 + 2)))
    vmax = max(0.1, np.max(np.abs(delta_mat)))
    im = ax.imshow(delta_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(categories)))
    ax.set_yticks(range(len(KEY_PATTERNS)))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(pattern_labels, fontsize=9)
    for i in range(len(KEY_PATTERNS)):
        for j in range(len(categories)):
            v = delta_mat[i, j]
            color = "black" if abs(v) < vmax * 0.6 else "white"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=7, color=color)
    ax.set_title("4.7 minus 4.6 pattern-rate delta, by seed category\n"
                 "(red = 4.7 uses more; blue = 4.7 uses less)",
                 fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig_path = FIG / "drift_by_category.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"→ {fig_path}")


if __name__ == "__main__":
    main()
