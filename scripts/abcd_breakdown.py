#!/usr/bin/env python3
"""
Classify all 100 patterns into A/B/C/D groups by behavior, output counts.

A = 本数据集 0 命中：所有模型 rate ≤ 1%（疑似单例引用）
B = 朝 GPT 漂移   ：opus-4-7 比 4-6 高 ≥ 1.5pp，且 gpt-5.4 ≥ 5%
C = 反向漂移      ：opus-4-7 比 4-6 低 ≥ 1.5pp
D = dilution       ：max(GPT) ≥ 2× max(Claude) 且 max(GPT) ≥ 10%
其他 = 以上都不满足

输入: analysis/stats.json （per-model pattern hit rates）
输出:
  - analysis/abcd_breakdown.json
  - stdout: markdown 片段（直接拷到 README）
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
STATS_PATH = ROOT / "analysis" / "stats.json"
OUT_PATH = ROOT / "analysis" / "abcd_breakdown.json"

GPT_MODELS = {"gpt-4o-2024-11-20", "gpt-5-chat-latest", "gpt-5.3-chat", "gpt-5.4"}
CLAUDE_MODELS = {"claude-opus-4-6", "claude-opus-4-7"}

# 阈值
A_MAX_RATE = 0.01           # 全模型 ≤ 1%
B_DELTA_PP = 0.015          # 4.7-4.6 ≥ +1.5pp
B_GPT54_MIN = 0.05          # gpt-5.4 ≥ 5%
C_DELTA_PP = -0.015         # 4.7-4.6 ≤ -1.5pp
D_RATIO = 2.0               # max(GPT) / max(Claude) ≥ 2
D_GPT_MIN = 0.10            # max(GPT) ≥ 10%


def classify(rates):
    """rates: dict[model -> hit rate]. 返回组名 A/B/C/D/其他."""
    gpt_rates = [rates.get(m, 0) for m in GPT_MODELS if m in rates]
    claude_rates = [rates.get(m, 0) for m in CLAUDE_MODELS if m in rates]
    all_rates = list(rates.values())

    max_gpt = max(gpt_rates) if gpt_rates else 0
    max_claude = max(claude_rates) if claude_rates else 0
    gpt54 = rates.get("gpt-5.4", 0)
    opus46 = rates.get("claude-opus-4-6", 0)
    opus47 = rates.get("claude-opus-4-7", 0)
    delta_47_46 = opus47 - opus46

    if all(r <= A_MAX_RATE for r in all_rates):
        return "A"
    if delta_47_46 >= B_DELTA_PP and gpt54 >= B_GPT54_MIN:
        return "B"
    if delta_47_46 <= C_DELTA_PP:
        return "C"
    if max_claude > 0 and max_gpt >= D_GPT_MIN and max_gpt / max_claude >= D_RATIO:
        return "D"
    return "其他"


def main():
    if not STATS_PATH.exists():
        sys.exit(f"missing {STATS_PATH} — run analyze.py first")
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))

    pattern_names = list(next(iter(stats.values()))["patterns"].keys())
    groups = {"A": [], "B": [], "C": [], "D": [], "其他": []}

    for pat in pattern_names:
        rates = {m: stats[m]["patterns"][pat]["rate"] for m in stats}
        groups[classify(rates)].append(pat)

    total = len(pattern_names)
    breakdown = {
        "total_patterns": total,
        "thresholds": {
            "A_max_rate": A_MAX_RATE,
            "B_delta_pp": B_DELTA_PP,
            "B_gpt54_min": B_GPT54_MIN,
            "C_delta_pp": C_DELTA_PP,
            "D_ratio": D_RATIO,
            "D_gpt_min": D_GPT_MIN,
        },
        "counts": {g: len(p) for g, p in groups.items()},
        "percentages": {g: round(len(p) / total * 100, 1) for g, p in groups.items()},
        "patterns": groups,
    }

    OUT_PATH.write_text(json.dumps(breakdown, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    # markdown 片段输出
    print(f"Total: {total} patterns\n")
    labels = {
        "A": "本数据集 0 命中（疑似单例）",
        "B": "朝 GPT 漂移",
        "C": "反向漂移",
        "D": "dilution baseline",
        "其他": "未触发分类",
    }
    print(f"| 组 | 含义 | 条数 | 占比 |")
    print(f"|---|---|---|---|")
    for g in ["A", "B", "C", "D", "其他"]:
        c = breakdown["counts"][g]
        p = breakdown["percentages"][g]
        print(f"| **{g}** | {labels[g]} | {c} | {p}% |")
    print()
    print(f"详细分组见 [analysis/abcd_breakdown.json]({OUT_PATH.relative_to(ROOT)})")


if __name__ == "__main__":
    main()
