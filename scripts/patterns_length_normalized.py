#!/usr/bin/env python3
"""
Length-normalized pattern rates: per-reply-boolean vs per-1000-char side-by-side.

stats.json records each pattern as:
  total         — sum of findall matches across all replies (counts repeats)
  replies_with  — number of replies containing ≥1 match
  rate          — replies_with / n_replies  ← boolean rate used throughout report

The boolean rate ignores reply length. 4.7 replies are 37% shorter than 4.6
(median 210 vs 333), so patterns that can occur multiple times in a long
reply (加粗, emoji) are systematically penalized for 4.7 under the boolean
metric, while patterns that are rare regardless of length (如果你愿意) are
only mildly affected.

per-1000-char rate = total / total_chars × 1000, where
  total_chars ≈ char_len.mean × n_replies  (stored in stats.json)

Compare the two metrics per pattern; flag sign flips in Δ(4.7 - 4.6) and
magnitude changes on the 5 key patterns tracked in Q3 (加粗, 帮你, 给你X,
如果你愿意, emoji_any).

Output:
  analysis/patterns_length_normalized.csv
  stdout: summary table + sign-flip count
"""

import csv
import json

from _common import ANALYSIS_DIR

STATS_PATH = ANALYSIS_DIR / "stats.json"
OUT_CSV = ANALYSIS_DIR / "patterns_length_normalized.csv"

MODELS = [
    "claude-opus-4-6",
    "claude-opus-4-7",
    "gpt-4o-2024-11-20",
    "gpt-5-chat-latest",
    "gpt-5.3-chat",
    "gpt-5.4",
]

KEY_PATTERNS = ["加粗", "帮你", "给你X", "如果你愿意", "emoji_any"]


def main():
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    patterns = list(stats["claude-opus-4-6"]["patterns"].keys())

    # total_chars per model, from stats.json
    total_chars = {
        m: stats[m]["char_len"]["mean"] * stats[m]["n_replies"]
        for m in MODELS
    }

    rows = []
    for pat in patterns:
        row = {"pattern": pat}
        for m in MODELS:
            pstat = stats[m]["patterns"][pat]
            row[f"{m}__bool"] = pstat["rate"]
            row[f"{m}__per1k"] = (
                pstat["total"] / total_chars[m] * 1000 if total_chars[m] else 0
            )
        # deltas 4.7 - 4.6 in both metrics
        row["delta_bool"] = row["claude-opus-4-7__bool"] - row["claude-opus-4-6__bool"]
        row["delta_per1k"] = row["claude-opus-4-7__per1k"] - row["claude-opus-4-6__per1k"]
        row["sign_flip"] = (
            (row["delta_bool"] > 0) != (row["delta_per1k"] > 0)
            if row["delta_bool"] != 0 and row["delta_per1k"] != 0
            else False
        )
        rows.append(row)

    # Write full CSV
    fieldnames = ["pattern"]
    for m in MODELS:
        fieldnames += [f"{m}__bool", f"{m}__per1k"]
    fieldnames += ["delta_bool", "delta_per1k", "sign_flip"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            out = {k: row[k] for k in fieldnames}
            for k in fieldnames:
                if isinstance(out[k], float):
                    out[k] = round(out[k], 5)
            w.writerow(out)

    # Summary: how many patterns flipped sign?
    n_flipped = sum(1 for r in rows if r["sign_flip"])
    print(f"Length normalization: {n_flipped} / {len(rows)} patterns flip sign of Δ(4.7 − 4.6)")
    print(f"(sign flip = direction of change reverses when using per-1000-char instead of per-reply-boolean)")
    print()

    # Detail: 5 key patterns
    print("5 key patterns — boolean vs per-1000-char:")
    print(f"{'pattern':<15} {'4.6 bool':>9} {'4.7 bool':>9} {'Δ bool pp':>10}  | "
          f"{'4.6 /1k':>8} {'4.7 /1k':>8} {'Δ /1k':>8}  | flip")
    for pat in KEY_PATTERNS:
        r = next((x for x in rows if x["pattern"] == pat), None)
        if r is None:
            continue
        print(
            f"{pat:<15} "
            f"{r['claude-opus-4-6__bool']*100:>8.1f}% "
            f"{r['claude-opus-4-7__bool']*100:>8.1f}% "
            f"{r['delta_bool']*100:>+9.1f}   | "
            f"{r['claude-opus-4-6__per1k']:>8.3f} "
            f"{r['claude-opus-4-7__per1k']:>8.3f} "
            f"{r['delta_per1k']:>+8.3f}  | {r['sign_flip']}"
        )
    print()

    # Detail: flipped patterns (should be mostly noise)
    flipped = [r for r in rows if r["sign_flip"]]
    if flipped:
        print(f"All {len(flipped)} sign-flipped patterns (boolean Δ vs per-1000-char Δ):")
        flipped.sort(key=lambda r: -abs(r["delta_bool"]))
        for r in flipped[:20]:
            print(
                f"  {r['pattern']:<20} "
                f"Δbool={r['delta_bool']*100:+.2f}pp  "
                f"Δper1k={r['delta_per1k']:+.4f}  "
                f"(4.6 bool={r['claude-opus-4-6__bool']*100:.2f}%, "
                f"4.7 bool={r['claude-opus-4-7__bool']*100:.2f}%)"
            )

    print(f"\n→ {OUT_CSV}")


if __name__ == "__main__":
    main()
