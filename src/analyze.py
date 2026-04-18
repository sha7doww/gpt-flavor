#!/usr/bin/env python3
"""
Analyze collected corpus. Produces frequency stats, signature-pattern hits,
and per-(model, condition) comparison tables.

Usage:
  python analyze.py                    # scan data/, write analysis/report.md
  python analyze.py --top-ngrams 30    # tune output
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import yaml

ROOT = Path(__file__).parent.parent
API_DIR = ROOT / "data"
PATTERNS_DIR = ROOT / "patterns"
OUT_DIR = ROOT / "analysis"


def load_patterns():
    """Glob patterns/*.yaml, return {name: compiled_re} and {name: meta-dict}."""
    compiled = {}
    meta = {}
    for path in sorted(PATTERNS_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for p in data.get("patterns", []):
            name = p["name"]
            flags = re.MULTILINE if p.get("multiline") else 0
            compiled[name] = re.compile(p["regex"], flags)
            meta[name] = {
                "desc": p.get("desc", ""),
                "source": p.get("source", ""),
                "regex": p["regex"],
                "file": path.name,
            }
    return compiled, meta


SIGNATURE_PATTERNS, PATTERN_META = load_patterns()


def split_sentences(text):
    sents = re.split(r"[。！？!?\n]+", text)
    return [s.strip() for s in sents if s.strip()]


def char_ngrams(text, n):
    text = re.sub(r"\s+", "", text)
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def load_jsonl(path):
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def scan_records(by="model"):
    """Group records by key. by='model' pools 3 conditions (default,
    primary analysis); by='bucket' keeps model × condition (18 buckets)."""
    groups = defaultdict(list)
    if not API_DIR.exists():
        return groups
    for model_dir in sorted(API_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        for jsonl in sorted(model_dir.glob("*.jsonl")):
            recs = load_jsonl(jsonl)
            for r in recs:
                if by == "model":
                    key = r["model"]
                elif by == "bucket":
                    key = f"{r['model']}/{r['condition']}"
                else:
                    raise ValueError(f"unknown by={by}")
                groups[key].append(r)
    return groups


def analyze_bucket(records, top_ngrams):
    replies = [r["reply"] for r in records if r.get("reply")]
    char_lens = [len(r) for r in replies]
    sent_counts = [len(split_sentences(r)) for r in replies]
    sent_lens = [len(s) for r in replies for s in split_sentences(r)]

    pattern_hits = {}
    for name, pat in SIGNATURE_PATTERNS.items():
        total = sum(len(pat.findall(r)) for r in replies)
        per_reply = sum(1 for r in replies if pat.search(r))
        pattern_hits[name] = {
            "total": total,
            "replies_with": per_reply,
            "rate": round(per_reply / len(replies), 3) if replies else 0,
        }

    all_text = "\n".join(replies)
    bi = Counter(char_ngrams(all_text, 2)).most_common(top_ngrams)
    tri = Counter(char_ngrams(all_text, 3)).most_common(top_ngrams)
    four = Counter(char_ngrams(all_text, 4)).most_common(top_ngrams)

    by_category = defaultdict(int)
    for r in records:
        by_category[r.get("category", "?")] += 1

    return {
        "n_replies": len(replies),
        "char_len": {
            "mean": round(mean(char_lens), 1) if char_lens else 0,
            "median": int(median(char_lens)) if char_lens else 0,
            "max": max(char_lens) if char_lens else 0,
        },
        "sentences_per_reply": {
            "mean": round(mean(sent_counts), 1) if sent_counts else 0,
            "median": int(median(sent_counts)) if sent_counts else 0,
        },
        "sentence_len_chars": {
            "mean": round(mean(sent_lens), 1) if sent_lens else 0,
            "median": int(median(sent_lens)) if sent_lens else 0,
        },
        "patterns": pattern_hits,
        "top_bigrams": bi,
        "top_trigrams": tri,
        "top_4grams": four,
        "by_category": dict(by_category),
    }


def render_markdown(groups, top_ngrams, by):
    lines = ["# Corpus Analysis", ""]
    lines.append(f"Grouping: by={by}  ({len(groups)} groups)")
    total_replies = sum(len(v) for v in groups.values())
    lines.append(f"Total replies: {total_replies}")
    lines.append("")

    lines.append(f"## Per-{by} summary")
    lines.append("")
    lines.append(f"| {by} | n | char_len.mean | char_len.median | sents/reply | sent_len |")
    lines.append("|---|---|---|---|---|---|")
    rows = []
    for key, recs in sorted(groups.items()):
        a = analyze_bucket(recs, top_ngrams)
        rows.append((key, a))
        lines.append(
            f"| {key} | {a['n_replies']} | "
            f"{a['char_len']['mean']} | {a['char_len']['median']} | "
            f"{a['sentences_per_reply']['mean']} | "
            f"{a['sentence_len_chars']['mean']} |"
        )
    lines.append("")

    lines.append("## Signature pattern rate (fraction of replies that match)")
    lines.append("")
    pattern_names = list(SIGNATURE_PATTERNS.keys())
    header = "| pattern | " + " | ".join(k for k, _ in rows) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(rows) + 1))
    for pn in pattern_names:
        row = [f"| {pn}"]
        for _, a in rows:
            row.append(f"{a['patterns'][pn]['rate']:.2f}")
        lines.append(" | ".join(row) + " |")
    lines.append("")

    lines.append(f"## Top char {top_ngrams}-grams per {by}")
    lines.append("")
    for key, a in rows:
        lines.append(f"### {key}")
        lines.append("")
        lines.append("**trigrams**: " + ", ".join(
            f"`{g}`×{c}" for g, c in a["top_trigrams"][:20]
        ))
        lines.append("")
        lines.append("**4-grams**: " + ", ".join(
            f"`{g}`×{c}" for g, c in a["top_4grams"][:20]
        ))
        lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-ngrams", type=int, default=40)
    ap.add_argument("--by", choices=["model", "bucket"], default="model",
                    help="'model' pools 3 conditions (default). 'bucket' = model×condition (18).")
    ap.add_argument("-o", "--output-stem", default=None,
                    help="Stem for output files (default: 'stats' for --by model, "
                         "'stats_by_bucket' for --by bucket).")
    args = ap.parse_args()

    groups = scan_records(by=args.by)
    if not groups:
        print("No data under data/. Run collect.py first.")
        return
    md = render_markdown(groups, args.top_ngrams, args.by)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem or ("stats" if args.by == "model" else "stats_by_bucket")
    (OUT_DIR / f"{stem}.md").write_text(md, encoding="utf-8") \
        if args.by != "model" else (OUT_DIR / "report.md").write_text(md, encoding="utf-8")

    stats_json = {}
    for key, recs in groups.items():
        stats_json[key] = analyze_bucket(recs, args.top_ngrams)
    (OUT_DIR / f"{stem}.json").write_text(
        json.dumps(stats_json, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Wrote {OUT_DIR}/{stem}.json ({len(groups)} groups, by={args.by})")


if __name__ == "__main__":
    main()
