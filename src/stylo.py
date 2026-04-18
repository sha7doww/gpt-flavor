#!/usr/bin/env python3
"""
Stylometry / authorship-attribution analysis on the collected corpus.

Produces three complementary views:

1. Weighted log-odds (Monroe et al. 2008 "Fightin' Words") — which n-grams
   are most characteristic of each model relative to the rest.
2. Char-ngram TF-IDF + cosine similarity matrix between buckets.
3. LinearSVC classifier: can a model tell the text apart from other buckets?

Usage:
  python stylo.py                          # full run
  python stylo.py --by model               # pool per-model (not per-bucket)
  python stylo.py --ngram-range 2 5        # char n-gram range
  python stylo.py --top-k 40               # log-odds top-k per group
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).parent.parent
API_DIR = ROOT / "data"
OUT_DIR = ROOT / "analysis"

WORD_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z]+|\d+")


def load_records():
    """Walk data/<model>/<condition>.jsonl and yield all records."""
    records = []
    if not API_DIR.exists():
        return records
    for model_dir in sorted(API_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        for jsonl in sorted(model_dir.glob("*.jsonl")):
            with jsonl.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("reply"):
                        records.append(r)
    return records


def char_ngrams(text, n_min=2, n_max=5):
    """Char n-grams including cross-n-gram mix. Uses tokens, not bytes."""
    text = re.sub(r"\s+", "", text)
    out = []
    for n in range(n_min, n_max + 1):
        for i in range(len(text) - n + 1):
            out.append(text[i:i + n])
    return out


# ---------------------------------------------------------------------------
# 1. Weighted log-odds (Monroe et al. 2008), informative Dirichlet prior
# ---------------------------------------------------------------------------

def weighted_log_odds(group_counts, total_counts, alpha=0.01):
    """
    Compute log-odds ratio with informative Dirichlet prior for each group
    vs. 'rest'. Returns {group: {ngram: z-score}}.

    group_counts: {group_name: Counter}
    total_counts: Counter over all groups combined
    alpha: smoothing strength relative to corpus-weighted prior
    """
    n_total = sum(total_counts.values())
    a0 = alpha * n_total  # prior mass; per-term prior ~ alpha * p(w)

    out = {}
    for g, g_counts in group_counts.items():
        rest_counts = Counter()
        for gg, gc in group_counts.items():
            if gg == g:
                continue
            rest_counts.update(gc)
        ng = sum(g_counts.values())
        nr = sum(rest_counts.values())

        scores = {}
        for w, c_total in total_counts.items():
            c_g = g_counts.get(w, 0)
            c_r = rest_counts.get(w, 0)
            # prior pseudo-count for w proportional to its corpus share
            a_w = alpha * c_total
            # log-odds with prior smoothing
            num = (c_g + a_w) / (ng + a0 - c_g - a_w)
            den = (c_r + a_w) / (nr + a0 - c_r - a_w)
            log_odds = np.log(num) - np.log(den)
            # Monroe et al. z-score: divide by sqrt of variance estimate
            var = 1.0 / (c_g + a_w) + 1.0 / (c_r + a_w)
            z = log_odds / np.sqrt(var)
            scores[w] = float(z)
        out[g] = scores
    return out


def run_log_odds(records, group_key, top_k, ngram_min, ngram_max, min_doc_freq):
    """Compute weighted log-odds per group and write top_k ngrams to CSV."""
    # 1. build per-group ngram counts
    group_ngrams = defaultdict(Counter)
    doc_freq = Counter()  # in how many docs each ngram appears
    for r in records:
        g = group_key(r)
        ngs = char_ngrams(r["reply"], ngram_min, ngram_max)
        group_ngrams[g].update(ngs)
        for ng in set(ngs):
            doc_freq[ng] += 1

    # 2. prune by min doc freq to avoid noise
    keep = {w for w, df in doc_freq.items() if df >= min_doc_freq}
    total_counts = Counter()
    for g in group_ngrams:
        group_ngrams[g] = Counter({w: c for w, c in group_ngrams[g].items() if w in keep})
        total_counts.update(group_ngrams[g])

    # 3. compute log-odds
    scores = weighted_log_odds(group_ngrams, total_counts)

    # 4. top-k per group
    top_per_group = {}
    for g, sc in scores.items():
        items = sorted(sc.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        top_per_group[g] = items

    return top_per_group, group_ngrams


# ---------------------------------------------------------------------------
# 2. TF-IDF + cosine similarity matrix between buckets
# ---------------------------------------------------------------------------

def run_cosine(records, group_key, ngram_min, ngram_max):
    """Pool replies per group, TF-IDF, return (groups, cosine_matrix)."""
    groups = defaultdict(list)
    for r in records:
        groups[group_key(r)].append(r["reply"])

    group_names = sorted(groups.keys())
    docs = ["\n".join(groups[g]) for g in group_names]

    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        min_df=2,
        max_features=20000,
        sublinear_tf=True,
    )
    X = vec.fit_transform(docs)
    sim = cosine_similarity(X)
    return group_names, sim


# ---------------------------------------------------------------------------
# 3. LinearSVC classifier: can styles be told apart at reply level?
# ---------------------------------------------------------------------------

def run_classifier(records, group_key, ngram_min, ngram_max,
                   min_per_class=10, split_by_seed=False):
    """Train LinearSVC on char-ngram TF-IDF of individual replies.

    If split_by_seed=True, use GroupShuffleSplit keyed on r['seed'] so the
    same prompt's replies don't leak across train/test. Required when the
    classifier has to generalize to unseen prompts (e.g. 4.6 vs 4.7, where
    every seed was shown to both models).
    """
    buckets = defaultdict(list)
    seeds_buckets = defaultdict(list)
    for r in records:
        buckets[group_key(r)].append(r["reply"])
        seeds_buckets[group_key(r)].append(r.get("seed", ""))
    keep = {k for k, v in buckets.items() if len(v) >= min_per_class}
    buckets = {k: buckets[k] for k in keep}
    seeds_buckets = {k: seeds_buckets[k] for k in keep}

    X_text = []
    y = []
    seeds = []
    for g, replies in buckets.items():
        X_text.extend(replies)
        y.extend([g] * len(replies))
        seeds.extend(seeds_buckets[g])
    y = np.array(y)
    seeds = np.array(seeds)
    if len(set(y)) < 2:
        return None

    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        min_df=3,
        max_features=30000,
        sublinear_tf=True,
    )
    X = vec.fit_transform(X_text)

    if split_by_seed:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, te_idx = next(gss.split(X, y, groups=seeds))
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    clf = LinearSVC(C=1.0, max_iter=5000, dual="auto")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    labels = sorted(set(y))
    cm = confusion_matrix(y_te, y_pred, labels=labels).tolist()
    return {
        "labels": labels,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {k: v for k, v in report.items() if k not in ("accuracy", "macro avg", "weighted avg")},
        "confusion_matrix": cm,
        "n_train": X_tr.shape[0],
        "n_test": X_te.shape[0],
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def group_key_bucket(r):
    return f"{r['model']}/{r['condition']}"


def group_key_model(r):
    return r["model"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by", choices=["bucket", "model"], default="model",
                    help="grouping: 'model' pools 3 conditions (default, 6 groups), "
                         "'bucket' = model×condition (18 groups)")
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--ngram-range", type=int, nargs=2, default=[2, 5])
    ap.add_argument("--min-doc-freq", type=int, default=5)
    args = ap.parse_args()

    records = load_records()
    print(f"Loaded {len(records)} records across "
          f"{len({(r['model'], r['condition']) for r in records})} buckets")

    gk = group_key_bucket if args.by == "bucket" else group_key_model
    nmin, nmax = args.ngram_range

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- log-odds ---
    print("\n[1/3] Weighted log-odds (Monroe et al.)...")
    top_per_group, _ = run_log_odds(
        records, gk, args.top_k, nmin, nmax, args.min_doc_freq
    )
    (OUT_DIR / "log_odds_top.json").write_text(
        json.dumps({g: [{"ngram": w, "z": z} for w, z in items] for g, items in top_per_group.items()},
                   ensure_ascii=False, indent=2)
    )
    print(f"  → {OUT_DIR / 'log_odds_top.json'}")
    # show a glimpse
    for g, items in sorted(top_per_group.items()):
        top10 = [w for w, _ in items[:10]]
        print(f"  {g}: {', '.join(top10)}")

    # --- cosine ---
    print("\n[2/3] TF-IDF char-ngram cosine similarity...")
    names, sim = run_cosine(records, gk, nmin, nmax)
    (OUT_DIR / "cosine_matrix.json").write_text(
        json.dumps({
            "groups": names,
            "matrix": sim.tolist(),
            "ngram_range": [nmin, nmax],
        }, ensure_ascii=False, indent=2)
    )
    print(f"  → {OUT_DIR / 'cosine_matrix.json'}  ({len(names)}×{len(names)})")
    # print top-off-diagonal pairs
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((sim[i, j], names[i], names[j]))
    pairs.sort(reverse=True)
    print("  top-5 most similar pairs:")
    for s, a, b in pairs[:5]:
        print(f"    {s:.3f}  {a}  ↔  {b}")

    # --- classifier ---
    print("\n[3/3] LinearSVC classifier...")
    cls_result = run_classifier(records, gk, nmin, nmax)
    if cls_result is None:
        print("  not enough classes, skipped")
    else:
        (OUT_DIR / "classifier.json").write_text(
            json.dumps(cls_result, ensure_ascii=False, indent=2)
        )
        print(f"  → {OUT_DIR / 'classifier.json'}")
        print(f"  accuracy: {cls_result['accuracy']:.3f}")
        print(f"  macro F1: {cls_result['macro_f1']:.3f}")
        print(f"  n_train={cls_result['n_train']}  n_test={cls_result['n_test']}")

    # --- also run binary: is this gpt-5.4? ---
    bin_records = [{**r, "y": "gpt-5.4" if r["model"] == "gpt-5.4" else "other"} for r in records]
    print("\n[3b/3] Binary: is this gpt-5.4?")
    bin_result = run_classifier(bin_records, lambda r: r["y"], nmin, nmax)
    if bin_result is not None:
        (OUT_DIR / "classifier_binary_gpt54.json").write_text(
            json.dumps(bin_result, ensure_ascii=False, indent=2)
        )
        print(f"  accuracy: {bin_result['accuracy']:.3f}")
        print(f"  macro F1: {bin_result['macro_f1']:.3f}")

    # --- also run binary: claude-opus-4-6 vs claude-opus-4-7 ---
    # Headline result of README §Q1 — must stay reproducible from run_all.sh.
    opus_records = [r for r in records
                    if r["model"] in ("claude-opus-4-6", "claude-opus-4-7")]
    print("\n[3c/3] Binary: claude-opus-4-6 vs claude-opus-4-7?")
    opus_result = run_classifier(opus_records, group_key_model, nmin, nmax,
                                 split_by_seed=True)
    if opus_result is not None:
        (OUT_DIR / "classifier_binary_opus46vs47.json").write_text(
            json.dumps(opus_result, ensure_ascii=False, indent=2)
        )
        print(f"  accuracy: {opus_result['accuracy']:.3f}")
        print(f"  macro F1: {opus_result['macro_f1']:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
