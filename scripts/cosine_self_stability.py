#!/usr/bin/env python3
"""
Self-cosine stability: within-model split-half cos baseline.

For each model, randomly split its 1,449 replies into two halves, pool each
into a mega-doc, and compute char-ngram TF-IDF cosine using the SAME feature
space as the main cross-model matrix (fit on 6 model-pooled mega-docs, per
src/stylo.py run_cosine). Repeat N_BOOTSTRAP times.

The resulting distribution is the **noise floor** against which cross-model
cos deltas (e.g., 4.6→4.7 ↔ gpt-5.4 +0.013) should be judged: if median
self_cos = 0.997, then noise ≈ 0.003 and +0.013 is ~4× noise; if self_cos
median is much lower, the Δ may be dominated by sampling variance.

Output:
  analysis/cosine_self_stability.json
"""

import json
import random
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from _common import ANALYSIS_DIR, load_records

N_BOOTSTRAP = 30
NGRAM_RANGE = (2, 5)
RANDOM_SEED = 42


def main():
    records = load_records()
    by_model = defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r["reply"])

    # Fit TF-IDF on the same feature space as the main cosine matrix:
    # one mega-doc per model. See src/stylo.py:run_cosine.
    model_names = sorted(by_model.keys())
    mega_docs = ["\n".join(by_model[m]) for m in model_names]
    vec = TfidfVectorizer(
        analyzer="char", ngram_range=NGRAM_RANGE,
        min_df=2, max_features=20000, sublinear_tf=True,
    )
    vec.fit(mega_docs)

    rng = random.Random(RANDOM_SEED)
    results = {}
    for m in model_names:
        replies = by_model[m]
        n = len(replies)
        samples = []
        for _ in range(N_BOOTSTRAP):
            idx = list(range(n))
            rng.shuffle(idx)
            half_a = "\n".join(replies[i] for i in idx[: n // 2])
            half_b = "\n".join(replies[i] for i in idx[n // 2:])
            X = vec.transform([half_a, half_b])
            samples.append(float(cosine_similarity(X[0:1], X[1:2])[0, 0]))
        arr = np.array(samples)
        results[m] = {
            "n_replies": n,
            "n_bootstrap": N_BOOTSTRAP,
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p05": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    # Cross-model deltas reported in REPORT §3.1, for contextual comparison.
    ctx = {
        "method": "split-half bootstrap: shuffle each model's replies, pool "
                  "halves into mega-docs, compute char-ngram(2-5) TF-IDF cos",
        "gpt_delta_4_6_to_4_7": {
            "gpt-5.4": 0.013,
            "gpt-5.3-chat": 0.029,
            "gpt-5-chat-latest": 0.036,
            "gpt-4o-2024-11-20": 0.036,
        },
        "interpretation": "Δ should be judged against (1 - self_cos median) "
                          "per model — that is the within-model noise floor.",
    }

    out = {"context": ctx, "self_cos": results}
    path = ANALYSIS_DIR / "cosine_self_stability.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print(f"{'model':<25} {'median':>8} {'p05':>8} {'p95':>8} {'noise':>8}")
    for m in model_names:
        r = results[m]
        noise = 1 - r["median"]
        print(f"{m:<25} {r['median']:>8.4f} {r['p05']:>8.4f} "
              f"{r['p95']:>8.4f} {noise:>8.4f}")
    print(f"\n→ {path}")


if __name__ == "__main__":
    main()
