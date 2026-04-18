#!/usr/bin/env python3
"""
Gap 3 — LinearSVC "opus-4-6 vs opus-4-7" binary classifier, per seed category.

Global accuracy = 96.7% (very high). Broken down by category, we can ask:
  - In which scenarios does the classifier do best (→ 4.6 vs 4.7 most different)?
  - In which scenarios does it barely beat random (→ 4.6 ≈ 4.7 in those)?

This pinpoints where the 4.6→4.7 style shift is most/least visible.

Output:
  analysis/classifier_by_category.json
    {category: {accuracy, macro_f1, n_train, n_test}}
"""

import json
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from _common import load_records, ANALYSIS_DIR

MIN_PER_CLASS = 20


def run_one(records):
    """Train LinearSVC on 4.6 vs 4.7 for these records."""
    X_text = []
    y = []
    for r in records:
        if r["model"] not in ("claude-opus-4-6", "claude-opus-4-7"):
            continue
        X_text.append(r["reply"])
        y.append(r["model"])
    y = np.array(y)
    if len(X_text) < MIN_PER_CLASS * 2:
        return None
    if len(set(y)) < 2:
        return None
    cnt = {c: int((y == c).sum()) for c in set(y)}
    if min(cnt.values()) < MIN_PER_CLASS:
        return None

    vec = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5),
        min_df=2, max_features=20000, sublinear_tf=True,
    )
    X = vec.fit_transform(X_text)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LinearSVC(C=1.0, max_iter=5000, dual="auto")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    rep = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": rep["accuracy"],
        "macro_f1": rep["macro avg"]["f1-score"],
        "n_train": X_tr.shape[0],
        "n_test": X_te.shape[0],
        "class_counts": cnt,
    }


def main():
    records = load_records()
    categories = sorted({r["category"] for r in records})

    # overall baseline
    overall = run_one(records)
    print(f"Overall 4.6 vs 4.7: accuracy={overall['accuracy']:.3f}")
    print()

    by_cat = {"__overall__": overall}
    for cat in categories:
        subset = [r for r in records if r["category"] == cat]
        result = run_one(subset)
        by_cat[cat] = result

    # print sorted by accuracy (most distinguishable first)
    rows = [(cat, data) for cat, data in by_cat.items() if data and cat != "__overall__"]
    rows.sort(key=lambda x: -x[1]["accuracy"])
    print(f"{'category':<24} {'acc':>6} {'F1':>6} {'n_test':>7}")
    for cat, d in rows:
        print(f"{cat:<24} {d['accuracy']:>6.3f} {d['macro_f1']:>6.3f} {d['n_test']:>7}")

    out = ANALYSIS_DIR / "classifier_by_category.json"
    out.write_text(json.dumps(by_cat, ensure_ascii=False, indent=2))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
