#!/usr/bin/env python3
"""
Gap 3 — LinearSVC "opus-4-6 vs opus-4-7" binary classifier, per seed category.

Global accuracy = 88.0% (GroupShuffleSplit by seed). Broken down by category,
we can ask:
  - In which scenarios does the classifier do best (→ 4.6 vs 4.7 most different)?
  - In which scenarios does it barely beat random (→ 4.6 ≈ 4.7 in those)?

This pinpoints where the 4.6→4.7 style shift is most/least visible. Uses
GroupShuffleSplit keyed on seed so same-prompt replies don't leak across
train/test (otherwise each seed's 18 rows — 3 conds × 3 runs × 2 models —
would overlap both splits and inflate accuracy ~10pp).

Output:
  analysis/classifier_by_category.json
    {category: {accuracy, macro_f1, n_train, n_test}}
"""

import json
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC

from _common import load_records, ANALYSIS_DIR

MIN_PER_CLASS = 20


def run_one(records):
    """Train LinearSVC on 4.6 vs 4.7 for these records."""
    X_text = []
    y = []
    groups = []
    for r in records:
        if r["model"] not in ("claude-opus-4-6", "claude-opus-4-7"):
            continue
        X_text.append(r["reply"])
        y.append(r["model"])
        groups.append(r["seed"])
    y = np.array(y)
    groups = np.array(groups)
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
    # Group by seed: each seed has up to 18 replies (3 conditions × 3 runs × 2
    # models). Without grouping, same-seed replies leak across train/test and
    # inflate accuracy by several pp.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
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
