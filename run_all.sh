#!/usr/bin/env bash
# One-shot: re-run the full pipeline from existing采集 data.
# (不会重采数据。要重采见 docs/REPRODUCE.md)
#
# Produces:
#   analysis/stats.json / stats_by_bucket.json
#   analysis/log_odds_top.json
#   analysis/cosine_matrix.json
#   analysis/classifier*.json
#   analysis/patterns_by_condition.csv
#   analysis/patterns_by_category.csv
#   analysis/classifier_by_category.json
#   analysis/abcd_breakdown.json
#   analysis/figures/*.png (4 张图)

set -euo pipefail

cd "$(dirname "$0")"

echo "== 1/4 analyze (per-model) =="
python src/analyze.py

echo "== 2/4 analyze (by bucket，给 prompt sensitivity 图用) =="
python src/analyze.py --by bucket

echo "== 3/4 stylo (log-odds + cosine + classifier) =="
python src/stylo.py

echo "== 4/4 visualize (4 张图) =="
python src/visualize.py

echo "== gap scripts =="
python scripts/patterns_by_condition.py
python scripts/patterns_by_category.py
python scripts/classifier_by_category.py
python scripts/abcd_breakdown.py
python scripts/cosine_self_stability.py

echo
echo "Done. 产出见 analysis/ 和 analysis/figures/"
