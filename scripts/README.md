# scripts/ — 补充分析脚本

主 pipeline 在 `src/`（`analyze.py`/`stylo.py`/`visualize.py`）。
这里装的是**针对具体问题**的细化分析，需要时再跑。

## 4 个 gap 脚本

| 脚本 | 回答的问题 | 输入 | 产出 |
|---|---|---|---|
| `patterns_by_condition.py` | 短语漂移是 A/B/C 哪个 prompt 条件下放大的？是模型内生还是 prompt 诱发？ | `data/**/*.jsonl` + `patterns.yaml` 全 100 条 | `analysis/patterns_by_condition.csv` |
| `patterns_by_category.py` | 4.6→4.7 的风格漂移是全场景还是只在特定 15 个 seed category 里发生？ | `data/**/*.jsonl` + `patterns/*.yaml` 里的 5 个 key pattern | `analysis/patterns_by_category.csv` + `analysis/figures/drift_by_category.png` |
| `classifier_by_category.py` | 按 category 训"4.6 vs 4.7"二分类，哪个场景区分度最高（= 风格差异最大）？ | `data/**/*.jsonl` 中所有 opus 数据 | `analysis/classifier_by_category.json` |
| `abcd_breakdown.py` | 100 条 pattern 按观测行为分 A/B/C/D 组（0 命中 / 真涨 / 反向 / dilution） | `analysis/stats.json` | `analysis/abcd_breakdown.json` |

## 运行

```bash
python scripts/patterns_by_condition.py
python scripts/patterns_by_category.py
python scripts/classifier_by_category.py
python scripts/abcd_breakdown.py
```

所有脚本都依赖 `_common.py`（共享 `load_records` / `load_patterns`），不要直接从项目根的
`analyze.py`/`stylo.py` import（那些脚本 import 时会执行 main）。

## 添加新脚本

- 放这里（不是根目录）
- 用 `_common.load_records()` / `_common.load_patterns()` 而不是重新写
- 输出到 `analysis/` 或 `analysis/figures/`
- 主 pipeline 的数字（stats.json / cosine_matrix.json）由主脚本维护，这里**不要**覆写
