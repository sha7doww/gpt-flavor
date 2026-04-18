# patterns/ — regex 的单一真理源

**一份文件**：`patterns.yaml`，100 条，flat list。`analyze.py` 会 `yaml.safe_load` 加载、按 `name` 去扫 reply。

**不做 pattern-level 分类**：结果分组（0 命中 / 真涨 / 反向 / dilution baseline）按实际观测行为在 [`../docs/REPORT.md`](../docs/REPORT.md) §4.1 里做——分类是数据驱动的，不是 pattern 属性。

## Schema

```yaml
patterns:
  - name: "短 ID"              # 作为 stats.json 里的 key
    regex: 'Python re 正则'
    desc: 'one-line English description'  # visualize 图例，避免 CJK 字体依赖
    expect_high_in: [model1, model2]      # 可选；预期高命中
    multiline: true                       # 可选；re.MULTILINE
```

## 添加一个新 pattern

1. 加条目：简短 `name` + `regex` + 英文 `desc`
2. 重跑 `bash run_all.sh`（或 `python src/analyze.py`），`stats.json` 里会自动多出一个 key

## 历史

- 2026-04：从 3 个 YAML 合并成单文件
- 2026-04-17：尝试按来源分 4 → 2 组，最后发现 pattern-level 分类对分析没帮助
- 2026-04-18：完全扁平化，去掉 `source` 字段。结果分组挪到 REPORT §4.1（按观测行为分 A/B/C/D）
