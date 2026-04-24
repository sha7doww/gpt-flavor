# 复现 / 贡献指南

> **注意**：`data/` 已随 git 分发（17 MB），`bash run_all.sh` 直接能跑。重新采集只在你想换模型 / seed / prompt 时需要。

## TL;DR

- 想验证我们的数字：clone + `bash run_all.sh`（约 5-10 分钟）
- 想换模型 / seed / 重新采集：环境 → 采集（约 \$15-20、15-25 分钟）→ `bash run_all.sh`
- 想加自己的 seed / pattern / 模型：见后面对应小节
- 想读所有数字 + 方法学：[REPORT.md](REPORT.md)

---

## 1. 环境配置

```bash
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 填 OPENAI_API_KEY（任意 OpenAI 兼容端点，需支持 Claude Opus 系列）
# 非默认端点同时设 OPENAI_BASE_URL
```

## 2. 数据采集（可选——data/ 已含）

> **⚠️ 重采会得到和原数据不同的绝对值**。本仓库 `data/` 的原始采集时间是 2026-04-17~18，正好在 Anthropic Claude Code "≤100 words" verbosity 指令窗口（04-16 ~ 04-20）内且通过第三方 OpenAI-compatible 中转站采——两个因素都会让今天重采的 median 字数、markdown boolean rate 等指标跟原数据不完全对得上。结构性结论（offer 漂移、按 seed 场景双向分裂、cosine 朝 `gpt-5-chat-latest`）在两种后端假设下都成立，绝对数字不保证复刻。详见 [REPORT §5.2](REPORT.md#52-我们这次实验的局限) + [§5.3](REPORT.md#53-与-anthropic-官方-postmortem-对照)。

```bash
python src/collect.py run \
  --models gpt-4o-2024-11-20 gpt-5-chat-latest gpt-5.3-chat gpt-5.4 claude-opus-4-6 claude-opus-4-7 \
  --concurrency 100
```

- 规模：6 模型 × 3 条件 × 161 seeds × 3 runs = **8,694 条**（每模型 1,449 条）
- 成本 / 时间：约 \$15-20、15-25 分钟（concurrency=100，受上游限速）
- 参数：`temperature=1.0`（`src/collect.py` 硬编码；部分模型不接受该值则自动 fallback 到 server 默认，基本也是 1.0）
- 幂等：按 `sha256(model|condition|seed|run)` 跳过已采的；中断后重跑只补缺
- 仓库自带的 data/ 就是按这条命令采的成果，结论可直接复现，不必重采

## 3. 全量分析

```bash
bash run_all.sh
```

约 5-10 分钟（瓶颈是 stylo.py 的 TF-IDF）。一键串起：

| 步骤 | 脚本 | 产出 |
|---|---|---|
| per-model stats | `src/analyze.py` | `analysis/stats.json` + `report.md` |
| per-bucket stats（model × condition） | `src/analyze.py --by bucket` | `analysis/stats_by_bucket.{json,md}` |
| log-odds + 6×6 cosine + 3 个 classifier | `src/stylo.py` | `analysis/{log_odds_top.json, cosine_matrix.json, classifier*.json}` |
| 3 张图（cosine / log_odds / prompt_sensitivity） | `src/visualize.py` | `analysis/figures/*.png` |
| pattern × 3 condition | `scripts/patterns_by_condition.py` | `analysis/patterns_by_condition.csv` |
| 5 key pattern × 15 category + drift heatmap（第 4 张图） | `scripts/patterns_by_category.py` | `analysis/patterns_by_category.csv` + `drift_by_category.png` |
| 4.6 vs 4.7 分 category 二分类 | `scripts/classifier_by_category.py` | `analysis/classifier_by_category.json` |
| 100 pattern 按 ABCD 行为分组 | `scripts/abcd_breakdown.py` | `analysis/abcd_breakdown.json` |
| cosine 自稳定性噪声底 | `scripts/cosine_self_stability.py` | `analysis/cosine_self_stability.json` |
| 每千字频次（长度归一化审计） | `scripts/patterns_length_normalized.py` | `analysis/patterns_length_normalized.csv` |

---

## 如何加 seed

`seeds/seeds.json` 是扁平 dict——key 是 category 名，value 是中文 prompt 字符串列表：

```json
{
  "emotional_comfort": [
    "我最近很累但又不想休息",
    "我好像一直在硬撑"
  ],
  "your_new_category": [
    "你的新 prompt 1",
    "你的新 prompt 2"
  ]
}
```

- 现有 15 类（详见 `seeds/seeds.json`）：`emotional_comfort` / `self_doubt` / `relationships` / `procrastination` / `meaning_existential` / `work_study` / `control_technical` / `control_casual` / `disagreement` / `analysis` / `refusal` / `summarization` / `creative` / `tech_deliberation` / `community_replication`
- 加完后重跑 `python src/collect.py run`——只采新 seed（已采的按 hash 跳过）
- 分析自动读全 `data/`，不用改 pipeline

## 如何加 pattern

编辑 `patterns/patterns.yaml`，追加一条：

```yaml
  - name: "短 ID"
    regex: '你的 Python 正则'
    desc: 'one-line English description (用于图 label)'
    expect_high_in: [model-id-1, model-id-2]   # 可选
    multiline: true                            # 可选
```

- 重跑 `bash run_all.sh`，`analysis/stats.json` 自动多出一个 key
- ABCD 分组由 `scripts/abcd_breakdown.py` 按规则自动算（见 [REPORT §4.1](REPORT.md#41-100-条-pattern-的-abcd-分组)）

## 如何加模型

1. 编辑 `scripts/_common.py` 的 `MODELS_MAIN`——加一个模型 ID（你用的 OpenAI 兼容端点支持的）
2. 跑 `python src/collect.py run --models <new-model-id>`——只采新模型
3. 跑 `bash run_all.sh`——分析自动适配（cosine 变 7×7、log-odds 多一组、6-way classifier 变 7-way）
4. **手动更新** `docs/REPORT.md` / `README.md` 里的表格 / 数字（这部分没自动化）

---

## 方法学 + 所有数字

完整研究细节（命题、4 类分析、3 个问题、局限、讨论）见 [REPORT.md](REPORT.md)。
