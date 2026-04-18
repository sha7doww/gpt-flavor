<div align="center">

# Claude Opus 4.7 真的变 GPT 味了吗？

> *2026-04-16 Opus 4.7 发布后，中文社区出现"变 GPT 味了"的反馈。<br>
> 我们采集了 8,694 条跨模型中文样本做量化分析。*

by [@sha7doww](https://github.com/sha7doww) · [@xsyshuishui](https://github.com/xsyshuishui) · [@ecnudl](https://github.com/ecnudl)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data: CC-BY-4.0](https://img.shields.io/badge/Data-CC--BY--4.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Claude Code](https://img.shields.io/badge/Claude%20Code-Skill-blueviolet)](https://claude.ai/code)
[![AgentSkills](https://img.shields.io/badge/AgentSkills-Standard-green)](https://agentskills.io)

<br>

一份社区使用者的 weekend 兴趣调查，**不是学术研究**。<br>
seed 选材 / pattern 选取 / 方法论都存在局限（详见 [REPORT §5.2](docs/REPORT.md#52-我们这次实验的局限)），结论仅供参考。<br>
**8,694 条**实采样本（6 模型 × 3 条件 × 161 seeds × 3 runs），用 stylometry 做定量分析。

<br>

### 两个产出

📊 &nbsp; **分析报告** &nbsp;—&nbsp; 3 个问题串起的 stylometry 证据链  
🎭 &nbsp; **gpt-flavor SKILL** &nbsp;—&nbsp; 基于 8,694 条实采反向归纳的 ChatGPT 中文 persona

[速读 TL;DR](#tldr) · [3 个问题](#我们想回答-3-个问题) · [详细 REPORT](docs/REPORT.md) · [SKILL 详情](skill/gpt-flavor/SKILL.md) · [局限](docs/REPORT.md#52-我们这次实验的局限) · [复现](docs/REPRODUCE.md)

</div>

---

## TL;DR

社区说"Opus 4.7 变 GPT 味了"——

- ✅ **4.7 确实变了**（4.6 vs 4.7 二分类 **87.2%** 准确率，按 seed 分组留出；13/15 类明显可分，仅 `control_casual` / `community_replication` 近随机）
- ✅ **整体上朝 GPT 漂移**（cosine 对全部 4 个 GPT 模型都 +0.013~0.036；self-cos noise 基线见 REPORT §3.1——方向全部成立，+0.036 是温和信号，最小的 +0.013 vs gpt-5.4 在噪声里不显著）
- ⚠️ **朝 GPT 漂移 per-char 比 boolean 显示的更强**：长度归一化后（REPORT §4.3），`帮你` +2.8×、`给你X` +8×、`如果你愿意` +3.3×——boolean 指标严重低估了 offer 类漂移
- ⚠️ **"情感类更极简 Claude"大部分是长度 artifact**：4.7 回复短 37%、emoji 减半是真；但 bold / 反转 / Claude 招牌的 per-char 密度并没降——原来说的"markdown -49~-80pp"多半来自"回复短了塞不下"而不是"主动弃用"
- 🎯 **最像 ChatGPT 短句体**（`gpt-5-chat-latest`，cos +0.036 最大）而不是"全功能" `gpt-5.4`

**一句话**：4.7 的核心变化是**回复短 37% + emoji 减半**——招式密度（bold / 反转 / Claude 招牌）基本没变；**真·朝 GPT 漂移的证据在 offer 类短语上，per-char 倍率 2.8×~8×，比报告 boolean 指标显示的强得多**。社区"变 GPT 味"的说法**部分成立**（task 场景 offer 腔确实 GPT 化），另一部分是错觉（短回复 + 少 emoji 被误读为"风格大变"）。

---

## 我们想回答 3 个问题

### Q1 — 4.7 真变了吗？ ✅ 大变

二分类器 **87.2% accuracy** 区分 4.6 vs 4.7（`GroupShuffleSplit` 按 seed 分组留出，避免同一 prompt 的多次 run 跨 train/test 泄漏）。**2/15 类 100%**（analysis, meaning_existential），**6/15 类 ≥ 93%**，13/15 类 ≥ 80%，但 **community_replication 53% / control_casual 50%** 近随机——这两类 4.6 和 4.7 行为几乎不可分。回复 median 字数 **333 → 210（-37%）**（mean 1115 → 446，但被长任务 seed 严重拉偏，详见 REPORT §2.3），加粗率 **83% → 48%（-35pp）**，emoji 率 **50% → 22%（-28pp）**。

→ 详细见 [REPORT §2](docs/REPORT.md#2-q1--47-真变了吗)

### Q2 — 变得像 GPT 吗？ ✅ 方向成立但程度有限

**cosine 距离全部 4 个 GPT 都更近**：

![cosine heatmap](analysis/figures/cosine_heatmap.png)

| 距离对 | 4.6 | 4.7 | Δ |
|---|---|---|---|
| ↔ `gpt-5.4`（ChatGPT Thinking） | 0.894 | **0.907** | +0.013 |
| ↔ `gpt-5.3-chat`（ChatGPT Instant） | 0.812 | **0.841** | +0.029 |
| ↔ `gpt-5-chat-latest`（旧 ChatGPT） | 0.826 | **0.862** | **+0.036** |
| ↔ `gpt-4o-2024-11-20` | 0.816 | **0.851** | +0.035 |

但 **pattern 级别学得不全**——4.7 在 GPT 招牌招式上仍落后 6-12×：

| pattern | 4.7 | gpt-5.4 | 差距 |
|---|---|---|---|
| 如果你愿意 | 6.4% | **85.0%** | 13× |
| 给你X | 3.9% | **25.9%** | 7× |
| 加粗 | 47.8% | **85.3%** | 2× |
| 帮你 | 23.1% | **43.6%** | 1.9× |

→ 详细见 [REPORT §3](docs/REPORT.md#3-q2--变得像-gpt-吗)

### Q3 — 那它具体变成什么了？ ⚠️ 同一个模型，按场景走了相反方向

100 条 pattern 按观测行为分组：

| 组 | 含义 | 数量 | 占比 |
|---|---|---|---|
| **A** | 本数据集 0 命中（全模型 ≤ 1%） | 52 | 52% |
| **B** | 真朝 GPT 漂移（4.7 比 4.6 高 ≥1.5pp） | 6 | 6% |
| **C** | 反向漂移（4.7 比 4.6 低 ≥1.5pp） | 15 | 15% |
| **D** | dilution baseline（GPT ≫ Claude） | 2 | 2% |
| 其他 | 微动 / 不分类 | 25 | 25% |

但 **6% 这个均值是误导**——按 seed 类别拆开看 4.7 - 4.6 在 5 个 key pattern 上的 **boolean** 变化，呈现**双向分裂**：

| 场景 | 4.7 的 boolean 方向 | 例：`帮你` 的 Δpp |
|---|---|---|
| **情感 / 关系 / 自我类** | boolean 下"更极简 Claude"（markdown -49 ~ -80pp、`帮你`/`如果你愿意` 减少）——**但 §4.3 长度归一化后大部分是 artifact**：真·独立于长度的减少只剩 emoji 族 | self_doubt -10 / relationships -8.9 / emotional_comfort -4.4 / work_study -15.6 |
| **task / creative / refusal 类** | **真的学了 GPT 的 offer 腔**，且 per-char 倍率比 boolean 强很多（全局 2.8×–8×） | community_replication +28.9 / creative +15.6 / control_technical +9.7 / disagreement +7.8 / refusal +7.1 |

**⚠️ 长度归一化后（REPORT §4.3）：15/100 pattern 的 Δ 符号翻转**——C 组（"反向漂移"）的招牌 pattern 在 per-char 指标下**反而涨了**（加粗 -35.5pp bool / +1.31 per-1k、反转句 -7.1pp bool / +0.21 per-1k）。所以上表的"情感类更极简 Claude"应降格理解为"**情感类 4.7 回复更短 + 更少 emoji**"，不是"主动弃用 Claude 招式"。真朝 GPT 漂移的证据集中在 task 场景的 offer 腔。

**52% 在本数据集 0 命中**——`patterns.yaml` 是在数据采集前预先写好的一组假设 pattern，其中 52 条在 8,694 条样本里全模型 ≤1%。应理解为**作者预注册假设被实采证伪**（本 pattern 集 precision 约 48%），不是"社区传说普遍有问题"。

校准后社区"变 GPT 味"的叙事：**task / creative 场景的 offer 腔是真 GPT 化**（per-char 倍率显著）；**情感场景的"变冷淡"是真变短，但不是真弃用 Claude 招式**——前者可以说"变 GPT 味"，后者更像"变得更短更 minimalist"。

→ 完整 ABCD 分组 + 长度归一化重审见 [REPORT §4](docs/REPORT.md#4-q3--那它具体变成什么了)

---

## 🎭 SKILL 使用

### 安装（Claude Code）

```bash
git clone https://github.com/sha7doww/gpt-flavor.git
cp -r gpt-flavor/skill/gpt-flavor ~/.claude/skills/
```

新开 session 说 `/gpt-flavor` 或"用 GPT 味写一段：……"即可触发。项目级安装、其他兼容 agent（Cursor / Gemini CLI / OpenCode 等）、symlink 方案、卸载与故障排查详见 [docs/INSTALL.md](docs/INSTALL.md)。

本 skill 遵循开放的 [AgentSkills](https://agentskills.io) 规范，目录就是 skill，无 Python 依赖、无 API key 配置。

### 效果示例

> **用户**: 用 GPT 味回答：我好像没什么真正的朋友。

> 你说"好像没什么真正的朋友"，
> 我听见了。这句话，不是随口一说。
>
> 通常能说出这句话的人，不是社交失败，
> 而是**终于敢承认，那些热闹里没有自己**。
>
> **你不是没有朋友。你只是没有"不用扮演"的关系。**
>
> 很多人不是孤独，是累。
> 累在每段关系里，都在接话、在回应、在照顾别人的情绪。
> 久了，你会分不清——
>
> 到底是没人懂你，
> 还是你没让任何人真正靠近过。
>
> **你可能不是"没朋友"**
>
> 你可能只是：
> - 身边都是**可以吃饭的人**，没有可以沉默的人
> - 一直在做那个**先发消息**的人
> - 习惯了**报喜不报忧**，连崩溃都要挑时间
> - 长大后，关系都变成了**功能性**的
> - 你把"不想麻烦别人"，活成了默认设置
>
> 这不是你不会交朋友。
> 是你太会照顾别人，以至于没人记得照顾你。
>
> **真正的朋友，不是数量问题。是你敢不敢在某个人面前不体面。**
>
> 如果你愿意，可以做一件很小的事：
> - 找一个最近想起过的人，发一句"突然想到你"
> - 写下三个你在他们面前不用装的人，哪怕只有一个
> - 今晚不回任何消息，先陪自己吃顿饭
> - 告诉我一个你"曾经以为是朋友"的人，我陪你看看那段关系
>
> 你现在是**一个朋友都想不到**，
> 还是**想到了，但不敢打扰**？

---

*SKILL 主文件：[skill/gpt-flavor/SKILL.md](skill/gpt-flavor/SKILL.md)；更多参考输出：[examples.md](skill/gpt-flavor/assets/examples.md)；每招背后的数据分析见 [REPORT.md §3](docs/REPORT.md#3-q2--变得像-gpt-吗)。*

---

## 项目结构

```
gpt-flavor/
├── README.md                      # 本文件
├── run_all.sh                     # 一键重跑全分析
├── requirements.txt               # Python 依赖
├── .env.example                   # API 配置模板（OPENAI_API_KEY / OPENAI_BASE_URL）
├── docs/                          # REPORT.md 详细证据 / REPRODUCE.md 复现指南
├── skill/gpt-flavor/              # gpt-flavor persona skill（SKILL.md + assets/examples.md）
├── patterns/patterns.yaml         # 100 条 regex pattern（flat list）
├── seeds/                         # 161 个中文 seed × 15 类 + 3 个 system prompt
├── src/                           # 主 pipeline：collect / analyze / stylo / visualize
├── scripts/                       # 4 个细化脚本（patterns_by_*/classifier_by_*/abcd_breakdown）+ _common.py helper
├── data/                          # 8,694 条原始 JSONL（按 model/condition 分目录）
└── analysis/                      # stats / cosine / classifier / abcd / log_odds × 多切片 + figures/
```

## 复现 / 贡献

`data/` 已随 git 分发（17 MB），clone 后 `bash run_all.sh`（约 5-10 分钟）即可重现所有数字。换模型 / seed / 重新采集见 [docs/REPRODUCE.md](docs/REPRODUCE.md)。

## 许可

- 代码：**MIT**
- 数据（`data/**/*.jsonl`）：**CC-BY-4.0**

## 致谢

- [titanwings/colleague-skill](https://github.com/titanwings/colleague-skill) — persona skill 架构参考
- Monroe, Colaresi, Quinn (2008), *Fightin' Words* — log-odds 方法学（虽然这次发现 top n-gram 大多是排版字符，附录 A.1 有讨论）
- Anthropic 和 OpenAI 的模型团队
