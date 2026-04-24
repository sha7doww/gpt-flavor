<div align="center">

# Claude Opus 4.7 真的变 GPT 味了吗？

> *2026-04-16 Opus 4.7 发布后，中文社区出现"变 GPT 味了"的反馈。<br>
> 我们采集了 8,694 条跨模型中文样本做量化分析。*

by [@sha7doww](https://github.com/sha7doww) · [@xsyshuishui](https://github.com/xsyshuishui) · [@ecnudl](https://github.com/ecnudl)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data: CC-BY-4.0](https://img.shields.io/badge/Data-CC--BY--4.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)

<br>

一份社区使用者的 weekend 兴趣调查，**不是学术研究**。<br>
seed 选材 / pattern 选取 / 数据采集 / 方法论都存在局限（详见 [REPORT §5.2](docs/REPORT.md#52-我们这次实验的局限)），结论仅供参考。<br>
**8,694 条**实采样本（6 模型 × 3 条件 × 161 seeds × 3 runs），用 stylometry 做定量分析。

<br>

[速读 TL;DR](#tldr) · [3 个问题](#我们想回答-3-个问题) · [详细 REPORT](docs/REPORT.md) · [局限](docs/REPORT.md#52-我们这次实验的局限) · [复现](docs/REPRODUCE.md)

</div>

---

> 📢 **2026-04-23 更新**：Anthropic 发布了关于 4.7 发布期部分质量回归的[官方 postmortem](https://www.anthropic.com/engineering/april-23-postmortem)。本报告的数据采集期（2026-04-17~18）与 postmortem 披露的 Claude Code "verbosity instruction" 窗口（2026-04-16 ~ 04-20）**完全重合**——两份结论从不同角度（中文文本 stylometry vs. Anthropic 内部 Claude Code 评测）相互印证。详见 [REPORT §5.3](docs/REPORT.md#53-与-anthropic-官方-postmortem-对照)。

---

## TL;DR

社区说"Opus 4.7 变 GPT 味了"——

- ✅ **4.7 确实变了**（4.6 vs 4.7 二分类 **87.2%** 准确率，按 seed 分组留出；13/15 类明显可分，仅 control_casual / community_replication 近随机）
- ✅ **整体上朝 GPT 漂移**（cosine 对全部 4 个 GPT 模型都 +0.013~0.036）
- ⚠️ **但按场景走了相反方向**：情感 / 关系类 → **更极简 Claude**（markdown -49~-80pp、offer 招式都下降）；task / creative / refusal → **真学了 GPT offer 腔**（`帮你` +4~+29pp、`给你X` 涨）
- 🎯 **最像 ChatGPT 短句体**（`gpt-5-chat-latest`，cos +0.036 最大）而不是"全功能" `gpt-5.4`

**一句话**：4.7 整体是 **压缩版 Claude + 一点 ChatGPT 风味**——但这个均值其实是**双向分裂的合成**：情感场景比 4.6 更冷淡极简（被误读成"GPT 化"），task / creative 场景才真的学了 GPT 的菜单口。

---

## 我们想回答 3 个问题

### Q1 — 4.7 真变了吗？ ✅ 是

二分类器 **87.2% accuracy** 区分 4.6 vs 4.7（`GroupShuffleSplit` 按 seed 留出，保证 test 的 prompt 不在 train 里；随机基线 50%）。按 15 个 seed 类别拆开：

- **2 类 100% 可分**（analysis / meaning_existential），**6 类 ≥ 93%**，12 类 ≥ 80%
- **但 control_casual（闲聊）和 community_replication（落地页复现）两类 ≈ 50%**——这两种场景下 4.6 和 4.7 几乎无法区分

表层变化：回复 median 字数 **333 → 210（-37%）**（mean 1115 → 446，被长任务 seed 严重拉偏，详见 REPORT §2.3）；加粗率 83% → 48%；emoji 率 50% → 22%。

→ 详细见 [REPORT §2](docs/REPORT.md#2-q1--47-真变了吗)

### Q2 — 变得像 GPT 吗？ ✅ 方向成立，程度温和

4.7 对全部 4 个 GPT 的 cosine 都比 4.6 更近：

![cosine heatmap](analysis/figures/cosine_heatmap.png)

| 距离对 | 4.6 | 4.7 | Δ |
|---|---|---|---|
| ↔ `gpt-5.4`（ChatGPT Thinking） | 0.894 | **0.907** | +0.013 |
| ↔ `gpt-5.3-chat`（ChatGPT Instant） | 0.812 | **0.841** | +0.029 |
| ↔ `gpt-5-chat-latest`（旧 ChatGPT） | 0.826 | **0.862** | **+0.036** |
| ↔ `gpt-4o-2024-11-20` | 0.816 | **0.851** | +0.035 |

同模型的 split-half self-cos 噪声底约 0.008–0.024（REPORT §3.1），所以这些 Δ 都是 **1–2× 噪声量级**的温和信号——方向都成立，但 `gpt-5.4` 的 +0.013 在噪声里不算显著，所以 4.7 更像的是**短句 ChatGPT**而不是长结构 Thinking。

绝对水平上 4.7 离 gpt-5.4 仍有很大距离：

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

但 **6% 这个均值是误导**——按 seed 话题拆开看 4.7 − 4.6 在 5 个 key pattern 上的 boolean 变化，呈现**双向分裂**：

| 场景 | 4.7 的实际方向 | 例：`帮你` Δpp |
|---|---|---|
| **情感 / 关系 / 自我类** | **更极简 Claude**（markdown 大砍 −49 ~ −80pp、`帮你` / `如果你愿意` 都减少） | self_doubt −10 / relationships −9 / emotional_comfort −4 / work_study −16 |
| **task / creative / refusal 类** | **真学了 GPT 的 offer 腔**（`帮你` / `给你X` / `如果你愿意` 显著上涨；markdown 在纯技术 / 创作场景保持，refusal / community_replication 仍大砍） | community_replication **+29** / creative **+16** / control_technical +10 / disagreement +8 / refusal +7 |

社区"变 GPT 味"的说法其实混淆了**两种相反的变化**：情感场景"变得更冷淡极简"被误读成"变 GPT"，task / creative 场景才是真的"学了 GPT 的菜单口"。

#### C 组 15 条（boolean 下降）：方向是朝"更极简"，不是朝 GPT

`patterns/patterns.yaml` 的每条 pattern 的 `expect_high_in` 字段都标了 `gpt-5.4 / gpt-5`——pattern 集测的就是"是不是 GPT 风格招式"。C 组 15 条 boolean 下降 = **4.7 比 4.6 更少用这些 GPT 风格招式**，方向是朝"零使用"压缩，不是朝 GPT 漂移。

按跨模型 boolean 拆：

| 类型 | 招式 | 跨模型位置 | 4.7 下降的方向含义 |
|---|---|---|---|
| GPT > Claude（7 条） | 不是X是Y / 本质上 / 一句话总结 / 先说结论 / 如果你 / 明确 / 感叹句 | gpt-5.4 `不是X是Y` 53% / `如果你` 89% / `本质上` 11%、gpt-4o `感叹句` 29% | **4.7 在这些 GPT 重词上离 GPT 更远** |
| 两家都高（1 条） | 加粗 | Claude 83% ≈ gpt-5.4 85% | 通用排版，下降主要是长度效应（§4.3 审计：per-char 反涨 +1.31） |
| Claude > 4 GPT（1 条） | emoji（Claude 4.6 50% vs 4 GPT 最高 22%） | pattern 集里这是异常——作者按 GPT 假设写进 yaml，实采发现 Claude 最高 | 4.7 照样砍到 22% |
| 弱 Claude / 边缘（6 条） | 真正的X / 接住 / 确实 / 诚实 / 拆解 / emoji_heart | 量级低或差距小 | 方向次要 |

所以**这和 B 组（6 条真朝 GPT 漂移的 offer 类）形成局部矛盾**：4.7 在 offer 类学了一点 GPT，在反转 / 强调 / 总结等 GPT 重词上反而更少用。Q2 的 "朝 GPT 漂移" 在 cosine 层面成立（整体短句 char n-gram 风格朝 gpt-5-chat-latest），在 pattern 层面方向分裂。

跨模型 boolean 详细分布见 [REPORT 附录 A.3](docs/REPORT.md#a3-c-组-15-条的跨模型-boolean-分布)。

#### 对 pattern 集的诚实交代

`patterns/patterns.yaml` 的 100 条 regex 是**数据采集前**根据社区印象写好的假设集。其中 **52 条在 8,694 条样本里全模型 ≤ 1%**——像"砍一刀 / 哪把刀 / 多嘴 / 翻译成人话 / 你开口我接 / 实话说"这些短语，更像从单条截图偶发引用放大出的社区印象，没在测试范围内观察到广义规律。作者预注册的一半假设**被实采证伪**，本 pattern 集 precision 约 48%。

→ 完整 ABCD 分组、per-category 细分、按 condition 拆分见 [REPORT §4](docs/REPORT.md#4-q3--那它具体变成什么了)

---

## 项目结构

```
gpt-flavor/
├── README.md                      # 本文件
├── run_all.sh                     # 一键重跑全分析
├── requirements.txt               # Python 依赖
├── .env.example                   # API 配置模板（OPENAI_API_KEY / OPENAI_BASE_URL）
├── docs/                          # REPORT.md 详细证据 / REPRODUCE.md 复现指南
├── patterns/patterns.yaml         # 100 条 regex pattern（flat list）
├── seeds/                         # 161 个中文 seed × 15 类 + 3 个 system prompt
├── src/                           # 主 pipeline：collect / analyze / stylo / visualize
├── scripts/                       # 6 个细化脚本（per-category / per-condition / per-1000-char / ABCD / cosine noise / classifier）+ _common.py helper
├── data/                          # 8,694 条原始 JSONL（按 model/condition 分目录）
└── analysis/                      # stats / cosine / classifier / abcd / log_odds × 多切片 + figures/
```

## 复现 / 贡献

`data/` 已随 git 分发（17 MB），clone 后 `bash run_all.sh`（约 5-10 分钟）即可重现所有数字。换模型 / seed / 重新采集见 [docs/REPRODUCE.md](docs/REPRODUCE.md)。

## 许可

- 代码：**MIT**
- 数据（`data/**/*.jsonl`）：**CC-BY-4.0**

## 致谢

- Monroe, Colaresi, Quinn (2008), *Fightin' Words* — log-odds 方法学（虽然这次发现 top n-gram 大多是排版字符，附录 A.1 有讨论）
- Anthropic 和 OpenAI 的模型团队
