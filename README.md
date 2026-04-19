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

社区说"Opus 4.7 变 GPT 味了"。

**先说大结论**：4.6 vs 4.7 二分类 **87.2%** 准确率（随机基线 50%）——4.7 确实变了。但"变成什么"得拆成社区的三种感受来看：

- **"变冷淡了"** ✅ 真的。emoji 少一半，感叹句、"本质上 / 确实 / 诚实"、"一句话总结 / 先说结论 / 拆解" 真·减少了——**所谓 Claude 味 = 表情 + 权威判断腔 + 总结收束腔**，这三件套被砍了。
- **"像 GPT 了"** ⚠️ 方向对，但**看场景**。4.7 在"温柔倾听者"这类 prompt 下长出 GPT 口头禅：`给你X` **8×**、`如果你愿意` **3.3×**、`帮你` **2.8×**。换到空 prompt 或金句体 prompt，漂移明显弱化——所以"像 GPT"是**情感陪聊场景**下最强。
- **"markdown 砍半"** ❌ 错觉。加粗、反转句（"不是 X，是 Y"）、起手招呼这些**排版骨架**没丢，只是回复 median 从 **333 → 210 字（-37%）**，一屏里自然见得少。

**一句话**：Claude 味不是被均匀淡化——**骨架保留**，**表情 + 判断 + 总结三件套被削弱**，同时在陪聊场景下长出 **GPT offer 腔**。

---

## 我们想回答 3 个问题

### Q1 — 4.7 真变了吗？ ✅ 是

87.2% 是整体数字，按 15 类 seed 拆开看更有意思：

- **2 类 100% 可分**（深度分析 / 存在意义类问题），**13 类 ≥ 80%**
- **但闲聊和落地页复现两类 ≈ 50%**——这两种场景下 4.6 和 4.7 几乎无法区分

表层变化一眼可见：回复 median **333 → 210 字（-37%）**，加粗率 **83% → 48%**，emoji 率 **50% → 22%**。

→ 详细见 [REPORT §2](docs/REPORT.md#2-q1--47-真变了吗)

### Q2 — 变得像 GPT 吗？ ✅ 方向成立，程度温和

![cosine heatmap](analysis/figures/cosine_heatmap.png)

风格距离上，4.7 对 4 个 GPT 全都比 4.6 更近——**方向是朝 GPT 漂移的**。但漂移量级温和，而且**最靠近的是 `gpt-5-chat-latest`（短句 ChatGPT）而不是 `gpt-5.4`（长结构 Thinking）**——所以 4.7 是在向轻量聊天版 ChatGPT 靠拢，不是向 Thinking 模式靠拢。

绝对差距上，4.7 离真正的 GPT 腔还远：

| 招式 | 4.7 | gpt-5.4 | 差距 |
|---|---|---|---|
| 如果你愿意 | 6.4% | **85.0%** | 13× |
| 给你X | 3.9% | **25.9%** | 7× |
| 加粗 | 47.8% | **85.3%** | 2× |
| 帮你 | 23.1% | **43.6%** | 1.9× |

→ 详细见 [REPORT §3](docs/REPORT.md#3-q2--变得像-gpt-吗)

### Q3 — 那它具体变成什么了？ ⚠️ Claude 味被有选择地削弱

4.6 → 4.7 的变化分三块看最清楚。

#### 1. 真·被砍掉的 Claude 味

Claude 招式池里有 **9 条真·减少**（不是"回复短了所以看起来少"——是按每千字频次算都下降了），按功能分组出奇整齐：

| 类型 | 招式 |
|---|---|
| **表情化** | emoji、爱心 |
| **权威判断腔** | 感叹句、"本质上"、"确实"、"诚实" |
| **总结收束腔** | "一句话总结"、"先说结论"、"拆解" |

这才是"Claude 味变淡"的真实构成——**表情 + 权威判断 + 总结收束三件套**被削弱。

#### 2. Claude 骨架其实保留了（看起来少是因为回复变短）

另外 6 条招式出现率大降，但按千字频次看**持平甚至略涨**——属于"招式没丢，只是分母变小"：

| 类型 | 招式 |
|---|---|
| **markdown 结构** | 加粗 |
| **反转句式** | "不是 X，是 Y" |
| **起手招呼** | "如果你"、"接住" |
| **修饰语** | "明确"、"真正的 X" |

这些是 Claude 的**句式骨架**——4.7 没动它们，只是因为回复 median 短了 37%，一条回复里见到的次数自然变少。"markdown 砍半"的视觉效应来自这里，不是招式被主动弃用。

#### 3. 只在"倾听者"prompt 下显形的 GPT offer 腔

GPT offer 招式在 4.7 涨幅显著：

| 招式 | 4.7 相对 4.6 |
|---|---|
| 给你X | **8×** |
| 如果你愿意 | **3.3×** |
| 帮你 | **2.8×** |

但按 system prompt 拆开看，漂移**高度不均匀**——只在"温柔倾听者"这类情感陪聊 prompt 下最强：

| 招式 | 空 prompt | 倾听者 prompt | 金句体 prompt |
|---|---|---|---|
| 如果你愿意 | +0.2 | **+4.2** | +0.2 |
| 帮你 | +3.1 | **+5.4** | 0 |
| 给你X | +3.3 | +1.8 | +2.9 |

*数字为出现率变化，单位百分点。*

情感陪聊 ≈ 倾听者 prompt——**这恰好是社区"变 GPT 味"感受最强的场景**。换成技术或海报 prompt，offer 漂移明显弱化甚至归零。

#### 对 pattern 集的诚实交代

`patterns/patterns.yaml` 的 100 条 regex 是**数据采集前**根据社区印象写好的假设集。其中 52 条在 8,694 条样本里全模型 ≤1%——作者预注册的一半假设**被实采证伪**，pattern 集 precision 约 48%。换句话说：社区传说有一半没根据，但另一半真有。

→ 完整 ABCD 分组、per-category 细分、按 condition 拆分见 [REPORT §4](docs/REPORT.md#4-q3--那它具体变成什么了)

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

- [titanwings/colleague-skill](https://github.com/titanwings/colleague-skill) — persona skill 架构参考
- Monroe, Colaresi, Quinn (2008), *Fightin' Words* — log-odds 方法学（虽然这次发现 top n-gram 大多是排版字符，附录 A.1 有讨论）
- Anthropic 和 OpenAI 的模型团队
