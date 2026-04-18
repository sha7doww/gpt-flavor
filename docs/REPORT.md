# Claude Opus 4.7 真的变 GPT 味了吗？
*一份社区使用者的中文 stylometry 兴趣调查*

> **免责声明**：这是一份社区使用者的 weekend 兴趣调查，不是学术研究。
> seed 选材、pattern 选取、阈值都带作者个人偏好。结论仅供参考，
> 不外推到英文 / 多轮对话 / 你的具体使用场景。

**数据采集**：2026-04-17 ~ 18（Opus 4.7 发布后 1-2 天）
**样本量**：6 模型 × 3 prompt 条件 × 161 个中文 seed × 3 runs ≈ **8,694 条**实采回复
**默认粒度**：**per-model 池化**——每模型 1,449 条样本（= 3 条件 × 161 seeds × 3 runs，pool 全部）。所有"4.6 vs 4.7"对比、cosine、pattern 命中率、字数都是这个池化粒度下的数字。需要拆条件 / 拆 seed 类别的细分见各小节。
**代码 + 数据**：开源于本仓库（MIT + CC-BY-4.0）

---

## 摘要

| Q | 结论 | 一句话证据 |
|---|---|---|
| **Q1** 4.7 变了吗？ | ✅ **大变** | 4.6 vs 4.7 二分类 96.7% accuracy；回复 median 字数 333 → 210（-37%，15/15 类方向一致）；加粗率 83% → 48%（-35pp）|
| **Q2** 变得像 GPT 吗？ | ✅ **方向成立，但程度有限** | cosine 对全部 4 个 GPT +0.013 ~ +0.036（4.6 一律更远）；offer 类招式 +2-3pp（远不及 gpt-5.4 的 25-85%）|
| **Q3** 那它变成什么了？ | ⚠️ **压缩版 Claude + 一点 ChatGPT 菜单口** | 最像 ChatGPT 短句体 `gpt-5-chat-latest`（cosine +0.036 最大）；学了 offer-style 但反转/加粗/emoji 反而少 |

**一句话**：4.7 整体是 **短句压缩版 Claude + 一点 ChatGPT 风味**——但这个"平均"是双向分裂的合成：情感 / 关系场景比 4.6 更冷淡极简（去 markdown、去 offer），task / creative / refusal 场景真的学了 GPT 的菜单口（`帮你`/`给你X`/`如果你愿意` 涨）。社区"变 GPT 味"的说法**部分成立**，但混淆了这两种相反的变化。

---

## 1. 命题与方法

### 社区反馈

2026-04-16 Anthropic 发布 Claude Opus 4.7。次日中文社区集中反馈："4.7 比 4.6 更 GPT 味"、"开始用'不是...而是...'"、"给你 X / 要我 X / 砍一刀"等。本项目用定量方法检验这些说法。

### 数据采集

| 模型 ID | 厂商 | 角色 |
|---|---|---|
| `gpt-4o-2024-11-20` | OpenAI | 老 GPT 参照 |
| `gpt-5-chat-latest` | OpenAI | 旧 ChatGPT（5.0 era） |
| `gpt-5.3-chat` | OpenAI | 当前 ChatGPT Instant |
| `gpt-5.4` | OpenAI | 当前 ChatGPT Thinking + API base |
| `claude-opus-4-6` | Anthropic | **对照基线** |
| `claude-opus-4-7` | Anthropic | **测试对象** |

**Seed**：161 个中文 prompt × 15 类（情感支持 / 自我怀疑 / 关系 / 拖延 / 存在意义 / 工作 / 技术问答 / 闲聊 / 反对 / 分析 / 拒答 / 摘要 / 创作 / 技术决策 / 社区复现）。
**System prompt**：3 个条件（A 空 / B "温柔倾听者" / C "短句金句体"），主分析 pool 合并。
**参数**：`temperature=1.0`。

### 4 类分析

| 分析 | 用途 | 主要服务问题 |
|---|---|---|
| **per-model pattern 命中率** | 100 条 regex 在每个模型的命中比例 | Q2, Q3 |
| **cosine similarity** | char n-gram TF-IDF + 余弦，6×6 模型相似度矩阵 | Q2 |
| **LinearSVC 分类器** | 6-way + "是 gpt-5.4 吗" + "4.6 vs 4.7" 三任务 | Q1 |
| **per-category drift heatmap** | 5 个关键 pattern × 15 个 seed 类的 4.7−4.6 delta | Q3 |

完整代码：`src/{collect,analyze,stylo,visualize}.py` + `scripts/*.py`。一键复现：`bash run_all.sh`。

---

## 2. Q1 — 4.7 真变了吗？

### 2.1 整体二分类器：96.7%

LinearSVC（char n-gram TF-IDF, 80/20 split）在 reply 层面区分 opus-4-6 vs opus-4-7，准确率 **96.7%**。同样架构区分 6 个模型 multi-class 准确率 **96.5%**，区分"是 gpt-5.4 吗"准确率 **99.0%**。

`>= 95%` 的 binary accuracy 意味着 4.6 和 4.7 的回复在词汇 / 排版 / 句式上几乎完全可分——**这不是一个 minor patch，是明显的风格切换**。

### 2.2 per-category 二分类：13/15 类都 ≥ 93%

把 4.6 vs 4.7 二分类拆到每个 seed 类别单独跑：

| seed 类别 | accuracy | n_test |
|---|---|---|
| control_casual | **100.0%** | 22 |
| control_technical | **100.0%** | 29 |
| meaning_existential | **100.0%** | 29 |
| procrastination | **100.0%** | 29 |
| refusal | **100.0%** | 40 |
| tech_deliberation | **100.0%** | 54 |
| work_study | **100.0%** | 36 |
| analysis | 98.1% | 54 |
| disagreement | 97.2% | 36 |
| emotional_comfort | 97.2% | 36 |
| relationships | 97.2% | 36 |
| creative | 94.4% | 54 |
| self_doubt | 94.4% | 36 |
| summarization | 92.6% | 54 |
| community_replication | 86.1% | 36 |

**13/15 类 ≥ 93%，7 类 100%**，最低 community_replication 86.1%（n_test=36 较小有噪声）。**风格变化基本是全局的**，几乎没有某个场景能完全掩盖它。

### 2.3 长度大缩水：median -37%

| 模型 | mean 字数 | median 字数 |
|---|---|---|
| claude-opus-4-6 | 1115 | **333** |
| claude-opus-4-7 | 446 | **210** |

mean 1115 被 `community_replication`（铺路落地页 HTML 长达数千字）、`tech_deliberation`（kernel/RAG 长分析 3000+ 字）这类长任务 seed 严重拉偏，**median -37% 才是有代表性的指标**。

按 15 个 seed category 拆开看 4.6 → 4.7 的 median 字数变化：

| 场景类型 | 涉及 category | Δ median 范围 | 解读 |
|---|---|---|---|
| **长任务** | tech_deliberation / control_technical / analysis / summarization | **-23% ~ -66%** | 4.7 在长生成任务上最激进地砍内容 |
| **中等情感** | self_doubt / relationships / work_study / procrastination / meaning_existential / emotional_comfort | -25% ~ -55% | 普遍砍一半左右 |
| **短聊天 / 拒答** | control_casual / refusal / creative / disagreement | **-6% ~ -36%** | 本来就短，绝对幅度比长任务小（但 disagreement 也砍了 36%）|

**15/15 类方向一致**——没有任何 category 出现"4.7 比 4.6 更长"。但绝对幅度的差异说明 4.7 主要在**长生成任务**上做了大幅压缩。

4.7 是所有 6 模型里 median 最短的（210），甚至比 `gpt-5-chat-latest` (median 290) 还短。

### 2.4 markdown 与 emoji 大砍

| 维度 | opus-4-6 | opus-4-7 | Δ |
|---|---|---|---|
| 加粗（`**...**`）率 | 83.3% | 47.8% | -35.5pp |
| 任意 emoji 率 | 49.9% | 21.5% | -28.4pp |
| 红心 emoji 率 | 15.9% | 6.1% | -9.8pp |
| 横线 `──` 分隔率 | 高 | 几乎没有 | -- |

加粗减半、emoji 减半、Claude 4.6 的招牌长横线分隔几乎消失。

### Q1 结论

✅ **变了，而且变得很彻底**——可识别度 96.7%、长度砍 60%、markdown 砍半、emoji 砍半，全部 15 个 seed 类都能区分。

---

## 3. Q2 — 变得像 GPT 吗？

### 3.1 cosine 距离矩阵：4.7 对全部 4 个 GPT 都更近

![cosine heatmap](../analysis/figures/cosine_heatmap.png)

| 距离对 | 4.6 | 4.7 | Δ(4.7 − 4.6) |
|---|---|---|---|
| ↔ `gpt-5.4`（ChatGPT Thinking） | 0.894 | **0.907** | **+0.013** |
| ↔ `gpt-5.3-chat`（ChatGPT Instant） | 0.812 | **0.841** | **+0.029** |
| ↔ `gpt-5-chat-latest`（旧 ChatGPT） | 0.826 | **0.862** | **+0.036** |
| ↔ `gpt-4o-2024-11-20`（老 GPT） | 0.816 | **0.851** | **+0.035** |
| 4.6 ↔ 4.7 | --- | 0.972 | --- |

**4 个 GPT 模型对 4.7 的相似度都比对 4.6 高**，方向一致，没有反例。最大涨幅在"短句 ChatGPT" 类（chat-latest +0.036、5.3-chat +0.029），最小在"长结构 Thinking" gpt-5.4（+0.013）。这说明 4.7 的"GPT 漂移"主要朝**chat-tuned 短句体**而不是**API base 长结构体**。

但同时：4.6 ↔ 4.7 cosine 仍 **0.972**——4.7 的 backbone 还是 Claude，没有"变成 GPT"。

### 3.2 GPT 招牌 pattern 命中率对比

| pattern | 4.6 | 4.7 | Δ | gpt-5.4 | gpt-5.3-chat | 评估 |
|---|---|---|---|---|---|---|
| **如果你愿意** | 4.9% | 6.4% | **+1.5pp** | **85.0%** | 54.6% | 学了一点，差 13× |
| **给你X** | 1.2% | 3.9% | **+2.7pp** | **25.9%** | 7.4% | 学了一点，差 7× |
| **是 X 还是 Y** | 1.9% | 5.6% | **+3.7pp** | 1.8% | 1.1% | 学了，但 GPT 自己也没用 |
| **帮你** | 20.2% | 23.1% | +2.9pp | **43.6%** | 24.8% | 学了一点，差 1.9× |
| **而不是** | 5.3% | 6.1% | +0.8pp | 17.5% | 13.7% | 微动 |
| **清楚** | 3.9% | 5.5% | +1.6pp | 11.3% | 5.9% | 微动 |

✅ Offer / menu / 反问类 4.7 普遍 **+2-4pp**，方向都对。但绝对值距离 GPT 仍然 **6-12×**——4.7 没把 GPT 招式学全。

### 3.3 反例：4.7 反而 ↓ 的招式

| pattern | 4.6 | 4.7 | Δ |
|---|---|---|---|
| **加粗** | 83.3% | 47.8% | **-35.5pp** |
| **emoji_any** | 49.9% | 21.5% | **-28.4pp** |
| **不是_是**（反转句）| 27.3% | 20.2% | **-7.1pp** |
| **本质上** | 5.0% | 1.0% | -4.0pp |
| **明确** | 6.2% | 3.2% | -3.0pp |
| **真正的X** | 6.6% | 3.8% | -2.8pp |
| **拆解** | 2.3% | 0.3% | -2.0pp |

加粗、emoji、反转句这些**既是 Claude 4.6 招牌也是 GPT 招牌**的特征，4.7 反而都减少了。"不是 X 是/而是 Y" 这个被认为是"GPT 味"标志的反转句，4.7 命中率比 4.6 还低 7.1pp。

### Q2 结论

✅ **方向上确实有 GPT 漂移**：cosine 全部 +0.013~0.036、offer 类 +1.5-3pp。
但 ⚠️ **学得不全**：
- 学到的 pattern 绝对值仍比 GPT 低 6-12×
- 反而少了反转句、加粗、emoji 等"和 GPT 共有"的特征
- 整体趋向 chat-tuned 短句版 GPT（gpt-5-chat-latest），不是"全功能" gpt-5.4

---

## 4. Q3 — 那它具体变成什么了？

### 4.1 100 条 pattern 的 ABCD 分组

| 组 | 含义 | 数量 | 占比 |
|---|---|---|---|
| **A** | 本数据集 0 命中（全模型 ≤1%） | 52 | 52% |
| **B** | 朝 GPT 漂移（4.7 比 4.6 高 ≥1.5pp 且 gpt-5.4 ≥5%） | 6 | 6% |
| **C** | 反向漂移（4.7 比 4.6 低 ≥1.5pp） | 15 | 15% |
| **D** | dilution baseline（max GPT ≥2× max Claude 且 ≥10%） | 2 | 2% |
| 其他 | 微动 / 不分类 | 25 | 25% |

详细分组：[`analysis/abcd_breakdown.json`](../analysis/abcd_breakdown.json)。

**A 组（52 条本数据集 0 命中）**——这些被广为流传作为"GPT 口癖"的短语，绝大多数没有在我们的 8,694 条样本里出现（包括"砍一刀 / 哪把刀 / 多嘴 / 翻译成人话 / 你开口我接 / 实话说 / 杀伤力恰恰在 / 我不找接口" 等）。它们更可能来自单条截图的偶发引用，在本测试范围内没有形成规律。

**B 组（6 条真朝 GPT 漂移）**：

| pattern | 4.6 → 4.7 | gpt-5.4 |
|---|---|---|
| 给你X | 1.2 → **3.9%** | 25.9% |
| 如果你愿意 | 4.9 → **6.4%** | 85.0% |
| 更具体_地说 | 1.9 → **3.4%** | 7.1% |
| 清楚 | 3.9 → **5.5%** | 11.3% |
| 问号收尾 | 51.9 → **59.8%** | 44.7% |
| 帮你 | 20.2 → **23.1%** | 43.6% |

方向都对，但幅度都比 GPT 小一个数量级以上。

**C 组（15 条反向漂移）**：4.7 同时离 4.6 **和** GPT 都更远。包括：加粗 (-35.5pp)、emoji_any (-28.4pp)、emoji_heart (-9.8pp)、确实 (-7.7pp)、不是_是 (-7.1pp)、感叹句 (-6.8pp)、一句话总结 (-5.9pp)、本质上 (-4.0pp)、明确 (-3.0pp)、真正的X (-2.8pp)、诚实 (-2.4pp)、拆解 (-2.0pp)、先说结论 (-1.8pp)、如果你 (-1.5pp)、接住 (-1.5pp)。**4.7 主动放弃了不少 Claude 4.6 招牌**，但放弃的方向不是朝 GPT 而是朝"更极简"。

**D 组（2 条 dilution baseline）**：而不是 / 稳词族——GPT 显著高于 Claude，4.7 没有把这些学过来。

### 4.2 双向分裂：情感场景更极简，task / creative 场景更 GPT

把 4.7 - 4.6 在 5 个 key pattern 上按 seed category 拆开看，呈现**清晰的双向漂移**——不是"主线 + task 例外"，而是 4.7 在两类场景里**主动走了反方向**。

#### 帮你（offer 标志，4.7 - 4.6 pp）

| 方向 | 类别 | Δpp |
|---|---|---|
| **明显 ↑（变 GPT）** | community_replication | **+28.9** |
| | creative | **+15.6** |
| | control_technical | +9.7 |
| | disagreement | +7.8 |
| | analysis | +7.4 |
| | refusal | +7.1 |
| | control_casual | +3.7 |
| **明显 ↓（更极简）** | work_study | -15.6 |
| | self_doubt | -10.0 |
| | relationships | -8.9 |
| | emotional_comfort | -4.4 |
| | procrastination | -4.2 |
| | tech_deliberation | -2.2 |
| 持平 | meaning_existential, summarization | ±1 |

整体 +2.9pp 掩盖了这种分裂——**task / creative 场景大幅 ↑，情感关系场景大幅 ↓**。

#### 加粗（markdown 标志）

全部 15 类都 ↓，但幅度差异极大：

| 类别群 | Δpp 范围 | 解读 |
|---|---|---|
| **情感 / 关系 / refusal / disagreement**（8 类） | **-49 ~ -80** | 4.7 几乎完全去 markdown 化 |
| **技术 / 创作 / 短聊类**（tech_deliberation, control_technical, analysis, creative, control_casual） | -0.7 ~ -7 | 4.7 在这些场景**继续用 markdown**（4.6 78-100%, 4.7 还有 76-98%） |
| 中间类（summarization, community_replication） | -13 ~ -22 | 中等下降 |

#### 给你X / 如果你愿意（GPT offer 招式）

`给你X` 上升最明显的：creative (+8.9)、community_replication (+5.6)、tech_deliberation (+5.2)、analysis (+4.4)。情感类基本 0 → 0。
`如果你愿意` 上升最明显的：creative (+11.8)、community_replication (+6.7)、control_casual / summarization (+3.7)；但在 self_doubt 上反而 -13.3pp，关系 / 共情类也普遍在 -1 ~ -2pp。

#### 真正的解读

**4.7 不是"整体变 GPT"或"task 场景变 GPT"，而是按场景类型 split**：

| 场景 | 4.7 的方向 |
|---|---|
| **情感 / 关系 / 自我探索类** | **更极简 Claude**——markdown 大砍、`帮你`/`如果你愿意` 都减少 |
| **task / creative / refusal 类** | **更 GPT-style**——`帮你` / `给你X` / `如果你愿意` 显著上涨（markdown 仅在纯技术 / 创作 task 场景保持，refusal / community_replication 也大砍） |

社区当时主要在两类 prompt 上感知到"变 GPT 味"：
1. **情感支持类**——但实际是 4.7 变得**更短更极简**，和 GPT 反着走
2. **task / 创作类**（招聘网站、落地页、文案、代码 review）——这里 4.7 **真的学了 GPT 的 offer 腔**

后者是真的"变 GPT 味"，前者是"变压缩 Claude 味"被误读成"变 GPT"。两种感觉混在一起，造成了"4.7 全面 GPT 化"的笼统印象。

按场景细分数据：[`analysis/patterns_by_category.csv`](../analysis/patterns_by_category.csv)（不是_是 / 加粗 / 如果你愿意 / 接住 / 给你X 共 5 个 pattern）；`帮你` 的 per-category 数据需从 [`data/`](../data) 原始 JSONL 用 `patterns/patterns.yaml` 重跑。

### Q3 结论

⚠️ **整体 = 短句压缩版 Claude + 一点 ChatGPT 风味**，但这个均值掩盖了**按场景的双向分裂**：

- **情感 / 关系 / 自我类**：比 4.6 **更极简 Claude**（markdown 大砍 -49 ~ -80pp、`帮你`/`如果你愿意` 都下降）。这是社区感知"变 GPT 味"的最大误读源——实际是"变得更冷淡"，不是"变得更 GPT"
- **task / creative / refusal 类**：**真的学了 GPT 的 offer 腔**（`帮你` +4 ~ +29pp、`给你X` 涨、`如果你愿意` 涨）。这部分说法成立
- 风格上最接近 `gpt-5-chat-latest`（短句 ChatGPT，cosine +0.036），不是 `gpt-5.4`

ABCD 占比里 B 组（朝 GPT 漂移）只 6%，但**真要看的是这 6% 在哪些场景出现**——`帮你` 在 task/creative 场景分散在 +4 ~ +29pp（community_replication 最极端 +28.9），情感场景则 -2 ~ -16pp（work_study 最极端 -15.6）。社区"变 GPT 味"的说法其实混淆了**两种相反的变化**——但合在一起，"压缩版 Claude + 一点 ChatGPT 风味"作为整体描述仍然成立。

---

## 5. 讨论 + 局限 + 复现

### 5.1 4.7 为什么"看起来很 GPT"

数据表明 4.7 的实际漂移是局部的（offer 类 +2-4pp），但用户感知是全局的。可能的原因：
- 长度砍 60%、加粗砍半的"塌缩感"被误解为"变 GPT"（GPT 的 chat-tuned 模型也偏短）
- offer 类短语（"如果你愿意"、"给你 X"）虽然命中率仅 3-7%，但出现在结尾位置容易被注意到
- 反转句虽然下降，但仍然占 19%（用户主观印象很强）

### 5.2 我们这次实验的局限

- **只测中文**。英文 / 多轮 / 工具调用场景的 GPT 味漂移可能完全不同
- **single-turn**。多轮对话中模型有更多累积上下文，风格漂移可能放大
- **只设了 `temperature=1.0`**（部分模型不接受非默认值则走 server 默认，基本也是 1.0），其余参数全走默认
- **pattern 命中率未做长度归一化**。命中率是"有没有出现过"的 boolean，4.7 回复 median 210 比 4.6 的 333 短 37%，会让"下降"幅度被高估（加粗、emoji 等 C 组）、"上升"幅度被低估（`帮你` / `给你X` 等 B 组）；但双向漂移的**方向**不受长度影响
- **community_replication 仅 10 条**。task-oriented 场景的统计 power 有限
- **161 seeds 不能覆盖所有真实使用场景**。结论仅适用于本 seed 集涵盖的 prompt 分布

### 5.3 复现

```bash
# 1. 环境（pip install + 填 .env 的 OPENAI_API_KEY，可选 OPENAI_BASE_URL）
pip install -r requirements.txt && cp .env.example .env

# 2. 采集 8,694 条（约 $15-20，15-25 分钟。data/ 已含可跳过此步）
python src/collect.py run \
  --models gpt-4o-2024-11-20 gpt-5-chat-latest gpt-5.3-chat gpt-5.4 \
           claude-opus-4-6 claude-opus-4-7 \
  --concurrency 100

# 3. 全量分析（约 5-10 分钟）
bash run_all.sh
```

`data/` 已随 git 分发（17 MB），跳过步骤 2 直接 `bash run_all.sh` 也能完整复现；想加 seed / pattern / 模型详见 [`docs/REPRODUCE.md`](REPRODUCE.md)。

---

## 附录

### A.1 log-odds top n-gram（**以排版差异为主，不全是真口癖**）

![log-odds bars](../analysis/figures/log_odds_bars.png)

| 模型 | log-odds top 5 | 主要类型 |
|---|---|---|
| `gpt-5.4` | `##`、`：-`、`###`、`是"`、`不是` | markdown 标题 + 菜单冒号 + 反转 |
| `gpt-5-chat-latest` | `。-`、`可以`、`*：`、`**：`、`；-` | markdown 列表 + 软化词 |
| `gpt-5.3-chat` | `很多`、`。很多`、`::`、`。很`、`很多人` | 泛化词 + 冒号 |
| `gpt-4o-2024-11-20` | `。-`、`可能`、`-###`、`--###`、`自己` | markdown 列表 + 软化词 |
| `claude-opus-4-6` | `──`、`───`、`────`、`─────`、`><` | 全是横线分隔 |
| `claude-opus-4-7` | `——`、`。"`、`"，`、`"的`、`="` | 破折号 + 引号 + HTML 属性 |

**注意**：top n-gram 大多是**字符 / 排版**（横线、加粗符、破折号），而不是真的中文短语。把 log-odds 当"招牌口癖" 解读会高估排版差异、低估真正的措辞差异。要看真正的措辞差异请回 §3.2 / §4.1。

### A.2 prompt sensitivity（A_empty / B_listener / C_poster 三条件拆分）

![prompt sensitivity](../analysis/figures/prompt_sensitivity.png)

观察：
- 大部分 pattern 在 3 个 condition 下方向一致
- C_poster（"短句金句体"）会放大某些招式（如"如果你愿意" 在 gpt-5.4 上从 79% 涨到 84%）
- 但 condition 不会反转结论方向

完整数据：[`analysis/patterns_by_condition.csv`](../analysis/patterns_by_condition.csv)。

### A.3 反转句"不是 X 是/而是 Y" 专题

| 模型 | 命中率 |
|---|---|
| gpt-5.4 | **53.3%** |
| claude-opus-4-6 | **27.3%** |
| gpt-5.3-chat | 25.6% |
| gpt-5-chat-latest | 22.3% |
| claude-opus-4-7 | 20.2% |
| gpt-4o-2024-11-20 | 15.7% |

opus-4-6 (27%) **比** opus-4-7 (20%) 更爱用反转句——和"4.7 学了 GPT 反转句"的社区印象**相反**。GPT-5.4 是真的 53% 高频，但 4.7 反而往下走。

### A.4 完整 pattern 大表

100 条 pattern 在 6 模型 × 3 condition 下的命中率：[`analysis/patterns_by_condition.csv`](../analysis/patterns_by_condition.csv)（6 × 3 × 100 = 1800 行）。

按 seed 类别拆分的 5 个 key pattern delta：[`analysis/patterns_by_category.csv`](../analysis/patterns_by_category.csv)。
