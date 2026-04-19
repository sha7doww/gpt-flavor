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
| **Q1** 4.7 变了吗？ | ✅ **大变** | 4.6 vs 4.7 二分类 87.2% accuracy（`GroupShuffleSplit` 按 seed 留出，baseline 50%）；回复 median 字数 333 → 210（-37%，15/15 类方向一致）；emoji 率 50% → 22% |
| **Q2** 变得像 GPT 吗？ | ✅ **方向成立，幅度温和** | cosine 对全部 4 个 GPT 都更近 (+0.013 ~ +0.036)，最接近 `gpt-5-chat-latest`（短句 ChatGPT），不是 `gpt-5.4`（长结构 Thinking）|
| **Q3** 那它变成什么了？ | ⚠️ **按 seed 话题双向分裂** | 情感 / 关系 / 自我探索类 seeds 下 4.7 **更极简 Claude**（加粗 self_doubt −80 / meaning_existential −79 / procrastination −74pp、`帮你` −10 ~ −4、`如果你愿意` self_doubt −13pp）；task / creative / refusal 类 seeds 下才 **真学 GPT offer 腔**（`帮你` community_replication **+29** / creative **+16** / control_technical +10 / refusal +7pp、`给你X` creative **+9** / community_replication +6pp） |

**一句话**：4.7 整体是 **压缩版 Claude + 一点 ChatGPT 风味**——但这个均值其实是**双向分裂**的合成：情感场景比 4.6 更冷淡极简（被误读成"GPT 化"），task / creative 场景才真的学了 GPT 的菜单口。

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

### 2.1 整体二分类器：87.2%

LinearSVC（char n-gram TF-IDF, 80/20 split，**`GroupShuffleSplit` 按 seed 分组留出**）在 reply 层面区分 opus-4-6 vs opus-4-7，准确率 **87.2%**（随机基线 50%）。同样架构下 6 模型 multi-class 准确率 **96.4%**，"是 gpt-5.4 吗" 99.0%。

> 分组策略：每个 seed 有 18 行（3 条件 × 3 runs × 2 模型）。如果直接用 `train_test_split(stratify=y)`，同一 prompt 的回复会在 train/test 两侧同时出现，分类器相当于"见过同一问题的另两次回答"再猜作者——会让准确率虚高近 10pp。`GroupShuffleSplit(groups=seed)` 保证测试集的 prompt 在训练里完全没见过，是对"未见过的 prompt"的真实留出泛化性能。

87% 的留出 binary accuracy 意味着 4.6 和 4.7 的回复在词汇 / 排版 / 句式上**对未见过的 prompt 仍明显可分**——**这不是一个 minor patch，是明显的风格切换**。

### 2.2 per-category 二分类：2 类接近随机，其余仍明显可分

把 4.6 vs 4.7 二分类拆到每个 seed 类别单独跑（同样 `GroupShuffleSplit`）：

| seed 类别 | accuracy | n_test |
|---|---|---|
| analysis | **100.0%** | 54 |
| meaning_existential | **100.0%** | 36 |
| self_doubt | 97.2% | 36 |
| work_study | 97.2% | 36 |
| tech_deliberation | 96.3% | 54 |
| procrastination | 94.4% | 36 |
| refusal | 90.7% | 54 |
| emotional_comfort | 88.9% | 36 |
| relationships | 86.1% | 36 |
| control_technical | 83.3% | 36 |
| creative | 81.5% | 54 |
| disagreement | 80.6% | 36 |
| summarization | 70.4% | 54 |
| community_replication | **52.8%** | 36 |
| control_casual | **50.0%** | 36 |

**2 类 100%、6 类 ≥ 93%、11 类 ≥ 80%**，但 **community_replication（社区复现短语）和 control_casual（闲聊短句）接近随机**——这两类里 4.6 和 4.7 的输出几乎不可分。这是一个有内容的 null：4.7 在"哈哈"一类短闲聊、和"帮我落地 landing page"一类 community_replication 场景下，输出风格与 4.6 高度重合；风格切换主要体现在较长的情感 / 分析 / 技术类生成里。

**这个 null 反过来解释了社区叙事**：用户感知的"4.7 变 GPT 味"几乎只能来自较长的生成场景（情感 / task / refusal）；用 4.7 写"哈哈好的"或赶 landing page 的人，在风格层面其实没机会感觉到任何切换——这部分用户也不会发推抱怨。社区抱怨样本因此天然偏向"长生成"那一侧，比真实总体更 dramatic。

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

加粗减半、emoji 减半、Claude 4.6 的招牌长横线分隔几乎消失。注：加粗在 Claude / GPT 两家都高（Claude 4.6 83% / gpt-5.4 85%），它的 boolean 下降主要是长度效应（§4.3 审计：per-char 反涨 +1.31）；emoji 和横线分隔才是跨模型看 Claude 显著独有的招牌。

### Q1 结论

✅ **变了，而且变得明显**——按 seed 留出的可识别度 87.2%（baseline 50%）、长度砍 60%、markdown 砍半、emoji 砍半，13/15 seed 类明显可分，仅短闲聊 / community_replication 两类接近随机。

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

**4 个 GPT 模型对 4.7 的相似度都比对 4.6 高**，方向一致，没有反例。最大涨幅在"短句 ChatGPT" 类（chat-latest +0.036、5.3-chat +0.029），最小在"长结构 Thinking" gpt-5.4（+0.013）。同时 4.6 ↔ 4.7 cosine 仍 **0.972**——远高于任何跨模型 cos，4.7 的 backbone 仍然是 Claude。

#### 噪声底：Δ 放在什么尺度下看

纯 Δ 值本身没有参照。对每个模型做 split-half bootstrap（随机把 1,449 条 reply 分两半各自 pool 后算 cos，重复 30 次，共用同一 TF-IDF 特征空间），得到**同一模型跟自己的 cos 分布**作为 noise floor：

| 模型 | self-cos median | 噪声 (1−median) |
|---|---|---|
| claude-opus-4-6 | 0.988 | 0.012 |
| claude-opus-4-7 | 0.971 | 0.029 |
| gpt-5.4 | 0.980 | 0.020 |
| gpt-5.3-chat | 0.979 | 0.021 |
| gpt-5-chat-latest | 0.972 | 0.028 |
| gpt-4o-2024-11-20 | 0.966 | 0.034 |

完整 bootstrap 分布：[`analysis/cosine_self_stability.json`](../analysis/cosine_self_stability.json)。half-pool 噪声是 full-pool 的 √2 倍，full-pool 真实噪声约为上表的 1/√2 ≈ **0.008-0.024**。

按这个尺度重判 4.6 → 4.7 的 Δ：

| 目标 | Δ | full-pool noise 参照 | 解读 |
|---|---|---|---|
| gpt-5-chat-latest | +0.036 | ≈ 0.020 | ~1.8× noise，温和但真实的信号 |
| gpt-4o | +0.035 | ≈ 0.024 | ~1.5× noise，温和信号 |
| gpt-5.3-chat | +0.029 | ≈ 0.020 | ~1.5× noise，borderline |
| gpt-5.4 | +0.013 | ≈ 0.020 | Δ 低于 noise，本数据集内**不显著** |

**结论**：Q2 "cos 对全部 4 个 GPT 都更近" 在方向上成立；但幅度应读作温和信号，最大的 +0.036 只是噪声底的约 2 倍；`gpt-5.4` 的 +0.013 在噪声里不显著，所以 "4.7 最像 gpt-5-chat-latest 短句体"这个判断是可靠的，"4.7 更像 gpt-5.4 长结构 Thinking"不成立。

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

| pattern | 4.6 | 4.7 | Δ | gpt-5.4 | 跨模型位置 |
|---|---|---|---|---|---|
| **加粗** | 83.3% | 47.8% | **-35.5pp** | 85.3% | Claude / GPT 都高，通用 |
| **emoji_any** | 49.9% | 21.5% | **-28.4pp** | 0.3% | Claude 4.6 **> 4 GPT**（pattern 集里的分布异常） |
| **不是_是**（反转句）| 27.3% | 20.2% | **-7.1pp** | **53.3%** | GPT 用得更多 |
| **本质上** | 5.0% | 1.0% | -4.0pp | **11.0%** | GPT 用得更多 |
| **明确** | 6.2% | 3.2% | -3.0pp | **15.6%** | GPT 用得更多 |
| **真正的X** | 6.6% | 3.8% | -2.8pp | 6.2% | 持平 |
| **拆解** | 2.3% | 0.3% | -2.0pp | 1.1% | 弱 Claude |

这 7 条下降的招式按跨模型位置分三类：

- **emoji（Claude 4.6 > 4 GPT）**：Claude 4.6 50% 高于 4 GPT 最高 22%——在 pattern 集里这是 Claude 4.6 > 所有 GPT 的唯一一条，4.7 砍到 GPT 的量级（22%）
- **加粗（两家都高）**：Claude 4.6 83% / gpt-5.4 85%——通用招式，boolean 下降主要是长度效应（§4.3 审计：per-char 反涨到 +1.31）
- **反转句 / 本质上 / 明确（GPT > Claude）**：这 3 条 4.7 下降意味着 4.7 在这些招式上**离 gpt-5.4 更远**——和 Q2 "朝 GPT 漂移" 的总基调**局部矛盾**

换句话说，pattern 命中率层面，4.7 只在 `如果你愿意` / `给你X` / `帮你` 等 **offer 类**上学了 GPT；在反转句 / 强调副词等 **GPT 重词**上反而更少用。§Q2 的"朝 GPT 漂移"主要靠 cosine（char n-gram 整体短句风格），而不是靠招式命中率——这也解释了为什么 cosine 最接近的是 `gpt-5-chat-latest`（短句 ChatGPT）而不是 `gpt-5.4`（长结构 Thinking，正好是重反转句的那个模型）。

### Q2 结论

✅ **方向上确实有 GPT 漂移**：cosine 全部 +0.013~0.036（均为温和信号，最大的 +0.036 只是噪声底的 ~1.8×；+0.013 vs gpt-5.4 不显著），offer 类 +1.5-3pp。
但 ⚠️ **学得不全 / 且方向混杂**：
- offer 类 pattern 学到了但绝对值仍比 GPT 低 6-12×
- 反转句 / 本质上 / 明确等 **GPT 重词**反而下降——pattern-level 上 4.7 在这些招式上**离 GPT 更远**
- emoji 大降，但 emoji 在 pattern 集里是 Claude 4.6 > 所有 GPT 的特例，它的下降和"朝 GPT"是不同方向
- 整体趋向 **chat-tuned 短句版 GPT**（`gpt-5-chat-latest`），不是"全功能" `gpt-5.4`——cosine 靠的是整体短句排版风格，不是具体招式

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

**A 组（52 条本数据集 0 命中）**——`patterns/patterns.yaml` 是**在数据采集之前**基于社区零散截图写好的，等价于一次轻量 pre-registration。52 条没命中的 pattern（"砍一刀 / 哪把刀 / 多嘴 / 翻译成人话 / 你开口我接 / 实话说 / 杀伤力恰恰在 / 我不找接口" 等）应被理解为**作者预先提出的假设被实采数据证伪**——即这些短语**没有在 8,694 条样本里形成广义规律**。这不是"社区传说有问题"（那些短语大概率存在于真实聊天中），而是"本研究的 100 条 pattern 集里一半过于宽泛 / 过于零散，没有覆盖到实际高频信号"。**52% 本身不是关于 GPT 味的 finding，是关于本 pattern 集 precision 的 finding**。

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

**C 组（15 条 boolean rate 下降 ≥ 1.5pp）**：包括加粗 (-35.5pp)、emoji_any (-28.4pp)、emoji_heart (-9.8pp)、确实 (-7.7pp)、不是_是 (-7.1pp)、感叹句 (-6.8pp)、一句话总结 (-5.9pp)、本质上 (-4.0pp)、明确 (-3.0pp)、真正的X (-2.8pp)、诚实 (-2.4pp)、拆解 (-2.0pp)、先说结论 (-1.8pp)、如果你 (-1.5pp)、接住 (-1.5pp)。

pattern 集每条 `expect_high_in` 都指向 `gpt-5.4 / gpt-5`，测的是"是不是 GPT 风格招式"。C 组 boolean 下降 = **4.7 比 4.6 更少用这些 GPT 风格招式**，方向朝"零使用"压缩，**不是朝 GPT 漂移**。

按跨模型位置拆（见附录 A.3）：
- **7 条 GPT > Claude**：不是X是Y / 本质上 / 一句话总结 / 先说结论 / 如果你 / 明确 / 感叹句（`反转句` gpt-5.4 53% / `如果你` gpt-5.4 89% / `感叹句` gpt-4o 29% / `本质上` gpt-5.4 11% / `一句话总结` gpt-5.4 11%）→ 4.7 在这些招式上**离 GPT 更远**
- **1 条两家都高**：加粗（Claude 83% ≈ gpt-5.4 85%）→ boolean 下降主要是长度效应（§4.3 审计：per-char 反涨 +1.31）
- **1 条 Claude > 4 GPT（pattern 集异常）**：emoji（Claude 4.6 50% vs 4 GPT 最高 22%——作者按 GPT 假设写进 yaml，实采发现 Claude 反而最高），4.7 照样砍到 22%
- **6 条弱 Claude 或边缘**：真正的X / 接住 / 确实 / 诚实 / 拆解 / emoji_heart——量级低或差距小，方向次要

§4.3 做了长度归一化审计：9 条 per-char 仍降（emoji 族 + 权威判断 + 总结），6 条 per-char 翻转（加粗 / 反转 / 招呼 / 修饰属于长度驱动假象）——这是 boolean 结论的 caveat 脚注，不是新叙事。

**D 组（2 条 dilution baseline）**：而不是 / 稳词族——GPT 显著高于 Claude，4.7 没有把这些学过来。

### 4.2 按 seed category 拆：boolean 下的双向漂移

把 4.7 − 4.6 在 5 个 key pattern 上按 seed category 拆开看 boolean rate 的变化。**注意本节所有 Δpp 都是 per-reply boolean**；§4.3 会用每千字频次重判这些招式的方向是否源自密度变化还是回复变短。per-category 的 per-char 分析未单独做，但从全局视角可以推断：加粗 per-category 的暴跌（如 self_doubt 下的 −80pp）相当一部分来自回复压缩（§4.3 全局 per-char 反涨），而 emoji / 判断腔 / 总结腔的下降在 per-char 下仍成立。

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

#### 场景层面的模式

boolean rate 下，5 个 key pattern 的按场景漂移方向如下：

| 场景 | 4.7 相对 4.6 的方向（boolean） |
|---|---|
| **情感 / 关系 / 自我探索类** | 回复更短、markdown 大降、`帮你`/`如果你愿意` 也降；但 per-char 密度基本持平，所以主要是长度效应 |
| **task / creative / refusal 类** | `帮你` / `给你X` / `如果你愿意` 显著上涨（markdown 仅在技术 / 创作场景保持） |

社区当时主要在两类 prompt 上感知到"变 GPT 味"：
1. **情感支持类**——实际是 4.7 回复变短 + emoji 减半，视觉上"塌缩"被误读成"变 GPT"
2. **task / 创作类**（招聘、落地页、文案、代码 review）——这里 4.7 **真的学了 GPT 的 offer 腔**

两种感觉混在一起，造成了"4.7 全面 GPT 化"的笼统印象。按场景细分数据：[`analysis/patterns_by_category.csv`](../analysis/patterns_by_category.csv)（不是_是 / 加粗 / 如果你愿意 / 接住 / 给你X 共 5 个 pattern）；`帮你` 的 per-category 数据需从 [`data/`](../data) 原始 JSONL 用 `patterns/patterns.yaml` 重跑。

### Q3 结论

4.7 相对 4.6 的核心观测（按重要性排序）：

- **按 seed 话题双向分裂**（§4.2，核心发现）：情感 / 关系 / 自我类 seed 下 4.7 **更极简 Claude**（加粗 self_doubt −80pp / procrastination −74pp、`帮你` 情感类 −4 ~ −16pp、`如果你愿意` self_doubt −13pp）；task / creative / refusal 类 seed 下 4.7 **真学 GPT offer 腔**（`帮你` community_replication +29pp / creative +16pp、`给你X` creative +9pp）。**社区"变 GPT 味"混淆了两种相反变化**——情感场景的"变冷淡"被误读成"变 GPT"，task 场景才是真的学 GPT 菜单口
- **长度压缩**：median 333 → 210（−37%），跨 prompt 稳定——这是 4.7 最 intrinsic 的变化
- **C 组 15 条方向：4.7 同时离 4.6 和 GPT 都更远，不是朝 GPT 漂移**：pattern 集每条 `expect_high_in` 都指向 GPT，所以 C 组 boolean 下降 = "4.7 少用 GPT 风格招式"。按跨模型位置拆：**7 条 GPT > Claude**（反转句 / 本质上 / 一句话总结 / 如果你 / 明确 / 感叹句 / 先说结论）→ 4.7 在这些 GPT 重词上**离 GPT 更远**；加粗两家都高（长度效应为主）；emoji 是 pattern 集里唯一 Claude > 4 GPT 的异常（4.6 50% vs 4 GPT 最高 22%），4.7 砍到 22%

cosine 上最接近 `gpt-5-chat-latest`（短句 ChatGPT），不是 `gpt-5.4`（长结构 Thinking）。

§4.3 用每千字频次做了**长度归一化审计**（§4.3 是 caveat 不是新发现）：15 条 C 组招式里 6 条在 per-char 下方向翻转（加粗 / 反转 / 招呼 / 修饰——它们的 boolean 下降完全是长度效应），9 条 per-char 也下降（emoji / 权威判断 / 总结收束——但后 7 条跨模型看 GPT 也用）。§4.4 按 system prompt 拆发现 B_listener 下 offer 放大最显著，但拆到 seed topic 层看这个放大主要来自 task seed——不是情感陪聊放大 offer。

### 4.3 数据审计：每千字频次对 §4.1–§4.2 boolean 结论的交叉验证

本节是**审计章节**（不是新发现）——§4.1 / §4.2 用的 boolean per-reply rate 不控制回复长度，4.7 median 210 字比 4.6 的 333 字短 37%，长回复天然有更多"空间"塞招式。用 per-1000-char rate（每千字出现几次）做一遍交叉验证，看哪些 boolean 结论经得起长度归一化。完整数据：[`analysis/patterns_length_normalized.csv`](../analysis/patterns_length_normalized.csv)。

**审计读法**：本报告主线叙事（§4.2 per-seed-category 双向分裂）建立在 boolean 指标之上，这对应社区感知（用户看到的是"一条回复里有没有 emoji / 加粗"，不是"每千字多少 emoji"）。本节目的是在 boolean 结论里分出**哪些是长度伪影**，而不是声明 per-char 才是"真相"。

把 15 条 C 组招式按 per-char 方向拆，**9 条 per-char 也下降，6 条 per-char 持平或涨**——后 6 条的 boolean 下降主要是长度效应：

#### 密度真降的 9 条（4.7 主动削减）

| 功能类型 | 招式 | Δ boolean pp | Δ per-1k |
|---|---|---|---|
| **表情化** | emoji_any | −28.4 | **−1.12** |
| | emoji_heart | −9.8 | −0.09 |
| **权威判断腔** | 感叹句 | −6.8 | −0.017 |
| | 确实 | −7.7 | −0.030 |
| | 本质上 | −4.0 | −0.022 |
| | 诚实 | −2.4 | −0.012 |
| **总结收束腔** | 一句话总结 | −5.9 | −0.022 |
| | 先说结论 | −1.8 | −0.012 |
| | 拆解 | −2.0 | −0.014 |

这 9 条在每千字频次上**真·减少**——即使控制了回复长度，4.7 还是比 4.6 更少用它们。按功能归类出奇整齐：**表情化 + 权威判断腔 + 总结收束腔**。但注意：这 9 条里只有 emoji 族在 pattern 集里是 Claude 4.6 > 所有 GPT 的（见 A.3），其余 7 条 GPT 也用甚至用得更多（`感叹句` gpt-4o 29% / `本质上` gpt-5.4 11% / `一句话总结` gpt-5.4 11% 等）。所以"9 条 per-char 下降"的真实含义是 **4.7 在整体压缩回复时，也压缩了这些共享招式**。

#### 长度驱动的 6 条（boolean 降，per-char 持平或涨）

| 功能类型 | 招式 | Δ boolean pp | Δ per-1k |
|---|---|---|---|
| **markdown 结构** | 加粗 | −35.5 | **+1.31** |
| **反转句式** | 不是_是 | −7.1 | +0.21 |
| **起手招呼** | 如果你 | −1.5 | +0.25 |
| | 接住 | −1.5 | +0.011 |
| **修饰语** | 真正的 X | −2.8 | +0.022 |
| | 明确 | −3.0 | +0.012 |

这 6 条 boolean 下的"暴跌"里藏着一个反直觉事实：4.7 每千字用加粗的次数反而比 4.6 多（5.84 → 7.15），反转句"不是 X，是 Y" 每千字也多（1.01 → 1.22），"如果你..." 起手式每千字更是涨了近一倍（0.44 → 0.69）。所以这些招式不是 4.7 主动放弃的；只因回复 median 从 333 字砍到 210 字，一条回复里自然塞不下那么多次——boolean 看起来的"砍半"主要是这个视觉效应。

跨模型看：`不是 X，是 Y` 在 gpt-5.4 高达 53%（Claude 4.6 只 27%，4.7 20%），`如果你` 在 GPT 阵营是 34–89%（Claude 4.6 只 20%），`明确` 在 GPT 是 10–16%（Claude 4.6 只 6%）——**反转 / 起手招呼 / 修饰语在 GPT 阵营命中率更高**（详见附录 A.3）。所以这 6 条 per-char 持平只说明 4.7 没主动弃用这些通用修辞。

#### 小结：C 组削减的方向是"朝零压缩"，不是朝 GPT

- **per-char 审计分两组**：9 条 per-char 仍降（emoji / 权威判断 / 总结），6 条 per-char 翻转（加粗 / 反转 / 招呼 / 修饰属于长度驱动）
- **跨模型位置分三组**（见 A.3）：
  - 7 条 GPT > Claude（`反转句` gpt-5.4 53% vs 4.6 27%、`如果你` gpt-5.4 89% vs 4.6 20%、`感叹句` gpt-4o 29% vs 4.6 14%、`本质上` gpt-5.4 11% vs 4.6 5%、`一句话总结` gpt-5.4 11% vs 4.6 9%、`先说结论` gpt-5.4 4% vs 4.6 2%、`明确` gpt-5.4 16% vs 4.6 6%）→ 4.7 在这些上**离 GPT 更远**
  - 1 条两家都高（加粗：Claude 83% ≈ gpt-5.4 85%）→ 长度效应
  - 1 条 Claude > 4 GPT（emoji：4.6 50% vs 4 GPT 最高 22%）→ 4.7 照样砍到 22%

Q3 用"朝 GPT 漂移"概括 4.7 方向**不准确**——B 组 6 条 offer 类是真朝 GPT（且主要在 task / creative / refusal seed 下显形），C 组 15 条大部分反而离 GPT 更远。

#### B 组 per-char 倍率远大于 boolean

朝 GPT offer 漂移的 3 条，per-char 倍率远超 boolean 单点变化：

| pattern | Δ bool (pp) | bool 倍率 | Δ per-1k | per-1k 倍率 |
|---|---|---|---|---|
| 帮你 | +2.9 | 1.1× | +0.49 | **2.8×** |
| 给你X | +2.7 | 3.3× | +0.077 | **8×** |
| 如果你愿意 | +1.5 | 1.3× | +0.10 | **3.3×** |

单条回复的出现概率只涨 1-3pp，但每千字出现次数涨到 2.8×-8×——从 per-char 视角看，这是一个幅度相当明显的真·漂移。

#### 对 Q3 的综合回答

4.7 相对 4.6 的变化分三条独立轨道：
- **长度压缩**：median 短 37%，这是内生风格变化（跨 prompt 稳定）
- **C 组 15 条方向朝"更极简"压缩，不是朝 GPT**：pattern 集 `expect_high_in` 都指向 GPT，C 组 boolean 下降 = "4.7 少用 GPT 风格招式"。跨模型拆：7 条 GPT > Claude（4.7 在这些 GPT 重词上离 GPT 更远）；加粗两家都高（长度效应）；emoji 是 pattern 集里 Claude > 4 GPT 的异常，4.7 照样砍到 22%（见 A.3）
- **GPT offer 腔长出**：`帮你` / `给你X` / `如果你愿意` per-char 2.8×-8×，但主要来自 task / creative / refusal seed（§4.2）；情感类 seed 下反而抑制

§4.2 的 per-category 表格（如 "self_doubt 加粗 -80pp"）用的也是 boolean per-reply rate，同样受长度效应影响。per-category 的 per-char 未单独做；按全局 per-char 方向推测：per-category 里"markdown 暴跌"相当一部分来自回复压缩，而真正独立于长度的 per-category "变冷淡"证据主要是 emoji。

### 4.4 prompt-conditional 还是 intrinsic？按 condition 拆

主分析把 3 个 system prompt 条件（A_empty = 空 / B_listener = 温柔倾听者 / C_poster = 短句金句体）pool 到一起。拆开看 4.7 − 4.6 在 5 个 key pattern 上按 condition 的 boolean Δpp：

| pattern | A_empty Δ | B_listener Δ | C_poster Δ | 模式 |
|---|---|---|---|---|
| 加粗 | −37.0 | −29.2 | −40.2 | **三条件一致大砍** |
| emoji_any | −25.9 | −34.6 | −24.9 | **三条件一致大砍** |
| 帮你 | +3.1 | **+5.4** | +0.0 | 倾听者放大，金句体抑制 |
| 给你X | +3.3 | +1.8 | +2.9 | 三条件都有 |
| 如果你愿意 | +0.2 | **+4.2** | +0.2 | **几乎全部来自 B_listener** |

完整 per-condition 数据：[`analysis/stats_by_bucket.json`](../analysis/stats_by_bucket.json)。

**解读**：
- **长度相关的变化（加粗、emoji）跨 prompt 保持 −25 ~ −40pp**——4.7 "回复变短、去 emoji" 是**模型内生**风格，不论给什么 system prompt 都是同方向同量级
- **GPT offer 漂移是 prompt-conditional**：`如果你愿意` 的 Δ **几乎全部**来自 B_listener；`帮你` 在 B_listener 放大 (+5.4) 而在 C_poster **完全抑制** (+0.0)；`给你X` 分散但 A_empty 最大

**但 B_listener ≠ 情感陪聊场景**——这是关键点。B_listener 是 system prompt（"你是温柔倾听者"persona），与 seed topic（用户问什么）**正交**。按 §4.2 的 per-seed-category 数据，`帮你` 在情感类 seed 下是 −4 ~ −16pp（下降），在 task / creative seed 下才是 +7 ~ +29pp（上涨）。所以 B_listener 下的 +5.4pp `帮你` 放大其实主要来自 **task seed 碰上倾听者 persona**（"温柔倾听者"人设讨论"改简历 / 写落地页"时，最容易长出"我帮你列 3 个建议"这种 GPT 菜单口）——和"情感咨询"场景正好相反。

所以 4.7 朝 GPT 的漂移拆成两种性质：**长度 / emoji 部分是 intrinsic**（跨 prompt、跨 seed 都稳定）；**offer 腔部分是 persona-dependent 且 topic-dependent**（倾听者 persona + task seed 组合下最强，换成情感 seed 或换成金句体 prompt 明显弱化甚至反转）。社区感知最强的"变 GPT 味"场景实际是 **task / creative seed**（改简历、文案、代码 review），而非"情感咨询"。

---

## 5. 讨论 + 局限 + 复现

### 5.1 4.7 为什么"看起来很 GPT"

用户感知的"整体变 GPT"是几种独立信号叠加：

- **回复塌缩的视觉效应**：median 字数砍 37%，加粗 boolean rate 砍半——第一眼"看起来不像 4.6 了"。但加粗 / 反转句 / 起手招呼 / 修饰语这些招式**在 GPT 阵营也常用**（反转句 / 起手招呼 gpt-5.4 甚至更多，见 A.3），per-char 看其实没降，只是字数少了塞不下那么多
- **emoji 真的少了**：Claude 4.6 emoji 50% → 4.7 22%。在 pattern 集里这是 Claude 4.6 命中率高于 4 个 GPT 的唯一一条（GPT 最高 22%），4.7 砍到了 GPT 的量级——社区说的"变冷淡"里最直观的部分就是这个
- **task / creative / refusal seed 下真学了 GPT offer 腔**：`帮你` 在 community_replication +29pp / creative +16pp、`给你X` 在 creative +9pp——改简历、写落地页、技术 review 这些场景下 4.7 确实开始说"我给你列 3 个建议"、"如果你愿意，我可以帮你"，这才是"变 GPT 味"的直接证据
- **情感 / 关系类 seed 下反而更极简**：markdown −49 ~ −80pp、offer 招式也下降——社区说"4.7 变冷淡"里一部分其实是这个。pattern 集里的 GPT 招式命中率 4.7 其实更低了，但直觉上容易被误读成"变 GPT"

这四件事叠加，再加上 cosine 方向上确实朝 chat-tuned GPT 偏移，造成了"4.7 全面 GPT 化"的笼统印象。实际拆开看是"emoji 没了 + task 场景学 GPT 菜单口 + 情感场景反而变极简 + 回复整体变短"四件独立的事合成的感觉。

### 5.2 我们这次实验的局限

- **只测中文**。英文 / 多轮 / 工具调用场景的 GPT 味漂移可能完全不同
- **single-turn**。多轮对话中模型有更多累积上下文，风格漂移可能放大
- **只设了 `temperature=1.0`**（部分模型不接受非默认值则走 server 默认，基本也是 1.0），其余参数全走默认
- **boolean 与 per-char 指标差异较大**。主报告表格使用 boolean per-reply rate（"一条回复是否出现过该 pattern"）。因 4.7 回复 median 短 37%，boolean 指标会低估短回复模型的招式密度；长度归一化后（§4.3 + [`analysis/patterns_length_normalized.csv`](../analysis/patterns_length_normalized.csv)），15/100 pattern 的 Δ(4.7−4.6) 方向相反——bold / 反转句 / 真正的 X / 明确 / 接住 的每千字密度基本持平，emoji 是唯一在两个指标下都真减少的 C 组招牌。读者引用"markdown 砍半"一类数字时请连带 §4.3 的 per-char 对比
- **community_replication 仅 10 条**。task-oriented 场景的统计 power 有限
- **161 seeds 不能覆盖所有真实使用场景**。结论仅适用于本 seed 集涵盖的 prompt 分布
- **cosine Δ 幅度在噪声范围内**。split-half bootstrap 显示各模型 self-cos noise 约 0.008-0.024（见 §3.1），所有 4 个 GPT 的 +0.013 ~ +0.036 Δ 都只是 1-2× 噪声量级，方向成立但幅度应读作温和信号；其中 +0.013（vs gpt-5.4）低于噪声底，本数据集不显著
- **cosine 未做长度控制**。4.7 回复 median 短 37%，短文本 char n-gram 分布更集中，会让"4.7 对任何模型的 cos 都天然偏高一点"——这是 §3.1 noise floor 之外的独立 confound，未单独校正。严格论证"4.7 在控制长度后仍更像 GPT"需要把所有 reply 截到同一长度（例如 median 200 字）后重算 cos matrix；本报告未做

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

### A.3 C 组 15 条的跨模型 boolean 分布

C 组 15 条在 6 个模型上的 boolean 分布拆开看，按跨模型位置可以分三类。下面给出几条代表性 pattern 的详细分布。

#### 反转句"不是 X 是/而是 Y"

| 模型 | 命中率 |
|---|---|
| gpt-5.4 | **53.3%** |
| claude-opus-4-6 | 27.3% |
| gpt-5.3-chat | 25.6% |
| gpt-5-chat-latest | 22.3% |
| claude-opus-4-7 | 20.2% |
| gpt-4o-2024-11-20 | 15.7% |

gpt-5.4 的 53% 比 Claude 两代都高；opus-4-6 (27%) 比 opus-4-7 (20%) 更爱用反转句——"4.7 学了 GPT 反转句"的社区印象和数据方向相反。

#### 起手招呼"如果你..."

| 模型 | 命中率 |
|---|---|
| gpt-5.4 | **89.0%** |
| gpt-5.3-chat | 63.4% |
| gpt-5-chat-latest | 38.6% |
| gpt-4o-2024-11-20 | 34.2% |
| claude-opus-4-6 | 19.7% |
| claude-opus-4-7 | 18.1% |

两个 Claude 都在 20% 以下，4 个 GPT 全部在 34% 以上，gpt-5.4 高达 89%——"如果你..."这个起手招呼在 GPT 阵营命中率远高于 Claude。

#### 修饰语"明确"

| 模型 | 命中率 |
|---|---|
| gpt-5.4 | **15.6%** |
| gpt-4o-2024-11-20 | 10.2% |
| gpt-5-chat-latest | 6.3% |
| claude-opus-4-6 | 6.2% |
| gpt-5.3-chat | 4.8% |
| claude-opus-4-7 | 3.2% |

gpt-5.4 和 gpt-4o 显著高于 Claude 4.6。

#### emoji

| 模型 | emoji 率 | 红心 emoji 率 |
|---|---|---|
| **claude-opus-4-6** | **49.9%** | **15.9%** |
| claude-opus-4-7 | 21.5% | 6.1% |
| gpt-4o-2024-11-20 | 21.9% | 8.1% |
| gpt-5-chat-latest | 17.8% | 6.6% |
| gpt-5.3-chat | 0.8% | 0.3% |
| gpt-5.4 | 0.3% | 0.0% |

Claude 4.6 的 50% 高于 4 个 GPT 最高 22%——在 C 组 15 条里这是 Claude > 所有 GPT 的唯一例外。其他 14 条 C 组 pattern 的分布要么是 GPT > Claude，要么两家都高。

#### C 组 15 条跨模型位置汇总

| 类别 | 数量 | 招式 | 跨模型特征 |
|---|---|---|---|
| GPT > Claude | 7 | 不是_是 / 如果你 / 明确 / 本质上 / 一句话总结 / 先说结论 / 感叹句 | gpt-5.4（或 gpt-4o）显著高于 Claude 4.6 |
| 两家都高 | 1 | 加粗 | Claude 83% ≈ gpt-5.4 85%（通用排版） |
| Claude > 4 GPT | 1 | emoji | Claude 4.6 50% vs 4 GPT 最高 22%（pattern 集里此分布是例外） |
| 弱 Claude / 边缘 | 6 | 真正的X / 接住 / 确实 / 诚实 / 拆解 / emoji_heart | 量级低或差距小 |

### A.4 完整 pattern 大表

100 条 pattern 在 6 模型 × 3 condition 下的命中率：[`analysis/patterns_by_condition.csv`](../analysis/patterns_by_condition.csv)（6 × 3 × 100 = 1800 行）。

按 seed 类别拆分的 5 个 key pattern delta：[`analysis/patterns_by_category.csv`](../analysis/patterns_by_category.csv)。
