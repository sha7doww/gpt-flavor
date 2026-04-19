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
| **Q3** 那它变成什么了？ | ⚠️ **更短、更少 emoji + 倾听者 prompt 下的 GPT offer 腔** | 回复短 37% + emoji 减半跨 prompt 稳定；按每千字频次算，`给你X` +8×、`如果你愿意` +3.3×、`帮你` +2.8×——这些涨幅主要由 B_listener 条件触发 |

**一句话**：4.7 = **更短、更少 emoji 的 Claude，在倾听者 prompt 下长出一小截 GPT offer 腔**。bold / 反转 / Claude 招牌的每千字密度与 4.6 基本持平——社区感知的"markdown 砍半 / Claude 味变淡"主要是回复变短的视觉效应，不是招式真·弱化。真·朝 GPT 的证据集中在 offer 类短语上（`给你X` / `如果你愿意` / `帮你`），per-char 倍率 2.8× ~ 8×，且高度依赖 system prompt。

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

加粗减半、emoji 减半、Claude 4.6 的招牌长横线分隔几乎消失。

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

但这只是 per-reply boolean rate 的下降；§4.3 用每千字频次重判后，**只有 emoji 族在 per-char 尺度下仍是真减少**，其他 pattern（bold / 不是_是 / 真正的X / 明确 / 接住 等）的 per-char 密度与 4.6 持平甚至略高——它们的 boolean 下降主要反映"4.7 回复短了、一条回复里塞不下那么多招式"，不是招式本身被弃用。

**D 组（2 条 dilution baseline）**：而不是 / 稳词族——GPT 显著高于 Claude，4.7 没有把这些学过来。

### 4.2 按 seed category 拆：boolean 下的双向漂移

把 4.7 − 4.6 在 5 个 key pattern 上按 seed category 拆开看 boolean rate 的变化。**注意本节所有 Δpp 都是 per-reply boolean**；pattern 是否更密集 per-char 需结合 §4.3。per-category 的 per-char 分析未单独做，但从全局的 per-char 对比可以推断：这里看到的大幅 markdown 下降里相当一部分来自"情感场景 4.7 回复变短"，真·独立于长度的方向性证据主要是 emoji。

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

整体上 4.7 = **短句压缩版 Claude + 一点 ChatGPT offer 腔**。拆成两块看最清楚：

- **跨 prompt 稳定的压缩**：回复短 37% + emoji 减半 + 长分隔线消失。但 bold / 反转 / 真正的X / 明确 / 接住 等 Claude 招式的每千字密度与 4.6 持平（§4.3）——"情感场景更极简 Claude"的 boolean 印象主要来自"回复短了塞不下"，不是真·弃用招式
- **持续朝 GPT 的 offer 漂移**：`帮你` / `给你X` / `如果你愿意` boolean +2 ~ 4pp，per-char 倍率 2.8×~8×（§4.3），按 system prompt 拆主要在 B_listener 条件下触发（§4.4）
- **风格归属**：cosine 上最接近 `gpt-5-chat-latest`（短句 ChatGPT），不是 `gpt-5.4`（长结构 Thinking）

ABCD 占比里 B 组（朝 GPT 漂移）只 6%，但**真要看的是这 6% 在哪些场景出现**——情感咨询式 prompt 下 offer 腔漂移最强（§4.4），这恰好对应社区抱怨最强的场景。

### 4.3 长度归一化：按每千字频次重判

boolean per-reply rate（"一条回复有没有命中 ≥1 次"）不控制回复长度。4.7 median 210 字比 4.6 的 333 字短 37%，长回复天然有更多"空间"塞招式，所以短回复模型在 boolean 指标下会系统性偏低。用每个 pattern 的**总命中次数 ÷ 总字符数 × 1000** 得到 **per-1000-char rate**（每千字出现几次），提供不受长度影响的对比。完整数据：[`analysis/patterns_length_normalized.csv`](../analysis/patterns_length_normalized.csv)。

**100 条 pattern 里 15 条的 Δ(4.7 − 4.6) 方向在两个指标下相反**，全部是 "boolean 负 → per-char 正" 这一侧。核心示例：

| pattern | boolean 4.6 → 4.7 | Δ bool (pp) | per-1k 4.6 → 4.7 | Δ per-1k | 方向 |
|---|---|---|---|---|---|
| 加粗 | 83.3% → 47.8% | **−35.5** | 5.84 → 7.15 | **+1.31** | 相反 |
| 不是_是 | 27.3% → 20.2% | −7.1 | 1.01 → 1.22 | +0.21 | 相反 |
| 真正的X | 6.6% → 3.8% | −2.8 | 0.043 → 0.066 | +0.022 | 相反 |
| 明确 | 6.2% → 3.2% | −3.0 | 0.056 → 0.069 | +0.012 | 相反 |
| 如果你 | 19.7% → 18.2% | −1.5 | 0.435 → 0.690 | +0.254 | 相反 |
| 接住 | 3.4% → 1.9% | −1.5 | 0.030 → 0.041 | +0.011 | 相反 |

也就是说：**4.7 的加粗、反转句、"真正的 X"、"明确"、"如果你"、"接住" 这些 Claude 招式在每千字密度上反而略多于 4.6**，只是回复变短了、一条回复里塞不下那么多——boolean 看起来的"暴跌"主要是这个长度效应。

**emoji 两个指标方向一致**：boolean 49.9% → 21.5%（−28.4pp），per-1k 2.45 → 1.33（−1.12）。**emoji 是真·减少**，和"招式密度持平"形成对比。

**B 组 per-1k 倍率远大于 boolean pp**：

| pattern | Δ bool (pp) | bool 倍率 | Δ per-1k | per-1k 倍率 |
|---|---|---|---|---|
| 帮你 | +2.9 | 1.1× | +0.49 | **2.8×** |
| 给你X | +2.7 | 3.3× | +0.077 | **8×** |
| 如果你愿意 | +1.5 | 1.3× | +0.10 | **3.3×** |

从 per-char 看，"4.7 学了 GPT offer 腔"的幅度相当明显——单条回复出现概率只涨了 1-3pp，但每千字里出现次数涨到了 2.8×-8×。

#### 综合 §4.1–§4.3 对 Q3 的回答

- 回复短 37%、emoji 减半——**这两件事在 boolean 和 per-char 两个指标下都成立**，是 4.7 的内生变化
- bold / 反转 / Claude 招式——**boolean 下降，per-char 持平**。只是字数少了，招式没丢
- GPT offer 短语（`帮你` / `给你X` / `如果你愿意`）——**boolean 和 per-char 都涨，per-char 涨得更快**。这是真朝 GPT 的漂移

§4.2 的 per-category 表格（如"self_doubt 加粗 −80pp"）用的也是 boolean per-reply rate，同样受长度效应影响。per-category 的 per-char 分析未单独做，但依全局 per-char 的方向可以推测："情感类 4.7 bold −49~−80pp" 里相当一部分来自回复压缩；真·独立于长度的"情感场景更冷淡"证据主要是 emoji。

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
- **GPT offer 漂移是 prompt-conditional**：`如果你愿意` 的 Δ **几乎全部**来自 B_listener；`帮你` 在 B_listener 放大 (+5.4) 而在 C_poster **完全抑制** (+0.0)；`给你X` 分散但 A_empty 最大。换个 system prompt 这个漂移会减弱或消失

这也解释了社区感知的差异：
- "情感咨询"场景 ≈ B_listener 条件，`如果你愿意` / `帮你` 在那里漂移最大——**社区"变 GPT 味"感受最强的场景**
- "写海报 / 技术问答"场景 ≈ C_poster / A_empty，offer 漂移弱化甚至归零

所以 4.7 朝 GPT 的漂移拆成两种性质：**长度 / emoji 部分是 intrinsic**（跨条件稳定，不依赖 system prompt）；**offer 腔部分是 persona-dependent**（主要靠倾听者 system prompt 触发，换成海报 / 技术 prompt 明显弱化）。读者依据自己常用的 prompt 风格判断哪一半适用于自己。

---

## 5. 讨论 + 局限 + 复现

### 5.1 4.7 为什么"看起来很 GPT"

用户感知的"整体变 GPT"是几种独立信号叠加的错觉：

- **长度塌缩的视觉效应**：回复 median 砍 37%，加粗 boolean rate 砍半——第一眼就"看起来不像 Claude 了"。但每千字密度上 bold / 反转 / Claude 招牌与 4.6 基本持平，所以"招式都没了"是错觉
- **倾听者 prompt 下的 offer 漂移**：`帮你` / `给你X` / `如果你愿意` 这些 GPT 招牌 per-char 倍率 2.8×-8×，而且出现在回复末尾位置特别显眼；情感咨询式对话（恰好最接近 B_listener 条件）命中最多，也是社区抱怨最集中的场景
- **emoji 真减少**：从 50% 砍到 22%，这一项 boolean 和 per-char 方向一致——4.7 确实少用 emoji，这也贡献了"更冷淡"的直观感

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
