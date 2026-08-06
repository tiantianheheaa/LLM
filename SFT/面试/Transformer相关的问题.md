下面按“**搜广推算法面试 + SFT Qwen3 项目**”的视角，整理 Transformer 相关高频考点，并给出可直接用于面试回答的答案。

---

# 一、整体架构类考点

## 1. Qwen3 这类大模型本质上是什么结构？

**面试答案：**

Qwen3 属于 Decoder-only Transformer 架构，本质上是一个自回归语言模型。它的训练目标是根据前面的 token 预测下一个 token，也就是最大化条件概率：

\[
P(x_1, x_2, ..., x_n)=\prod_{t=1}^{n}P(x_t|x_{<t})
\]

在 SFT 阶段，模型仍然是做 next token prediction，只不过训练数据从通用语料变成了带有 instruction、input、response 的监督样本。对于每个样本，通常只对 assistant response 部分计算 loss，而 prompt/user 部分只作为上下文参与 attention，不参与监督损失。

---

## 2. Encoder-only、Decoder-only、Encoder-Decoder 有什么区别？为什么 Qwen3 用 Decoder-only？

**面试答案：**

Transformer 架构大致可以分为三类：

| 架构 | 代表模型 | 特点 | 适合任务 |
|---|---|---|---|
| Encoder-only | BERT | 双向注意力，理解能力强 | 分类、匹配、检索、NER |
| Decoder-only | GPT、LLaMA、Qwen | 单向 causal attention，自回归生成 | 对话、生成、指令跟随 |
| Encoder-Decoder | T5、BART | Encoder 理解输入，Decoder 生成输出 | 翻译、摘要、Seq2Seq |

Qwen3 用 Decoder-only 主要是因为它面向通用生成和对话场景。Decoder-only 架构天然适合自回归生成，每一步根据历史上下文预测下一个 token。同时它可以统一处理问答、摘要、推理、代码、推荐文案生成等任务，只需要组织不同的 prompt 格式即可。

---

## 3. Transformer 的核心模块有哪些？

**面试答案：**

一个典型的 Decoder-only Transformer block 主要包括：

1. **Token Embedding**
2. **位置编码**，如 RoPE
3. **Masked Multi-Head Self-Attention**
4. **残差连接**
5. **归一化层**，常见是 RMSNorm 或 LayerNorm
6. **前馈网络 FFN**，大模型中常用 SwiGLU
7. **输出 LM Head**，预测下一个 token 的概率分布

Decoder-only 模型中，每一层都会通过 causal mask 保证当前位置只能看到自己及之前的 token，不能看到未来 token。

---

# 二、Self-Attention 高频考点

## 4. Self-Attention 的计算过程是什么？

**面试答案：**

Self-Attention 的核心是让序列中每个 token 根据与其他 token 的相关性动态聚合上下文信息。

给定输入 hidden states \(X\)，通过三个线性变换得到：

\[
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
\]

然后计算注意力分数：

\[
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
\]

其中：

- \(Q\)：当前 token 想查询什么信息；
- \(K\)：每个 token 提供什么索引特征；
- \(V\)：每个 token 实际被聚合的内容；
- \(\sqrt{d_k}\)：防止点积过大导致 softmax 饱和。

在 Decoder-only 模型中，还会加 causal mask，让当前位置不能关注未来位置。

---

## 5. 为什么 attention 要除以 \(\sqrt{d_k}\)？

**面试答案：**

因为当 \(d_k\) 较大时，\(Q\) 和 \(K\) 的点积方差会随维度增大而增大。如果不缩放，attention logits 会变得很大，softmax 容易进入饱和区间，导致梯度很小，训练不稳定。

除以 \(\sqrt{d_k}\) 后，可以让 logits 的尺度相对稳定，使 softmax 的分布不会过早变得极端，有利于训练。

---

## 6. Multi-Head Attention 为什么要多头？

**面试答案：**

多头注意力的核心作用是让模型从不同子空间学习不同类型的关系。

单头 attention 只能学习一种相似度模式，而多头可以并行学习多种关系，例如：

- 某些 head 关注局部上下文；
- 某些 head 关注长距离依赖；
- 某些 head 关注语法结构；
- 某些 head 关注实体、属性、数值关系；
- 在推荐/广告场景里，可能有 head 关注用户意图，有 head 关注商品属性，有 head 关注行为序列。

最终多个 head 的结果 concat 后再经过线性变换融合。

---

## 7. MHA、MQA、GQA 有什么区别？为什么大模型常用 GQA？

**面试答案：**

三者主要区别在于 Query、Key、Value 的 head 数量是否一致。

| 类型 | Q head | K/V head | 特点 |
|---|---:|---:|---|
| MHA | 多个 | 多个 | 表达能力强，但 KV cache 大 |
| MQA | 多个 | 1 组 | 推理快，KV cache 小，但表达能力可能下降 |
| GQA | 多个 | 少数组 | 折中方案，兼顾效果和推理效率 |

在自回归推理时，每生成一个 token 都要读取历史 token 的 K/V cache。长上下文场景下，KV cache 会占用大量显存。GQA 通过让多组 Query 共享较少的 K/V head，显著降低 KV cache 显存和带宽开销，同时相比 MQA 保留更强表达能力。

如果面试官问到 Qwen3，你可以说：  
**Qwen 系列大模型通常会使用现代 Decoder-only 架构设计，其中常见优化包括 RoPE、RMSNorm、SwiGLU、GQA/MQA 等，用于提升长上下文建模和推理效率。具体是否使用 GQA，要以所使用的 Qwen3 模型配置为准。**

---

# 三、Causal Mask 与训练目标

## 8. Decoder-only 为什么需要 causal mask？

**面试答案：**

因为 Decoder-only 语言模型是自回归生成模型，训练和推理都要求当前位置只能依赖历史 token，不能看到未来 token。

如果训练时不加 causal mask，模型预测第 \(t\) 个 token 时可以看到 \(t+1, t+2\) 等未来 token，会造成信息泄露。这样训练 loss 会很低，但推理时无法使用未来信息，训练和推理分布不一致，模型生成能力会崩。

所以 causal mask 保证：

\[
x_t \text{ only attends to } x_{\leq t}
\]

---

## 9. SFT 训练时，prompt 部分要不要计算 loss？

**面试答案：**

通常不计算 prompt 部分的 loss，只计算 assistant response 部分的 loss。

原因是 SFT 的目标是让模型在给定 instruction/user query 的情况下生成期望回答，而不是让模型去复现用户问题或系统提示词。如果对 prompt 部分也计算 loss，模型会浪费容量去学习“如何生成用户输入”，甚至影响指令跟随能力。

实际实现中，一般会构造 label：

- prompt/user/system 部分 label 置为 `-100`；
- assistant response 部分保留真实 token id；
- loss 函数忽略 `-100` 位置。

---

## 10. SFT 和预训练的 loss 有什么区别？

**面试答案：**

二者底层目标都是 next token prediction，即交叉熵损失：

\[
L=-\sum_t \log P(y_t|y_{<t},x)
\]

区别主要在数据和 loss mask：

1. **预训练**  
   使用大规模通用文本，通常对所有 token 计算 next token loss。

2. **SFT**  
   使用指令-回答格式数据，输入包含 system、user、assistant 等角色信息，通常只对 assistant 回答部分计算 loss。

3. **目标差异**  
   预训练让模型学习语言、知识和通用模式；SFT 让模型学习指令跟随、业务风格、回答格式和领域知识表达方式。

---

# 四、位置编码与长上下文

## 11. Transformer 为什么需要位置编码？

**面试答案：**

Self-Attention 本身对输入 token 的顺序不敏感。也就是说，如果不加位置信息，attention 只看到一组 token 的集合，而不知道它们的先后顺序。

但语言模型必须知道顺序，例如：

- “用户点击了 A 后购买 B”
- “用户购买 B 后点击 A”

这两个序列语义不同。因此 Transformer 需要位置编码来注入 token 的位置信息。

---

## 12. RoPE 是什么？相比绝对位置编码有什么优势？

**面试答案：**

RoPE，全称 Rotary Position Embedding，是旋转位置编码。它不是简单地把位置向量加到 token embedding 上，而是在 attention 的 \(Q\)、\(K\) 上引入与位置相关的旋转变换。

RoPE 的一个核心特点是：  
**两个 token 的 attention 分数会自然包含相对位置信息。**

相比绝对位置编码，RoPE 的优势包括：

1. 更适合自回归语言模型；
2. 能更好建模相对距离；
3. 对长上下文外推更友好；
4. 不需要为每个位置学习独立的位置向量。

在大模型中，RoPE 是非常常见的位置编码方案。

---

## 13. 长上下文训练或推理时，Transformer 的瓶颈在哪里？

**面试答案：**

主要瓶颈有两个：

### 1. Attention 计算复杂度

标准 self-attention 的复杂度是：

\[
O(n^2d)
\]

其中 \(n\) 是序列长度，\(d\) 是 hidden size。序列长度翻倍，attention 矩阵大小大约变成 4 倍。

### 2. KV Cache 显存开销

自回归推理时需要缓存每一层历史 token 的 K/V。上下文越长，KV cache 越大，显存和带宽压力越明显。

因此大模型通常会结合：

- GQA/MQA；
- FlashAttention；
- RoPE scaling；
- Sliding Window Attention；
- KV cache 量化；
- 分块 prefill；
- paged attention；

来优化长上下文效率。

---

# 五、归一化、残差与稳定训练

## 14. 为什么 Transformer 要用残差连接？

**面试答案：**

残差连接的作用是改善深层网络的梯度传播，缓解梯度消失和训练退化问题。

在 Transformer 中，每个 attention 和 FFN 模块外面都有残差连接：

\[
x_{l+1}=x_l+F(x_l)
\]

这样即使某一层学得不好，模型也可以近似保留上一层表示。对于几十层甚至上百层的大模型，残差连接是稳定训练的关键。

---

## 15. Pre-LN 和 Post-LN 有什么区别？大模型为什么常用 Pre-LN？

**面试答案：**

Post-LN 是原始 Transformer 的设计：

\[
x_{l+1}=LN(x_l+F(x_l))
\]

Pre-LN 是先归一化再进入子模块：

\[
x_{l+1}=x_l+F(LN(x_l))
\]

大模型更常用 Pre-LN，因为它的梯度传播路径更稳定。残差路径上没有 LayerNorm 阻断，梯度可以更直接地从深层传到浅层，因此训练更稳定。

不过 Post-LN 在某些设置下最终效果可能更好，但训练深层模型时更难稳定。

---

## 16. RMSNorm 和 LayerNorm 有什么区别？

**面试答案：**

LayerNorm 会同时减均值、除标准差：

\[
LN(x)=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\gamma+\beta
\]

RMSNorm 只根据均方根进行缩放，不减均值：

\[
RMSNorm(x)=\frac{x}{\sqrt{mean(x^2)+\epsilon}}\gamma
\]

RMSNorm 的优点是：

1. 计算更简单；
2. 速度更快；
3. 参数更少；
4. 在大模型中通常足够稳定。

因此很多现代 LLM 会使用 RMSNorm 替代 LayerNorm。

---

# 六、FFN 与激活函数

## 17. Transformer 里的 FFN 起什么作用？

**面试答案：**

Self-Attention 负责 token 之间的信息交互，而 FFN 负责对每个 token 的表示进行非线性变换和特征抽取。

可以粗略理解为：

- Attention 做“跨 token 信息聚合”；
- FFN 做“单 token 表示加工”。

在大模型中，FFN 通常占据大量参数，是模型记忆知识和进行复杂变换的重要部分。

---

## 18. SwiGLU 是什么？为什么大模型常用它？

**面试答案：**

SwiGLU 是 GLU 的一种变体，形式可以理解为：

\[
FFN(x)=W_2(SiLU(W_1x) \odot W_3x)
\]

相比普通 ReLU/GELU FFN，SwiGLU 引入了门控机制，可以让模型动态控制哪些信息通过。它通常具有更强的表达能力，在大模型中表现更好，因此很多现代 LLM 使用 SwiGLU 作为 FFN 激活结构。

---

# 七、推理效率与工程优化

## 19. KV Cache 是什么？为什么能加速推理？

**面试答案：**

在自回归生成中，每一步都会生成一个新 token。如果每一步都重新计算所有历史 token 的 K/V，计算量会非常大。

KV Cache 的思想是：  
历史 token 的 Key 和 Value 一旦计算出来，在后续生成步骤中不会变化，因此可以缓存起来。下一步生成时，只需要计算当前 token 的 Q/K/V，然后用当前 Q 去 attend 历史缓存的 K/V。

这样可以把每步重复计算历史上下文的开销省掉，大幅提升推理速度。

---

## 20. Prefill 和 Decode 阶段有什么区别？

**面试答案：**

LLM 推理通常分为两个阶段：

### 1. Prefill 阶段

输入完整 prompt，一次性计算所有 prompt token 的 hidden states 和 KV cache。这个阶段通常是并行计算，计算量大，但并行度高。

### 2. Decode 阶段

每次生成一个 token，利用 KV cache 逐步生成。这个阶段通常是串行的，容易受显存带宽和 KV cache 读取影响。

所以优化方向不同：

- Prefill 更关注 attention 计算效率；
- Decode 更关注 KV cache 访问、batching、显存带宽和并发调度。

---

## 21. FlashAttention 解决什么问题？

**面试答案：**

标准 attention 会显式构造 \(n \times n\) 的 attention 矩阵，长序列下显存占用非常大。

FlashAttention 的核心是通过分块计算和 IO-aware 优化，避免完整 attention 矩阵落到 HBM 显存中，减少显存读写，提高速度和显存效率。

它不改变 attention 的数学结果，只是改变计算实现方式。因此 FlashAttention 是精确 attention 的高效实现，不是近似 attention。

---

# 八、SFT 项目强相关考点

## 22. SFT 时 Transformer 的哪些参数会被更新？

**面试答案：**

这取决于微调方式。

### 如果是全参 SFT

模型所有参数都会更新，包括：

- embedding；
- attention 的 \(W_Q,W_K,W_V,W_O\)；
- FFN；
- norm；
- LM head。

### 如果是 LoRA / QLoRA

通常冻结原模型参数，只在部分线性层上加低秩适配矩阵，例如：

- q_proj；
- k_proj；
- v_proj；
- o_proj；
- gate_proj；
- up_proj；
- down_proj。

训练时只更新 LoRA 参数，原模型参数不变。这样显著降低显存和训练成本。

---

## 23. LoRA 为什么有效？和 Transformer 线性层有什么关系？

**面试答案：**

LoRA 的核心假设是：下游任务对大模型参数的更新具有低秩特性。也就是说，不需要更新完整的权重矩阵 \(W\)，只需要学习一个低秩增量：

\[
W' = W + \Delta W
\]

其中：

\[
\Delta W = BA
\]

\(A\) 和 \(B\) 是低秩矩阵，rank 远小于原始矩阵维度。

Transformer 中 attention 和 FFN 包含大量线性投影矩阵，因此 LoRA 通常加在这些线性层上，能以很小的参数量调整模型行为。

---

## 24. 为什么 SFT 后模型可能出现幻觉或能力退化？

**面试答案：**

可能原因包括：

1. **训练数据质量差**  
   错误答案、格式混乱、噪声样本会直接影响模型行为。

2. **数据分布过窄**  
   过度拟合某类业务样本，导致通用能力下降。

3. **学习率过大**  
   参数更新过猛，破坏预训练知识。

4. **训练轮数过多**  
   出现过拟合，模型倾向死记训练集风格。

5. **loss mask 错误**  
   如果 prompt 也参与训练，可能影响指令跟随。

6. **模板不一致**  
   训练 chat template 和推理 chat template 不一致，会造成分布偏移。

7. **样本答案过长或风格单一**  
   模型容易学到啰嗦、模板化或过度自信的回答。

---

## 25. SFT 里 sequence length 对训练有什么影响？

**面试答案：**

sequence length 直接影响显存、速度和上下文建模能力。

对于 Transformer attention，复杂度近似是 \(O(n^2)\)，所以序列长度越长，显存和计算开销增长越快。

如果 max_seq_len 设置太短：

- 长样本会被截断；
- response 可能被截断；
- 模型学不到完整回答格式；
- 多轮对话上下文信息丢失。

如果设置太长：

- 训练显存压力大；
- batch size 变小；
- 训练速度下降；
- padding 浪费可能增加。

实际项目中通常要结合业务数据长度分布，统计 prompt 和 response 的 token 长度，选择覆盖大部分样本但不过度浪费的 max_seq_len。

---

# 九、搜广推场景结合类问题

## 26. 在搜广推算法场景里，为什么要用 LLM / Qwen3 做 SFT？

**面试答案：**

在搜广推场景中，LLM 可以补充传统模型在语义理解、生成和复杂推理上的能力。SFT Qwen3 可能用于：

1. **Query 理解**  
   识别用户搜索意图、改写 query、提取属性和约束。

2. **商品理解**  
   生成商品卖点、类目解释、属性补全。

3. **广告创意生成**  
   根据商品信息生成标题、卖点、短文案。

4. **推荐理由生成**  
   为用户生成个性化推荐解释。

5. **人群/意图标签生成**  
   根据用户行为序列总结兴趣偏好。

6. **排序特征增强**  
   用 LLM 生成语义特征、item 表征或 query-item 匹配解释。

SFT 的目标不是让模型重新学习全部知识，而是让它更好地遵循业务任务格式、理解领域术语、输出稳定结构化结果。

---

## 27. LLM 和传统搜广推模型怎么结合？

**面试答案：**

可以从三个层面结合：

### 1. 数据层

用 LLM 做数据增强、标签生成、样本清洗、query 改写、商品标题补全。

### 2. 特征层

将 LLM 生成的语义 embedding、意图标签、商品卖点、用户兴趣摘要作为特征输入排序或召回模型。

### 3. 系统层

在线使用 LLM 完成复杂 query 理解、推荐解释、广告文案生成等任务；离线则用于高成本语义理解和内容生产。

在高 QPS 的搜广推主链路中，一般不会直接把大模型放在强实时排序链路里，而是更常见地离线生成特征、蒸馏到小模型、或用于低频高价值场景。

---

## 28. 用户行为序列能不能直接喂给 Decoder-only Transformer？

**面试答案：**

可以，但要设计合适的序列化方式。

例如把用户行为序列组织成自然语言：

```text
用户最近点击了：A、B、C；
购买了：D；
加购了：E；
请总结用户当前兴趣，并给出推荐商品类型。
```

或者组织成结构化文本：

```text
[click] 商品A 类目=手机 价格=3000
[cart] 商品B 类目=耳机 价格=299
[buy] 商品C 类目=充电器 价格=99
```

然后通过 SFT 让模型输出用户兴趣、意图标签或推荐理由。

但要注意：

1. 序列长度限制；
2. 行为时间顺序；
3. 行为类型权重；
4. 商品信息噪声；
5. 在线推理成本；
6. 输出结果是否能被下游稳定消费。

---

# 十、训练细节与排查类问题

## 29. SFT 训练 loss 很低，但线上效果不好，可能是什么原因？

**面试答案：**

可能原因包括：

1. **训练集和线上分布不一致**  
   训练样本可能过于干净，线上 query 更口语化、更噪声。

2. **评估指标不匹配**  
   loss 低不代表业务指标好，尤其生成任务还要看格式、事实性、可用性。

3. **数据泄露或模板泄露**  
   模型可能学会了固定模板，而不是任务能力。

4. **过拟合**  
   训练 loss 下降，但验证集或人工评估变差。

5. **推理参数不合适**  
   temperature、top_p、max_new_tokens 设置不合理。

6. **chat template 不一致**  
   训练和推理使用的 system/user/assistant 格式不同。

7. **label mask 有问题**  
   如果监督区域错误，模型可能没有真正学习目标 response。

---

## 30. 如何判断 SFT 是否真的有效？

**面试答案：**

我会从四个层面评估：

### 1. 离线自动指标

根据任务类型设计指标，例如：

- 分类：Accuracy、F1；
- 结构化抽取：字段级 Precision/Recall/F1；
- 生成：ROUGE、BLEU 仅作参考；
- 排序增强：AUC、GAUC、NDCG；
- 语义匹配：Recall@K、MRR。

### 2. 人工评估

重点看：

- 指令遵循；
- 事实正确性；
- 输出格式；
- 业务术语理解；
- 是否幻觉；
- 是否过度生成。

### 3. Case 对比

对比 base model 和 SFT model 在典型业务 case 上的表现，看是否解决核心痛点。

### 4. 线上 A/B

如果模型结果进入业务链路，最终要看 CTR、CVR、GMV、广告收入、用户体验指标、人工审核通过率等。

---

# 十一、容易被追问的底层问题

## 31. Transformer 为什么比 RNN 更适合大模型？

**面试答案：**

主要有三点：

1. **并行训练能力强**  
   RNN 必须按时间步递归计算，Transformer 可以并行处理整个序列。

2. **长距离依赖建模更好**  
   Attention 可以让任意两个 token 直接交互，而 RNN 长距离信息容易衰减。

3. **扩展性更强**  
   Transformer 结构规则，适合大规模参数、大规模数据和 GPU/TPU 并行计算。

因此现代 LLM 基本都采用 Transformer 架构。

---

## 32. Attention 的复杂度是多少？有什么优化方向？

**面试答案：**

标准 attention 的时间和空间复杂度通常是：

\[
O(n^2d)
\]

其中 \(n\) 是序列长度，\(d\) 是 hidden size。

优化方向包括：

1. **工程优化**  
   FlashAttention、算子融合、混合精度、张量并行。

2. **结构优化**  
   MQA、GQA、Sliding Window Attention、Sparse Attention。

3. **缓存优化**  
   KV cache、PagedAttention、KV cache 量化。

4. **长度优化**  
   样本 packing、动态 padding、长度分桶、截断策略。

---

## 33. 为什么 instruction tuning 能提升模型指令跟随能力？

**面试答案：**

预训练模型主要学习通用语言建模能力，并不一定知道用户希望它如何回答问题。Instruction tuning 通过大量“指令-回答”样本，让模型学习：

1. 用户问题和模型回答之间的映射；
2. 不同任务的输出格式；
3. 对系统提示词的遵循；
4. 多轮对话中的角色边界；
5. 业务场景下的表达规范。

从 Transformer 角度看，SFT 改变的是模型在给定 prompt 上的条件分布，使模型更倾向于输出符合人类或业务期望的 response。

---

# 十二、面试中可以主动强调的项目亮点

如果你想把项目讲得更像真实业务项目，可以这样组织：

## 项目回答模板

**我在项目中使用 Qwen3 做 SFT，核心目标不是从零训练模型，而是在已有通用语言能力基础上，让模型适配搜广推业务场景。模型底座是 Decoder-only Transformer，训练目标仍然是 next token prediction。我们把业务样本组织成 instruction、input、response 格式，并通过 label mask 只对 response 部分计算 loss。**

**在训练侧，我重点关注了 sequence length、chat template、loss mask、学习率、LoRA target modules、样本质量和数据分布。因为 Transformer attention 对序列长度是平方复杂度，所以会先统计样本 token 长度分布，再决定 max_seq_len，并通过 packing 或动态 padding 降低训练浪费。**

**在推理侧，我关注 prefill 和 decode 两个阶段。长 prompt 会增加 prefill 计算，而长输出会增加 decode 阶段 KV cache 读取压力。因此如果模型结果进入搜广推链路，会更倾向离线生成特征、蒸馏到小模型，或者用于低频高价值的生成任务，而不是直接放到高 QPS 主排序链路。**

---

# 十三、建议重点准备的 Top 15 问题

如果时间有限，建议优先准备这些：

1. Transformer self-attention 公式和含义  
2. 为什么除以 \(\sqrt{d_k}\)  
3. Multi-head attention 的作用  
4. Causal mask 的作用  
5. Decoder-only 为什么适合生成  
6. SFT 和预训练 loss 的区别  
7. 为什么 prompt 部分通常不算 loss  
8. RoPE 的作用和优势  
9. KV cache 的原理  
10. Prefill 和 decode 的区别  
11. GQA/MQA/MHA 的区别  
12. FlashAttention 解决什么问题  
13. RMSNorm 和 LayerNorm 的区别  
14. SwiGLU 的作用  
15. SFT 后效果不佳如何排查  

---

# 十四、简历项目中可用的一句话总结

你可以在面试里这样总结：

> 这个项目里我主要基于 Qwen3 这类 Decoder-only Transformer 做 SFT，底层训练目标仍然是自回归 next token prediction。项目重点不只是调参，而是围绕业务任务构造高质量 instruction 数据，正确处理 chat template 和 label mask，并结合 Transformer 的序列长度复杂度、KV cache、LoRA 微调机制来控制训练和推理成本，最终让模型在搜广推场景下具备更稳定的意图理解、内容生成或结构化输出能力。
