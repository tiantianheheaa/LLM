### 问题
- qwen3-0.6B模型 为什么以下这几个数据对不上？ 下面这几个数据不应该是相同的吗？
- tokenizer.vocab_size是151643
- len(tokenizer)=151669，是**基础词汇表大小 + qwen3自带的26个special token**
- model.get_input_embeddings().weight.shape[0]是151936，模型的emb tabel size >= 词汇表大小即可，多的部分可以不查不用
- 最大词汇表id是151668，说明是从id是从0开始的
- **add_new_token是从151669开始的**

这是一个非常敏锐且专业的观察！你遇到的情况在大型语言模型（尤其是Qwen、Llama等）中非常常见，这并不是 Bug，而是由 **Tokenizer 基础词表**、**模型嵌入层对齐** 以及 **特殊 Token 预留槽位** 三者之间的差异造成的。

简单来说：**Tokenizer 看到的“词表大小”和 Model 实际分配的“嵌入矩阵大小”并不是同一个概念。**

下面我为你拆解这几个数字背后的逻辑：

### 1. 核心概念拆解

| 参数 | 你的数值 | 含义 |
| :--- | :--- | :--- |
| **`tokenizer.vocab_size`** | **151,643** | **基础词表大小 (Base Vocab Size)**。这是 SentencePiece/BPE 算法在训练数据上学到的“有效”Token 数量（ID 从 0 到 151642）。 |
| **`model.get_input_embeddings().weight.shape[0]`** | **151,936** | **模型嵌入层大小 (Embedding Size)**。这是 PyTorch 模型实际分配的矩阵行数。它必须 $\ge$ 词表大小。 |
| **最大词汇表 ID** | **151,638** | 基础词表中实际被占用的最大 ID（通常比 vocab_size 小一点，因为要预留 control 字符）。 |
| **`add_new_token` 起始 ID** | **151,669** | **特殊 Token 占用区**。这说明在 151,643 之后，有一段空间被系统预留给了特殊 Token（如 `<|endoftext|>` 等）。 |

---

### 2. 为什么会有 `151936 - 151643 = 293` 的差距？

这 **293** 个槽位（Slot）主要由以下两部分组成：

#### A. 特殊 Token 的占位 (Special Tokens Placeholder)
Qwen 模型在初始化时，会在词表末尾强行插入一些特殊的控制 Token。这些 Token 在基础的 SentencePiece 词表中可能不存在，或者 ID 不对，所以模型架构层面直接扩展了 Embedding 层来容纳它们。

常见的“隐形”占用包括：
*   **`<|endoftext|>` (EOS)**: 通常占用 1 个 ID。
*   **`<|pad|>` (PAD)**: 通常占用 1 个 ID（用于批处理对齐）。
*   **Chat Template Tokens**: 如 `<|system|>`, `<|user|>`, `<|assistant|>`, `im_start`, `im_end` 等。
    *   注意：有些 Tokenizer 会把这些映射到基础词表的高位 ID（比如 151600+），有些则直接在模型层硬加。Qwen 属于后者。

#### B. 硬件对齐 (Hardware Alignment / Padding)
为了让 GPU 计算更快（利用 Tensor Core），Embedding 层的维度通常会被对齐到 **8 的倍数**、**16 的倍数** 甚至 **128 的倍数**。
*   151,643 并不是一个整齐的数。
*   为了优化显存读写，PyTorch 可能会自动把 Embedding 层扩充到最近的“友好”数字，比如 151,680 或 151,936（具体取决于配置文件中的 `pad_to_multiple_of`）。

---

### 3. 深度解析：为什么 `add_new_token` 从 151,669 开始？

这是最关键的证据。它揭示了 Qwen Tokenizer 的内部结构：

```text
[ 0 ~ 151642 ]  : 基础词表 (SentencePiece 学到的 Token)
[ 151643 ~ 151668 ] : 预留给系统特殊 Token 的区域 (Reserved Special Tokens)
[ 151669 ~ ... ]   : 用户自定义添加 Token 的起始位置 (User Added Tokens)
[ ... ~ 151935 ]   : 模型 Embedding 层的总容量 (实际分配的显存)
```

**验证方法：**
你可以运行以下代码查看那些“看不见”的特殊 Token 到底是什么：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

local_model_path = "你的本地路径" # 替换为你的路径
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(local_model_path, trust_remote_code=True)

# 1. 查看基础词表大小
print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

# 2. 查看模型嵌入层大小
print(f"Model embedding size: {model.get_input_embeddings().weight.shape[0]}")

# 3. 查找特殊 Token 的 ID
print("\n--- 特殊 Token ID 检查 ---")
special_tokens_list = [
    "<|endoftext|>
