我们来分步完成这个任务：从Hugging Face下载Qwen2.5-0.5B模型（注：截至目前，Qwen3尚未发布，这里使用最新的Qwen2.5-0.5B-Instruct作为示例，流程完全相同），添加一个新的token，然后保存，并详细解释文件差异。

**重要提示**：`Qwen/Qwen3-0.6B` 这个模型ID目前在Hugging Face上不存在。Qwen系列最新的是Qwen2.5。为了让代码可以实际运行，我将使用 `Qwen/Qwen2.5-0.5B-Instruct` 作为示例。您只需将代码中的模型ID替换为您想使用的任何模型ID即可。

---

### 第一步：环境准备

首先，请确保您已经安装了必要的库。`accelerate` 库可以帮助更高效地加载模型。

```bash
pip install transformers torch accelerate -U
```

如果您的模型需要登录Hugging Face Hub（例如，一些gated模型），请先在终端登录：
```bash
huggingface-cli login
# 或者在Python/Jupyter Notebook中运行
# from huggingface_hub import notebook_login
# notebook_login()
```

---

### 第二步：具体代码实现

下面的Python脚本将完成所有操作：下载、添加token、调整模型、保存和验证。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# --- 1. 定义模型和保存路径 ---
# 注意：Qwen3-0.6B 尚未发布，这里使用 Qwen2.5-0.5B-Instruct 作为示例
# 请将下面的 model_id 替换为您想要下载的任何模型ID
model_id = "Qwen/Qwen2.5-0.5B-Instruct" 
local_save_path = "./qwen2.5-0.5b-with-new-token"
new_token_to_add = "<new_special_token>" # 我们要添加的新token

# --- 2. 加载原始模型和Tokenizer ---
print(f"正在从Hugging Face下载模型: {model_id}...")

# 加载tokenizer
# trust_remote_code=True 是加载Qwen等模型所必需的
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

# 加载模型
# device_map="auto" 会自动将模型加载到可用的GPU或CPU上
# torch_dtype=torch.bfloat16 可以节省显存并加速推理（如果硬件支持）
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, # 如果不支持bfloat16，可以使用 torch.float16 或 torch.float32
    device_map="auto",
    trust_remote_code=True
)

print("模型和Tokenizer加载成功！")
print(f"原始Tokenizer词汇表大小: {tokenizer.vocab_size}")
print(f"原始模型嵌入层大小: {model.get_input_embeddings().weight.shape[0]}")

# --- 3. 添加新Token并调整模型 ---
# 使用 add_tokens 方法添加新token。这会返回新token的ID。
# 这个方法会更新tokenizer的词汇表，但不会改变模型的嵌入层
new_token_id = tokenizer.add_tokens(new_token_to_add)
print(f"\n成功添加新Token: '{new_token_to_add}'")
print(f"新Token的ID是: {new_token_id}")
print(f"添加后Tokenizer词汇表大小: {len(tokenizer)}") # 或者 tokenizer.vocab_size

# 关键步骤：调整模型的嵌入层（embedding layer）大小以匹配新的词汇表
# 如果不执行这一步，模型在遇到新token时会报错
model.resize_token_embeddings(len(tokenizer))

print(f"调整后模型嵌入层大小: {model.get_input_embeddings().weight.shape[0]}")

# (可选) 为新token初始化权重
# 新添加的token的嵌入向量是随机初始化的。在实际应用中，你可能需要用其他词的嵌入向量的平均值来初始化它，以获得更好的效果。
# 这里我们简单地用现有词汇的平均嵌入来初始化
with torch.no_grad():
    # 获取所有旧token的嵌入向量
    old_embeddings = model.get_input_embeddings().weight[:-1, :] # 排除最后一个（新添加的）
    # 计算平均值
    mean_embedding = old_embeddings.mean(dim=0)
    # 将新token的嵌入向量设置为这个平均值
    model.get_input_embeddings().weight[-1, :] = mean_embedding
    # 同样，也要更新输出层的权重（对于因果语言模型，输入和输出嵌入层通常是共享的）
    model.get_output_embeddings().weight[-1, :] = mean_embedding

print(f"已为新Token '{new_token_to_add}' 初始化嵌入权重。")


# --- 4. 保存新的模型和Tokenizer ---
print(f"\n正在将新模型和Tokenizer保存到: {local_save_path}")

# 创建保存目录（如果不存在）
os.makedirs(local_save_path, exist_ok=True)

# 保存tokenizer
tokenizer.save_pretrained(local_save_path)

# 保存模型
# save_pretrained会保存模型权重、配置文件等
model.save_pretrained(local_save_path)

print("保存完成！")

# --- 5. 验证保存的模型 ---
print("\n--- 验证新模型 ---")
# 从本地路径加载
print(f"正在从本地路径加载模型: {local_save_path}")
new_tokenizer = AutoTokenizer.from_pretrained(local_save_path, trust_remote_code=True)
new_model = AutoModelForCausalLM.from_pretrained(
    local_save_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print(f"加载后的Tokenizer词汇表大小: {len(new_tokenizer)}")
print(f"加载后的模型嵌入层大小: {new_model.get_input_embeddings().weight.shape[0]}")

# 检查新token是否存在
new_token_id_verify = new_tokenizer.convert_tokens_to_ids(new_token_to_add)
print(f"验证：新Token '{new_token_to_add}' 的ID是: {new_token_id_verify}")
assert new_token_id_verify == new_token_id

# 简单测试生成
prompt = f"Hello, what is the meaning of {new_token_to_add}?"
inputs = new_tokenizer(prompt, return_tensors="pt").to(new_model.device)
outputs = new_model.generate(**inputs, max_new_tokens=20)
generated_text = new_tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n生成测试:")
print(generated_text)

print("\n所有步骤成功完成！")
```

---

### 第三步：保存后的新模型与原始模型的文件差别

当你运行上述代码后，`qwen2.5-0.5b-with-new-token` 目录下会生成一系列文件。我们来对比一下它和从Hugging Face下载的原始模型缓存目录（通常在 `~/.cache/huggingface/hub/` 下）的文件差异。

假设原始模型目录为 `original_model_cache`，新模型目录为 `qwen2.5-0.5b-with-new-token`。

| 文件名 | 原始模型 (`original_model_cache`) | 新模型 (`qwen2.5-0.5b-with-new-token`) | 差别说明 |
| :--- | :--- | :--- | :--- |
| **`tokenizer.json`** | 包含原始词汇表的tokenizer文件。 | **已更新**。包含了新token的词汇表。这是最核心的变化之一。文件内容不同。 |
| **`tokenizer_config.json`** | `vocab_size` 为原始大小（如 151936）。 | **已更新**。`vocab_size` 的值增加了1（变为 151937）。 |
| **`special_tokens_map.json`** | 定义了原始的特殊token（如`<|endoftext|>`）。 | **可能更新**。如果新添加的token被用作特殊token，这里会反映出来。通常保持不变。 |
| **`tokenizer.model`** | (如果存在) SentencePiece模型文件。 | **已更新**。如果tokenizer是基于SentencePiece的，这个二进制文件会被重新生成以包含新token。 |
| **`config.json`** | 模型的配置文件，如`hidden_size`, `num_layers`等。 | **通常不变**。模型的结构超参数没有改变。 |
| **`model-00001-of-00002.safetensors`**<br/>(或 `.bin` 文件) | 模型权重文件。**关键**：嵌入层（embedding）的权重矩阵形状为 `[原始词汇表大小, hidden_size]`。 | **已更新**。嵌入层权重矩阵的形状变为 `[新词汇表大小, hidden_size]`。新增的一行是为新token初始化的权重。**这是最核心的变化**。其他层的权重保持不变。 |
| **`model.safetensors.index.json`** | 权重文件的索引，记录了每个权重张量在哪个文件中。 | **已更新**。因为嵌入层权重的形状变了，这个索引文件也需要更新以反映新的权重布局。 |
| **`generation_config.json`** | 生成配置，如`temperature`, `top_p`等。 | **通常不变**。除非你在保存前修改了它。 |
| **其他文件**<br/>(如 `README.md`, `.gitattributes`) | 模型的说明文档等。 | **通常不变**。`save_pretrained` 不会修改这些文件，除非你明确指定。 |

#### 总结核心差别：

1.  **词汇表文件 (`tokenizer.json`, `tokenizer.model`)**: 这些文件被**更新**，以包含新添加的token及其ID。这是为了让tokenizer能够正确地将新token转换为模型能理解的数字ID。

2.  **模型权重文件 (`.safetensors`)**: 这是最关键的变化。模型的**词嵌入层（Token Embedding Layer）**的权重矩阵被**扩展**了。
    *   **原始**: 形状为 `[V, D]`，其中 `V` 是原始词汇表大小，`D` 是隐藏层维度。
    *   **新模型**: 形状为 `[V+1, D]`。
    *   新增加的第 `V+1` 行就是新token的嵌入向量。这个向量在我们的代码中被初始化为所有旧token嵌入向量的平均值。模型的其他所有层（如Transformer块、输出层等）的权重都保持原样，没有改变。

3.  **配置文件 (`tokenizer_config.json`)**: `vocab_size` 字段的值被**更新**，以反映词汇表的新大小。

简单来说，你并没有改变模型的“大脑”（Transformer层的权重），你只是给它的“字典”（词汇表）里增加了一个新词，并为这个新词在“字典”的索引（嵌入层）里分配了一个位置和一个初始含义（嵌入向量）。
