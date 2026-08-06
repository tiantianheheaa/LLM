可以把 RoPE scaling / YaRN 理解成：**不改模型权重，只改位置编码的外推方式和推理框架的最大长度配置**，让原本按 32K 训练的位置编码能够外推到更长，例如 64K / 96K / 128K。

对 Qwen3 文本模型，常见配置是：

```text
原始上下文长度 original_max_position_embeddings = 32768
目标上下文长度 target_length = original_max_position_embeddings × factor

factor = 2.0  -> 65536
factor = 3.0  -> 98304
factor = 4.0  -> 131072
```

Qwen3 文本模型通常原生是 32,768 tokens；要扩展到 131,072 tokens，一般设置 `factor=4.0`，并使用 YaRN 作为 RoPE scaling 方法。[〔1〕](https://deepwiki.com/guquan/Qwen3/5.2-context-scaling-and-yarn)

---

# 1. vLLM 里怎么操作？

如果你用 vLLM 部署 Qwen3，比如 Qwen3-32B，可以这样启动：

```bash
vllm serve Qwen/Qwen3-32B \
  --max-model-len 131072 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
```

如果多卡部署，可以加：

```bash
vllm serve Qwen/Qwen3-32B \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
```

几个参数含义：

| 参数 | 含义 |
|---|---|
| `rope_type: "yarn"` | 使用 YaRN 方式扩展 RoPE |
| `factor: 4.0` | 将上下文长度扩展为原来的 4 倍 |
| `original_max_position_embeddings: 32768` | 原始预训练上下文长度 |
| `--max-model-len 131072` | 推理服务允许的最大上下文长度 |

vLLM 文档里也说明，RoPE context extension 主要通过 `rope_type`、`factor`、`original_max_position_embeddings` 和 `max_model_len` 控制。[〔2〕](https://docs.vllm.ai/en/latest/features/context_extension/)

---

# 2. Transformers 里怎么操作？

如果你用 Hugging Face Transformers 直接推理，有两种方式。

## 方式一：修改模型目录下的 `config.json`

在 `config.json` 里加入或修改：

```json
{
  "rope_scaling": {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768
  }
}
```

有些 Transformers 版本字段可能叫：

```json
{
  "rope_scaling": {
    "type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768
  }
}
```

也就是说，`rope_type` 和 `type` 取决于 Transformers / vLLM 版本。新版本一般更推荐 `rope_type`。

---

## 方式二：代码里动态改 config

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

model_name = "Qwen/Qwen3-32B"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

config = AutoConfig.from_pretrained(
    model_name,
    trust_remote_code=True
)

config.rope_scaling = {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768
}

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

messages = [
    {"role": "user", "content": "请总结下面这篇长文：..."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=2048
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Transformers 也支持通过修改 `config.json` 或在加载时覆盖配置的方式启用 YaRN。[〔1〕](https://deepwiki.com/guquan/Qwen3/5.2-context-scaling-and-yarn)

---

# 3. llama.cpp 里怎么操作？

如果你用 GGUF + llama.cpp，本质也是设置目标上下文长度、RoPE scaling 方法、scale factor 和原始上下文长度。

示例：

```bash
./llama-cli \
  -m qwen3-32b.gguf \
  -c 131072 \
  --rope-scaling yarn \
  --rope-scale 4 \
  --yarn-orig-ctx 32768
```

参数含义：

| 参数 | 含义 |
|---|---|
| `-c 131072` | 目标上下文长度 |
| `--rope-scaling yarn` | 使用 YaRN |
| `--rope-scale 4` | 扩展 4 倍 |
| `--yarn-orig-ctx 32768` | 原始上下文长度 |

llama.cpp 对 YaRN 的配置通常就是通过 `-c`、`--rope-scaling`、`--rope-scale`、`--yarn-orig-ctx` 这几个参数完成。[〔1〕](https://deepwiki.com/guquan/Qwen3/5.2-context-scaling-and-yarn)

---

# 4. SGLang 里怎么操作？

SGLang 一般也是类似配置：

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-32B \
  --context-length 131072 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
```

不同版本的 SGLang 参数名可能略有差异，有的版本会使用类似：

```bash
--rope-type yarn
--rope-scale 4
```

所以实际使用时需要以当前 SGLang 版本的 help 或文档为准。

---

# 5. 面试里怎么解释 YaRN 的原理？

可以这样讲：

> Qwen3 使用 RoPE 位置编码。RoPE 本身对位置有一定外推能力，但模型预训练时只见过固定长度，比如 32K。如果直接把长度拉到 128K，远距离位置编码分布会偏离训练分布，模型容易出现注意力退化、重复、定位不准等问题。  
>  
> YaRN 是一种 RoPE scaling 方法，它通过对 RoPE 的频率进行缩放，让位置编码可以更平滑地外推到更长上下文。工程上通常不需要重新训练模型，只需要在推理框架中配置 `rope_scaling`，比如 `rope_type=yarn`、`factor=4.0`、`original_max_position_embeddings=32768`，再把 `max_model_len` 设置为目标长度，比如 131072。

---

# 6. 需要注意什么？

## 1. 不是改了配置就一定效果无损

YaRN 是外推，不是原生长上下文训练。越接近 128K，越可能出现：

- 长距离信息召回下降；
- 回答质量下降；
- 注意力定位不准；
- 重复输出；
- 延迟和显存显著增加。

---

## 2. 短文本场景不建议默认开启 YaRN

很多实现是 **static YaRN**，也就是无论输入长短都使用固定 scaling factor。这样对短文本任务未必有收益，甚至可能影响短上下文表现。[〔1〕](https://deepwiki.com/guquan/Qwen3/5.2-context-scaling-and-yarn)

所以线上实践中可以：

- 短上下文服务：不开 YaRN，保持 32K；
- 长文档服务：单独起一个 128K YaRN 服务；
- 根据请求长度路由到不同服务。

---

## 3. 显存主要涨在 KV Cache

长上下文推理时，模型权重显存不一定变，但 **KV Cache 显存会随上下文长度近似线性增长**。例如从 32K 拉到 128K，KV Cache 大约变成 4 倍，具体还和层数、hidden size、KV heads、batch size、dtype 有关。

---

## 4. factor 不要盲目拉太大

推荐从需求出发：

| 目标长度 | factor |
|---|---|
| 64K | 2.0 |
| 96K | 3.0 |
| 128K | 4.0 |

计算公式：

```text
target_length = 32768 × factor
```

如果业务只需要 60K，不一定要直接上 128K。

---

# 7. 最推荐的生产方式

如果你只是想在服务侧支持 Qwen3-32B 的 128K，上 vLLM 最直接：

```bash
vllm serve Qwen/Qwen3-32B \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
```

然后做三类验证：

1. **短上下文回归**：1K、4K、8K 输入下是否退化；
2. **长上下文召回**：32K、64K、96K、128K needle-in-a-haystack；
3. **业务长文评测**：长文总结、长代码问答、多文档检索增强生成。

面试里你可以总结成一句：

> 具体操作就是在推理框架中启用 YaRN 类型的 RoPE scaling，设置原始长度 `32768`、扩展倍数 `factor`，并把服务最大上下文 `max_model_len` 设置为目标长度。比如 Qwen3 从 32K 扩到 128K，就是 `factor=4.0`，`original_max_position_embeddings=32768`，`max_model_len=131072`。
