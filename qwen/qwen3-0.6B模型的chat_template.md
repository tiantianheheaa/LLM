- im_start中的im是 instant message（即时消息）的缩写。

---

# Qwen3-0.6B 的 `chat_template` 详解与示例

`chat_template` 是 **Qwen3-0.6B tokenizer 配置中的对话格式化模板**，通常定义在 Hugging Face 仓库的 `tokenizer_config.json` 中。它的核心作用是：把结构化的多轮消息 `messages` 转换成模型训练时熟悉的单一文本 Prompt，使模型能区分 **system / user / assistant / tool** 等角色，并正确进入生成状态。Qwen3 系列模板基于 **Jinja2** 编写，支持多轮对话、工具调用、思考内容处理等场景。[〔1〕](https://m.blog.csdn.net/ljp1919/article/details/154756841)[〔2〕](https://blog.csdn.net/qq128252/article/details/147755244)[〔3〕](https://m.blog.csdn.net/silverfoxlynx45/article/details/156958755)

---

## 1. `chat_template` 解决什么问题？

大模型本质上接收的是一串 token，而不是天然理解“多轮聊天对象”。因此，类似下面的结构化消息：

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么我可以帮你的吗？"},
    {"role": "user", "content": "介绍一下 Qwen3-0.6B"}
]
```

需要被转换成模型能理解的文本格式，例如：

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么我可以帮你的吗？<|im_end|>
<|im_start|>user
介绍一下 Qwen3-0.6B<|im_end|>
<|im_start|>assistant
```

最后一行 `<|im_start|>assistant` 表示：**现在轮到 assistant 生成回复**。这正是 `add_generation_prompt=True` 的作用。[〔3〕](https://m.blog.csdn.net/silverfoxlynx45/article/details/156958755)[〔7〕](https://m.blog.csdn.net/SilverfoxLynx45/article/details/156958755)

---

# 2. Qwen3-0.6B 的基础对话格式

Qwen3 系列典型的基础格式是：

```text
<|im_start|>{role}
{content}<|im_end|>
```

其中：

| 组成部分 | 含义 |
|---|---|
| <|im_start|> | 一条消息的开始标记 |
| `role` | 消息角色，如 `system`、`user`、`assistant`、`tool` |
| 换行符 `\n` | 分隔角色名和正文内容 |
| `content` | 消息正文 |
| `<|im_end|>` | 一条消息的结束标记 |

例如：

```text
<|im_start|>user
你好<|im_end|>
```

表示一条用户消息。Qwen 系列通常使用 `<|im_start|>` / `<|im_end|>` 这类 ChatML 风格特殊 token，而不是 Llama 常见的 `[INST]...[/INST]` 格式；不同模型的 chat template 不能随意混用。[〔7〕](https://m.blog.csdn.net/SilverfoxLynx45/article/details/156958755)[〔9〕](https://m.blog.csdn.net/KIKI3666/article/details/156544210)

---

# 3. 核心模板逻辑拆解

Qwen3 的 `chat_template` 并不只是简单拼接字符串，它包含多个分支，主要处理以下几类场景：[〔2〕](https://blog.csdn.net/qq128252/article/details/147755244)[〔7〕](https://m.blog.csdn.net/SilverfoxLynx45/article/details/156958755)

## 3.1 无工具调用时：普通多轮对话

简化后的核心逻辑可以理解为：

```jinja2
{% for message in messages %}
{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{% endfor %}
{% if add_generation_prompt %}
{{ '<|im_start|>assistant\n' }}
{% endif %}
```

含义是：

1. 遍历 `messages` 中的每一条消息；
2. 每条消息前加 `<|im_start|>`；
3. 写入角色名 `role`；
4. 换行后写入消息正文 `content`；
5. 消息末尾加 `<|im_end|>`；
6. 如果 `add_generation_prompt=True`，在最后追加 `<|im_start|>assistant\n`，提示模型开始生成。[〔3〕](https://m.blog.csdn.net/silverfoxlynx45/article/details/156958755)[〔7〕](https://m.blog.csdn.net/SilverfoxLynx45/article/details/156958755)

---

## 3.2 有 `system` 消息时

`system` 消息一般用于定义模型行为，例如身份、风格、安全边界、输出格式要求等。

示例：

```python
messages = [
    {"role": "system", "content": "你是一个严谨、简洁的技术助手。"},
    {"role": "user", "content": "什么是 chat_template？"}
]
```

格式化后：

```text
<|im_start|>system
你是一个严谨、简洁的技术助手。<|im_end|>
<|im_start|>user
什么是 chat_template？<|im_end|>
<|im_start|>assistant
```

`system` 通常放在对话最前面，对后续所有回复产生全局约束。Qwen3 模板会优先处理首条 `system` 消息；如果存在工具定义，模板还会把工具说明合并到系统提示区域。[〔2〕](https://blog.csdn.net/qq128252/article/details/147755244)[〔9〕](https://m.blog.csdn.net/KIKI3666/article/details/156544210)

---

## 3.3 `add_generation_prompt`

`add_generation_prompt` 是推理时非常关键的参数。

| 参数值 | 作用 |
|---|---|
| `True` | 在末尾追加 `<|im_start|>assistant\n`，让模型开始回答 |
| `False` | 只格式化已有对话，不额外添加 assistant 开始标记 |

示例：

```python
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

如果忘记设置 `add_generation_prompt=True`，模型可能把输入当作未完成文本继续补写，而不是作为 assistant 回复用户。[〔3〕](https://m.blog.csdn.net/silverfoxlynx45/article/details/156958755)[〔7〕](https://m.blog.csdn.net/SilverfoxLynx45/article/details/156958755)

---

# 4. Qwen3 的思考模式：`enable_thinking`、`<think>` 与 `reasoning_content`

Qwen3 系列引入了思考/非思考相关的模板逻辑。在对话历史中，assistant 消息可能包含：

```text
<think>
这里是模型的推理过程
</think>

这里是最终回答
```

模板会尝试区分两部分：

| 字段/标记 | 作用 |
|---|---|
| `reasoning_content` | 显式存放 assistant 的推理内容 |
| `<think>...</think>` | 文本中的思考内容标记 |
| `content` | assistant 最终可见回答内容 |
| `enable_thinking` | 控制是否启用思考模式，具体行为取决于推理框架和模板版本 |

Qwen3 的模板会处理 assistant 历史消息中的 `reasoning_content`，或者从 `content` 中解析 `<think>...</think>`，从而在多轮对话中保留或剥离推理内容。相关资料也提到 Qwen3 模板支持解析助手推理内容，并可结合思考模式使用。[〔2〕](https://blog.csdn.net/qq128252/article/details/147755244)[〔7〕](https://m.blog.csdn.net/SilverfoxLynx45/article/details/156958755)

---

## 示例：关闭思考模式

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

messages = [
    {"role": "user", "content": "请用一句话解释什么是机器学习。"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

print(prompt)
```

可能得到类似格式：

```text
<|im_start|>user
请用一句话解释什么是机器学习。<|im_end|>
<|im_start|>assistant
<think>

</think>

```

这里空的 `<think></think>` 通常表示：**不要求模型展开思考过程，直接进入最终回答**。实际输出会受 Transformers 版本、模型仓库 `tokenizer_config.json` 版本和推理框架实现影响，最稳妥的方式是通过 `print(tokenizer.chat_template)` 查看当前环境加载到的真实模板。[〔5〕](https://m.blog.csdn.net/weixin_33670640/article/details/157190578)[〔9〕](https://m.blog.csdn.net/KIKI3666/article/details/156544210)

---

## 示例：启用思考模式

```python
prompt = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "小明有5个苹果，吃了2个，又买了3个，现在有几个？"}
    ],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

print(prompt)
```

格式化后的输入一般会以 assistant 开始标记结尾：

```text
<|im_start|>user
小明有5个苹果，吃了2个，又买了3个，现在有几个？<|im_end|>
<|im_start|>assistant
```

启用思考模式后，模型可能先生成 `<think>...</think>`，再给出最终答案。资料中也提到 Qwen3 支持通过 `enable_thinking`、`return_reasoning` 等扩展参数启用思维链相关能力。[〔5〕](https://m.blog.csdn.net/weixin_33670640/article/details/157190578)[〔10〕](https://blog.csdn.net/weixin_42515842/article/details/157424967)

---

# 5. 工具调用场景：`tools` 与 `<tool_call>`

Qwen3 的 `chat_template` 还支持工具调用，也就是 Function Calling / Tool Calling。模板在检测到 `tools` 参数时，会在系统消息中插入工具说明，结构大致为：

```text
<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
...
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
```

这段模板告诉模型：可用工具有哪些、工具参数是什么、如果要调用工具应如何输出。Qwen3 模板使用 `<tools></tools>` 包裹工具签名，使用 `<tool_call></tool_call>` 包裹函数调用 JSON。[〔2〕](https://blog.csdn.net/qq128252/article/details/147755244)[〔7〕](https://m.blog.csdn.net/SilverfoxLynx45/article/details/156958755)

---

## 工具调用示例

### Python 输入

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_sku_price",
            "description": "查询指定 SKU 的当前价格",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku_id": {
                        "type": "string",
                        "description": "商品 SKU ID"
                    }
                },
                "required": ["sku_id"]
            }
        }
    }
]

messages = [
    {"role": "system", "content": "你是一个电商智能助手。"},
    {"role": "user", "content": "帮我查一下 SKU 10099320302155 的价格。"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    tokenize=False,
    add_generation_prompt=True
)

print(prompt)
```

### 可能的格式化结果

```text
<|im_start|>system
你是一个电商智能助手。

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"get_sku_price","description":"查询指定 SKU 的当前价格","parameters":{"type":"object","properties":{"sku_id":{"type":"string","description":"商品 SKU ID"}},"required":["sku_id"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
帮我查一下 SKU 10099320302155 的价格。<|im_end|>
<|im_start|>assistant
```

如果模型判断需要调用工具，理想输出可能是：

```text
<tool_call>
{"name": "get_sku_price", "arguments": {"sku_id": "10099320302155"}}
</tool_call>
```

然后外部系统执行工具，把工具结果作为 `tool` 消息或工具响应重新放入对话，再让模型生成最终回答。Qwen3 模板对工具调用参数 JSON 格式化和 XML 标签封装有专门处理逻辑。[〔2〕](https://blog.csdn.net/qq128252/article/details/147755244)[〔7〕](https://m.blog.csdn.net/SilverfoxLynx45/article/details/156958755)

---

# 6. `messages` 中各字段含义

| 字段 | 类型 | 是否常用 | 含义 |
|---|---:|---:|---|
| `role` | string | 是 | 消息角色，如 `system`、`user`、`assistant`、`tool` |
| `content` | string | 是 | 消息正文 |
| `reasoning_content` | string | 可选 | assistant 的推理内容，常与 `<think>` 相关 |
| `tool_calls` | list | 可选 | assistant 发起的工具调用列表 |
| `name` | string | 可选 | 某些工具或函数消息中可能使用 |
| `tool_call_id` | string | 可选 | 多工具调用时用于关联请求和响应，具体依赖框架 |

在普通聊天中，最常用的是 `role` 和 `content`。在 Agent 或 Function Calling 场景中，才会进一步使用 `tools`、`tool_calls`、`tool` 消息等。Qwen3 模板专门覆盖了系统提示、用户查询、助手回复、工具调用和工具响应等复杂场景。[〔2〕](https://blog.csdn.net/qq128252/article/details/147755244)[〔7〕](https://m.blog.csdn.net/SilverfoxLynx45/article/details/156958755)

---

# 7. `apply_chat_template()` 常用参数

Hugging Face Transformers 提供 `tokenizer.apply_chat_template()` 方法，用来把消息列表转换成模型所需输入格式。该方法的价值是确保输入格式与模型训练时一致，避免手工拼接错误。[〔9〕](https://m.blog.csdn.net/KIKI3666/article/details/156544210)

```python
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
```

| 参数 | 作用 |
|---|---|
| `messages` | 对话消息列表 |
| `tokenize=False` | 返回字符串 Prompt |
| `tokenize=True` | 返回 token ids，可直接喂给模型 |
| `add_generation_prompt=True` | 在末尾追加 assistant 开始标记，用于推理生成 |
| `tools=tools` | 注入工具定义，用于工具调用场景 |
| `enable_thinking=True/False` | 控制 Qwen3 思考模式相关模板行为 |
| `return_tensors="pt"` | 当 `tokenize=True` 时，可返回 PyTorch Tensor |

---

# 8. 完整推理示例：Transformers 方式

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

messages = [
    {"role": "system", "content": "你是一个严谨、简洁的技术助手。"},
    {"role": "user", "content": "请解释 Qwen3-0.6B 的 chat_template 是什么。"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.6,
    top_p=0.9
)

response = tokenizer.decode(
    outputs[0][inputs.shape[-1]:],
    skip_special_tokens=True
)

print(response)
```

这里的关键点是：不要手写 `<|im_start|>`、`<|im_end|>`，而是优先使用 `apply_chat_template()`，让 tokenizer 根据仓库中的真实 `chat_template` 自动处理。资料中也强调，不同模型的模板不同，直接手工拼接容易出现兼容性和生成质量问题。[〔7〕](https://m.blog.csdn.net/SilverfoxLynx45/article/details/156958755)[〔9〕](https://m.blog.csdn.net/KIKI3666/article/details/156544210)

---

# 9. OpenAI 兼容接口中的关系

如果你通过 vLLM、TGI、FastChat 或自建 OpenAI 兼容服务调用 Qwen3-0.6B，通常发送的是标准 `messages`：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

completion = client.chat.completions.create(
    model="Qwen-0.6B",
    messages=[
        {"role": "system", "content": "你是一个简洁的技术助手。"},
        {"role": "user", "content": "用一句话说明 chat_template 的作用。"}
    ],
    temperature=0.5,
    extra_body={
        "enable_thinking": False
    }
)

print(completion.choices[0].message.content)
```

在这种模式下，通常由服务端 tokenizer 自动应用 `chat_template`，调用方不需要手动拼接 `<|im_start|>` 等特殊标记。相关资料提到 Qwen3 可通过 OpenAI 兼容接口接入，并可传入 `enable_thinking`、`return_reasoning` 等扩展参数。[〔5〕](https://m.blog.csdn.net/weixin_33670640/article/details/157190578)[〔10〕](https://blog.csdn.net/weixin_42515842/article/details/157424967)

---

# 10. 常见错误与排查

| 问题 | 现象 | 原因 | 建议 |
|---|---|---|---|
| 忘记 `add_generation_prompt=True` | 模型不回答或续写用户输入 | 没有追加 assistant 起始标记 | 推理时设置为 `True` |
| 手动漏写 `<|im_end|>` | 输出混乱、角色错乱 | 消息边界不清 | 使用 `apply_chat_template()` |
| 角色名写错 | 模板无法正确识别 | 如写成 `useer` | 只使用 `system/user/assistant/tool` |
| 混用其他模型模板 | 回答质量下降 | Qwen、Llama、Mistral 模板不同 | 使用当前模型自带 tokenizer |
| 上下文过长 | 截断、报错或效果变差 | 多轮历史累计 token 超限 | 控制历史长度，必要时摘要压缩 |
| 工具调用 JSON 不合法 | 外部工具无法解析 | 参数格式错误 | 对 `<tool_call>` 内容做 JSON 校验 |

资料中也列举了类似问题，例如漏掉 `<|im_end|>`、角色名错误、忘记 `add_generation_prompt=True`、输入超出上下文窗口等。[〔3〕](https://m.blog.csdn.net/silverfoxlynx45/article/details/156958755)[〔7〕](https://m.blog.csdn.net/SilverfoxLynx45/article/details/156958755)

---

# 11. 推荐实践

1. **始终优先使用模型仓库自带的 tokenizer 和 chat template**：  
   ```python
   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
   print(tokenizer.chat_template)
   ```

2. **推理时使用 `add_generation_prompt=True`**，训练或构造完整样本时则视任务而定。

3. **不要跨模型复用模板**：Qwen 系列、Llama 系列、Mistral 系列的对话模板差异明显。

4. **工具调用场景要让模型只生成结构化 `<tool_call>`**，外部系统执行工具后再把结果回填给模型。

5. **如果做 SFT 微调，训练和推理必须使用一致模板**，否则模型学到的角色边界、回复起点、特殊 token 语义会不一致，影响效果。资料也强调训练与推理时模板一致性非常关键。[〔7〕](https://m.blog.csdn.net/SilverfoxLynx45/article/details/156958755)[〔9〕](https://m.blog.csdn.net/KIKI3666/article/details/156544210)

---

## 总结

Qwen3-0.6B 的 `chat_template` 本质上是 **对话消息到模型 Prompt 的格式化引擎**。它用 `<|im_start|>role\ncontent<|im_end|>` 标记多轮对话，用 `add_generation_prompt` 指示模型开始回复，并进一步支持 **思考模式、工具调用、推理内容解析、多轮工具响应** 等复杂 Agent 场景。实际开发中，建议始终通过 `tokenizer.apply_chat_template()` 使用仓库内置模板，而不是手工拼接 Prompt。
