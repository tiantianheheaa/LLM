Qwen3 的 `chat_template` 可以理解成：**把结构化的多轮对话 messages 转成模型真正看到的一串 token / 文本格式的规则**。

简单说：

```python
messages = [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "解释一下 RoPE"},
]
```

经过 Qwen3 的 `chat_template` 后，会被转换成类似：

```text
<|im_start|>system
你是一个有帮助的助手<|im_end|>
<|im_start|>user
解释一下 RoPE<|im_end|>
<|im_start|>assistant
```

模型实际训练和推理看到的是后面这种带特殊 token 的序列，而不是 Python 里的 `messages` 对象。

---

# 1. Qwen3 chat_template 具体是什么？

Qwen3 主要沿用 **ChatML 风格** 的对话格式，用特殊 token 标记不同角色和轮次。Qwen3 / Qwen3-Coder 的文档中也说明，`apply_chat_template()` 会把人类可读的 message list 转成模型可理解的 ChatML 格式，并通过 `<|im_start|>`、`<|im_end|>` 标记不同 role。[〔1〕](https://deepwiki.com/QwenLM/Qwen3-Coder/3.2-text-generation-and-chat-templates)

核心格式大致是：

```text
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_response}<|im_end|>
```

多轮对话就是不断追加：

```text
<|im_start|>user
第一轮用户问题<|im_end|>
<|im_start|>assistant
第一轮助手回答<|im_end|>
<|im_start|>user
第二轮用户问题<|im_end|>
<|im_start|>assistant
第二轮助手回答<|im_end|>
```

推理时，如果你设置：

```python
add_generation_prompt=True
```

模板会在最后自动追加：

```text
<|im_start|>assistant
```

意思是告诉模型：

> 现在轮到 assistant 回答了。

---

# 2. Qwen3 的 thinking / non-thinking 和 chat_template 有什么关系？

Qwen3 一个比较特殊的点是支持 thinking / non-thinking 模式。Hugging Face 对 Qwen3 chat template 的分析里提到，Qwen3 可以通过 `enable_thinking` 控制是否启用思考模式；当 `enable_thinking=False` 时，模板会插入一个空的 `<think></think>`，引导模型跳过显式推理过程。[〔2〕](https://github.com/huggingface/blog/blob/main/qwen-3-chat-template-deep-dive.md)

例如非思考模式下，最后可能变成类似：

```text
<|im_start|>user
解释一下 RoPE<|im_end|>
<|im_start|>assistant
<think>

</think>

RoPE 是一种旋转位置编码……
```

也就是说，Qwen3 的 chat_template 不只是简单拼接 role，还可能处理：

- 是否启用 thinking；
- 是否保留或清理历史 `<think>` 内容；
- tool call 的序列化；
- assistant 起始 prompt；
- role 边界；
- EOS / stop token。

所以面试时可以说：

> Qwen3 的 chat_template 是一个 Jinja 模板，存放在 tokenizer 配置里，用于把 system / user / assistant / tool 等结构化消息渲染成模型训练和推理时真正使用的 ChatML token 序列。它还支持 Qwen3 的 thinking 控制，例如通过 `enable_thinking` 决定是否插入 `<think>` 相关标记。

---

# 3. 实际代码怎么用？

一般不要自己手写模板，而是用 tokenizer 自带的：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-8B",
    trust_remote_code=True
)

messages = [
    {"role": "system", "content": "你是一个专业的算法面试助手。"},
    {"role": "user", "content": "解释一下 SFT 的 loss。"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

print(text)
```

如果你要直接得到 token ids：

```python
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt"
)
```

`apply_chat_template()` 是 Hugging Face Transformers 里处理 chat model 输入格式的标准方式，官方文档也说明 chat template 的作用就是把 `{"role": ..., "content": ...}` 这样的对话结构转换成模型需要的 token 序列。[〔3〕](https://huggingface.co/docs/transformers/chat_templating)

---

# 4. chat_template 的作用是什么？

## 4.1 标记谁在说话

没有 chat_template，模型看到的可能只是：

```text
你是一个助手
解释一下 RoPE
RoPE 是……
```

它不一定知道哪部分是 system，哪部分是 user，哪部分是 assistant。

使用模板后：

```text
<|im_start|>system
你是一个助手<|im_end|>
<|im_start|>user
解释一下 RoPE<|im_end|>
<|im_start|>assistant
RoPE 是……<|im_end|>
```

模型可以明确区分角色。

---

## 4.2 告诉模型什么时候开始回答

推理时最后追加：

```text
<|im_start|>assistant
```

就是告诉模型：

> 前面是上下文，现在该 assistant 生成回复了。

如果这个提示缺失，模型可能继续生成 user 内容、重复 prompt，或者角色错乱。

---

## 4.3 对齐训练和推理分布

Chat 模型在 SFT / RLHF / DPO 阶段通常就是按某种模板训练的。

如果训练时用：

```text
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
...
<|im_end|>
```

但推理时你手写成：

```text
User: ...
Assistant:
```

这就产生了格式分布不一致。

结果可能是：

- 输出质量下降；
- 角色边界混乱；
- stop token 不生效；
- 模型重复生成 `<|im_start|>`；
- 多轮对话理解变差；
- thinking 模式控制失效；
- 工具调用格式异常。

---

## 4.4 支持多轮对话

多轮对话不能简单拼接文本，最好保留 role 边界：

```text
<|im_start|>user
第一轮问题<|im_end|>
<|im_start|>assistant
第一轮回答<|im_end|>
<|im_start|>user
第二轮问题<|im_end|>
<|im_start|>assistant
```

这样模型知道第二轮问题是在前文上下文基础上的追问。

---

## 4.5 支持 Qwen3 的 thinking 控制和 tool calling

Qwen3 的模板比 Qwen2.5 更复杂，一个重要变化就是支持通过 `enable_thinking` 控制 reasoning 行为；此外模板里还会处理工具调用参数序列化等细节。[〔2〕](https://github.com/huggingface/blog/blob/main/qwen-3-chat-template-deep-dive.md)

这意味着你自己手写模板时，很容易漏掉这些细节。

---

# 5. SFT 是否一定需要 chat_template？

结论分情况：

> **如果你是在做 Qwen3-Instruct / Chat 模型的对话式 SFT，基本上应该使用官方 chat_template。**  
> **如果你是在做 base model 的纯 completion SFT，或者任务不是 chat 形式，则不一定需要 chat_template。**

---

# 6. 哪些情况必须或强烈建议用 chat_template？

## 6.1 基于 Qwen3-Instruct 继续 SFT

如果你用的是：

```text
Qwen3-8B
Qwen3-14B
Qwen3-32B
Qwen3-30B-A3B
```

这类 instruct/chat 模型做二次 SFT，那么强烈建议使用官方 `apply_chat_template()`。

因为这些模型已经习惯了对应的 ChatML 格式。

训练数据最好统一成：

```python
messages = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
]
```

然后用：

```python
tokenizer.apply_chat_template(...)
```

转成训练文本。

---

## 6.2 多轮对话 SFT

比如客服、助手、Agent、工具调用、多轮任务规划。

这种必须保留 role 和轮次边界，否则模型很容易不知道：

- 哪句话是用户说的；
- 哪句话是助手说的；
- 哪些历史需要参考；
- 当前应该生成哪个 role。

---

## 6.3 需要和线上推理格式一致

如果线上部署用：

```python
tokenizer.apply_chat_template(...)
```

训练时也应该用同一个模板。

原则是：

```text
训练格式 = 推理格式
```

这是 SFT 里非常重要的工程原则。

---

# 7. 哪些情况不一定需要 chat_template？

## 7.1 训练 base model 做 completion

如果你不是用 instruct 模型，而是用 base model 做纯续写任务，例如：

```text
输入：蒙古国的首都是乌兰巴托。冰岛的首都是雷克雅未克。埃塞俄比亚的首都是
输出：亚的斯亚贝巴。
```

这种是 completion 格式，不一定需要 ChatML。

可以直接构造：

```text
蒙古国的首都是乌兰巴托。
冰岛的首都是雷克雅未克。
埃塞俄比亚的首都是亚的斯亚贝巴。
```

然后做 causal LM 训练。

---

## 7.2 强结构化任务，且你自定义了稳定格式

比如你从零定义一种格式：

```text
### Instruction:
抽取品牌、品类和价格

### Input:
我想买 5000 元左右的苹果手机

### Response:
{"品牌": "苹果", "品类": "手机", "价格": "5000元左右"}
```

这种也可以训练，但有两个前提：

1. 训练和推理必须完全一致；
2. 最好是基于 base model，或者明确接受和原 chat 模板不一致带来的风险。

如果你基于 Qwen3-Instruct，却不用官方 chat_template，而用 Alpaca 模板，也不是绝对不能训，但可能会损失模型已有的 chat 对齐能力。

---

# 8. SFT 时使用 chat_template 的关键细节

## 8.1 只对 assistant 部分算 loss

训练样本渲染后类似：

```text
<|im_start|>user
解释一下 RoPE<|im_end|>
<|im_start|>assistant
RoPE 是一种旋转位置编码……<|im_end|>
```

通常 label mask 是：

```text
user 部分：-100，不算 loss
assistant 内容：正常 label，算 loss
```

也就是只训练模型生成 assistant response。

---

## 8.2 不要让模型学习生成 user

如果你把所有 token 都算 loss，模型会被训练去预测：

```text
<|im_start|>user
解释一下 RoPE
```

这不符合目标。

SFT 的目标是：

```text
P(assistant_answer | system, user, history)
```

不是：

```text
P(system, user, assistant)
```

---

## 8.3 训练和推理的 `enable_thinking` 要一致

如果你的 SFT 数据不包含推理过程，只希望模型直接回答，那么训练和推理都应尽量使用 non-thinking 风格，例如：

```python
enable_thinking=False
```

如果你训练数据包含 `<think>...</think>`，那就是另一种范式，需要在评估和部署时保持一致。

---

## 8.4 不要手写特殊 token，优先用 tokenizer

不推荐自己拼：

```text
<|im_start|>user
...
```

更推荐：

```python
tokenizer.apply_chat_template(...)
```

原因是不同模型、不同版本的模板可能有细节差异。比如 Qwen3 相比 Qwen2.5，在 thinking、上下文管理和 tool call 序列化上都有差异。[〔2〕](https://github.com/huggingface/blog/blob/main/qwen-3-chat-template-deep-dive.md)

---

# 9. 如果 chat_template 用错了会怎样？

常见问题：

1. **角色混乱**
   - 模型生成 user；
   - 模型自己问自己答；
   - 多轮对话边界不清。

2. **输出格式异常**
   - 反复输出 `<|im_start|>`；
   - 忘记停止；
   - 多输出 `<|im_end|>`；
   - assistant 回答前出现奇怪模板符号。

3. **训练效果变差**
   - loss 可能下降，但推理效果不好；
   - 模型学到错误 role pattern；
   - 对原有 instruct 能力造成破坏。

4. **推理和训练不一致**
   - 训练用 Alpaca；
   - 推理用 ChatML；
   - 模型实际看到的格式分布不同，效果不稳定。

5. **thinking 控制失效**
   - 想关闭思考，但模型仍输出 `<think>`；
   - 或者模型输出空思考标签后格式混乱。

---

# 10. 面试推荐回答

你可以这样回答：

> Qwen3 的 chat_template 是 tokenizer 里的一个 Jinja 模板，用来把结构化 messages 转成模型实际输入的 ChatML 格式。典型格式是用 `<|im_start|>system/user/assistant` 和 `<|im_end|>` 标记不同角色和轮次。推理时设置 `add_generation_prompt=True`，会在最后追加 `<|im_start|>assistant`，提示模型开始生成 assistant 回复。  
>  
> 它的作用主要是统一训练和推理格式、明确角色边界、支持多轮对话，并处理 Qwen3 的 thinking / non-thinking 以及 tool call 等特殊逻辑。  
>  
> SFT 是否一定需要 chat_template 要看任务。如果是基于 Qwen3-Instruct 做对话式 SFT，基本必须使用官方 chat_template，否则训练分布和模型原有对齐格式不一致，容易导致角色混乱和效果下降。如果是 base model 的 completion 训练，或者你从头定义固定任务格式，则不一定必须用 chat_template。但无论哪种情况，训练格式和推理格式必须保持一致。

---

# 11. 最短记忆版

```text
chat_template = 把 messages 转成模型真正输入格式的模板。

Qwen3 典型格式：
<|im_start|>user
...<|im_end|>
<|im_start|>assistant
...<|im_end|>

作用：
1. 区分 system/user/assistant
2. 标记多轮对话边界
3. 提示模型什么时候开始回答
4. 保证训练和推理格式一致
5. 支持 Qwen3 thinking / tool call 等特殊逻辑

SFT 是否一定需要：
- Qwen3-Instruct 对话式 SFT：强烈需要，基本应使用官方模板。
- Base model completion SFT：不一定需要。
- 自定义格式可以，但训练和推理必须一致。
```
