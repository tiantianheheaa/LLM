在 Transformers 库中，input_ids、labels 和 attention_mask 是训练 NLP 模型的核心输入参数。以下是它们的详细解释、使用方式和代码示例：

1. input_ids‌
含义‌：文本的数字表示，将分词后的 token 映射到词汇表中的唯一整数 ID。
作用‌：模型通过 input_ids 理解文本的语义信息。
生成方式‌：通过分词器（Tokenizer）将原始文本转换为整数序列。
python
Copy Code
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "How are you?"
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]  # 例如 tensor([[101, 2129, 2024, 2017, 1029, 102]])

2. labels‌
含义‌：模型训练的目标值，用于计算损失函数。
作用‌：
生成任务‌（如翻译）：目标文本的 token ID。
分类任务‌：类别标签的整数。
语言模型任务‌：通常与 input_ids 相同，但可屏蔽部分位置（如用 -100 忽略填充符的损失）。
python
Copy Code
# 机器翻译示例（目标文本的法语 token ID）
fr_text = "Comment ça va?"
labels = tokenizer(fr_text, return_tensors="pt").input_ids

# 分类任务示例（假设类别为 2）
labels = torch.tensor()

3. attention_mask‌
含义‌：二进制掩码，指示模型应关注哪些 token（1=有效 token，0=填充符）。
作用‌：
屏蔽填充符（如 [PAD]），防止模型关注无效位置。
在生成任务中避免“偷看”未来信息（因果掩码）。
python
Copy Code
# 处理变长序列时自动生成
inputs = tokenizer(["Hello", "Hi there!"], padding=True, return_tensors="pt")
attention_mask = inputs["attention_mask"]  # 例如 tensor([[1,1,0], [1,1,1]])

三者的协同工作流程‌
python
Copy Code
from transformers import AutoModelForSeq2SeqLM

# 示例：机器翻译任务
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
inputs = tokenizer(en_text, return_tensors="pt", padding=True)
labels = tokenizer(fr_text, return_tensors="pt").input_ids

outputs = model(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    labels=labels
)
loss = outputs.loss  # 计算损失

关键注意事项‌
input_ids 与 labels 的关系‌：
在生成任务中，labels 是目标序列；在分类任务中，labels 是类别标签。
使用 -100 忽略 labels 中特定位置的损失（如填充符）。
attention_mask 类型‌：
填充掩码‌：屏蔽填充符（默认生成）。
因果掩码‌：防止模型关注未来信息（需手动生成）。
模型输入‌：
训练时需同时传入三者；推理时只需 input_ids 和 attention_mask。

通过合理配置这三个参数，可确保模型高效处理变长序列并准确学习任务目标。
