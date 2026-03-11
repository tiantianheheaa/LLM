要从 Hugging Face 下载 **Qwen-0.6B** 模型并添加新 token，你需要使用 `transformers` 库加载模型，调整 tokenizer，然后重新训练或保存修改后的模型。以下是详细步骤和代码示例：

---

### **1. 安装依赖**
确保已安装 `transformers`、`torch` 和 `tokenizers`：
```bash
pip install transformers torch tokenizers
```

---

### **2. 下载 Qwen-0.6B 模型**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 下载模型和tokenizer
model_name = "Qwen/Qwen-0.6B"  # 替换为实际模型路径（如官方未发布，需确认）
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
```

> **注意**：截至 2024 年 3 月，Qwen-0.6B 可能尚未正式发布到 Hugging Face。如果模型名无效，请替换为类似模型（如 `Qwen/Qwen-7B`）或检查官方文档。

---

### **3. 添加新 Token**
#### **(1) 定义新 Token 列表**
```python
new_tokens = ["<new_token1>", "<new_token2>"]  # 替换为你的自定义token
```

#### **(2) 扩展 Tokenizer 的词汇表**
```python
# 方法1：直接添加新token
tokenizer.add_tokens(new_tokens)

# 方法2：合并原有词汇表和新token（更灵活）
# from tokenizers import Tokenizer
# tokenizer = Tokenizer.from_pretrained(model_name)
# tokenizer.add_tokens(new_tokens)
```

#### **(3) 调整模型嵌入层**
```python
# 扩展模型词汇表大小并初始化新token的权重
model.resize_token_embeddings(len(tokenizer))
```

---

### **4. 验证 Token 添加**
```python
# 测试新token是否能被正确编码
input_text = "这是一个测试<new_token1>"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
print("输入ID:", input_ids)

# 解码验证
decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
print("解码结果:", decoded_text)
```

---

### **5. 保存修改后的模型和 Tokenizer**
```python
# 保存tokenizer
tokenizer.save_pretrained("./qwen-0.6b-custom")

# 保存模型
model.save_pretrained("./qwen-0.6b-custom")
```

---

### **6. 重新加载自定义模型（可选）**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

custom_tokenizer = AutoTokenizer.from_pretrained("./qwen-0.6b-custom", trust_remote_code=True)
custom_model = AutoModelForCausalLM.from_pretrained("./qwen-0.6b-custom", trust_remote_code=True, device_map="auto")
```

---

### **完整代码示例**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载原始模型
model_name = "Qwen/Qwen-0.6B"  # 确认模型是否存在
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")

# 2. 添加新token
new_tokens = ["<new_token1>", "<new_token2>"]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

# 3. 测试
input_text = "测试新token<new_token1>"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
print("输入ID:", input_ids)
print("解码结果:", tokenizer.decode(input_ids[0]))

# 4. 保存
tokenizer.save_pretrained("./qwen-0.6b-custom")
model.save_pretrained("./qwen-0.6b-custom")
```

---

### **注意事项**
1. **模型兼容性**：Qwen 模型可能需要 `trust_remote_code=True`（因其代码托管在 Hugging Face Hub）。
2. **显存需求**：0.6B 模型约需 1.2GB 显存，确保硬件足够。
3. **新 Token 的初始化**：新增 token 的嵌入权重是随机初始化的，需通过**微调（Fine-tuning）**使其适应任务。
4. **官方支持**：如果 Qwen-0.6B 未发布，可先用类似模型（如 `Qwen/Qwen-7B`）测试流程。

---

### **微调建议（可选）**
若需让新 token 具备实际语义，需进行微调：
```python
from transformers import Trainer, TrainingArguments

# 准备微调数据（示例）
train_data = [{"text": "包含<new_token1>的句子"}, ...]

# 训练参数
training_args = TrainingArguments(
    output_dir="./finetuned-qwen",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    save_steps=100,
)

# 初始化Trainer并训练（需自定义数据集类）
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_data,  # 需转换为Dataset格式
)
trainer.train()
```

通过以上步骤，你可以成功扩展 Qwen 模型的词汇表并适配新 token。
