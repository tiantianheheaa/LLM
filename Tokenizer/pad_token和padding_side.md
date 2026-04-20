```python
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'
```



<img width="1576" height="884" alt="image" src="https://github.com/user-attachments/assets/2725e857-6eda-41e9-983b-cc9ef9bf127b" />
<img width="1518" height="1252" alt="image" src="https://github.com/user-attachments/assets/8722e3ee-b204-4bc4-afc3-aa06b0191f4b" />

<img width="1700" height="916" alt="image" src="https://github.com/user-attachments/assets/ff4f3391-e3ce-4a9a-8f9e-b079b9480732" />

| ‌参数‌ | ‌必要性‌ | ‌推荐值‌ | ‌主要作用‌ |
| --- | --- | --- | --- |
| tokenizer.pad_token | ‌必须显式设置‌ | = tokenizer.eos_token | 提供批处理的填充标记 |
| tokenizer.padding_side | 根据任务选择 | 训练：'right'生成：'left' | 控制填充位置，避免干扰模型逻辑 |
