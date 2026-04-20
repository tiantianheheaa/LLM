

<img width="1514" height="1562" alt="image" src="https://github.com/user-attachments/assets/6378f7ce-6825-4b61-ac10-ae0b8b69aa7f" />



| ‌超参数‌ | ‌作用描述‌ | ‌推荐值或范围‌ | ‌配置示例‌ |
| --- | --- | --- | --- |
| ‌lora_alpha‌ | 缩放因子，控制低秩矩阵更新的幅度，与r协同调整表达力 | 通常设为r的倍数：r、2r或4r；常用16-32 | alpha=32（当r=16时） |
| ‌lora_dropout‌ | Dropout率，防止过拟合，尤其小数据场景 | 0.05-0.2；小数据取0.1-0.2，大数据取0.05-0.1 | dropout=0.1 |
| ‌target_modules‌ | 指定应用LoRA的模块，影响参数效率 | Transformer注意力层常用["query", "value"]；全注意力层可扩展至["query","key","value"] | target_modules=["query","value"] |
| ‌bias‌ | 控制偏置项训练，正则化选项 | "none"（默认不训练偏置）；资源充足或性能不足时试"lora_only" | bias="none" |
| ‌task_type‌ | 定义任务类型，确保损失函数匹配 | 如"CAUSAL_LM"（自回归语言模型）、"SEQ_CLS"（序列分类） | task_type="CAUSAL_LM"（GPT微调） |
| ‌modules_to_save‌ | 指定额外解冻训练的模块（非LoRA层），增强灵活性 | 可选，如["classifier"]或["lm_head"]；分类任务添加可提升1-2%准确率 | modules_to_save=["classifier"] |
| ‌init_lora_weights‌ | 初始化LoRA权重方式 | 通常True（默认初始化）；特殊场景可自定义 | init_lora_weights=True |



<img width="1518" height="838" alt="image" src="https://github.com/user-attachments/assets/6f761a1d-ad4f-42de-97c2-310a0badb262" />
