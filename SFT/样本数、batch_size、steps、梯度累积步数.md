- 梯度累积步数：gradient_accumulation_steps
- step：模型参数更新一次
- epoch：所有的样本过一遍
- 梯度累积步数相当于变相增大batch_size，延迟更新参数。



<img width="1558" height="1542" alt="image" src="https://github.com/user-attachments/assets/d507dc45-8657-496d-9bc0-dfd73bd491ce" />
<img width="1530" height="1056" alt="image" src="https://github.com/user-attachments/assets/da799890-b4ea-442d-8717-6a0347c5c545" />

