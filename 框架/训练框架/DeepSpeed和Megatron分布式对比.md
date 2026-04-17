
- DeepSpeed是ZeRo-0、1、2、3。
- Megatron是
  - DP：数据并行
  - TP：tensor 并行。 将一个矩阵乘法 分为2个或多个部分，放到不同的gpu上执行。执行结束后merge结果。
  - PP： pipeline并行。 将模型分块，在不同的gpu上执行不同的块，不同的gpu之间是`串行`的关系。
  - EP: expert 并行。适用于moe模型。 不同的专家放到不同的gpu上。
  - SP：sequence并行。SP通过将序列维度也进行切分，使得每张卡只处理序列的一部分，从而成比例地减少单卡所需存储的激活值大小。
 



<img width="740" height="674" alt="image" src="https://github.com/user-attachments/assets/9f135d7c-e000-4621-bc7c-bd7134f8cef0" />

<img width="704" height="686" alt="image" src="https://github.com/user-attachments/assets/0dd7ce96-8f4b-4283-87ed-c71c8e32586f" />

