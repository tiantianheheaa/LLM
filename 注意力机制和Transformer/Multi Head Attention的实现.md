### **Multi-Head Attention（多头注意力）的本质原理**
Multi-Head Attention 是 Transformer 架构的核心组件，其本质是通过**多组独立的注意力头（Attention Heads）并行计算**，捕获输入序列中不同位置的**不同特征表示子空间**的信息。具体来说：

1. **维度拆分与并行计算**：
   - 将输入词向量的维度 `dim` 拆分为 `num_heads` 个较小的维度（`head_dim = dim // num_heads`）。
   - 每个头独立计算注意力，最后将结果拼接并通过线性层恢复原始维度。

2. **本质原理**：
   - **多视角建模**：不同头可以学习到不同的特征（如语法、语义、长距离依赖等）。
   - **表达能力增强**：类似卷积神经网络中的多通道机制，通过并行计算捕获更丰富的上下文信息。
   - **降低计算复杂度**：单头注意力若直接扩展到高维，计算量会剧增，多头拆分后更高效。

3. **为什么这样做？**
   - **避免信息混合**：单头注意力可能将所有特征混在一个高维空间，多头允许不同特征在独立子空间中计算。
   - **捕获多样化依赖**：例如一个头关注语法结构，另一个头关注核心词，提升模型鲁棒性。

---

### **数学表示**
给定输入张量 `X`（形状为 `(batch_size, seq_len, dim)`）：
1. **线性变换**：通过 `W_q, W_k, W_v` 分别投影到 `Q, K, V`（形状均为 `(batch_size, num_heads, seq_len, head_dim)`）。
2. **缩放点积注意力**：每个头独立计算：
   \[
   \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i
   \]
   其中 \(d_k = \text{head\_dim}\)。
3. **拼接与输出**：将所有头的输出拼接并通过 `W_o` 线性变换：
   \[
   \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_o
   \]
<img width="875" height="470" alt="image" src="https://github.com/user-attachments/assets/35485348-2e72-4edc-9b65-8416ab753aeb" />

---

### **示例代码（PyTorch 实现）**
```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        
        # 线性变换矩阵：Q, K, V
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        # 线性变换并拆分多头
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, seq_len, head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, seq_len, seq_len)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, seq_len, head_dim)
        
        # 拼接多头并输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        output = self.out_proj(attn_output)
        return output

# 测试
batch_size, seq_len, dim = 2, 10, 512
num_heads = 8
x = torch.randn(batch_size, seq_len, dim)
mha = MultiHeadAttention(dim, num_heads)
output = mha(x)
print("Input shape:", x.shape)  # torch.Size([2, 10, 512])
print("Output shape:", output.shape)  # torch.Size([2, 10, 512])
```

---

### **关键点解析**
1. **维度拆分**：
   - 输入维度 `dim=512`，`num_heads=8` → 每个头的 `head_dim=64`。
   - 通过 `view` 和 `transpose` 将 `(B, seq_len, dim)` 转换为 `(B, num_heads, seq_len, head_dim)`。

2. **并行计算**：
   - 每个头独立计算注意力，互不干扰。
   - 最终拼接时通过 `transpose` 和 `contiguous().view()` 恢复原始维度。

3. **与单头注意力的区别**：
   - 单头注意力：`Q, K, V` 形状为 `(B, seq_len, dim)`，直接计算全局注意力。
   - 多头注意力：通过拆分维度，在多个子空间中并行计算。

---

### **为什么多头有效？**
- **直观例子**：在机器翻译中，一个头可能专注“主语-动词”关系，另一个头专注“形容词-名词”修饰关系。
- **对比实验**：
  - 若 `num_heads=1`，模型可能难以同时捕捉语法和语义。
  - 若 `num_heads=dim`（每个头维度为1），表达能力不足。
  - 适中 `num_heads`（如8）能平衡计算效率和模型能力。

---

### **总结**
- **多头注意力的本质**：通过维度拆分和并行计算，在多个子空间中捕获多样化的上下文特征。
- **核心优势**：增强模型表达能力，避免信息混合，类似“分治策略”。
- **代码关键点**：`view` + `transpose` 实现维度拆分，`softmax` 计算注意力权重，最后拼接输出。
