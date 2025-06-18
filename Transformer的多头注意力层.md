### 单头注意力机制

单头注意力机制是Transformer模型中最基本的注意力计算单元。它通过计算查询（Query）、键（Key）和值（Value）之间的相似度，来决定每个位置的重要性。单头注意力机制只计算一次注意力，适用于简单的上下文建模。

#### 单头注意力代码示例

```python
import torch
import torch.nn.functional as F
from math import sqrt

def single_head_attention(query, key, value, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))
    weights = F.softmax(scores, dim=-1)
    output = torch.bmm(weights, value)
    return output

# 示例输入
batch_size, seq_len, dim = 2, 5, 64
query = torch.randn(batch_size, seq_len, dim)
key = torch.randn(batch_size, seq_len, dim)
value = torch.randn(batch_size, seq_len, dim)
mask = torch.ones(batch_size, seq_len, seq_len)  # 假设没有需要屏蔽的位置

output = single_head_attention(query, key, value, mask)
print(output.shape)  # 输出: torch.Size([2, 5, 64])
```

### 多头注意力机制

多头注意力机制通过将输入拆分为多个头（heads），在每个头上独立计算注意力，然后将结果拼接起来，以捕获不同方面的上下文信息。这种方法允许模型在不同的子空间中学习不同的注意力模式，从而提高模型的表达能力。

#### 多头注意力代码示例

```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query_proj = torch.nn.Linear(dim, dim)
        self.key_proj = torch.nn.Linear(dim, dim)
        self.value_proj = torch.nn.Linear(dim, dim)
        self.out_proj = torch.nn.Linear(dim, dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换并拆分为多个头
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力得分
        scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float("inf"))
        weights = F.softmax(scores, dim=-1)

        # 计算输出
        output = torch.matmul(weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # 线性变换输出
        output = self.out_proj(output)
        return output

# 示例输入
dim = 64
num_heads = 8
model = MultiHeadAttention(dim, num_heads)

output = model(query, key, value, mask)
print(output.shape)  # 输出: torch.Size([2, 5, 64])
```

### 区别

1. **计算复杂度**：
   - 单头注意力：计算一次注意力，复杂度较低。
   - 多头注意力：计算多次注意力（每个头一次），复杂度较高，但能捕获更多上下文信息。

2. **表达能力**：
   - 单头注意力：只能捕获一种上下文信息。
   - 多头注意力：通过多个头，可以捕获不同子空间中的上下文信息，提升模型的表达能力。

3. **应用场景**：
   - 单头注意力：适用于简单的上下文建模任务。
   - 多头注意力：适用于复杂的上下文建模任务，如自然语言处理中的Transformer模型。

多头注意力机制通过增加模型的复杂性，提供了更强的上下文建模能力，是Transformer模型成功的关键之一。
