### VQ-VAE作为图像Tokenizer的原理、公式与代码详解
**优秀解读**：https://zhuanlan.zhihu.com/p/14423487388

#### **一、核心原理：离散表征与向量量化**
VQ-VAE（Vector Quantised Variational Autoencoder）通过**向量量化（Vector Quantization）**将连续的图像特征离散化为有限集合的向量（码本向量），实现图像的“分词化”（Tokenization）。其核心思想类似于自然语言处理（NLP）中的词嵌入（Word Embedding），但将离散单元从单词扩展到图像特征向量。

1. **编码器（Encoder）**  
   输入图像 \( x \) 通过CNN编码器映射到连续潜在空间，生成特征图 \( z_e(x) \)，形状为 \( (B, C, H, W) \)（\( B \) 为批次大小，\( C \) 为通道数，\( H/W \) 为特征图高宽）。

2. **向量量化（Vector Quantization）**  
   - **码本（Codebook）**：维护一个可学习的离散码本 \( E = \{e_1, e_2, \dots, e_K\} \)，其中 \( K \) 是码本大小，\( e_i \) 是维度为 \( C \) 的向量。
   - **最近邻查找**：对 \( z_e(x) \) 中的每个空间位置向量（形状 \( (C,) \)），在码本中找到欧氏距离最近的向量 \( e_k \)：
     \[
     e_k = \arg\min_{e_i \in E} \|z_e(x)_{h,w} - e_i\|^2
     \]
   - **索引映射**：将每个空间位置向量替换为对应码本向量的索引 \( k \)，生成离散索引图 \( \text{indices} \in \mathbb{R}^{B \times H \times W} \)。

3. **解码器（Decoder）**  
   根据离散索引图从码本中检索对应向量，重构为连续特征图 \( z_q(x) \)，再通过转置卷积（Transposed Conv）还原为图像 \( \hat{x} \)。

#### **二、关键公式与损失函数**
VQ-VAE的优化目标包含三项损失，共同优化编码器、解码器和码本：

1. **重建损失（Reconstruction Loss）**  
   最小化原始图像 \( x \) 与重构图像 \( \hat{x} \) 的均方误差（MSE）：
   \[
   \mathcal{L}_{\text{recon}} = \|x - \hat{x}\|^2
   \]

2. **码本损失（Codebook Loss）**  
   使码本向量 \( e_i \) 接近编码器输出的向量 \( z_e(x) \)（通过停止梯度操作避免双向优化冲突）：
   \[
   \mathcal{L}_{\text{codebook}} = \|\text{sg}[z_e(x)] - e\|^2
   \]
   其中 \( \text{sg}[\cdot] \) 表示停止梯度（Stop Gradient），前向传播保留值，反向传播梯度为0。

3. **承诺损失（Commitment Loss）**  
   约束编码器输出 \( z_e(x) \) 接近码本向量 \( e \)，防止编码器输出发散：
   \[
   \mathcal{L}_{\text{commit}} = \beta \|z_e(x) - \text{sg}[e]\|^2
   \]
   其中 \( \beta \) 是超参数（通常设为0.25）。

**总损失函数**：
\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{codebook}} + \mathcal{L}_{\text{commit}}
\]

#### **三、代码实现（PyTorch）**
以下代码展示VQ-VAE的核心组件，包括向量量化层和完整模型结构：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        self.beta = beta  # 承诺损失系数

    def forward(self, z_e):
        # 计算输入与码本向量的距离
        B, C, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        z_e_flat = z_e_flat.view(-1, C)  # [B*H*W, C]

        # 计算L2距离
        distances = (torch.sum(z_e_flat**2, dim=1, keepdim=True) + 
                     torch.sum(self.embedding.weight**2, dim=1) - 
                     2 * torch.matmul(z_e_flat, self.embedding.weight.T))
        
        # 最近邻查找
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # [B*H*W, 1]
        encodings = torch.zeros(encoding_indices.shape[0], self.embedding.num_embeddings, device=z_e.device)
        encodings.scatter_(1, encoding_indices, 1)  # 独热编码 [B*H*W, K]

        # 量化向量
        z_q_flat = torch.matmul(encodings, self.embedding.weight).view(B, H, W, C)  # [B, H, W, C]
        z_q = z_q_flat.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        # 梯度直通估计（Straight-Through Estimator）
        loss = F.mse_loss(z_q.detach(), z_e) + self.beta * F.mse_loss(z_e.detach(), z_q)
        z_q = z_e + (z_q - z_e).detach()  # 反向传播时跳过量化步骤

        # 计算码本使用率（Perplexity）
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, loss, perplexity

class VQVAE(nn.Module):
    def __init__(self, input_dim=1, latent_dim=64, num_embeddings=512, embedding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, latent_dim, 4, 2, 1), nn.ReLU()
        )
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, input_dim, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, x):
        z_e = self.encoder(x)  # [B, C, H, W]
        z_q, vq_loss, perplexity = self.vq_layer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, perplexity
```

#### **四、图像Tokenization流程**
1. **编码与量化**  
   输入图像 \( x \) 通过编码器生成 \( z_e(x) \)，经向量量化得到离散索引图 \( \text{indices} \)。

2. **索引到Token**  
   将 \( \text{indices} \) 视为图像的“Token序列”，每个空间位置对应一个Token（码本索引）。

3. **生成应用**  
   - **自回归生成**：结合Transformer等模型，以Token序列为输入生成新图像（如DALL-E）。
   - **压缩与传输**：仅传输离散索引和码本，实现高效图像压缩。

#### **五、与其他模型的对比**
| **特性**         | **AE**               | **VAE**              | **VQ-VAE**           |
|------------------|----------------------|----------------------|----------------------|
| **潜在空间**     | 连续、确定性         | 连续、概率化         | 离散、码本映射       |
| **生成质量**     | 模糊（过拟合）       | 模糊（KL散度约束）   | 清晰（离散表征）     |
| **典型应用**     | 数据压缩             | 图像生成             | 图像分词、自回归生成 |

#### **六、总结**
VQ-VAE通过向量量化将图像离散化为可学习的码本向量，解决了传统VAE生成模糊的问题，并为图像生成任务提供了高效的Tokenization方案。其核心创新在于：
1. **离散表征**：码本向量捕捉图像的关键特征，提升生成清晰度。
2. **梯度直通估计**：解决量化操作不可导问题，实现端到端训练。
3. **可扩展性**：与Transformer结合（如VQ-GAN、DALL-E），推动生成模型发展。
