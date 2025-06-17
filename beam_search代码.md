# Beam Search 解码过程详解

Beam Search 是一种常用的解码策略，用于在生成序列时平衡生成质量和计算效率。与贪婪搜索（每次选择概率最高的词）不同，beam search 会维护多个候选序列（称为"beam"），并在每一步选择最有可能的候选序列。

## Beam Search 的基本原理

1. **初始化**：从起始标记（如 `<s>`）开始，初始 beam 包含一个空序列。
2. **扩展**：在每一步，对当前 beam 中的每个候选序列，生成下一个可能的所有标记的概率。
3. **选择**：从所有可能的扩展中，选择概率最高的 `beam_width` 个序列作为新的 beam。
4. **终止**：当达到最大序列长度或遇到结束标记（如 `</s>`）时停止。

## 为什么使用 Beam Search？

- **贪婪搜索**：简单但容易陷入局部最优
- **穷举搜索**：理论上最优但计算复杂度高（O(V^T)，V 是词汇量，T 是序列长度）
- **Beam Search**：在两者之间提供折中，保持较好的生成质量同时计算可行

## 代码实现

以下是使用 PyTorch 实现 Beam Search 的示例代码，代码注释使用中文：

```python
import torch
import torch.nn.functional as F

def beam_search_decode(model, tokenizer, input_ids, max_length=20, beam_width=5):
    """
    Beam search 解码实现
    
    参数:
        model: Transformer 模型
        tokenizer: 分词器实例
        input_ids: 输入的 token IDs (batch_size, seq_length)
        max_length: 生成的最大序列长度
        beam_width: 维护的 beam 数量
        
    返回:
        生成的序列列表及其得分
    """
    # 初始化
    batch_size = input_ids.size(0)
    device = input_ids.device
    
    # 维护一个 beams 列表，每个元素是一个元组: (序列, 得分, 长度归一化因子)
    beams = [([], 0.0, 1.0) for _ in range(batch_size)]
    
    # 初始输入
    current_ids = input_ids.clone()
    
    for _ in range(max_length):
        # 获取当前序列的模型预测
        with torch.no_grad():
            outputs = model(current_ids)
            logits = outputs.logits[:, -1, :]  # 我们只需要最后一个 token 的预测
        
        # 转换为概率并获取每个 beam 的 top k 候选
        probs = F.softmax(logits, dim=-1)
        top_probs, top_tokens = probs.topk(beam_width, dim=-1)
        
        new_beams = []
        
        for batch_idx in range(batch_size):
            # 当前 batch 元素的 beam 状态
            current_beam = beams[batch_idx]
            current_seq, current_score, current_len = current_beam
            
            # 如果已经到达 EOS，保持 beam 不变
            if len(current_seq) > 0 and current_seq[-1] == tokenizer.eos_token_id:
                new_beams.append((current_seq, current_score, current_len))
                continue
                
            # 用每个可能的下一个 token 扩展每个 beam
            for token_idx, (prob, token) in enumerate(zip(top_probs[batch_idx], top_tokens[batch_idx])):
                new_seq = current_seq.copy()
                new_seq.append(token.item())
                
                # 计算新得分（对数概率）
                new_score = current_score + torch.log(prob)
                
                # 长度归一化因子（可选）
                # 防止偏向短序列
                length_penalty = ((5.0 + len(new_seq)) / 6.0) ** 0.65  # 来自 GNMT 论文
                normalized_score = new_score / length_penalty
                
                new_beams.append((new_seq, normalized_score, length_penalty))
        
        # 从所有可能性中选择 top beam_width 个序列
        # 展平所有 beams 并按得分排序
        all_beams = []
        for beam in new_beams:
            all_beams.append(beam)
            
        # 按得分降序排序
        all_beams.sort(key=lambda x: x[1], reverse=True)
        
        # 选择 top beam_width 个 beams
        selected_beams = all_beams[:beam_width]
        
        # 更新 beams 用于下一次迭代
        beams = selected_beams
        
        # 准备下一步的输入
        next_ids = torch.full((batch_size, 1), tokenizer.pad_token_id, dtype=torch.long, device=device)
        for i, beam in enumerate(selected_beams):
            seq, _, _ = beam
            next_ids[i, 0] = seq[-1] if len(seq) > 0 else tokenizer.pad_token_id
            
        current_ids = torch.cat([current_ids, next_ids], dim=1)
        
        # 检查所有 beams 是否都到达了 EOS
        all_done = True
        for beam in selected_beams:
            if len(beam[0]) > 0 and beam[0][-1] != tokenizer.eos_token_id:
                all_done = False
                break
        if all_done:
            break
    
    # 返回最佳序列（取消得分归一化）
    final_sequences = []
    for beam in selected_beams:
        seq, score, length_penalty = beam
        final_score = score * length_penalty  # 取消归一化
        final_sequences.append((seq, final_score))
    
    return final_sequences
```

## 代码说明

1. **初始化**：从输入序列开始，初始化 beams 列表。
2. **循环生成**：
   - 使用模型预测下一个 token 的概率
   - 对每个 beam 扩展 top-k 个候选 token
   - 计算每个新序列的得分（对数概率）
   - 应用长度归一化（可选）
3. **选择 top beams**：从所有可能的扩展中选择得分最高的 beam_width 个序列
4. **终止条件**：当所有 beams 都生成结束标记或达到最大长度时停止

## 优化技巧

1. **长度归一化**：防止模型偏向生成短序列
2. **禁止重复 n-gram**：避免生成重复内容
3. **采样策略**：可以使用 top-k 采样或 nucleus 采样与 beam search 结合
4. **早停**：当所有 beams 都生成结束标记时提前终止

Beam search 是生成任务中的基础算法，许多现代模型（如 GPT、BERT 等）都使用其变体进行解码。
