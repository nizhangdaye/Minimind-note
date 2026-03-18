import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import MiniMindConfig
from Pos_emb import apply_rotary_pos_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复 Key/Value heads 以实现 GQA (Grouped Query Attention)

    GQA 是一种注意力机制优化，使用较少的 KV heads 来匹配更多的 Query heads。
    例如：8 个 Query heads 对应 2 个 KV heads，每个 KV head 需要重复 4 次。

    这样可以减少 KV 缓存的大小，在推理时节省显存。

    Args:
        x: Key 或 Value 张量 [batch, seq_len, num_kv_heads, head_dim]
        n_rep: 每个 KV head 需要重复的次数（n_rep = num_heads / num_kv_heads）

    Returns:
        重复后的张量 [batch, seq_len, num_heads, head_dim]
    """
    bs, slen, num_key_val_heads, head_dim = x.shape

    # 如果不需要重复，直接返回
    if n_rep == 1:
        return x

    # 在维度 3 插入新维度，然后扩展并重塑
    # 例如：[B, L, 2, D] -> [B, L, 2, 1, D] -> [B, L, 2, 4, D] -> [B, L, 8, D]
    return (
        x[:, :, :, None, :]  # 在维度三插入新维度
        .expand(-1, -1, -1, n_rep, -1)  # 扩展新维度以重复 KV heads
        .reshape(bs, slen, num_key_val_heads * n_rep, head_dim)  # 合并第三四维
    )


class Attention(nn.Module):
    """
    多头注意力机制（支持 GQA 和 Flash Attention）

    实现了标准的缩放点积注意力（Scaled Dot-Product Attention），支持：
    1. GQA (Grouped Query Attention): 使用较少的 KV heads 匹配更多的 Query heads
    2. Flash Attention: 使用 PyTorch 2.0+ 的优化注意力实现
    3. RoPE: 通过旋转位置编码将位置信息注入 Q 和 K
    4. KV Cache: 支持推理时的 KV 缓存加速

    注意力公式：
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """

    def __init__(self, args: MiniMindConfig):
        super().__init__()
        # ========== GQA 配置 ==========
        # 处理GQA：如果没有指定kv头数，则使用与query相同的头数
        self.num_key_value_heads = (
            args.num_key_value_heads
            if args.num_key_value_heads is not None
            else args.num_attention_heads
        )

        # 确保query头数能被kv头数整除（GQA的基本要求）
        assert args.num_attention_heads % self.num_key_value_heads == 0

        self.n_loacl_heads = args.num_attention_heads  # Q heads 数量
        self.n_local_kv_heads = self.num_key_value_heads  # KV heads 数量
        self.n_rep = self.n_loacl_heads // self.n_local_kv_heads  # 每个KV head需要重复的次数
        self.head_dim = args.hidden_size // args.num_attention_heads  # 每个头的维度

        # ========== 线性投影层 ==========
        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        # ========== Dropout ==========
        self.attn_dropout = nn.Dropout(args.dropout)  # 注意力权重的 dropout
        self.resid_dropout = nn.Dropout(args.dropout)  # 残差连接的 dropout
        self.dropout = args.dropout  # dropout 率

        # ========== Flash Attention 配置 ==========
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention") and args.flash_attn
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        前向传播

        Args:
            x: 输入张量 [batch, seq_len, hidden_size]
            position_embeddings: RoPE 位置编码 (cos, sin) 元组
            past_key_value: 缓存的 KV 值，用于增量解码 [batch, past_len, num_kv_heads, head_dim]
            use_cache: 是否返回 KV 缓存供下次使用
            attention_mask: 注意力掩码 [batch, seq_len]，1 表示有效位置，0 表示掩码位置

        Returns:
            output: 注意力输出 [batch, seq_len, hidden_size]
            past_kv: 新的 KV 缓存（如果 use_cache=True），否则为 None
        """
        bsz, seq_len, _ = x.shape

        # ========== 步骤 1：Q/K/V 投影 ==========
        # 将输入 x 投影到 Q/K/V 空间
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 将 Q/K/V 张量重塑为多头格式 [batch, seq_len, num_heads, head_dim]
        xq = xq.view(bsz, seq_len, self.n_loacl_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # GQA
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # GQA

        # ========== 步骤 2：应用 RoPE 位置编码 ==========
        # 通过旋转位置编码将位置信息注入 Q 和 K
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # ========== 步骤 3：KV Cache 处理 ==========
        # 如果有缓存的 KV 值（增量解码），将其与当前 KV 拼接
        if past_key_value is not None:
            # 在（时间）序列维度（dim=1）上拼接：[batch, past_len+seq_len, num_kv_heads, head_dim]
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        # 如果需要缓存，保存当前的 KV 值
        past_kv = (xk, xv) if use_cache else None

        # ========== 步骤 4：重复 KV heads（GQA） ==========
        # 调整维度顺序为 [batch, num_heads, seq_len, head_dim]（Flash Attention 格式）
        # 对于 KV，需要重复 heads 以匹配 Query heads 数量
        xq = xq.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]

        # ========== 步骤 5：计算注意力 ==========
        if (
            self.flash
            and (seq_len > 1)
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            # Flash Attention：使用 PyTorch 优化的注意力实现
            # 条件：序列长度 > 1，没有 KV cache，没有复杂掩码
            attn_output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=None,
                dropout_p=self.dropout,
                is_causal=True,  # 自动应用因果掩码
            )
        else:
            # 标准注意力计算
            # 步骤 5.1：计算注意力权重 QK^T / sqrt(d_k)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(
                self.head_dim
            )  # [batch, num_heads, seq_len, kv_len]

            # 步骤 5.2：应用因果掩码
            # 上三角（对角线以上）置为 -inf，防止看到未来信息
            scores[:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1
            )

            # 步骤 5.3：应用额外的注意力掩码（如果提供）
            if attention_mask is not None:
                # 将掩码扩展到 [batch, 1, 1, seq_len] 并转换为分数掩码
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (
                    1.0 - extended_attention_mask
                ) * -1e9  # 0 -> -inf, 1 -> 0
                scores = scores + extended_attention_mask

            # 步骤 5.4：Softmax 归一化
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)

            # 步骤 5.5：应用 dropout
            scores = self.attn_dropout(scores)

            # 步骤 5.6：加权求和得到注意力输出
            output = scores @ xv  # [batch, num_heads, seq_len, head_dim]

        # ========== 步骤 6：合并多头 & 输出投影 ==========
        # 重塑并投影回 hidden_size 维度
        output = output.transpose(1, 2).reshape(
            bsz, seq_len, -1
        )  # [batch, seq_len, num_heads*head_dim]
        output = self.resid_dropout(self.out_proj(output))  # 输出投影和残差 dropout

        return output, past_kv
