import math
from typing import Optional

import torch


def precompute_freqs_cis(
    dim: int,
    end: int = (32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    """
    预计算 RoPE 的频率矩阵（cos 和 sin 值）。

    数学流程：
        1. θ_j = base^(-2j/d)           → 基础频率向量
        2. [YaRN] θ'_j = θ_j * ((1-γ) + γ/s)  → 分段频率缩放
        3. Φ_{m,j} = m * θ_j            → 角度矩阵（外积）
        4. 输出 cos(Φ), sin(Φ)           → 供 apply_rotary_pos_emb 使用

    Args:
        dim: 每个注意力头的维度（head_dim），即公式中的 d
        end: 最大序列长度（默认 32768）
        rope_base: 基频参数（默认 1e6），即公式中的 base
        rope_scaling: YaRN 外推配置字典，None 则不使用外推

    Returns:
        freqs_cos: [end, dim] 的 cos 值
        freqs_sin: [end, dim] 的 sin 值
    """
    # ===== 步骤 1：计算基础频率 =====
    # θ_j = base^(-2j/d)
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1,
    )

    # ===== 步骤 2：YaRN 外推（如果启用） =====
    if rope_scaling:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),  # 扩展倍数 s
            rope_scaling.get("beta_fast", 32.0),  # 高频边界参数 β_fast
            rope_scaling.get("beta_slow", 1.0),  # 低频边界参数 β_slow
            rope_scaling.get("attention_factor", 1.0),  # 温度修正因子
        )

        # 如果目标长度超过训练长度，应用 YaRN 外推
        if end / orig_max > 1.0:
            # 计算分段边界：inv_dim(β) = d·ln(L_train / (β·2π)) / (2·ln(base))
            inv_dim = lambda b: (
                (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            )
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)

            # 斜坡函数：γ(j) = clamp((j - low) / (high - low), 0, 1)
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            # 应用 YaRN 分段缩放：θ'_j = θ_j · [(1-γ) + γ/s]
            freqs = freqs * (1 - ramp + ramp / factor)

    # ===== 步骤 3：计算所有位置的频率 =====
    # Φ_{m,j} = m · θ_j（外积生成角度矩阵）
    t = torch.arange(end, device=freqs.device)  # 位置索引 [0, 1, 2, ..., end-1]
    freqs = torch.outer(t, freqs)  # shape = [end, dim//2]

    # ===== 步骤 4：输出 cos(Φ) 和 sin(Φ)，乘以温度修正因子 =====
    # 由于 RoPE 使用复数旋转，需要将 dim/2 的频率复制到完整的 dim 维度
    freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1) * attn_factor
    freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1) * attn_factor

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    对 Query 和 Key 应用旋转位置编码。

    数学公式：
        RoPE(x, m) = x ⊙ cos(mθ) + rotate_half(x) ⊙ sin(mθ)

    这是复数旋转 x · e^{imθ} 的实数等价形式。

    Args:
        q: Query [batch, seq_len, num_heads, head_dim]
        k: Key   [batch, seq_len, num_kv_heads, head_dim]
        cos: 预计算的 cos 值 [seq_len, head_dim]
        sin: 预计算的 sin 值 [seq_len, head_dim]
        position_ids: 位置索引（未使用，cos/sin 已包含位置信息）
        unsqueeze_dim: 扩展维度以匹配 q/k 形状（默认 1）

    Returns:
        q_embed: 应用 RoPE 后的 Query [batch, seq_len, num_heads, head_dim]
        k_embed: 应用 RoPE 后的 Key [batch, seq_len, num_kv_heads, head_dim]
    """

    def rotate_half(x):
        """
        将向量后半部分取负放到前面：[a,b,c,d] → [-c,-d,a,b]
        这实现了复数旋转的实部/虚部交换。
        """
        return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)

    # 调整 cos 和 sin 的形状以匹配 q/k：[seq_len, head_dim] -> [seq_len, 1, head_dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # RoPE(x, m) = x ⊙ cos(mθ) + rotate_half(x) ⊙ sin(mθ)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


if __name__ == "__main__":
    # 测试预计算频率矩阵
    dim = 64
    end = 2048  # 假设训练时最大长度为 2048
    rope_base = 10000
    freqs_cos, freqs_sin = precompute_freqs_cis(dim, end, rope_base)
    print("预计算的 cos 矩阵形状:", freqs_cos.shape)  # [end, dim]
    print("预计算的 sin 矩阵形状:", freqs_sin.shape)  # [end, dim]

    # 测试应用 RoPE
    batch_size = 2
    seq_len = 2047  # 测试接近最大长度的情况
    num_heads = 4

    q = torch.randn(batch_size, seq_len, num_heads, dim)
    k = torch.randn(batch_size, seq_len, num_heads, dim)

    q_embed, k_embed = apply_rotary_pos_emb(q, k, freqs_cos[:seq_len], freqs_sin[:seq_len])
    print("应用 RoPE 后的 Query 形状:", q_embed.shape)  # [batch_size, seq_len, num_heads, head_dim]
    print("应用 RoPE 后的 Key 形状:", k_embed.shape)  # [batch_size, seq_len, num_heads, head_dim]
