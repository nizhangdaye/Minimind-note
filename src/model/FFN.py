import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import MiniMindConfig
from torch.nn import init

ACT2FN = {
    "relu": F.relu,
    "silu": F.silu,  # Swish 激活函数
    "gelu": F.gelu,
    # 可以根据需要添加更多激活函数
}


class FeedForward(nn.Module):
    """
    SwiGLU 前馈网络

    实现了 SwiGLU (Swish-Gated Linear Unit) 激活函数的前馈网络。
    SwiGLU 是 GLU (Gated Linear Unit) 的变体，使用 Swish/SiLU 作为门控激活函数。

    公式：
        FFN(x) = down_proj(Swish(gate_proj(x)) * up_proj(x))

    其中：
        - gate_proj: 门控投影，用于生成门控信号
        - up_proj: 上投影，用于生成特征
        - Swish(x) = x * sigmoid(x) = x * silu(x)
        - down_proj: 下投影，将中间维度映射回 hidden_size

    相比标准 FFN (ReLU(xW1)W2)，SwiGLU 通常有更好的性能。

    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()

        # ========== 中间层维度计算 ==========
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 为了更好地利用 GPU 的并行计算能力（特别是 TensorCore、SIMD 等），中间维度通常会做 64 对齐
            # 向上取整到最近的 64 的倍数
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        # ========== 投影层 ==========
        # 门控投影：hidden_size -> intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # 下投影：intermediate_size -> hidden_size
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # 上投影：hidden_size -> intermediate_size
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

        # ========== Dropout 和激活函数 ==========
        self.dropout = nn.Dropout(config.dropout)
        # 激活函数：通常是 SiLU（Swish）
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        前向传播
        SwiGLU 公式：FFN(x) = down_proj(Swish(gate_proj(x)) * up_proj(x))

        Args:
            x: 输入张量 [batch, seq_len, hidden_size]

        Returns:
            输出张量 [batch, seq_len, hidden_size]
        """
        # 计算门控信号和特征
        gate = self.gate_proj(x)  # [batch, seq_len, intermediate_size]
        up = self.up_proj(x)  # [batch, seq_len, intermediate_size]

        # SwiGLU：Swish(gate) * up
        #   Swish(x) = x * sigmoid(x) = silu(x)
        activated = self.act_fn(gate) * up  # [batch, seq_len, intermediate_size]

        # 下投影回 hidden_size，并应用 dropout
        return self.dropout(self.down_proj(activated))  # [batch, seq_len, hidden_size]


class MoEFeedForward(nn.Module):
    """
    MoE (Mixture of Experts) 前馈网络

    使用多个专家（FeedForward）处理不同的 token，通过门控网络动态选择专家。
    支持路由专家（routed experts）和共享专家（shared experts）两种类型。

    工作流程：
        1. 门控网络为每个 token 选择 top-k 个路由专家
        2. 每个 token 被路由到选中的专家处理
        3. 专家输出按权重加权求和
        4. 共享专家处理所有 token 并添加到输出
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config

        # ========== 路由专家 ==========
        # 路由专家：通过门控网络动态选择，每个 token 只使用 top-k 个专家
        self.expers = nn.ModuleList([FeedForward(config) for _ in range(config.n_routed_experts)])

        # ========== 门控网络 ==========
        # 负责为每个 token 选择专家并计算权重
        self.gate = MoEGate(config)

        # ========== 共享专家 ==========
        # 共享专家：处理所有 token，不经过门控网络
        #   用于提供通用特征，增强模型表达能力
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [batch, seq_len, hidden_size]

        Returns:
            输出张量 [batch, seq_len, hidden_size]
        """
        identity = x  # 保存原始输入
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape

        # ========== 步骤 1：门控网络选择专家 ==========
        # 为每个 token 选择 top-k 个专家，并计算权重
        # topk_idx: [batch*seq_len, top_k] 专家索引
        # topk_weight: [batch*seq_len, top_k] 专家权重
        topk_idx, topk_weight, aux_loss = self.gate(x)

        # ========== 步骤 2：路由到专家处理 ==========
        x = x.view(-1, x.shape[-1])  # [batch*seq_len, hidden_size]
        flat_topk_idx = topk_idx.view(-1)  # [batch*seq_len*top_k]  展平

        if self.training:
            # 训练模式：为每个 token 的每个选中专家复制输入
            #   例如：top_k=2，每个 token 需要处理 2 次
            # x: [batch*seq_len, hidden_size] -> [batch*seq_len*top_k, hidden_size]
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)

            y = torch.empty_like(x, dtype=x.dtype)

            # 对每个专家，处理分配给他们的 token
            for i, expert in enumerate(self.expers):
                # 找得到分配给专家 i 的 token 索引
                mask = flat_topk_idx == i
                expert_out = expert(x[mask])

                if expert_out.shape[0] > 0:
                    # 如果有 token 被分配给专家 i，保存输出
                    y[mask] = expert_out.to(y.dtype)
                else:
                    # 没有 token 被分配给专家 i，创建空输出（保持梯度流）
                    y[mask] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())

            # 加权求和：将每个 token 的 top-k 个专家输出加权平均
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(
                dim=1
            )  # [batch*seq_len, hidden_size]
            y = y.view(orig_shape)  # [batch, seq_len, hidden_size]
        else:
            # 推理模式：使用优化的推理函数
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        # ========== 步骤 添加共享专家处理 ==========
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)  # 残差链接

        # 保存辅助损失（用于训练时的负载平衡）
        self.aux_loss = aux_loss

        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        优化的 MoE 推理函数（仅推理时使用）

        通过批量处理每个专家的所有 token，减少计算开销。
        工作流程：
            1. 按专家索引排序 token
            2. 统计每个专家处理的 token 数量
            3. 批量处理每个专家的所有 token
            4. 按权重加权并累加到输出缓存

        Args:
            x: 输入张量 [batch*seq_len, hidden_size]
            flat_expert_indices: 展平的专家索引 [batch*seq_len*top_k]
            flat_expert_weights: 展平的专家权重 [batch*seq_len*top_k, 1]

        Returns:
            输出张量 [batch*seq_len, hidden_size]
        """
        expert_cache = torch.zeros_like(x)  # 输出缓存

        # ========== 步骤 1：按专家索引排序 ==========
        # 将 token 按专家索引排序，使同一专家的 token 聚集在一起
        idxs = flat_expert_indices.argsort()  # 排序后的索引

        # ========== 步骤 2：统计每个专家处理的 token 数量 ==========
        # bincount: 统计每个专家被选中的次数
        # cumsum: 累积和，得到每个专家的 token 范围
        #   例如：[6, 15, 20, 26] 表示：
        #     - 专家 0 处理前 6 个 token
        #     - 专家 1 处理第 6-15 个 token
        #     - 专家 2 处理第 15-20 个 token
        #     - 专家 3 处理第 20-26 个 token
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)

        # 计算每个 token 的原始索引（去除 top_k 的重复）
        token_idxs = idxs // self.config.num_experts_per_tok

        # ========== 步骤 3：批量处理每个专家 ==========
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]

            # 如果该专家没有处理的 token，跳过
            if start_idx == end_idx:
                continue

            # 获取该专家处理的 token 索引
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]  # 原始 token 索引
            expert_tokens = x[exp_token_idx]  # 该专家需要处理的 token

            # 批量处理该专家的所有 token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)

            # 应用权重
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            # 累加到输出缓存（使用 scatter_add 处理同一 token 被多个专家处理的情况）
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
            )

        return expert_cache


class MoEGate(nn.Module):
    """
    MoE (Mixture of Experts) 门控网络

    负责为每个 token 选择 top-k 个专家，并计算专家权重。
    使用辅助损失（auxiliary loss）来鼓励专家负载均衡，防止专家退化。

    工作流程：
        1. 计算每个专家对每个 token 的分数（logits）
        2. 使用 softmax 转换为概率
        3. 选择 top-k 个专家
        4. 计算辅助损失（训练时）
    """

    def __init__(self, config: MiniMindConfig):
        """
        初始化 MoE 门控网络

        Args:
            config: MiniMindConfig 配置对象
        """
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  # 每个 token 选择的专家数量
        self.n_routed_experts = config.n_routed_experts  # 专家总数

        self.scoring_func = config.scoring_func  # 评分函数（'softmax'）
        self.alpha = config.aux_loss_alpha  # 辅助损失权重
        self.seq_aux = config.seq_aux  # 是否在序列级别计算辅助损失

        self.norm_topk_prob = config.norm_topk_prob  # 是否标准化 top-k 概率
        self.gating_dim = config.hidden_size  # 门控网络输入维度

        # 门控网络权重：[n_routed_experts, hidden_size]
        #   每一行对应一个专家的权重向量
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """使用 Kaiming 均匀分布初始化权重"""
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        前向传播：为每个 token 选择专家

        Args:
            hidden_states: 输入张量 [batch, seq_len, hidden_size]

        Returns:
            topk_idx: 选择的专家索引 [batch*seq_len, top_k]
            topk_weight: 专家权重 [batch*seq_len, top_k]
            aux_loss: 辅助损失（标量），用于鼓励负载均衡
        """

        # hidden_states: 输入数据。
        # 形状是 [batch(批次大小), seq_len(句子长度), h(隐藏层维度)]
        # 例如: [2, 10, 512] 表示 2 句话，每句 10 个词，每个词用 512 维向量表示。
        bsz, seq_len, h = hidden_states.shape

        # ========== 步骤 1：计算专家分数 ==========

        # view(-1, h): 改变张量形状（Reshape）。
        # -1 的意思是“自动计算这一维”。
        # 结果形状变为 [batch * seq_len, h]。
        # 含义：把所有句子的所有词平铺开，变成一个长长的列表，因为我们对每个词是独立处理的。
        hidden_states = hidden_states.view(-1, h)

        # F.linear(input, weight): 线性层计算，数学公式是 Y = XW^T。
        # hidden_states 形状 [Total_Tokens, h]
        # self.weight 形状 [n_experts, h]
        # 结果 logits 形状 [Total_Tokens, n_experts]
        # 含义：计算每个 Token 和每个 Expert 的匹配分数（原始分数，未归一化）。
        logits = F.linear(hidden_states, self.weight, None)

        # ========== 步骤 2：转换为概率 ==========
        if self.scoring_func == "softmax":
            # 使用 softmax 将 logits 转换为概率分布
            scores = logits.softmax(dim=-1)  # [batch*seq_len, n_routed_experts]
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        # ========== 步骤 3：选择 top-k 专家 ==========
        # torch.topk: 寻找张量中最大的 k 个值。
        # scores: 来源张量。
        # k=self.top_k: 要选几个（比如 2 个）。
        # dim=-1: 在专家维度上选。
        # sorted=False: 不需要对选出来的结果排序（为了速度）。
        # 返回值：
        #   topk_weight: [batch*seq_len, top_k] 选中的那 k 个专家的概率值。
        #   topk_idx: [batch*seq_len, top_k] 选中的那 k 个专家的索引（ID 号）。
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # ========== 步骤 4：标准化 top-k 概率（可选） ==========
        if self.top_k > 1 and self.norm_topk_prob:
            # 将 top-k 权重标准化，使其和为 1
            #   这样确保每个 token 的专家权重分布是归一化的
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # ========== 步骤 5：计算辅助损失（训练时） ==========
        # 辅助损失用于鼓励专家负载均衡，防止某些专家被过度使用或完全不用
        # 难点来了，坐稳了
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores  # 也就是所有专家原本的概率分布
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)  # [batch, seq_len*top_k]

            if self.seq_aux:
                # === 方案 A：序列级辅助损失 (DeepSeek-V2/V3 常用) ===
                # 这种计算方式更精细，在每条样本内部看负载均衡。

                # 变形回 [batch, seq_len, n_experts]
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)

                # 计算每个专家的使用频率（期望负载）
                # 创建一个全 0 矩阵用来统计次数
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                # scatter_add_: 这是一个复杂的“散射加法”操作。
                # 形象理解：这是在“投票”。
                # topk_idx_for_aux_loss 里的值是专家 ID，它告诉我们每个 Token 投给了谁。
                # 这行代码统计：在这个 Batch 里，每个专家被选中了多少次。
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                # .div_(...): 除以期望的平均次数，将其归一化。
                # 如果 ce = 1，说明该专家被选中的频率正好等于平均水平。

                # 计算损失：(实际使用频率 * 专家平均概率得分)
                # 这种损失设计会迫使模型倾向于让所有专家的使用频率和平均得分趋于一致。
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # === 方案 B：Token 级辅助损失 (传统的 Switch Transformer 做法) ===
                # 这种是全局统计所有 Token。

                # F.one_hot: 独热编码。如果 ID 是 3，变成 [0, 0, 0, 1, 0...]
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )  # 这一行看不懂的话可以问问AI
                ce = mask_ce.float().mean(0)  # [n_routed_experts] - 每个专家的平均使用频率

                # 计算每个专家得到的平均分（模型“想”选它的程度）。
                Pi = scores_for_aux.mean(0)  # [n_routed_experts] - 每个专家的平均分数

                # 计算负载均衡分数
                fi = ce * self.n_routed_experts  # 归一化因子

                # 经典的负载均衡损失公式：
                # minimize (N * sum(Pi * fi))
                # 只有当概率分布是均匀分布时，这个点积最小。
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            # 如果不在训练，或者不需要辅助损失，损失为 0
            aux_loss = scores.new_zeros(1).squeeze()

        return topk_idx, topk_weight, aux_loss


if __name__ == "__main__":
    # 简单测试
    config = MiniMindConfig()
    ffn = FeedForward(config)
    moe_ffn = MoEFeedForward(config)

    x = torch.randn(2, 10, config.hidden_size)  # [batch, seq_len, hidden_size]
    out_ffn = ffn(x)
    out_moe = moe_ffn(x)

    print("FFN output shape:", out_ffn.shape)  # 应该是 [2, 10, hidden_size]
    print("MoE FFN output shape:", out_moe.shape)  # 应该也是 [2, 10, hidden_size]
    print("MoE FFN auxiliary loss:", moe_ffn.aux_loss.item())
