import torch
import torch.nn as nn


class RSMNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-5):
        """
        Args:
            dim: 输入张量的最后一个维度大小
            eps: 防止除零的数值稳定性常数
        """
        super().__init__()
        self.eps = eps

        # 可学习的缩放参数，初始化为全 1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        公式: y = x / sqrt(mean(x^2) + eps)
        Args:
            x: 输入张量，形状为 (batch_size, seq_length, dim)
        """
        # x.pow(2)：计算 x 的平方
        # .mean(dim=-1, keepdim=True)：在最后一个维度上计算均值，保持维度不变
        # torch.rsqrt()：计算输入的平方根的倒数，即 1/sqrt(x)
        # 直接调用 rsqrt 比先 sqrt 再 1 / 更高效，尤其在 GPU 上
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_length, dim)
        """
        # 先转为 float32 进行归一化计算（提高数值稳定性）
        # 然后转回原始精度

        return self.weight * self._norm(x.float()).type_as(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        """
        公式: y = (x - mean) / sqrt(var + eps) * weight + bias
        Args:
            x: 输入张量，形状为 (batch_size, seq_length, dim)
        """
        mean = x.mean(dim=-1, keepdim=True)  # 均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 方差
        normed = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * normed + self.bias


if __name__ == "__main__":
    # 测试 RSMNorm 的功能
    norm = RSMNorm(dim=4)
    x = torch.randn(2, 3, 4)  # (batch_size=2, seq_length=3, dim=4)
    output = norm(x)
    print("Input:\n", x)
    print("Output:\n", output)

    # 测试 LayerNorm 的功能
    layer_norm = LayerNorm(dim=4)
    output_layer = layer_norm(x)
    print("LayerNorm Output:\n", output_layer)
