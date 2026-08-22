import torch
import torch.nn as nn


def fused_rms_norm(x: torch.Tensor, weight: nn.Parameter, eps: float):
    """
    Applies a fused Root Mean Square (RMS) normalization to the input tensor.
    Args:
        x (torch.Tensor): The input tensor to be normalized. Expected to have
            at least one dimension.
        weight (nn.Parameter): A learnable parameter used to scale the normalized
            tensor. Its shape must be broadcastable to the shape of `x`.
        eps (float): A small constant added to the denominator for numerical
            stability during normalization.
    Returns:
        torch.Tensor: The normalized and scaled tensor with the same shape as `x`.
    """

    x = x.float()
    return (x * torch.rsqrt((x * x).mean(-1, keepdim=True).add_(eps))) * weight


class LayerNorm(nn.LayerNorm):
    def forward(self, input: torch.Tensor):
        """
        Wrapper to ensure that the input tensor is cast to float before normalization.
        """
        y = super().forward(input.float())
        return y.type_as(input)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True):
        """
        Initializes the normalization layer.
        Args:
            dim (int): The number of features in the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-5.
            elementwise_affine (bool, optional): If True, this layer will have learnable per-element affine parameters. Defaults to True.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=elementwise_affine)

    def forward(self, x):
        return fused_rms_norm(x, weight=self.weight, eps=self.eps).type_as(x)
