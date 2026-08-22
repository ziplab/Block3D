import math

import torch
import torch.nn as nn

from block3d.model.transformers.norm import LayerNorm, RMSNorm


def init_linear(module, embed_dim: int):
    """
    Initializes the weights and biases of a given linear module.
    Args:
        module (nn.Module): The module to initialize. Expected to be an instance of nn.Linear.
        embed_dim (int): The embedding dimension used to calculate the standard deviation
                         for weight initialization.
    Returns:
        None
    """

    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=math.sqrt(1.0 / embed_dim))
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


def init_tfixup(module: nn.Module, num_layers: int):
    """T-Fixup initialization for transformer layers.

    Args:
        module (nn.Module): decoder/encoder module
        num_layers (int): number of layers in the module
    """
    with torch.no_grad():
        for pn, p in module.named_parameters():
            if (
                pn.endswith("c_proj.weight")
                or pn.endswith("up_proj.weight")
                or pn.endswith("down_proj.weight")
            ):
                p *= (4 * num_layers) ** (-0.25)
            elif pn.endswith("c_v.weight"):
                p *= (4 * num_layers) ** (-0.25) * math.sqrt(2)


class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, bias=True, approximate="none"):
        """
        MLP with GELU activation function."
        """

        super().__init__()
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.act_fn = nn.GELU(approximate=approximate)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        eps: float = 1e-6,
    ):
        """
        Initializes the self attention mechanism.
        Args:
            embed_dim (int): The dimensionality of the embedding space.
            num_heads (int): The number of attention heads.
            bias (bool, optional): Whether to include bias terms in the linear layers. Defaults to True.
            eps (float, optional): A small value added for numerical stability. Defaults to 1e-6.
        Raises:
            AssertionError: If `embed_dim` is not divisible by `num_heads`.
        """

        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.c_qk = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)
        self.c_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        head_dim = embed_dim // num_heads
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(
        self,
        x,
        attn_mask=None,
        is_causal: bool = False,
    ):
        """
        Performs the forward pass of the attention mechanism.
        Args:
            x (torch.Tensor): Input tensor.
            attn_mask (Optional[torch.Tensor]): Attention mask to apply. Default is None.
            is_causal (bool): If True, applies a causal mask to prevent attending to future positions.
                              Default is False.
        Returns:
            torch.Tensor: Output tensor after applying
                          the attention mechanism and projection.
        """

        b, l, d = x.shape

        q, k = self.c_qk(x).chunk(2, dim=-1)
        v = self.c_v(x)

        q = q.view(b, l, self.num_heads, -1).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(b, l, self.num_heads, -1).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(b, l, self.num_heads, -1).transpose(1, 2)  # (B, nh, T, hs)

        q = self.q_norm(q)
        k = self.k_norm(k)
        effective_is_causal = is_causal and attn_mask is None

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=effective_is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(b, l, d)

        y = self.c_proj(y)
        return y


class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        q_dim=None,
        kv_dim=None,
        bias: bool = True,
    ):
        """
        Initializes the cross attention mechanism.
        Args:
            embed_dim (int): The dimensionality of the embedding space.
            num_heads (int): The number of attention heads.
            q_dim (int, optional): The dimensionality of the query input. Defaults to `embed_dim`.
            kv_dim (int, optional): The dimensionality of the key and value inputs. Defaults to `embed_dim`.
            bias (bool, optional): Whether to include a bias term in the linear projections. Defaults to True.
        Raises:
            AssertionError: If `embed_dim` is not divisible by `num_heads`.
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        q_dim = q_dim or embed_dim
        kv_dim = kv_dim or embed_dim

        self.c_q = nn.Linear(q_dim, embed_dim, bias=bias)
        self.c_k = nn.Linear(kv_dim, embed_dim, bias=bias)
        self.c_v = nn.Linear(kv_dim, embed_dim, bias=bias)
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.num_heads = num_heads

    def forward(self, x, c, attn_mask=None, is_causal: bool = False):
        """
        Forward pass for the attention mechanism.
        Args:
            x (torch.Tensor): Input tensor of shape.
            c (torch.Tensor): Context tensor.
            attn_mask (torch.Tensor, optional): Attention mask.
                Defaults to None.
            is_causal (bool, optional): Whether to apply causal masking. Defaults to False.
        Returns:
            torch.Tensor: Output tensor.
        """

        q, k = self.c_q(x), self.c_k(c)
        v = self.c_v(c)

        b, l, d = q.shape
        s = k.shape[1]

        q = q.view(b, l, self.num_heads, -1).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(b, s, self.num_heads, -1).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(b, s, self.num_heads, -1).transpose(1, 2)  # (B, nh, T, hs)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=(attn_mask is not None) and is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(b, l, d)

        y = self.c_proj(y)
        return y


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        eps: float = 1e-6,
    ) -> None:
        """
        Initializes the EncoderLayer module.
        Args:
            embed_dim (int): The dimensionality of the embedding space.
            num_heads (int): The number of attention heads.
            bias (bool, optional): Whether to include bias terms in the layers. Defaults to True.
            eps (float, optional): A small value added for numerical stability in normalization layers. Defaults to 1e-6.
        """
        super().__init__()
        self.ln_1 = LayerNorm(embed_dim, elementwise_affine=False, eps=eps)
        self.attn = SelfAttention(embed_dim, num_heads, bias=bias, eps=eps)
        self.ln_2 = LayerNorm(embed_dim, elementwise_affine=False, eps=eps)
        self.mlp = MLP(embed_dim=embed_dim, hidden_dim=embed_dim * 4, bias=bias)

    def forward(
        self,
        x,
        attn_mask=None,
        is_causal: bool = False,
    ):
        """
        Performs the forward pass of the transformer block.
        Args:
            x (torch.Tensor): The input tensor.
            attn_mask (torch.Tensor, optional): An optional attention mask tensor to apply during the
                attention computation. Default is None.
            is_causal (bool, optional): If True, applies a causal mask to prevent attention to future
                positions. Default is False.
        Returns:
            torch.Tensor: The output tensor of the same shape as the input.
        """

        x = x + self.attn(
            self.ln_1(x),
            attn_mask=attn_mask,
            is_causal=is_causal,
        )
        x = x + self.mlp(self.ln_2(x))
        return x


class EncoderCrossAttentionLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        q_dim=None,
        kv_dim=None,
        bias: bool = True,
        eps: float = 1e-6,
    ) -> None:
        """
        Initializes the EncoderAttentionLayer module with cross-attention,
        and a feed-forward MLP.
        Args:
            embed_dim (int): The dimensionality of the embedding space.
            num_heads (int): The number of attention heads.
            q_dim (int, optional): Dimensionality of the query input. Defaults to `embed_dim`.
            kv_dim (int, optional): Dimensionality of the key and value inputs. Defaults to `embed_dim`.
            bias (bool, optional): Whether to include bias terms in the layers. Defaults to True.
            eps (float, optional): A small value added to the denominator for numerical stability
                in layer normalization. Defaults to 1e-6.
        """
        super().__init__()

        q_dim = q_dim or embed_dim
        kv_dim = kv_dim or embed_dim

        self.attn = CrossAttention(
            embed_dim,
            num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
            bias=bias,
        )

        self.ln_1 = LayerNorm(q_dim, elementwise_affine=False, eps=eps)
        self.ln_2 = LayerNorm(kv_dim, elementwise_affine=False, eps=eps)

        self.ln_f = LayerNorm(embed_dim, elementwise_affine=False, eps=eps)
        self.mlp = MLP(embed_dim=embed_dim, hidden_dim=embed_dim * 4, bias=bias)

    def forward(self, x, c, attn_mask=None, is_causal: bool = False):
        """
        Forward pass for the attention mechanism.
        Args:
            x (torch.Tensor): The input tensor to the attention mechanism.
            c (torch.Tensor): The context tensor used for cross-attention.
            attn_mask (torch.Tensor, optional): An optional attention mask to control
                which positions can attend to others. Defaults to None.
            is_causal (bool, optional): If True, applies a causal mask to prevent
                attending to future positions. Defaults to False.
        Returns:
            torch.Tensor: The output tensor after applying attention and MLP layers.
        """

        x = x + self.attn(
            self.ln_1(x), self.ln_2(c), attn_mask=attn_mask, is_causal=is_causal
        )
        x = x + self.mlp(self.ln_f(x))
        return x
