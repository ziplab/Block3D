from typing import Optional

import torch
import torch.nn.functional as F


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    curr_pos_id: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.
    Args:
        x (torch.Tensor): The input tensor.
        freqs_cis (torch.Tensor): A tensor containing the precomputed rotary
            frequency components.
        curr_pos_id (Optional[torch.Tensor]): An optional tensor specifying the
            current position IDs to use for selecting a subset of `freqs_cis`.
            If None, the function uses the last `seq_len` positions.
    Returns:
        torch.Tensor: The input tensor `x` with rotary positional embeddings
        applied.
    """
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    if curr_pos_id is None:
        freqs_cis = freqs_cis[:, -x.shape[2] :].unsqueeze(1)
    else:
        freqs_cis = freqs_cis[:, curr_pos_id, :].unsqueeze(1)
    y = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return y.type_as(x)


@torch.no_grad
def precompute_freqs_cis(dim: int, t: torch.Tensor, theta: float = 10000.0):
    """Calculate rotary embedding cos & sin, this is useful when every blocks in the network use same positional embedding.

    Args:
        dim (int): dimension of the single head of the transformer block
        t (torch.Tensor): position ids [..., L]
        theta (int, optional): rope theta. Defaults to 10000.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: tuple of cos and sin of rope
    """
    assert dim % 2 == 0, (
        "RoPE only supports embedding dimensions that are multiples of 2"
    )
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=t.device) / dim)
    )
    # [batch_size, seq_len, num_freqs]
    if t.numel() == 0:
        return torch.empty(
            (*t.shape, freqs.shape[0]),
            dtype=torch.complex64,
            device=t.device,
        )
    freqs = torch.outer(t.contiguous().view(-1), freqs).reshape(*t.shape, -1)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def scaled_dot_product_attention_with_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    freqs_cis: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    curr_pos_id: Optional[torch.Tensor] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Computes scaled dot product attention on query, key and value tensors
    with rotary position embeddings on query and key.

    Without caching enabled,
        q should be (bs, nh, seqlen, hd).
        k and v should stay unchanged, (bs, nh, seqlen, hd).
    With caching enabled,
        q should be (bs, nh, 1, hd).
        k and v should stay unchanged, (bs, nh, 1, hd).
        causal_mask must be False.
    """
    q = apply_rotary_emb(q, freqs_cis, curr_pos_id=curr_pos_id)  # (bs, nh, l, hd)
    k = apply_rotary_emb(k, freqs_cis, curr_pos_id=None)  # (bs, nh, s + l, hd)

    x = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal and attn_mask is None,
    )
    return x
