import torch
import torch.nn.functional as F


def top_p_filtering(logits, top_p: float = 1.0):
    """
    Filter a distribution of logits using top-p filtering.
    The input logits tensor is modified in-place.

    Args:
        logits (torch.Tensor): A tensor of logits to be filtered. Expected shape is [..., vocab_size].
        top_p (float, optional): The cumulative probability threshold for top-p sampling.
               If < 1.0, only keep the smallest set of tokens whose
               cumulative probability does not exceed this threshold.

    Returns:
        torch.Tensor: logits where values outside the top-p threshold are set to -∞.
    """
    if top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum(dim=-1) > top_p
        sorted_idx_to_remove[..., 0] = False

        idx_to_remove = sorted_idx_to_remove.scatter(
            -1, sorted_idx, sorted_idx_to_remove
        )
        logits.masked_fill_(idx_to_remove, -torch.inf)

    return logits


def sample_from_logits(
    logits: torch.Tensor,
    top_p: float = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select candidate token ids and return their probabilities.

    Args:
        logits: Tensor shaped [..., vocab_size].
        top_p: Optional nucleus sampling threshold.

    Returns:
        Tuple of:
            - token ids shaped [..., 1]
            - selected token probabilities shaped [..., 1]
    """
    if top_p is None:
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        probs = F.softmax(logits, dim=-1)
    else:
        filtered_logits = top_p_filtering(logits.clone(), top_p=top_p)
        probs = F.softmax(filtered_logits, dim=-1)
        flat_probs = probs.reshape(-1, probs.shape[-1])
        next_id = torch.multinomial(flat_probs, num_samples=1, replacement=True)
        next_id = next_id.view(*probs.shape[:-1], 1)

    next_prob = probs.gather(dim=-1, index=next_id)
    return next_id, next_prob


def process_logits(
        logits,
        top_p: float = None,
    ):
    """
    Process logits by optionally applying nucleus (top-p) filtering and token selection.

    If `top_p` is None, the token with the highest probability (argmax) is selected.
    If `top_p` is provided, smallest set of tokens with cumulative probability ≥ top_p are kept, then softmax is applied to obtain
    probabilities. A token is sampled from this filtered distribution using `torch.multinomial`.

    Args:
        logits (torch.Tensor): A tensor of logits to process.
        top_p (float, optional): The cumulative probability threshold for nucleus sampling.
            If None, argmax selection is performed (deterministic generation). Otherwise, smallest set of tokens with cumulative probability ≥ top_p are kept (stochastic generation).

    Returns:
        torch.Tensor: selected token index.
    """
    next_id, _ = sample_from_logits(logits, top_p=top_p)
    return next_id
