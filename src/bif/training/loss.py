"""Loss functions for language model training and evaluation."""

from __future__ import annotations

import torch
from torch import nn

from bif.constants import IGNORE_INDEX


def per_example_causal_lm_loss(
    labels: torch.Tensor,
    logits: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Compute per-example causal LM loss.

    Args:
        labels: Token labels of shape (batch, seq_len).
        logits: Model logits of shape (batch, seq_len, vocab_size).
        ignore_index: Label value to ignore in loss computation.

    Returns:
        Per-example loss tensor of shape (batch,).
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_mask = shift_labels.ne(ignore_index)
    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)
    vocab_size = shift_logits.size(-1)
    flat_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    per_token_loss = flat_loss.view_as(shift_labels)
    valid_counts = valid_mask.sum(dim=1).clamp(min=1)
    return per_token_loss.sum(dim=1) / valid_counts
