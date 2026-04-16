from __future__ import annotations

import os
from typing import Any

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def cross_entropy_loss(logits: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """
    logits: (..., vocab_size)
    targets: (...) integer class indices

    returns:
        scalar tensor = average cross-entropy over all batch-like dimensions
    """
    
    # Numerical stability: subtract max logit
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_logits
    
    # logsumexp, compute stable log-sum-exp
    log_denom = torch.log(torch.exp(shifted).sum(dim=-1))
    
    # Gather the target logit
    target_logits = shifted.gather(
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)
    
    # CE = -log softmax(target) = logsumexp - target_logit (subtract the target class logit)
    loss = log_denom - target_logits

    # average over all batch-like dimensions
    return loss.mean()



    