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


'''

Perplexity explanation

Perplexity is often a better evaluation metric because it is easier to interpret: 
it measures the model’s effective uncertainty, or roughly how many plausible next-token choices the model is considering on average. 
Lower perplexity means the model assigns higher probability to the true sequence, making it a more intuitive summary of language-model quality than raw cross-entropy

'''

    
    