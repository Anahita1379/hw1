
from __future__ import annotations

import os
from typing import Any

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

import math
import torch

from eecs148b_hw1.softmax import softmax

'''
Your implementation
should handle keys and queries of shape (batch size, ..., seq len, d k) and values
of shape (batch size, ..., seq len, d v), where ... represents any number of other
batch-like dimensions (if provided). The implementation should return an output with the
shape (batch size, ..., seq len, d v).
Your implementation should also support an optional user-provided boolean mask of shape
(seq len, seq len). The attention probabilities of positions with a mask value of True
should collectively sum to 1, and the attention probabilities of positions with a mask value
of False should be zero

'''

'''
For the masking insight:

Set every masked-out pre-softmax score to negative infinity (or a very large negative number), 
so after softmax those positions get probability 0.
'''



def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Q: (batch size,..., seq_len_q, d_k)
    K: (batch size,..., seq_len_k, d_k)
    V: (batch size,..., seq_len_k, d_v)
    mask: optional boolean tensor of shape (seq_len_q, seq_len_k)
          True = attention probabilities of positions should collectively sum to 1
          False = the attention probabilities of positions  should be zero
    Returns:
        (..., seq_len_q, d_v)
    """
    d_k = Q.shape[-1]
    d_v = V.shape[-1]
    
    scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k) #(..., seq_len_q, seq_len_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
        
    attn = softmax(scores, dim=-1)
    output = torch.matmul(attn, V) #(..., seq_len_q, d_v)
    return output   
    
    