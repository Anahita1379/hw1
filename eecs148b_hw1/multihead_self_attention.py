from __future__ import annotations
import math
import torch
import torch.nn as nn
import json
from typing import Iterable, Iterator
import regex as re
from eecs148b_hw1.linear import Linear
from eecs148b_hw1.scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        '''
        d model: int Dimensionality of the Transformer block inputs.
        num heads: int Number of heads to use in multi-head self-attention
        '''
        
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.d_v = self.d_model // self.num_heads
        
        self.q_proj = Linear(
            in_features=d_model,
            out_features=num_heads * self.d_k,
            device=device,
            dtype=dtype,
        )
        self.k_proj = Linear(
            in_features=d_model,
            out_features=num_heads * self.d_k,
            device=device,
            dtype=dtype,
        )
        self.v_proj = Linear(
            in_features=d_model,
            out_features=num_heads * self.d_v,
            device=device,
            dtype=dtype,
        )
        
        self.o_proj = Linear(
            in_features=num_heads * self.d_v,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        returns: (batch_size, seq_len, d_model)  # check the dimentionality
        """
        batch_size, seq_len, _ = x.shape

        # 1) Project once each for Q, K, V
        q = self.q_proj(x)  # (B, S, H*d_k)
        k = self.k_proj(x)  # (B, S, H*d_k)
        v = self.v_proj(x)  # (B, S, H*d_v)

        # 2) Split heads
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
        # shapes now:
        # q, k: (B, H, S, d_k)
        # v:    (B, H, S, d_v)

        # 3) Causal mask: query i can only attend to keys j <= i
        mask = ~torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )  # (S, S)

        # 4) Attention over each head: headi​=Attention(Qi​,Ki​,Vi​)
        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)
        # (B, H, S, d_v)

        # 5) Concatenate heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_heads * self.d_v
        )
        # (B, S, H*d_v)

        # 6) Output projection
        out = self.o_proj(attn_out)  # (B, S, d_model)
        return out
    