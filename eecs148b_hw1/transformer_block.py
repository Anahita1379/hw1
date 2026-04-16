from __future__ import annotations
import math
import torch
import torch.nn as nn
import json
from typing import Iterable, Iterator
import regex as re

import torch
import torch.nn as nn

from eecs148b_hw1.layernorm import LayerNorm
from eecs148b_hw1.multihead_self_attention import MultiHeadSelfAttention
from eecs148b_hw1.positionwise_feedforward import PositionWiseFeedForward




class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, use_layernorm:bool = True, device=None, dtype=None):
        '''
        Implement the pre-norm Transformer block
        '''
        super().__init__()
        self.use_layernorm = use_layernorm
        

        self.layerNorm1 = LayerNorm(d_model, device=device, dtype=dtype)
        self.multiHeadAttn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
        )
        
        self.layerNorm2 = LayerNorm(d_model, device=device, dtype=dtype)
        self.ffn = PositionWiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.use_layernorm: 
            # First sublayer:
            # y = x + MultiHeadSelfAttention(LayerNorm(x)) with residual connection
            out1 = x + self.multiHeadAttn(self.layerNorm1(x))
            
            # Second sublayer: 
            # z = y + FFN(LayerNorm(y)) with residual connection
            out2 = out1 + self.ffn(self.layerNorm2(out1))
            
        else: 
            out1 = x + self.multiHeadAttn(x)
            out2 = out1 + self.ffn(out1)

        return out2
