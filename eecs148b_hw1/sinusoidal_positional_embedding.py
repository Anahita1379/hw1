from __future__ import annotations
import math
import torch
import torch.nn as nn
import json
from typing import Iterable, Iterator
import regex as re


class SinusoidalPositionalEncoding(nn.Module):
    def __init__ (self, d_model: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        '''
        Construct the sinusoidal positional embedding module and precompute the positional
        embedding buffer
        '''
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        position = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(1)   # shape = (max_seq_len, 1)
        
        
        denominator = torch.exp(torch.arange( 0, d_model, 2,
                device=device,
                dtype=torch.float32,
            ) * (-math.log(10000.0) / d_model)) # shape = (d_model/2,)
        
        pe = torch.empty(
            (max_seq_len, d_model),
            device=device,
            dtype=torch.float32,
        )
        
        # position * div_term shape : (max_seq_len, d_model/2)
        pe[:, 0::2] = torch.sin(position * denominator)
        pe[:, 1::2] = torch.cos(position * denominator)
        
        if dtype is not None:
            pe = pe.to(dtype)
            
        self.register_buffer(
            "pe",
            pe,
            persistent=False,
        )
        
    def forward(self, token_positions: torch.Tensor) -> torch.Tensor: 
    
        '''
            Given token
        positions of shape (..., seq len), return positional embeddings of shape (...,
        seq len, d model)
        '''
        return self.pe[token_positions]