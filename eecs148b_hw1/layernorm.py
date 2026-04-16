from __future__ import annotations
import math
import torch
import torch.nn as nn
import json
from typing import Iterable, Iterator
import regex as re


class LayerNorm(nn.Module):
    def __init__ (self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        '''
        Construct the LayerNorm module
        '''
        
        self.d_model = d_model
        self.eps = eps
        
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

        self.bias = nn.Parameter(
            torch.zeros(d_model, device=device, dtype=dtype)
        )
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        '''
            Process an input tensor of
            shape (batch size, sequence length, d model) and return a tensor of the same
            shape
        
        '''
    
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        # Your code here performing LayerNorm
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        result = x_norm * self.weight + self.bias
        
        # Return the result in the original dtype
        return result.to(in_dtype)
    
    
    
    

        
        