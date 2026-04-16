from __future__ import annotations
import math
import torch
import torch.nn as nn
import json
from typing import Iterable, Iterator
import regex as re

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        '''
        Construct a linear transformation module
        '''
        
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(
            torch.empty(
                (out_features, in_features),
                device=device,
                dtype=dtype,
            )
        )
            
        self.initialize_weights()
    
    def initialize_weights(self):
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Apply the linear transformation to the input
        '''
        
        return x @ self.weight.T
        
        