import torch
import torch.nn as nn

import numpy as np

from eecs148b_hw1.linear import Linear

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff = None,  device=None, dtype=None):
        super().__init__()
        
        self.d_model = d_model
        if d_ff is not None: 
            self.d_ff = d_ff
        else:
            self.d_ff = 4 * d_model

        self.fc1 = Linear(
            in_features=self.d_model,
            out_features=self.d_ff,
            device=device,
            dtype=dtype,
        ) # W1 ∈ Rdff×dmodel
        
        self.fc2 = Linear(
            in_features=self.d_ff,
            out_features=self.d_model,
            device=device,
            dtype=dtype,
        ) #W2 ∈ Rdmodel×dff
        
    def relu(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, torch.zeros_like(x))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x