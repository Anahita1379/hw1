from __future__ import annotations
import math
import torch
import torch.nn as nn
import json
from typing import Iterable, Iterator
import regex as re


class Embedding(nn.Module):
    def __init__ (self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        '''
        Construct an embedding module
        '''
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        #  embedding matrix of shape (vocab size,d_model)
        self.embedding = nn.Parameter(
            torch.empty(
                (self.num_embeddings, self.embedding_dim ),
                device=device,
                dtype=dtype,
            )
        )
    
        self.initialize_weights()
    
    def initialize_weights(self):
        std = 1
        nn.init.trunc_normal_(self.embedding, mean=0.0, std=std, a=-3 * std, b=3 * std)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        '''
        Lookup the embedding vectors for the given token IDs
        '''
        return self.embedding[token_ids]
        
        