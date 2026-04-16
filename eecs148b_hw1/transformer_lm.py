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
from eecs148b_hw1.transformer_block import TransformerBlock
from eecs148b_hw1.linear import Linear
from eecs148b_hw1.embedding import Embedding
from eecs148b_hw1.sinusoidal_positional_embedding import SinusoidalPositionalEncoding




class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, 
                 context_length: int, 
                 d_model: int, 
                 num_layers: int,
                 num_heads: int, 
                 d_ff: int, 
                 use_layernorm:bool = True,
                 use_position_embeddings: bool = True,
                 device=None,dtype=None,):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_layernorm = use_layernorm
        self.use_position_embeddings = use_position_embeddings
        
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        
        self.position_embeddings = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_seq_len=context_length,
            device=device,
            dtype=dtype,
        )

        
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                use_layernorm=use_layernorm,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        
        self.layerNorm_final = LayerNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch_size, seq_len)
        returns: (batch_size, seq_len, vocab_size)
        """        
        batch_size, seq_len = token_ids.shape
        
        if seq_len > self.context_length:
            raise ValueError(
                f"seq_len={seq_len} exceeds context_length={self.context_length}"
            )
            
        # Token embeddings: (B, S, d_model)
        t_em = self.token_embeddings(token_ids)
        
        if self.use_position_embeddings:
            # Positional embeddings: positions shape (B, S) or (S,) both are fine
            positions = torch.arange(seq_len, device=token_ids.device)
            p_em = self.position_embeddings(positions)  # (S, d_model)
            
            # Broadcast across batch
            x = t_em + p_em.unsqueeze(0)  # (B, S, d_model)
        else:
            x = t_em
        
        # pass through num_layers Transformer blocks
        for layer in self.layers:
            x = layer(x)
            
        # Final LayerNorm
        # if self.use_layernorm:
        x = self.layerNorm_final(x)
    
        # LM head -> logits over vocab
        logits = self.lm_head(x)  # (B, S, vocab_size)
            
        # else: 
        #     # LM head -> logits over vocab
        #     logits = self.lm_head(x)  # (B, S, vocab_size)
            
        return logits
        
        
