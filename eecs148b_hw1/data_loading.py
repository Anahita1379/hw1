import numpy as np
import torch


def get_batch( x: np.ndarray, batch_size: int, context_length: int, device: str,):
    
    """
    x: 1D numpy array of token IDs
    returns:
        inputs:  (batch_size, context_length)
        targets: (batch_size, context_length)
    both on the requested device
    
    For example, for B = 1, m = 3, 
    ([x2, x3, x4], [x3, x4, x5]) would be one potential batch
    """
    # make sure x is 1D
    if x.ndim != 1:
        raise ValueError("x must be a 1D numpy array")
    
    if len(x) < context_length + 1:
        raise ValueError("x is too short for the requested context_length")
    
    max_start = len(x) - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    
    inputs = np.stack([x[s : s + context_length] for s in starts])
    targets = np.stack([x[s + 1 : s + context_length + 1] for s in starts])

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets