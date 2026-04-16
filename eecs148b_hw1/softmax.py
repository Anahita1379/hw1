'''
Write a function to apply the softmax operation on a tensor. Your function
should take two parameters: a tensor and a dimension i, and apply softmax to the i-th
dimension of the input tensor. The output tensor should have the same shape as the input
tensor, but its i-th dimension will now have a normalized probability distribution
'''

import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x = x - x_max

    exp_x = torch.exp(x)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)

    return exp_x / sum_exp