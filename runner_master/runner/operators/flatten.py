__all__ = ['get_flatten_op']


import torch
from torch import nn
from ..env import FLATTEN_OP
from .functional import Functional


def get_flatten_op():
    if FLATTEN_OP == 'nn':
        op = nn.Flatten(start_dim=1)
    elif FLATTEN_OP == 'torch':
        op = Functional(lambda x: torch.flatten(x, start_dim=1))
    elif FLATTEN_OP == 'view':
        op = Functional(lambda x: x.view(x.size(0), -1))
    elif hasattr(nn, 'Flatten'):
        op = nn.Flatten(start_dim=1)
    elif hasattr(torch, 'flatten'):
        op = Functional(lambda x: torch.flatten(x, start_dim=1))
    else:
        op = Functional(lambda x: x.view(x.size(0), -1))
    return op
