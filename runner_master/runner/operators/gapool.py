__all__ = ['get_gapool_op']


import torch
from torch import nn
from .functional import Functional
from .flatten import get_flatten_op
from ..env import CEIL_MODE, GAPOOL_OP


def get_mean(keepdim):
    return Functional(lambda x: x.mean([2, 3], keepdim=keepdim))


def get_ada(keepdim):
    if keepdim:
        op = nn.AdaptiveAvgPool2d(1)
    else:
        op = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            get_flatten_op(),
        )
    return op


def get_avg(input_size, keepdim):
    if keepdim:
        op = nn.AvgPool2d(input_size, stride=1, ceil_mode=CEIL_MODE)
    else:
        op = nn.Sequential(
            nn.AvgPool2d(input_size, stride=1, ceil_mode=CEIL_MODE),
            get_flatten_op(),
        )
    return op


def get_gapool_op(input_size=None, keepdim=True):
    # test if this torch version supports reduce-mean on multiple dimensions
    try:
        torch.zeros(1, 1).mean([0, 1])
        can_do_multiple_mean = True
    except:
        can_do_multiple_mean = False

    if GAPOOL_OP == 'mean':
        op = get_mean(keepdim)
    elif GAPOOL_OP == 'ada':
        op = get_ada(keepdim)
    elif GAPOOL_OP == 'avg':
        op = get_avg(input_size, keepdim)
    elif can_do_multiple_mean:
        op = get_mean(keepdim)
    elif input_size is None:
        op = get_ada(keepdim)
    else:
        op = get_avg(input_size, keepdim)

    return op
