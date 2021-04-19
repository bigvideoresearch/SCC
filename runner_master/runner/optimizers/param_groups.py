__all__ = ['default', 'bias_bn_no_decay']


import torch
from collections import OrderedDict


bn_classes = []

from torch.nn.modules.batchnorm import _BatchNorm
bn_classes.append(_BatchNorm)

bn_classes = tuple(bn_classes)


def default(network, **kwargs):
    return [dict(params=list(network.parameters()), **kwargs)]


def bias_bn_no_decay(network, **kwargs):
    wd_params = OrderedDict()
    no_params = OrderedDict()
    for name, param in network.named_parameters():
        if name.endswith('bias'):
            no_params[param] = None
        else:
            wd_params[param] = None
    for module in network.modules():
        if isinstance(module, bn_classes):
            del wd_params[module.weight]
            no_params[module.weight] = None
    no_kwargs = dict(**kwargs)
    no_kwargs['weight_decay'] = 0
    param_group = [
        dict(params=list(no_params.keys()), **no_kwargs),
        dict(params=list(wd_params.keys()), **kwargs),
    ]
    return param_group
