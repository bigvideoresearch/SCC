import torch
from . import misc
from torch.nn.modules.batchnorm import _BatchNorm

def sync_grad_sum(network):
    if misc.get_world_size() == 1: return
    misc.all_reduce_sum([param.grad.data for param in network.parameters() if param.grad is not None])

def sync_bn_stat(network):
    if misc.get_world_size() == 1: return
    tensor_list = []
    for mod in network.modules():
        if isinstance(mod, _BatchNorm):
            tensor_list.append(mod.running_mean)
            tensor_list.append(mod.running_var)
    misc.all_reduce_mean(tensor_list)
