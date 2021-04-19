__all__ = ['nothing', 'load_ckpt', 'load_ckpt_no_fc', 'load_ckpt_fix_backbone', 'load_ckpt_no_fc_fix_backbone']


import torch
from torch import nn


def nothing(network, network_name='main'):
    pass


def load_ckpt(network, ckpt, network_name='main'):
    state_dict = torch.load(ckpt, map_location='cuda:{}'.format(torch.cuda.current_device()))
    if 'network' in state_dict:
        state_dict = state_dict['network']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    if network_name in state_dict:
        state_dict = state_dict[network_name]
    network.load_state_dict(state_dict)


def load_ckpt_no_fc(network, ckpt, network_name='main'):
    state_dict = torch.load(ckpt, map_location='cuda:{}'.format(torch.cuda.current_device()))
    if 'network' in state_dict:
        state_dict = state_dict['network']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    if network_name in state_dict:
        state_dict = state_dict[network_name]

    if not hasattr(network, 'fc'):
        raise RuntimeError('network.fc not exists')
    if not isinstance(network.fc, nn.Module):
        raise RuntimeError('network.fc is not a torch.nn.Module, but {}'.format(type(network.fc)))

    state_dict['fc.weight'] = network.fc.weight.data
    state_dict['fc.bias'] = network.fc.bias.data

    network.load_state_dict(state_dict)


def load_ckpt_fix_backbone(network, ckpt, network_name='main'):
    load_ckpt(network, ckpt, network_name)

    if not hasattr(network, 'fc'):
        raise RuntimeError('network.fc not exists')
    if not isinstance(network.fc, nn.Module):
        raise RuntimeError('network.fc is not a torch.nn.Module, but {}'.format(type(network.fc)))

    for name, param in network.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad_(False)


def load_ckpt_no_fc_fix_backbone(network, ckpt, network_name='main'):
    load_ckpt_no_fc(network, ckpt, network_name)

    for name, param in network.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad_(False)
