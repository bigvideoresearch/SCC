__all__ = ['DefaultCollate']


import torch.utils.data.dataloader


def DefaultCollate():
    return torch.utils.data.dataloader.default_collate
