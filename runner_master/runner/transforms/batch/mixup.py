__all__ = ['Mixup']


import torch
import numpy as np
from .tensor import TensorDeepTransform


class Mixup(TensorDeepTransform):

    def __init__(self, alpha=0.2, randperm=True, multimix=True, keepleft=False,
                 keys_to_transform=('data', 'soft_label')):
        super(Mixup, self).__init__(keys_to_transform)
        self.a = alpha
        self.b = alpha
        self.randperm = randperm
        self.multimix = multimix
        self.keepleft = keepleft

    def __call__(self, batch):
        if self.a == 0 and self.b == 0:
            return batch

        with torch.no_grad():
            first_key = self.keys_to_transform[0]
            device = batch[first_key].device
            N = batch[first_key].size(0)

            if self.randperm:
                permute_index = torch.randperm(N).to(device)
            else:
                permute_index = torch.arange(N - 1, -1, -1).to(device)

            if self.multimix:
                mix1 = torch.tensor(np.random.beta(self.a, self.b, N).astype(np.float32)).to(device)
                if self.keepleft:
                    mix1 = (mix1 - 0.5).abs() + 0.5
                mix2 = 1 - mix1
            else:
                mix1 = np.random.beta(self.a, self.b)
                if self.keepleft:
                    mix1 = abs(mix1 - 0.5) + 0.5
                mix2 = 1 - mix1

            self.N = N
            self.permute_index = permute_index
            self.mix1 = mix1
            self.mix2 = mix2
            batch = super(Mixup, self).__call__(batch)
            del self.N
            del self.permute_index
            del self.mix1
            del self.mix2
            return batch

    def transform_tensor(self, tensor):
        N = tensor.size(0)
        assert N == self.N
        permute_tensor = tensor[self.permute_index]
        if self.multimix:
            sizes = [N] + [1] * (tensor.dim() - 1)
            result = tensor * self.mix1.view(*sizes) + permute_tensor * self.mix2.view(*sizes)
        else:
            result = tensor * self.mix1 + permute_tensor * self.mix2
        return result


class OldMixup:

    def __init__(self, alpha=0.2, randperm=True, multimix=True, keepleft=False,
                 extra_mix_keys=None, extra_mix_strict=False):
        self.a = alpha
        self.b = alpha
        self.randperm = randperm
        self.multimix = multimix
        self.keepleft = keepleft
        self.extra_mix_keys = extra_mix_keys or tuple()
        self.extra_mix_strict = extra_mix_strict

    def __call__(self, batch):
        if self.a == 0 and self.b == 0:
            return batch
        with torch.no_grad():
            device = batch['data'].device
            N = batch['data'].size(0)
            if self.randperm:
                permute_index = torch.randperm(N).to(device)
            else:
                permute_index = torch.arange(N - 1, -1, -1).to(device)
            permute_data = batch['data'][permute_index]
            permute_soft_label = batch['soft_label'][permute_index]
            if self.multimix:
                mix1 = torch.tensor(np.random.beta(self.a, self.b, N).astype(np.float32)).to(device)
                if self.keepleft:
                    mix1 = (mix1 - 0.5).abs() + 0.5
                mix2 = 1 - mix1
                batch['data'] = mix1.view(N, 1, 1, 1) * batch['data'] + mix2.view(N, 1, 1, 1) * permute_data
                batch['soft_label'] = mix1.view(N, 1) * batch['soft_label'] + mix2.view(N, 1) * permute_soft_label
                for key in self.extra_mix_keys:
                    if key in batch:
                        value = batch[key]
                        permute_value = value[permute_index]
                        view_mix1 = mix1.view(mix1.size(0), *[1 for _ in range(value.dim())]).type_as(value)
                        view_mix2 = mix2.view(mix1.size(0), *[1 for _ in range(value.dim())]).type_as(value)
                        batch[key] = view_mix1 * value * view_mix2 * permute_value
                    elif self.extra_mix_strict:
                        raise RuntimeError('extra_mix_key [{}] not in batch'.format(key))
            else:
                mix1 = np.random.beta(self.a, self.b)
                if self.keepleft:
                    mix1 = abs(mix1 - 0.5) + 0.5
                mix2 = 1 - mix1
                batch['data'] = mix1 * batch['data'] + mix2 * permute_data
                batch['soft_label'] = mix1 * batch['soft_label'] + mix2 * permute_soft_label
                for key in self.extra_mix_keys:
                    if key in batch:
                        value = batch[key]
                        permute_value = value[permute_index]
                        batch[key] = mix1 * value * mix2 * permute_value
                    elif self.extra_mix_strict:
                        raise RuntimeError('extra_mix_key [{}] not in batch'.format(key))
        return batch
