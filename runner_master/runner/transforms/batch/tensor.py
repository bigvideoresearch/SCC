__all__ = [
    'TensorDeepTransform',
    'TensorCuda', 'TensorCPU', 'TensorFloat', 'TensorHalf', 'TensorDetach',
    'BatchFancyPCA', 'BatchNormalize',
]


import torch
from .. import image as run_trans
from collections import Mapping, Sequence


class TensorDeepTransform:

    def __init__(self, keys_to_transform=None):
        self.check_keys_to_transform(keys_to_transform)
        self.keys_to_transform = keys_to_transform

    def check_keys_to_transform(self, keys_to_transform):
        error = False
        if keys_to_transform is None:
            pass
        elif isinstance(keys_to_transform, (str, int)):
            pass
        elif isinstance(keys_to_transform, Sequence):
            for keys in keys_to_transform:
                if isinstance(keys, (str, int)):
                    pass
                elif isinstance(keys, Sequence):
                    for k in keys:
                        if isinstance(k, (str, int)):
                            pass
                        else:
                            error = True
                else:
                    error = True
        else:
            error = True
        if error:
            msg = '\n'.join([
                'argument [keys_to_transform] shoud be either:',
                '- None',
                '- an instance of str',
                '- an instance of int',
                '- a sequence of either:',
                '    - instances of str',
                '    - instances of int',
                '    - sequences of either:',
                '        - instances of str',
                '        - instances of int',
                'but got:',
                repr(keys_to_transform),
            ])
            raise ValueError(msg)

    def transform_tensor(self, tensor):
        raise NotImplementedError

    def __call__(self, batch):
        if self.keys_to_transform is None:
            batch = self.transform_default(batch)
        elif isinstance(self.keys_to_transform, (str, int)):
            self.transform_on_keys(batch, self.keys_to_transform)
        else:
            for keys in self.keys_to_transform:
                self.transform_on_keys(batch, keys)
        return batch

    def transform_default(self, batch):
        error_msg = 'batch must be tensor, number, string, dict og list; found {}'
        if isinstance(batch, torch.Tensor):
            return self.transform_tensor(batch)
        elif isinstance(batch, (int, float, str, bytes)):
            return batch
        elif isinstance(batch, Mapping):
            return {key: self.transform_default(value) for key, value in batch.items()}
        elif isinstance(batch, Sequence):
            return [self.transform_default(value) for value in batch]
        raise TypeError(error_msg.format(type(batch)))

    def transform_on_keys(self, batch, keys):
        try:
            if isinstance(keys, (int, str)):
                key = keys
                parent = batch
                child = parent[key]
            else:
                child = None
                for i, key in enumerate(keys):
                    parent = batch if i == 0 else child
                    child = parent[key]
            if not isinstance(child, torch.Tensor):
                raise TypeError('the object indexed by keys {} should has type torch.Tensor, but got {}'.format(
                    keys, type(child),
                ))
            parent[key] = self.transform_tensor(child)
        except Exception as e:
            msg = '\n'.join([
                'got exception message:',
                repr(e),
                'when applying keys:',
                repr(keys),
                'on batch:',
                repr(batch),
            ])
            raise RuntimeError(msg)


class TensorCuda(TensorDeepTransform):

    def __init__(self, keys_to_transform=None, non_blocking=False):
        super(TensorCuda, self).__init__(keys_to_transform)
        self.non_blocking = non_blocking

    def transform_tensor(self, tensor):
        return tensor.cuda(non_blocking=self.non_blocking)


class TensorCPU(TensorDeepTransform):

    def transform_tensor(self, tensor):
        return tensor.cpu()


class TensorFloat(TensorDeepTransform):

    def transform_tensor(self, tensor):
        if tensor.is_floating_point():
            return tensor.float()
        else:
            return tensor


class TensorHalf(TensorDeepTransform):

    def transform_tensor(self, tensor):
        if tensor.is_floating_point():
            return tensor.half()
        else:
            return tensor


class TensorDetach(TensorDeepTransform):

    def transform_tensor(self, tensor):
        return tensor.detach()


class BatchFancyPCA(TensorDeepTransform):

    def __init__(self, *args, keys_to_transform=('data',), **kwargs):
        super(BatchFancyPCA, self).__init__(keys_to_transform)
        self.transform_tensor = run_trans.FancyPCA(*args, **kwargs)


class BatchNormalize(TensorDeepTransform):

    def __init__(self, *args, keys_to_transform=('data',), **kwargs):
        super(BatchNormalize, self).__init__(keys_to_transform)
        self.transform_tensor = run_trans.Normalize(*args, **kwargs)
