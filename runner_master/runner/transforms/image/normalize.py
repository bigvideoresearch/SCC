__all__ = ['Normalize']


import torch
import torchvision


class Normalize(torchvision.transforms.Normalize):

    def __call__(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError('tensor is not a torch image.')
        if tensor.dim() == 3:
            for i, (m, s) in enumerate(zip(self.mean, self.std)):
                tensor[i].sub_(m).div_(s)
        elif tensor.dim() == 4:
            for i, (m, s) in enumerate(zip(self.mean, self.std)):
                tensor[:, i].sub_(m).div_(s)
        else:
            raise RuntimeError('Normalize requires input tensor to be 3D or 4D, but got {}D'.format(tensor.dim()))
        return tensor
