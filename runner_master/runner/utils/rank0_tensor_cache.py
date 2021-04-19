__all__ = ['Rank0TensorCache']


import math
import torch
from queue import deque
from collections import OrderedDict
from .. import distributed as run_dist


class Rank0TensorCache:

    def __init__(self, cache_size=None, cuda_cache=False, cuda_output=False):
        assert cache_size is None or isinstance(cache_size, int), \
            'invalid cache_size: {} (should be None or int)'.format(cache_size)
        self.cache_size = cache_size
        self.cuda_cache = cuda_cache
        self.cuda_output = cuda_output
        self.world_size = run_dist.get_world_size()
        self.rank = run_dist.get_rank()
        self.caches = OrderedDict()

    def reset(self):
        for name, cache in self.caches.items():
            cache.clear()
        self.caches.clear()

    def add(self, psudo=None, **kwargs):
        if psudo is not None:
            psudo = self.gather_tensor(psudo, None)
        for name, tensor in sorted(kwargs.items()):  # bug without sorted
            tensor = self.gather_tensor(tensor, psudo)
            if self.rank == 0:
                if tensor.is_cuda and not self.cuda_cache:
                    tensor = tensor.cpu().clone()
                elif not tensor.is_cuda and self.cuda_cache:
                    tensor = tensor.cuda()
                else:
                    tensor = tensor.clone()
                if name not in self.caches:
                    if self.cache_size is None:
                        self.caches[name] = deque()
                    else:
                        self.caches[name] = deque(maxlen=self.cache_size)
                self.caches[name].append(tensor)

    def gather_tensor(self, tensor, psudo):
        if self.world_size > 1:
            tensor = run_dist.all_gather_cat([tensor])[0]
        if psudo is not None:
            tensor = tensor[psudo == 0]
        if self.rank == 0:
            return tensor
        else:
            return None

    # 目前不需要
    # def gather_cat(self, tensor):
    #     if self.world_size == 1:
    #         return tensor
    #     size = torch.LongTensor(list(tensor.size())).cuda()
    #     size_list = [size.new(size.size()) for _ in range(self.world_size)]
    #     torch.distributed.all_gather(size_list, size)
    #     all_rank_sizes = torch.stack(size_list)
    #     max_size = torch.Size(all_rank_sizes.max(0, False)[0].tolist())
    #     all_rank_slices = []
    #     for rank, size in enumerate(size_list):
    #         slices = []
    #         for s in size.tolist():
    #             slices.append(slice(0, s))
    #         all_rank_slices.append(slices)
    #     max_size_tensor = tensor.new(max_size)
    #     max_size_tensor[all_rank_slices[self.rank]].copy_(tensor)
    #     max_size_tensor_list = [tensor.new(max_size) for _ in range(self.world_size)]
    #     torch.distributed.all_gather(max_size_tensor_list, max_size_tensor)
    #     if self.rank == 0:
    #         tensor_list = [t[s] for t, s in zip(max_size_tensor_list, all_rank_slices)]
    #         return torch.cat(tensor_list)
    #     else:
    #         return None

    def cat(self, cuda_output=None):
        if cuda_output is None:
            cuda_output = self.cuda_output
        results = OrderedDict()
        if self.rank == 0:
            for name, cache in self.caches.items():
                tensor = torch.cat(list(cache))
                if cuda_output:
                    tensor = tensor.cuda()
                else:
                    tensor = tensor.cpu()
                results[name] = tensor
        return results
