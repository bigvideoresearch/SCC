import os
import math
import torch
from .. import env
import multiprocessing
import torch.distributed as dist



try:
    reduce_op = dist.ReduceOp
except:
    reduce_op = dist.reduce_op


def get_world_size():
    return int(os.environ.get('SLURM_NTASKS', 1))


def get_rank():
    return int(os.environ.get('SLURM_PROCID', 0))


def get_jobid():
    return int(os.environ['SLURM_JOBID'])


def get_backend():
    return os.environ.get('DISTRIBUTED_BACKEND', None)


# work as a virtual barrier
def barrier():
    if get_world_size() > 1:
        sync_tensor = torch.ones(1).cuda()
        dist.all_reduce(sync_tensor)
        sync_tensor.item()


def synchronize():
    torch.cuda.synchronize()


def all_reduce_mean(tensor_list):
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')
    if get_world_size() == 1:
        return
    for tensor in tensor_list:
        dist.all_reduce(tensor, op=reduce_op.SUM)
        tensor.div_(get_world_size())


def all_reduce_sum(tensor_list):
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')
    if get_world_size() == 1:
        return
    for tensor in tensor_list:
        dist.all_reduce(tensor, op=reduce_op.SUM)


def all_reduce_max(tensor_list):
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')
    if get_world_size() == 1:
        return
    for tensor in tensor_list:
        dist.all_reduce(tensor, op=reduce_op.MAX)


def all_reduce_min(tensor_list):
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')
    if get_world_size() == 1:
        return
    for tensor in tensor_list:
        tensor.neg_()
        dist.all_reduce(tensor, op=reduce_op.MAX)
        tensor.neg_()


def broadcast(tensor_list, src):
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')
    if get_world_size() == 1:
        return
    for tensor in tensor_list:
        dist.broadcast(tensor, src)


def all_gather_cat(tensor_list, cat_dim=0):
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')
    world_size = get_world_size()
    if world_size == 1:
        return tensor_list
    result_list = []
    for tensor in tensor_list:
        gather_list = [tensor.new(*tensor.size()) for _ in range(world_size)]
        if env.DISTRIBUTED_MODULE == 'torch':
            dist.all_gather(gather_list, tensor)
            gather_tensor = torch.cat(gather_list, cat_dim)
        else:
            raise RuntimeError('unknown env.DISTRIBUTED_MODULE: {}'.format(env.DISTRIBUTED_MODULE))
        result_list.append(gather_tensor)
    return result_list


def dist_segment(full_size, world_size=None, rank=None):
    if world_size is None:
        world_size = get_world_size()
    if rank is None:
        rank = get_rank()
    interval = math.ceil(full_size / world_size)
    offset = interval * rank
    part_size = min(full_size, offset + interval) - offset
    return offset, part_size


def get_group(group_size=None, num_groups=None, major='row'):
    world_size = get_world_size()
    rank = get_rank()

    if (group_size is not None) and (num_groups is not None):
        raise RuntimeError('Both group_size and num_groups are not None, but only one should be used.')
    if (group_size is None) and (num_groups is None):
        return dist.group.WORLD

    if group_size is not None:
        if group_size == world_size:
            return dist.group.WORLD
        if world_size % group_size != 0:
            raise RuntimeError('world_size ({}) % group_size ({}) != 0'.format(world_size, group_size))
        num_groups = world_size // group_size
    if num_groups is not None:
        if num_groups == 1:
            return dist.group.WORLD
        if world_size % num_groups != 0:
            raise RuntimeError('world_size ({}) % num_groups ({}) != 0'.format(world_size, num_groups))
        group_size = world_size // num_groups

    if major == 'row':
        beg = rank // group_size * group_size
        end = beg + group_size
        step = 1
    elif major == 'column':
        beg = rank % num_groups
        end = beg + group_size * num_groups
        step = num_groups
    else:
        raise RuntimeError('Argument major should be "row" or "column", but got {}'.format(major))

    ranks = list(range(beg, end, step))
    return dist.new_group(ranks)


def dist_init(port, backend, mp_method='spawn'):
    os.environ['DISTRIBUTED_BACKEND'] = backend
    # start_method默认是fork，不会重新读取dataset源码，但是多机会卡死
    # 设置为spawn之后，多机不会卡死了，但是会重新读取dataset源码
    # 为了避免修改源码引发错误，将整个code目录拷贝一份副本，然后只运行副本
    if multiprocessing.get_start_method(allow_none=True) != mp_method:
        multiprocessing.set_start_method(mp_method, force=True)
    rank = get_rank()
    world_size = get_world_size()
    # node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus
    torch.cuda.set_device(gpu_id)
    rank, world_size = 0, 1


    return rank, world_size
