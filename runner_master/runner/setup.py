__all__ = ['setup']


import os
import time
import torch
import random
import numpy as np
from .config import setup_config
from .distributed import dist_init


def setup(seed=1024,
          port=23333,
          backend='nccl',
          mp_method='spawn',
          cudnn_benchmark=True,
          cudnn_deterministic=False,
          config_first=True,
          verbose=True):

    config = setup_config()

    if config_first:
        # if an argument is already specified in config, use config, else use function's input
        seed = config.get('seed', seed)
        port = config.get('port', port)
        backend = config.get('backend', backend)
        mp_method = config.get('mp_method', mp_method)
        cudnn_benchmark = config.get('cudnn_benchmark', cudnn_benchmark)
        cudnn_deterministic = config.get('cudnn_deterministic', cudnn_deterministic)

    # random seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    # setup distributed environment
    rank, world_size = dist_init(port, backend, mp_method)

    # cudnn setup
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic

    if rank == 0 and verbose:
        print('=' * 80)
        print('runner config:')
        print(config)
        print('=' * 80)
        print('runner setup:')
        print('    date [{}]'.format(time.strftime('%Y-%m-%d-%H:%M:%S')))
        print('    seed [{}]'.format(config.seed))
        print('    port [{}]'.format(port))
        print('    backend [{}]'.format(backend))
        print('    mp_method [{}]'.format(mp_method))
        print('    world_size [{}]'.format(world_size))
        print('    cudnn_benchmark [{}]'.format(cudnn_benchmark))
        print('    cudnn_deterministic [{}]'.format(cudnn_deterministic))
        print('=' * 80, flush=True)

    return config, rank, world_size
