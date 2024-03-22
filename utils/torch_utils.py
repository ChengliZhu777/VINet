import os
import torch
import random
import logging

import numpy as np
import torch.distributed as torch_dist
import torch.backends.cudnn as cudnn

from contextlib import contextmanager

logger = logging.getLogger(__name__)


def select_torch_device(device='', batch_size=None, prefix=''):
    info = f'{prefix}torch {torch.__version__} '

    is_cpu = device.lower() == 'cpu'
    if is_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        logger.info(info + 'CPU')
        return torch.device('cpu')
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available(), f"Error: Cuda unavailable, " \
                                          f"invalid torch device '{device}' requested."

        num_cuda = torch.cuda.device_count()
        device_list = device.split(',')

        if len(device_list) > num_cuda:
            raise ValueError(f"Error: specified torch device count larger than actual cuda device count.")
        if num_cuda > 1 and batch_size:
            assert batch_size % num_cuda == 0, \
                f"Error: batch_size ({batch_size}) not multiple of cuda count ({num_cuda})."

        space = ' ' * 4
        available_device = 'cuda:'
        try:
            for i, d in enumerate(device_list):
                p = torch.cuda.get_device_properties(int(d))
                available_device += d
                info += f"{'' if i == 0 else space}CUDA: {d} ({p.name}, {p.total_memory / 1024 ** 2}MB)"
        except Exception as e:
            raise Exception(f"Error: specified torch device ( {device} ) unavailable.")
        logger.info(info)
        return torch.device(available_device)
    else:
        raise ValueError("Error: torch device must be specified, e.g, 'cpu', '0', or '0, 1, 2'.")


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def init_torch_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = False, True  # using default conv, more reproducible


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    if local_rank not in [-1, 0]:
        torch_dist.barrier()
    yield
    if local_rank == 0:
        torch_dist.barrier()
        
