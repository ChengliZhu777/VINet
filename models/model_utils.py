import logging

import torch.nn as nn

from utils.general import colorstr
from models.common import Conv

logger = logging.getLogger(__name__)


def parse_model(module_cfg):
    module_name, module_struct = module_cfg['type'], module_cfg['config']['structure']
    out_channel_list = [module_cfg['config']['input_dimension']]

    modules, save_layers = [], []
    logger.info(f"{colorstr(f'{module_name} structure')}")
    logger.info('{0:^10}{1:^20}{2:^15}{3:^15}{4:^30}{5:^80}'.format(
        'number', 'module from', 'module number', 'module params', 'module name', 'module arguments'))
    for i, (submodule_from, submodule_number, submodule_name, submodule_args) in \
            enumerate(module_struct):
        submodule = eval(submodule_name) if isinstance(submodule_name, str) else submodule_name
        for j, arg in enumerate(submodule_args):
            try:
                if arg not in ['ReLU-inplace']:
                    submodule_args[j] = eval(arg) if isinstance(arg, str) else arg
            except Exception as e:
                logger.error(e)

        if submodule in [Conv]:
            in_channels, out_channels = out_channel_list[submodule_from], submodule_args[0]
            submodule_args = [in_channels, out_channels, *submodule_args[1:]]
        else:
            out_channels = out_channel_list[submodule_from]

        submodule_ = nn.Sequential(*[submodule(*submodule_args) for _ in range(submodule_number)]) \
            if submodule_number > 1 else submodule(*submodule_args)
        submodule_type = str(submodule)[8:-2].replace('__main__', '')
        num_params = sum(x.numel() for x in submodule_.parameters())
        submodule_.module_index, submodule_.module_from, submodule_.module_type, submodule_.num_params = \
            i, submodule_from, submodule_type, num_params
        logger.info('{0:^10}{1:^20}{2:^15}{3:^15}{4:^30}{5:^80}'.format(
            i + 1, str(submodule_from), submodule_number, num_params, submodule_type, str(submodule_args)))
        modules.append(submodule_)

        save_layers.extend(x % i for x in ([submodule_from] if isinstance(submodule_from, int)
                                           else submodule_from) if x != -1)

        if i == 0:
            out_channel_list = []
        out_channel_list.append(out_channels)

    return nn.Sequential(*modules), sorted(save_layers)
    
