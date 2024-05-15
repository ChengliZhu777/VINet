import logging
import torch.nn as nn

import models
from utils.general import colorstr
from .common import Conv, MaxPool, ResBasicBlock

logger = logging.getLogger(__name__)


def build_model(cfg):
    param = dict()
    for key in cfg:
        if key in ['type', 'is_recognize']:
            continue
        param[key] = cfg[key]

    if cfg['type'] in ['RTLSTD']:
        return models.__dict__[cfg['type']](**param)
    elif cfg['type'] in ['ResNet']:
        return models.backbone.__dict__[cfg['type']](**param)
    elif cfg['type'] in ['PatchRPN']:
        return models.region_proposal.__dict__[cfg['type']](**param)
    else:
        raise KeyError


def parse_module(module_cfg, module_name='Orange'):
    module_struct, out_channel_list = module_cfg['structure'], [module_cfg['input_dimension']]

    submodules, output_layers = [], []
    logger.info(colorstr(f"{module_name} structure ..."))
    logger.info('{0:^10}{1:^20}{2:^15}{3:^30}{4:^80}{5:^15}'.format(
        'number', 'module-from', 'module-number', 'module-name', 'module-settings', 'module-params'))
    for i, (submodule_from, submodule_number, submodule_name, submodule_args) in enumerate(module_struct):
        submodule = eval(submodule_name) if isinstance(submodule_name, str) else submodule_name
        for j, arg in enumerate(submodule_args):
            try:
                submodule_args[j] = eval(arg) if isinstance(arg, str) else arg
            except Exception as e:
                logger.error(e)

        if submodule in [Conv, ResBasicBlock]:
            in_channels, out_channels = out_channel_list[submodule_from], submodule_args[0]
            submodule_args = [in_channels, *submodule_args]
        elif submodule in [MaxPool]:
            out_channels = out_channel_list[submodule_from]
        else:
            raise ValueError

        submodule_ = nn.Sequential(*[submodule(*submodule_args) for _ in range(submodule_number)]) \
            if submodule_number > 1 else submodule(*submodule_args)
        submodule_type = str(submodule)[8:-2].replace('__main__', '')
        num_params = sum(x.numel() for x in submodule_.parameters())
        submodule_.module_index, submodule_.module_from, submodule_.module_type, submodule_.num_params = \
            i, submodule_from, submodule_type, num_params
        logger.info('{0:^10}{1:^20}{2:^15}{3:^30}{4:^80}{5:^15}'.format(
            i + 1, str(submodule_from), submodule_number, submodule_type, str(submodule_args), num_params))
        submodules.append(submodule_)

        if i == 0:
            out_channel_list = []
        out_channel_list.append(out_channels)

    return nn.Sequential(*submodules)
