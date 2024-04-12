import logging

from utils.general import colorstr

logger = logging.getLogger(__name__)


def parse_model(module_cfg):
    module_name, module_struct = module_cfg['type'], module_cfg['config']['structure']
    logger.info(f"{colorstr(f'{module_name} structure')}")
    logger.info('{0:^10}{1:^20}{2:^15}{3:^15}{4:^30}{5:^80}'.format(
        'number', 'module from', 'module number', 'module params', 'module name', 'module arguments'))
    for i, (submodule_from, submodule_number, submodule_name, submodule_args, is_output) in \
            enumerate(module_struct):
        pass
              
