import yaml
import logging
import argparse

from pathlib import Path

from utils.torch_utils import select_torch_device
from utils.general import set_logging, get_options, get_latest_run, check_filepath, \
    increment_path, colorstr, load_file


logger = logging.getLogger(__name__)


def train(train_opts):

    hypers, model_cfg = load_file(train_opts.hyp), load_file(train_opts.model_cfg)
    device = select_torch_device(train_opts.device, batch_size=hypers['batch_size'],
                                 prefix=colorstr('device: '))


if __name__ == '__main__':
    set_logging(-1)

    train_options = get_options(resume=False)

    if train_options.resume:
        train_ckpt = train_options.resume if isinstance(train_options.resume, str) else get_latest_run()
        with open(Path(train_ckpt).parent.parent / 'opt.yaml') as f:
            train_options = argparse.ArgumentParser(**yaml.load(f, Loader=yaml.SafeLoader))

        train_options.weights, train_options.hyp, train_options.model_cfg, train_options.resume = \
            train_ckpt, str(Path(train_ckpt).parent.parent / 'hyp.yaml'), \
            str(Path(train_ckpt).parent.parent / 'model.yaml'), True
    else:
        train_options.hyp, train_options.model_cfg, train_options.data = check_filepath(train_options.hyp), \
                                                                         check_filepath(train_options.model_cfg), \
                                                                         check_filepath(train_options.data)
        train_options.save_dir = increment_path(Path(train_options.project) / train_options.name,
                                                exist_ok=False)

    logger.info(colorstr('Train Options: ') + str(train_options))

    train(train_options)
