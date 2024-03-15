import yaml
import logging
import argparse

from pathlib import Path

from utils.general import set_logging, get_options, check_filepath, colorstr


logger = logging.getLogger(__name__)


def train(train_opts):
    pass


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
