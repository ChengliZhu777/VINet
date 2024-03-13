import yaml
import argparse

from pathlib import Path

from utils.general import set_logging, get_options


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
        train_options.save_dir = increment_path(Path(train_options.project) / train_options.name,
                                                exist_ok=False)
        
