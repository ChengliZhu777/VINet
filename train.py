import yaml
import logging
import argparse

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from utils.torch_utils import select_torch_device, init_seeds
from utils.general import set_logging, get_options, get_latest_run, check_filepath, \
    increment_path, colorstr, load_file


logger = logging.getLogger(__name__)


def train(train_opts):

    hypers, model_cfg, data_dict = \
        load_file(train_opts.hyp), load_file(train_opts.model_cfg), load_file(train_opts.data)
    logger.info(colorstr('Hyper-parameters: ') + ', '.join(f'{k}={v}' for k, v in hypers.items()))

    device = select_torch_device(hypers['device'], batch_size=hypers['batch_size'],
                                 prefix=colorstr('Device: '))

    save_dir, epochs, batch_size = Path(train_opts.save_dir), hypers['epochs'], hypers['batch_size']
    weight_dir = save_dir / 'weights'
    last_weight, best_weight = weight_dir / 'last.pth', weight_dir / 'best.pth'

    if not train_opts.resume:
        init_seeds(666)

        save_dir.mkdir(parents=True, exist_ok=False)
        weight_dir.mkdir(parents=True, exist_ok=False)

        with open(save_dir / 'opt.yaml', 'w') as f1:
            yaml.dump(vars(train_opts), f1, sort_keys=False)
        with open(save_dir / 'hyp.yaml', 'w') as f1:
            yaml.dump(hypers, sort_keys=False)
        with open(save_dir / 'model.yaml', 'w') as f1:
            yaml.dump(model_cfg, sort_keys=False)

    logger.info(f"{colorstr('Tensorboard: ')}Start with 'tensorboard --logdir {train_opts.project}', "
                f"view at http://localhost:6006/")
    writer = SummaryWriter(train_opts.save_dir)

    results_writer = open(str(save_dir / 'record.txt'), 'a+', encoding='utf-8')

    train_loader, train_dataset, standard_char2image, standard_char2rot_image = \
        create_dataloader(data_dict['train'], data_dict['character'], workers=hypers['workers'],
                          image_width=256, image_height=32, batch_size=batch_size, is_train=True,
                          standard_char_path=data_dict['standard_char'], prefix=colorstr('Train-dataset'))

    valid_loader, valid_dataset = create_dataloader(data_list['valid'], data_list['character'],
                                                    workers=hypers['workers'], image_width=image_height,
                                                    image_height=image_width, batch_size=batch_size, is_train=False,
                                                    prefix=colorstr('Valid-dataset'))

    vertical_loader, vertical_dataset = create_dataloader(data_list['Vertical'], data_list['character'],
                                                          workers=hypers['workers'], image_width=image_width,
                                                          image_height=image_height, batch_size=batch_size,
                                                          is_train=False, prefix=colorstr('Vertical-dataset'))
    
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
