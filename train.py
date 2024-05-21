import yaml
import logging
import argparse

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from utils.torch_utils import select_torch_device, init_seeds
from utils.general import set_logging, get_options, get_latest_run, check_filepath, \
    increment_path, colorstr, load_file
from utils.dataset import load_char_table, create_dataloader, get_standard_char_image

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

    char_table = load_char_table(train_opts.data_char)
    train_loader, train_dataset = create_dataloader(data_list['train'], image_width=image_width,
                                                    image_height=image_height, batch_size=batch_size,
                                                    char_table=char_table, workers=hypers['workers'],
                                                    is_train=True, prefix=colorstr('Train-dataset'))

    valid_loader, valid_dataset = create_dataloader(data_list['valid'], image_width=image_width,
                                                    image_height=image_height, batch_size=batch_size,
                                                    char_table=char_table, workers=hypers['workers'],
                                                    is_train=False, prefix=colorstr('Valid-dataset'))

    vertical_loader, vertical_dataset = create_dataloader(data_list['Vertical'], image_width=image_width,
                                                          image_height=image_height, batch_size=batch_size,
                                                          char_table=char_table, workers=hypers['workers'],
                                                          is_train=False, prefix=colorstr('Vertical-dataset'))
    desirable_reconstructed_image = get_standard_char_image(data_list['standard_char'], char_table, image_height)

    char_table_size, epoch_iter = train_dataset.datasets[0].get_char_table_size() + 1, len(train_loader)
    model_cfg['model']['decoder']['config']['input_dimension'] = char_table_size

    model = nn.DataParallel(build_model(model_cfg['model']), device_ids=[0])

    optimizer = torch.optim.Adadelta(model.parameters(), lr=hypers['learning_rate'], rho=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

    recognition_loss, reconstruction_loss = nn.CrossEntropyLoss().to(device), nn.MSELoss().to(device)
    info_str = '{0:^11}{1:^12}{2:^18}{3:^36}{4:^12}'.format(
        'Epoch', 'GPU Memory', 'LearningRate', 'Recognition/Reconstruction loss', 'Accuracy')

    if train_opts.resume:
        ckpt = torch.load(train_opts.weights, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['lr_schedule'])
        start_epoch = ckpt['epoch'] + 1

        restore_rng_state(ckpt['random_state'], ckpt['np_ran_state'],
                          torch.from_numpy(ckpt['torch_state']).byte(),
                          torch.from_numpy(ckpt['torch_cuda_state']).byte())
        test_accuracy, best_accuracy = ckpt['best_accuracy']
        logger.info(colorstr(f'Continue training from epoch {start_epoch} ...'))
        del ckpt
    else:
        start_epoch, best_accuracy, test_accuracy = 0, 0.0, 0.0
        logger.info(colorstr('Start training from scratch ...'))
        results_writer.write(info_str + '\n')

    logger.info(info_str)

    for epoch in range(start_epoch, epochs):

        model.train()
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=epoch_iter, ncols=180)
        mean_loss = torch.zeros(3, device=device)
        for iter_index, (images, labels, labels_length, labels_ids, labels_ids_merge) in pbar:
            images, labels_length, labels_ids, labels_ids_merge = \
                images.to(device), labels_length.to(device), labels_ids.to(device), labels_ids_merge.to(device)

            outputs = model(images, labels_length, labels_ids)
        

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
