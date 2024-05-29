import yaml
import torch
import random
import logging
import argparse

import numpy as np
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

from test import test
from models import build_model
from utils.torch_utils import select_torch_device, init_seeds, restore_rng_state
from utils.general import set_logging, get_options, get_latest_run, check_filepath, \
    increment_path, colorstr, load_file
from utils.dataset import load_char_table, create_dataloader, get_standard_char_image

logger = logging.getLogger(__name__)


def train(train_opts, test_opts=None):

    hypers, model_cfg, data_list = \
        load_file(train_opts.hyp), load_file(train_opts.model_cfg), load_file(train_opts.data)
    logger.info(colorstr('Hyper-parameters: ') + ', '.join(f'{k}={v}' for k, v in hypers.items()))

    device = select_torch_device(hypers['device'], batch_size=hypers['batch_size'],
                                 prefix=colorstr('Device: '))

    save_dir, epochs, batch_size, image_width, image_height = \
        Path(train_opts.save_dir), hypers['epochs'], hypers['batch_size'], hypers['image_width'], hypers['image_height']

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
    logger.info(colorstr('Train-dataset: '))
    train_loader, train_dataset = create_dataloader(data_list['train'], image_width=image_width,
                                                    image_height=image_height, batch_size=batch_size,
                                                    char_table=char_table, workers=hypers['workers'], is_train=True)

    logger.info(colorstr('Test-dataset: '))
    test_datasets, test_dataloaders = dict(), dict()
    for data_name in list(data_list['valid']):
        test_loader, test_dataset = create_dataloader({data_name: data_list['valid'][data_name]},
                                                      image_width=image_width, image_height=image_height,
                                                      batch_size=batch_size, char_table=char_table,
                                                      workers=hypers['workers'], is_train=False)
        test_datasets[data_name] = test_dataset
        test_dataloaders[data_name] = test_loader

    desirable_reconstructed_image = get_standard_char_image(data_list['standard_char'], char_table, image_height)

    char_table_size, epoch_iter = train_dataset.datasets[0].get_char_table_size() + 1, len(train_loader)
    model_cfg['model']['recognizer']['config']['input_dimension'] = char_table_size

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
            predictions, constructed_images = outputs['predictions'], outputs['reconstructed_images']

            labels_merge = ''.join([label[:-1] for label in labels])  # remove $
            desirable_images = []
            for i in range(len(labels_merge)):
                desirable_images.append(desirable_reconstructed_image[labels_merge[i]])
            desirable_images = torch.cat(desirable_images, dim=0)

            loss_reconstruction = reconstruction_loss(constructed_images, desirable_images.to(device))
            loss_recognition = recognition_loss(predictions, labels_ids_merge)

            loss = loss_recognition + 5.0 * loss_reconstruction

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_items = torch.cat((loss_recognition.view(1), loss_reconstruction.view(1),
                                    loss.view(1))).detach()
            mean_loss = (mean_loss * iter_index + loss_items) / (iter_index + 1)

            memory_usage = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)

            info_str = ('{0:^11}{1:^12}{2:^18}{3:^27}'
                        ).format(f'{epoch + 1:>03}/%3g' % epochs,
                                 memory_usage,
                                 '%.8f' % optimizer.param_groups[0]['lr'],
                                 '%.4f / %.4f' % (mean_loss[0], mean_loss[1]))

            pbar.set_description(info_str + '{0:^12}'.format('%.4f' % test_accuracy))

        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()

        test_opts.epoch = epoch
        test_results = dict()
        for data_name in list(test_datasets):
            test_result = test(opts=test_opts,
                               hypers=hypers,
                               model=model,
                               model_cfg=model_cfg,
                               dataset=test_datasets[data_name],
                               dataloader=test_dataloaders[data_name],
                               prefix=info_str)
            test_results[data_name] = test_result

        ckpt = {'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_schedule': scheduler.state_dict(),
                'random_state': random.getstate(),
                'np_ran_state': np.random.get_state(),
                'torch_state': torch.get_rng_state().numpy(),
                'torch_cuda_state': torch.cuda.get_rng_state().numpy(),
                'best_accuracy': best_accuracy}

        if test_results['Fudan'] > best_accuracy:
            best_accuracy = test_results['Fudan']
            ckpt['best_accuracy'] = best_accuracy
            torch.save(ckpt, best_weight)
