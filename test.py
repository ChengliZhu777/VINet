import os
import torch
import logging

import torch.nn as nn

from tqdm import tqdm
from nltk.metrics.distance import edit_distance

from utils.general import set_logging, get_options, load_file, \
    colorstr, char_indices2string
from utils.torch_utils import select_torch_device
from utils.dataset import create_dataloader, load_char_table
from models import build_model

from util import get_alphabet

alphabet = get_alphabet()

logger = logging.getLogger(__name__)


@torch.no_grad()
def test(test_opts, model=None, datasets=None, dataloaders=None, prefix=''):
    is_training = model is not None

    if is_training:
        device = next(model.parameters()).device
        char_table = None
    else:
        hypers, model_cfg, data_list = \
            load_file(test_opts.hyp), load_file(test_opts.model_cfg), load_file(test_opts.data)
        device = select_torch_device(hypers['device'], batch_size=hypers['batch_size'],
                                     prefix=colorstr('Device: '))
        char_table = load_char_table(test_opts.data_char)
        char_table_size = len(char_table)
        logger.info(colorstr('Test-dataset: '))
        datasets, dataloaders = dict(), dict()
        for data_name in list(data_list['valid']):
            test_loader, test_dataset = create_dataloader({data_name: data_list['valid'][data_name]},
                                                          image_width=hypers['image_width'],
                                                          image_height=hypers['image_height'],
                                                          batch_size=hypers['batch_size'], char_table=char_table,
                                                          workers=hypers['workers'], is_train=False)
            datasets[data_name] = test_dataset
            dataloaders[data_name] = test_loader

        model_cfg['model']['recognizer']['config']['input_dimension'] = char_table_size  # duplicated '-'
        model = nn.DataParallel(build_model(model_cfg['model']), device_ids=[0])

        ckpt = torch.load(test_opts.weight)
        model.load_state_dict(ckpt['model'])

        logger.info(colorstr(f"Start Test from checkpoint saved at epoch {ckpt['epoch']} ..."))

        logger.info('%24s' % 'ACC/NED')

    torch.cuda.empty_cache()
    model.eval()

    test_results = dict()
    for data_name in list(datasets):
        test_results[data_name] = dict()
        cur_dataset, cur_dataloader = datasets[data_name], dataloaders[data_name]
        num_samples, num_true_positive, norm_edit_distance = len(cur_dataset), 0, 0.0
        pbar = tqdm(enumerate(cur_dataloader), total=len(cur_dataloader), ncols=180)

        if is_training:
            char_table = cur_dataset.datasets[0].char_table
        for batch_index, (images, labels_length, labels_ids_merge) in pbar:
            batch_size, max_label_length = images.shape[0], max(labels_length)
            image_features = None
            predictions, probability = \
                torch.zeros(batch_size, 1).long().to(device), torch.zeros(batch_size, max_label_length).float()
            cur_labels_length = torch.zeros(batch_size).long().to(device)

            for i in range(max_label_length):
                cur_labels_length += 1
                cur_result = model(images, cur_labels_length, predictions, vi_conv_feature=image_features)
                cur_prediction = torch.max(torch.softmax(cur_result['predictions'], 2), 2)
                probability[:, i] = cur_prediction[0][:, -1]
                predictions = torch.cat((predictions, cur_prediction[1][:, -1].view(-1, 1)), 1)
                image_features = cur_result['vi_conv_feature']

            gt_labels, start_index = [], 0
            for label_length in labels_length:
                gt_labels.append(labels_ids_merge[start_index: start_index + label_length])
                start_index += label_length

            pred_labels, pred_probs = [], []
            for i in range(batch_size):
                actual_pred = []
                for j in range(max_label_length):
                    if predictions[i][j] != cur_dataset.datasets[0].get_char_table_size():
                        actual_pred.append(predictions[i][j])
                    else:
                        break

                pred_labels.append(torch.Tensor(actual_pred)[1:].long().to(device))

                overall_prob = 1.0
                for j in range(len(actual_pred)):
                    overall_prob *= probability[i][j]
                pred_probs.append(overall_prob)

            for i in range(batch_size):
                gt, pred = char_indices2string(gt_labels[i], char_table), \
                    char_indices2string(pred_labels[i], char_table)

                if pred == gt:
                    num_true_positive += 1

                if len(gt) == 0 or len(pred) == 0:
                    norm_edit_distance += 0
                elif len(gt) > len(pred):
                    norm_edit_distance += 1 - edit_distance(pred, gt) / len(gt)
                else:
                    norm_edit_distance += 1 - edit_distance(pred, gt) / len(pred)

            info_str = prefix + '{0:^34}'.format('%.4f/%.4f' % (num_true_positive / num_samples,
                                                                norm_edit_distance / num_samples))
            pbar.set_description(info_str)

        test_results[data_name]['ACC'] = num_true_positive / num_samples
        test_results[data_name]['NED'] = norm_edit_distance / num_samples

    if is_training:
        model.float()
        return test_results


if __name__ == '__main__':
    set_logging(-1)
    test_options = get_options(weight='last.pth')
    assert test_options.weight and os.path.isfile(test_options.weight), f'Error: invalid model weight path.'
    logger.info(colorstr('Test Options: ') + str(test_options))
    test(test_options)
