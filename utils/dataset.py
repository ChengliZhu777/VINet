import os
import lmdb
import torch
import logging

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

from utils.torch_utils import torch_distributed_zero_first

from utils.dataset_utils import lmdb_image_buffer_parse, fullwidth_to_halfwidth

logger = logging.getLogger(__name__)


def load_char_table(path):
    char_table = ['START', '\xad']
    with open(path, encoding='utf-8') as mf:
        char_table.extend(mf.read().strip())
    char_table.append('END')

    return char_table


class LoadImageAndLabels(Dataset):
    def __init__(self, path, char_table, image_width=256, image_height=32, prefix='Loading Data'):
        self.interpolation = Image.BILINEAR

        self.char_table = char_table
        self.char2id = dict(zip(self.char_table, range(len(self.char_table))))

        self.target_width, self.target_height = image_width, image_height
        self.prefix = prefix

        self.env = lmdb.open(path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        assert self.env, f"({prefix}) Error: Fail to create lmdb from {path}."

        with self.env.begin(write=False) as txn:
            self.num_samples = int(txn.get('num-samples'.encode()))

        self.width, self.height = image_width, image_height

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        index += 1
        with self.env.begin(write=False) as txn:
            image_name = 'image-%09d' % index
            image_buf = txn.get(image_name.encode())

            is_ok, image = lmdb_image_buffer_parse(image_buf)

            if not is_ok:
                logger.info(f'({self.prefix}) WARNING: Ignoring corrupted data, lmdb index {index}.')
                return self[index]

            label_name = 'label-%09d' % index
            label = str(txn.get(label_name.encode()).decode('utf-8'))

            label = fullwidth_to_halfwidth(label)
            label += '$'
            label = label.lower()

        actual_width, actual_height = image.size
        vert_mark = False if actual_width * 1.5 >= actual_height else True

        image = self.resize_image(image, is_vert=vert_mark)
        label_ids = self.convert(label)

        return image, label, len(label), label_ids, int(vert_mark), self.char2id['END']

    def resize_image(self, image, is_vert):

        if is_vert:
            image = image.transpose(Image.ROTATE_90)  # counterclockwise

        image = image.resize((self.target_width, self.target_height), self.interpolation)

        image = transforms.ToTensor()(image)
        image.sub_(0.5).div_(0.5)

        return image

    def convert(self, label):
        label_ids = []
        for i in range(len(label) - 1):
            try:
                label_ids.append(self.char2id[label[i]])
            except:
                label_ids.append(self.char2id['-'])

        return label_ids

    def get_char_table_size(self):
        return len(self.char2id)

    @staticmethod
    def collate_fn(batch):
        images, labels, label_length, label_ids, vert_marks, end_marks = zip(*batch)

        images = torch.cat([img.unsqueeze(0) for img in images], 0)  # dict -> tensor
        label_length = torch.from_numpy(np.array(label_length)).long()
        _label_ids = np.zeros([len(label_ids), len(max(label_ids, key=lambda x:len(x))) + 1])  # start from 1

        label_ids_merge, vert_marks_merge = [], []

        for i, j in enumerate(label_ids):
            _label_ids[i][1:len(j) + 1] = j
            label_ids_merge.extend(j)
            label_ids_merge.append(end_marks[i])
            vert_marks_merge.extend([vert_marks[i]] * (label_length[i] - 1))

        return images, labels, label_length, torch.from_numpy(_label_ids).long(), \
            torch.from_numpy(np.array(label_ids_merge)).long(), vert_marks, vert_marks_merge
        
