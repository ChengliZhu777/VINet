import lmdb
import logging

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

        self.width, self.height = image_width, image_height
        self.prefix = prefix

        self.env = lmdb.open(path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        assert self.env, f"({prefix}) Error: Fail to create lmdb from {path}."

        with self.env.begin(write=False) as txn:
            self.num_samples = int(txn.get('num-samples'.encode()))

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
      
def create_dataloader(paths, base_character, workers, image_width, image_height, batch_size, is_train=False,
                      standard_char_path=None, prefix='Loading Data', rank=-1):
    with torch_distributed_zero_first(rank):
        datasets = []
        for dataset_format, dataset_path in paths.items():
            datasets.append(LoadImageAndLabels(dataset_path, base_character,
                                               image_width, image_height, prefix=prefix))              
