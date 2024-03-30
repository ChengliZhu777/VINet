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
            label += '$'
            label = label.lower()

        actual_width, actual_height = image.size
        if actual_width * 1.5 >= actual_height:
            vert_mark, org2horiz = False, False
        else:
            vert_mark, org2horiz = True, True

        image = self.resize_image(image, is_vert=vert_mark, is_rotate=org2horiz)
        label_length, label_ids = self.convert(label)

        return image, label, label_length, label_ids, self.char2id['END'], int(vert_mark)

    def resize_image(self, image, is_vert, is_rotate):

        if is_vert:
            image = image.transpose(Image.ROTATE_90)  # counterclockwise

        image = image.resize((self.target_width, self.target_height), self.interpolation)
      
def create_dataloader(paths, base_character, workers, image_width, image_height, batch_size, is_train=False,
                      standard_char_path=None, prefix='Loading Data', rank=-1):
    with torch_distributed_zero_first(rank):
        datasets = []
        for dataset_format, dataset_path in paths.items():
            datasets.append(LoadImageAndLabels(dataset_path, base_character,
                                               image_width, image_height, prefix=prefix))              


