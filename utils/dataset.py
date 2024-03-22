import logging

from PIL import Image
from torch.utils.data import Dataset

from utils.torch_utils import torch_distributed_zero_first

logger = logging.getLogger(__name__)


class LoadImageAndLabels(Dataset):
    def __init__(self, path, base_character, image_width=256, image_height=32, prefix='Loading Data'):
        self.interpolation = Image.BILINEAR
      
def create_dataloader(paths, base_character, workers, image_width, image_height, batch_size, is_train=False,
                      standard_char_path=None, prefix='Loading Data', rank=-1):
    with torch_distributed_zero_first(rank):
        datasets = []
        for dataset_format, dataset_path in paths.items():
            datasets.append(LoadImageAndLabels(dataset_path, base_character,
                                               image_width, image_height, prefix=prefix))              
