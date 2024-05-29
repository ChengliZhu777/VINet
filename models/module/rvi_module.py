import torch.nn as nn

from ..model_utils import parse_model
from ..common import Reconstructor


class RVIModule(nn.Module):
    def __init__(self, config):
        super(RVIModule, self).__init__()

        self.content_extractor, _ = parse_model(config, module_name='RVI-Module')
        self.reconstructor = Reconstructor()

    def forward(self, x):
        return self.content_extractor(x)

    def reconstruct(self, vi_conv_feature, ve_conv_feature, attention_map, labels_length):
        return self.reconstructor(vi_conv_feature, ve_conv_feature, attention_map, labels_length)
