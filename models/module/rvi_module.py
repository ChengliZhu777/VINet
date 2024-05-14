import torch.nn as nn

from ..model_utils import parse_model


class RVIModule(nn.Module):
    def __init__(self, config):
        super(RVIModule, self).__init__()

        self.content_extractor, _ = parse_model(config, module_name='RVI-Module')

    def forward(self, x):
        return self.content_extractor(x)
