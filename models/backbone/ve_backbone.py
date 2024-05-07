import torch.nn as nn

from .resnet import ResNet
from ..model_utils import parse_model


class VEBackbone(nn.Module):
    def __init__(self, config):
        super(VEBackbone, self).__init__()

        self.model, self.save_layers = parse_model(config, module_name='VEBackbone')

    def forward(self, x):
        return self.model(x)
        
