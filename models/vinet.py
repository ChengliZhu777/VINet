import torch.nn as nn

from .model_utils import parse_model
from models.builder import build_model


class VINet(nn.Module):
    def __init__(self, backbone, decoder):
        super(VINet, self).__init__()
        self.encoder = build_model(backbone)
      
