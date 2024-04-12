import torch.nn as nn

from .model_utils import parse_model


class VINet(nn.Module):
    def __init__(self, backbone, decoder):
        super(VINet, self).__init__()
        self.encoder = parse_model(backbone)
      
