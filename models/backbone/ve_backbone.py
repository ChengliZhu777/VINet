import torch.nn as nn

from ..model_utils import parse_model
from ..orn.models import upgrade_to_orn


class VEBackbone(nn.Module):
    def __init__(self, config):
        super(VEBackbone, self).__init__()

        self.model, self.save_layers = parse_model(config, module_name='VEBackbone')

        upgrade_to_orn(self.model, num_orientation=4, scale_factor=2,
                       classifier=None, features=None, invariant_encoding='align')

    def forward(self, x):
        return self.model(x)
