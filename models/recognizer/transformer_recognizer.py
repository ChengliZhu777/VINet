import torch.nn as nn

from ..model_utils import parse_model


class TransformerRecognitionHead(nn.Module):
    def __init__(self, config):
        super(TransformerRecognitionHead, self).__init__()

        self.model, self.save = parse_model(config)

    def forward(self, x, conv_feature):
        y, outputs = [], []
        for module in self.model:
            if module.module_from != -1:
                x = y[module.module_from] if isinstance(module.module_from, int) else \
                    [x if j == -1 else y[j] for j in module.module_from[::-1]]
            if module.module_index == 3:
                x = module(x, conv_feature)
            else:
                x = module(x)

            y.append(x if module.module_index in self.save else None)

        return [x]
    
