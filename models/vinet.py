import torch.nn as nn

from .model_utils import parse_model
from models.builder import build_model


class VINet(nn.Module):
    def __init__(self, backbone, decoder):
        super(VINet, self).__init__()
        self.encoder = build_model(backbone)
        self.rvi_module = build_model(invariant_module)
        self.recognition_head = build_model(recognizer)

    def forward(self, images, labels_length, labels_ids, vi_conv_feature=None):

        if vi_conv_feature is None:
            ve_conv_feature = self.backbone(images)
            vi_conv_feature = self.rvi_module(ve_conv_feature)
        else:
            ve_conv_feature = None

        decoder_output, attention_map = self.recognition_head(labels_ids, vi_conv_feature)[0]
