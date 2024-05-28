import torch
import torch.nn as nn

from models.builder import build_model


class VINet(nn.Module):
    def __init__(self, backbone, invariant_module, recognizer):
        super(VINet, self).__init__()

        self.backbone = build_model(backbone)
        self.rvi_module = build_model(invariant_module)
        self.recognition_head = build_model(recognizer)

        self.num_chars = recognizer['config']['input_dimension']

    def forward(self, images, labels_length, labels_ids, vi_conv_feature=None):

        if vi_conv_feature is None:
            ve_conv_feature = self.backbone(images)
            vi_conv_feature = self.rvi_module(ve_conv_feature)
        else:
            ve_conv_feature = None

        predicted_results, attention_map = self.recognition_head(labels_ids, vi_conv_feature)[0]

        if self.training:
            reconstructed_images = self.rvi_module.reconstruct(vi_conv_feature, ve_conv_feature,
                                                               attention_map, labels_length)

            predictions = torch.zeros(torch.sum(labels_length).data, self.num_chars).type_as(predicted_results.data)
            start_idx = 0
            for label_idx, label_length in enumerate(labels_length):
                label_length = label_length.data
                predictions[start_idx: start_idx + label_length, :] = predicted_results[label_idx, 0: label_length, :]
                start_idx += label_length

            return dict({'predictions': predictions,
                         'reconstructed_images': reconstructed_images})
        else:
            return dict({'predictions': predicted_results,
                         'vi_conv_feature': vi_conv_feature})
