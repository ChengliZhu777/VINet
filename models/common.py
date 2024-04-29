import torch.nn as nn


def auto_padding(kernel_size, padding=None):
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [ks // 2 for ks in kernel_size]
    return padding


class Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride,
                 padding=None, groups=1, bias=True, bn=False, act=None):
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride,
                              padding=auto_padding(kernel_size, padding), groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(output_channels) if bn else None

        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'ReLU-inplace':
            self.act = nn.ReLU(inplace=True)
        elif act == 'SiLU':
            self.act = nn.SiLU()
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x
      
