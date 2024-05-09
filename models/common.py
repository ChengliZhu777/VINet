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


class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride, padding=0, dilation=1):
        super(MaxPool, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                                     padding=padding, dilation=dilation)

    def forward(self, x):
        return self.max_pool(x)
        

class ResBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample):
        super(ResBasicBlock, self).__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bn=True)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bn=True)
        self.act, self.down_sample = nn.ReLU(inplace=True), down_sample

    def forward(self, x):
        residual = x
        out = self.conv2(self.act(self.conv1(x)))
        if self.down_sample is not None:
            residual = self.down_sample(residual)

        out += residual
        return self.act(out)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(ResBlock, self).__init__()

        ds_conv = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bn=True) \
            if in_channels != out_channels else None
        top_block = ResBasicBlock(in_channels, out_channels, ds_conv)

        blocks = [top_block]
        for _ in range(1, num_blocks):
            blocks.append(ResBasicBlock(out_channels, out_channels))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

