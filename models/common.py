import math
import copy
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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


class Embeddings(nn.Module):
    def __init__(self, char_table_size, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(char_table_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)  # batch-size, num_chars, d_model(256)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=7000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div_term), torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, label_embeddings):
        x = torch.zeros(label_embeddings.shape).cuda()
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)

        return self.dropout(x)
        

class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


def attention(query, key, value, mask=None, dropout=None):
    k_dim = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(k_dim)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    p_attn = scores.softmax(dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, compress_attention=True):
        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0, f'Error: embedding-dim must be multiple of num-heads.'
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.compress_attention = compress_attention

        self.linear_layers = clones(nn.Linear(embed_dim, embed_dim), 4)
        self.dropout = nn.Dropout(p=dropout)
        self.compressor = nn.Linear(num_heads, 1)

    def forward(self, query, key, value, mask=None):

        batch_size = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)

        query, key, value = [linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                             for linear, x in zip(self.linear_layers, (query, key, value))]
        x, attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        if self.compress_attention:
            attn = attn.permute(0, 2, 3, 1).contiguous()
            attn = self.compressor(attn).permute(0, 3, 1, 2).contiguous()

        return self.linear_layers[-1](x), attn
        

