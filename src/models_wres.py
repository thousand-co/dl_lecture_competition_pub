import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import numpy as np
import math
from einops import rearrange


# Residual blockの定義(Bottleneck Building Block: 1×1、3×3、1×1 の3つの畳み込み層で構成。ResNet-50、ResNet-101、ResNet-152 の residual block として使用。)
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv1d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=False
    )


class DoBottleneck(nn.Module):
    expansion = 4  # 出力のチャンネル数を入力のチャンネル数の何倍に拡大するか

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout1d(0.3)
        self.conv2 = conv3x3(channels , channels * self.expansion)
        self.bn2 = nn.BatchNorm1d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # 入力と出力のチャンネル数が異なる場合、x をダウンサンプリングする。
        # 1×1 の畳み込みを利用して、線形変換を行い、形状を一致させる。
        if in_channels != channels * self.expansion:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, channels * self.expansion, stride),
                nn.BatchNorm1d(channels * self.expansion),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out += self.shortcut(x)

        out = self.relu(out)

        return out


# (GAP:Global Average Pooling)
# 全体のネットワークの定義
class WideResNet(nn.Module):
    def __init__(self, block, layers, num_classes: int, seq_len: int, in_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = 256
        self.conv1 = nn.Conv1d(
            in_channels, self.out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Residual Block は、直前で Max Pooling でダウンサンプリングを行っているので、畳み込みによるダウンサンプリングは不要。
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) 
        self.conv2 = self._make_layer(block, 64, layers[0], stride=1)
        # stride=2 で畳み込みを行い、ダウンサンプリング。
        self.conv3 = self._make_layer(block, 128, layers[1], stride=2)
        self.conv4 = self._make_layer(block, 256, layers[2], stride=2)
        self.conv5 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 重みを初期化する。He initialization (torch.nn.init.kaimingnormal) を使用。
        # Batch Normalization 層の初期化は重み1、バイアス0で初期化。
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channels, blocks, stride):
        layers = []

        # 最初の Residual Block
        layers.append(block(self.in_channels, channels, stride))

        # 残りの Residual Block
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x