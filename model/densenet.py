#!/usr/bin/env python3
"""Implement DenseNet architecture.
"""
import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()

        if in_channels != out_channels:
            self.bottleneck = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        if hasattr(self, bottleneck):
            x = self.bottleneck(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)

        return x

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate, layers):
        super(DenseBlock, self).__init__()

        self.layers = layers

        for l in range(1, self.layers + 1):
            channels = growth_rate * (l - 1) + in_channels # k * (l - 1) + k0
            setattr(self,
                    f"layer{l}",
                    DenseLayer(channels, channels + growth_rate))

    def forward(self, x):
        out = [x]

        for l in range(1, self.layers + 1):
            layer = getattr(self, f"layer{l}")
            out.append(layer(torch.cat(out, axis=1)))

        return torch.cat(out, axis=1)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
 
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.pool = nn.AvgPool2d(out_channels, out_channels, 2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)

        return x

class DenseNet(nn.Module):
    """
    """
    depth_blocks = {
        40: [12, 12, 12],
        100: [32, 32, 32],
        250: [82, 82, 82],
        190: [62, 62, 62],
        121: [6, 12, 24, 16],
        169: [6, 12, 32, 32],
        201: [6, 12, 48, 32],
        161: [6, 12, 36, 24]}

    def __init__(self, depth, growth_rate, bottleneck=True, compression=0.5):
        super(DenseNet, self).__init__()

        try:
            self.blocks = self.depth_blocks[depth]
        except KeyError:
            raise RuntimeError(
                ("Unrecognized depth. Known values are: "
                 + ", ".join(self.depth_blocks.keys())))

        self.conv0 = nn.Conv2d(3, 3, 7, stride=2)
        self.pool0 = nn.MaxPool2d(3, stride=2)

        in_channels = 3

        for b in range(1, len(self.blocks)):
            layers = self.blocks[b]
            out_channels = growth_rate * layers + in_channels

            setattr(self, f"block{b}", DenseBlock(
                in_channels, out_channels, growth_rate, layers))
            setattr(self, f"trans{b}", TransitionLayer(
                out_channels, int(out_channels * compression))

            in_channels = int(out_channels * compression)

        layers = self.blocks[b+1]
        out_channels = growth_rate * layers + in_channels
        setattr(self, f"block{b+1}", DenseBlock(in_channels, out_channels))
        
        self.pool = nn.AvgPool2d()
        self.fn = nn.Linear(1000)

    def forward(self, x):
        x = self.conv0(x)
        x = self.pool0(x)

        for b in range(1, len(self.layers)):
            block = getattr(self, f"block{b}")
            x = block(x)

            trans = getattr(self, f"trans{b}")
            x = trans(x)

        block = getattr(self, f"block{b+1}")
        x = block(x)

        x = self.pool(x)
        x = self.fn(x)

        return x
