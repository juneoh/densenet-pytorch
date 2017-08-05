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
        if self.bottleneck:
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
                    f"layer{i}",
                    DenseLayer(channels, channels + growth_rate))

    def forward(self, x):
        out = [x]

        for l in range(1, self.layers + 1):
            layer = getattr(self, f"layer{i}")
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
    def __init__(self, depth, growth_rate, bottleneck=True, compression=0.5):
        super(DenseNet, self).__init__()

    def forward(self, x):
        pass
