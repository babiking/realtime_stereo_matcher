from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        deconv=False,
        is_3d=False,
        bn=True,
        relu=True,
        **kwargs
    ):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(
                    in_channels, out_channels, bias=False, **kwargs
                )
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(
                    in_channels, out_channels, bias=False, **kwargs
                )
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU(0.2)(x)  # , inplace=True)
        return x
