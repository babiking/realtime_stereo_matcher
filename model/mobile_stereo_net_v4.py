# MobileStereoNetV2 implementation based on: https://github.com/zjjMaiMai/TinyHITNet

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_3x3(in_c, out_c, s=1, d=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, s, d, dilation=d, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )


def conv_1x1(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )


class Difference3DCostVolume(nn.Module):
    def __init__(self, hidden_dim, max_disp, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_dim = hidden_dim
        self.max_disp = max_disp

        self.l_kernels = self.get_conv2d_kernels(reverse=False)
        self.r_kernels = self.get_conv2d_kernels(reverse=True)

    def get_conv2d_kernels(self, reverse=True):
        kernels = []
        for i in range(1, self.max_disp):
            kernel = np.zeros(shape=[self.hidden_dim, 1, 1, i + 1], dtype=np.float32)
            kernel[:, 0, 0, (0 if reverse else -1)] = 1.0
            kernel = torch.tensor(kernel, dtype=torch.float32)

            kernels.append(kernel)
        return kernels

    def forward(self, l_fmap, r_fmap):
        cost_volume = []
        for d in range(self.max_disp):
            if d == 0:
                cost_volume.append((l_fmap - r_fmap).unsqueeze(2))
            else:
                x = F.conv2d(
                    l_fmap,
                    self.l_kernels[d - 1].to(l_fmap.device),
                    stride=1,
                    padding=0,
                    groups=self.hidden_dim,
                )
                y = F.conv2d(
                    r_fmap,
                    self.r_kernels[d - 1].to(r_fmap.device),
                    stride=1,
                    padding=0,
                    groups=self.hidden_dim,
                )

                cost_volume.append(F.pad(x - y, [d, 0, 0, 0], value=1.0).unsqueeze(2))
        cost_volume = torch.concat(cost_volume, dim=2)
        return cost_volume


class ResBlock(nn.Module):
    def __init__(self, c0, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            conv_3x3(c0, c0, d=dilation),
            conv_3x3(c0, c0, d=dilation),
        )

    def forward(self, input):
        x = self.conv(input)
        return x + input


class RefineNet(nn.Module):
    def __init__(self, hidden_dim, refine_dilates):
        super().__init__()
        self.in_dim = 1 + hidden_dim
        self.hidden_dim = hidden_dim
        self.refine_dilates = refine_dilates
        self.conv0 = nn.Sequential(
            conv_3x3(self.in_dim, self.hidden_dim),
            *[ResBlock(self.hidden_dim, d) for d in refine_dilates],
            nn.Conv2d(self.hidden_dim, 1, 3, 1, 1),
        )

    def forward(self, disp, l_fmap, r_fmap=None):
        # disp: 1 x 1 x 60 x 80
        # l_fmap: 1 x 32 x 120 x 160
        # r_fmap: 1 x 32 x 120 x 160

        # disp: 1 x 1 x 120 x 160
        disp = (
            F.interpolate(disp, scale_factor=2, mode="bilinear", align_corners=False)
            * 2
        )

        x = torch.cat((disp, l_fmap), dim=1)

        # x: 1 x 1 x 120 x 160
        x = self.conv0(x)
        # x: 1 x 1 x 120 x 160, x >= 0.0
        return F.relu(disp + x)


def same_padding_conv(x, w, b, s):
    out_h = math.ceil(x.size(2) / s[0])
    out_w = math.ceil(x.size(3) / s[1])

    pad_h = max((out_h - 1) * s[0] + w.size(2) - x.size(2), 0)
    pad_w = max((out_w - 1) * s[1] + w.size(3) - x.size(3), 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    x = F.conv2d(x, w, b, stride=s)
    return x


class SameConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return same_padding_conv(x, self.weight, self.bias, self.stride)


class UpsampleBlock(nn.Module):
    def __init__(self, c0, c1):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(c0, c1, 2, 2),
            nn.LeakyReLU(0.2),
        )
        self.merge_conv = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c1, c1, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c1, c1, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input, sc):
        x = self.up_conv(input)
        x = torch.cat((x, sc), dim=1)
        x = self.merge_conv(x)
        return x


class UNetFeatureExtractor(nn.Module):
    def __init__(self, hidden_dims, up_factor):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.down_factor = len(hidden_dims) - 1
        self.up_factor = up_factor
        self.down_layers = nn.ModuleList([])
        self.up_layers = nn.ModuleList([])

        for i in range(self.down_factor + 1):
            if i == 0:
                layer = nn.Sequential(
                    nn.Conv2d(3, self.hidden_dims[0], 3, 1, 1),
                    nn.LeakyReLU(0.2),
                )
            elif i > 0 and i < self.down_factor:
                layer = nn.Sequential(
                    SameConv2d(self.hidden_dims[i - 1], self.hidden_dims[i], 4, 2),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i], 3, 1, 1),
                    nn.LeakyReLU(0.2),
                )
            elif i == self.down_factor:
                layer = nn.Sequential(
                    SameConv2d(self.hidden_dims[i - 1], self.hidden_dims[i], 4, 2),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i], 3, 1, 1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i], 3, 1, 1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i], 3, 1, 1),
                    nn.LeakyReLU(0.2),
                )
            self.down_layers.append(layer)

        for i in range(self.up_factor):
            j = self.down_factor - i

            layer = UpsampleBlock(self.hidden_dims[j], self.hidden_dims[j - 1])
            self.up_layers.append(layer)

    def forward(self, x):
        down_pyramid = []
        for i, down_layer in enumerate(self.down_layers):
            x = down_layer(x)
            down_pyramid.append(x)

        up_pyramid = [down_pyramid[-1]]
        for i, up_layer in enumerate(self.up_layers):
            j = self.down_factor - i

            y = up_layer(up_pyramid[i], down_pyramid[j - 1])
            up_pyramid.append(y)
        return up_pyramid


class Difference3DCostVolume(nn.Module):
    def __init__(self, hidden_dim, max_disp, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_dim = hidden_dim
        self.max_disp = max_disp

        self.l_kernels = self.get_conv2d_kernels(reverse=False)
        self.r_kernels = self.get_conv2d_kernels(reverse=True)

    def get_conv2d_kernels(self, reverse=True):
        kernels = []
        for i in range(1, self.max_disp):
            kernel = np.zeros(shape=[self.hidden_dim, 1, 1, i + 1], dtype=np.float32)
            kernel[:, 0, 0, (0 if reverse else -1)] = 1.0
            kernel = torch.tensor(kernel, dtype=torch.float32)

            kernels.append(kernel)
        return kernels

    def make_cost_volume_naive(self, l_fmap, r_fmap, max_disp):
        # left: 1 x 32 x 60 x 80
        # right: 1 x 32 x 60 x 80
        # max_disp: 24
        # cost_volume: 1 x 32 x 24 x 60 x 80
        n, c, h, w = l_fmap.shape

        cost_volume = torch.ones(
            size=[n, c, max_disp, h, w], dtype=l_fmap.dtype, device=l_fmap.device
        )

        # for any disparity d:
        #   cost_volume[:, :, d, :, :d] = 1.0
        #   cost_volume[:, :, d, :, d:] = left[:, :, :, d:] - right[:, :, :, :-d]
        cost_volume[:, :, 0, :, :] = l_fmap - r_fmap
        for d in range(1, max_disp):
            cost_volume[:, :, d, :, d:] = l_fmap[:, :, :, d:] - r_fmap[:, :, :, :-d]

        # cost_volume: 1 x 32 x 24 x 60 x 80
        return cost_volume

    def make_cost_volume_conv2d(self, l_fmap, r_fmap, max_disp):
        cost_volume = []
        for d in range(max_disp):
            if d == 0:
                cost_volume.append((l_fmap - r_fmap).unsqueeze(2))
            else:
                x = F.conv2d(
                    l_fmap,
                    self.l_kernels[d - 1].to(l_fmap.device),
                    stride=1,
                    padding=0,
                    groups=self.hidden_dim,
                )
                y = F.conv2d(
                    r_fmap,
                    self.r_kernels[d - 1].to(r_fmap.device),
                    stride=1,
                    padding=0,
                    groups=self.hidden_dim,
                )

                cost_volume.append(F.pad(x - y, [d, 0, 0, 0], value=1.0).unsqueeze(2))
        cost_volume = torch.concat(cost_volume, dim=2)
        return cost_volume

    def forward(self, l_fmap, r_fmap, use_naive=True):
        return (
            self.make_cost_volume_naive(l_fmap, r_fmap, self.max_disp)
            if use_naive
            else self.make_cost_volume_conv2d(l_fmap, r_fmap, self.max_disp)
        )


class MobileStereoNetV4(nn.Module):
    def __init__(
        self,
        extract_dims=[16, 24, 48, 64],
        up_factor=3,
        max_disp=192,
        refine_dilates=[1, 2, 4, 8],
        filter_dim=32,
    ):
        super().__init__()
        self.up_factor = up_factor

        self.down_factor = len(extract_dims) - 1
        self.max_disp = (max_disp + 1) // (2**self.down_factor)

        self.extract_dims = extract_dims
        self.feature_extractor = UNetFeatureExtractor(extract_dims, up_factor)

        self.cost_builder = Difference3DCostVolume(
            hidden_dim=self.extract_dims[-1], max_disp=self.max_disp
        )

        self.filter_dim = filter_dim
        self.cost_filter = nn.Sequential(
            nn.Conv3d(self.extract_dims[-1], self.filter_dim, 3, 1, 1),
            nn.BatchNorm3d(self.filter_dim),
            nn.ReLU(),
            nn.Conv3d(self.filter_dim, self.filter_dim, 3, 1, 1),
            nn.BatchNorm3d(self.filter_dim),
            nn.ReLU(),
            nn.Conv3d(self.filter_dim, self.filter_dim, 3, 1, 1),
            nn.BatchNorm3d(self.filter_dim),
            nn.ReLU(),
            nn.Conv3d(self.filter_dim, self.filter_dim, 3, 1, 1),
            nn.BatchNorm3d(self.filter_dim),
            nn.ReLU(),
            nn.Conv3d(self.filter_dim, 1, 3, 1, 1),
        )

        self.refine_dilates = refine_dilates
        self.refine_layers = nn.ModuleList(
            [
                RefineNet(
                    hidden_dim=self.extract_dims[self.down_factor - 1 - i],
                    refine_dilates=self.refine_dilates,
                )
                for i in range(self.up_factor)
            ]
        )

    def forward(self, l_img, r_img, is_train=True):
        # l_fmaps:
        #   [1] 1 x 32 x 60 x 80, i.e. 8x downsample
        #   [2] 1 x 32 x 120 x 160
        #   [3] 1 x 32 x 240 x 320
        #   [4] 1 x 32 x 480 x 640
        l_fmaps = self.feature_extractor(l_img)
        r_fmaps = self.feature_extractor(r_img)

        # max_disp: 192 // 8 = 24
        # cost_volume: 1 x 32 x 24 x 60 x 80
        cost_volume = self.cost_builder(l_fmaps[0], r_fmaps[0], use_naive=is_train)

        # cost_volume: 1 x 24 x 60 x 80
        cost_volume = self.cost_filter(cost_volume).squeeze(1)

        x = F.softmax(cost_volume, dim=1)
        d = torch.arange(0, self.max_disp, device=x.device, dtype=x.dtype)
        # x: 1 x 1 x 60 x 80
        x = torch.sum(x * d.view(1, -1, 1, 1), dim=1, keepdim=True)

        multi_scale = [x] if is_train else []
        for i, refine in enumerate(self.refine_layers):
            # x: 1 x 1 x 60 x 80
            # l_fmaps[i + 1]: 1 x 32 x 120 x 160
            x = refine(x, l_fmaps[i + 1], r_fmaps[i + 1])

            if is_train or i == len(self.refine_layers) - 1:
                multi_scale.append(x)
        return (multi_scale, l_fmaps, r_fmaps) if is_train else multi_scale
